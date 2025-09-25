// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/types.h>
#include <mutex>
#include <vector>

#include "src/torchcodec/_core/BetaCudaDeviceInterface.h"

#include "src/torchcodec/_core/DeviceInterface.h"
#include "src/torchcodec/_core/FFMPEGCommon.h"
#include "src/torchcodec/_core/NVDECCache.h"

#include <cuda_runtime.h> // For cudaStreamSynchronize
#include "src/torchcodec/_core/nvcuvid_include/cuviddec.h"
#include "src/torchcodec/_core/nvcuvid_include/nvcuvid.h"

extern "C" {
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {

namespace {

// Register the BETA CUDA interface with 'beta' variant
static bool g_cuda_beta = registerDeviceInterface(
    DeviceInterfaceKey(torch::kCUDA, "beta"),
    [](const torch::Device& device) {
      return new BetaCudaDeviceInterface(device);
    });

static int CUDAAPI
HandleVideoSequence(void* pUserData, CUVIDEOFORMAT* pVideoFormat) {
  BetaCudaDeviceInterface* decoder =
      static_cast<BetaCudaDeviceInterface*>(pUserData);
  return decoder->handleVideoSequence(pVideoFormat);
}

static int CUDAAPI
HandlePictureDecode(void* pUserData, CUVIDPICPARAMS* pPicParams) {
  BetaCudaDeviceInterface* decoder =
      static_cast<BetaCudaDeviceInterface*>(pUserData);
  return decoder->handlePictureDecode(pPicParams);
}

static UniqueCUvideodecoder createDecoder(CUVIDEOFORMAT* video_format) {
  auto codec_type = video_format->codec;
  unsigned height = video_format->coded_height;
  unsigned width = video_format->coded_width;
  auto num_decode_surfaces = video_format->min_num_decode_surfaces;
  auto chroma_format = video_format->chroma_format;
  auto bit_depth_luma_minus8 = video_format->bit_depth_luma_minus8;

  if (num_decode_surfaces == 0) {
    num_decode_surfaces = 20;
  }

  // Check decoder capabilities
  auto caps = CUVIDDECODECAPS{};
  caps.eCodecType = codec_type;
  caps.eChromaFormat = chroma_format;
  caps.nBitDepthMinus8 = bit_depth_luma_minus8;
  CUresult caps_result = cuvidGetDecoderCaps(&caps);
  TORCH_CHECK(
      caps_result == CUDA_SUCCESS, "Failed to get decoder caps: ", caps_result);

  TORCH_CHECK(
      caps.bIsSupported,
      "Codec configuration not supported on this GPU. "
      "Codec: ",
      static_cast<int>(codec_type),
      ", chroma format: ",
      static_cast<int>(chroma_format),
      ", bit depth: ",
      bit_depth_luma_minus8 + 8);

  TORCH_CHECK(
      width >= caps.nMinWidth && height >= caps.nMinHeight,
      "Video is too small in at least one dimension. Provided: ",
      width,
      "x",
      height,
      " vs supported:",
      caps.nMinWidth,
      "x",
      caps.nMinHeight);

  TORCH_CHECK(
      width <= caps.nMaxWidth && height <= caps.nMaxHeight,
      "Video is too large in at least one dimension. Provided: ",
      width,
      "x",
      height,
      " vs supported:",
      caps.nMaxWidth,
      "x",
      caps.nMaxHeight);

  TORCH_CHECK(
      width * height / 256 <= caps.nMaxMBCount,
      "Video is too large (too many macroblocks). "
      "Provided (width * height / 256): ",
      width * height / 256,
      " vs supported:",
      caps.nMaxMBCount);

  // Create new decoder
  CUVIDDECODECREATEINFO decoder_info;
  memset(&decoder_info, 0, sizeof(CUVIDDECODECREATEINFO));

  decoder_info.bitDepthMinus8 = bit_depth_luma_minus8;
  decoder_info.ChromaFormat = chroma_format;
  decoder_info.CodecType = codec_type;
  decoder_info.ulHeight = height;
  decoder_info.ulWidth = width;
  decoder_info.ulMaxHeight = height;
  decoder_info.ulMaxWidth = width;
  decoder_info.ulTargetHeight =
      video_format->display_area.bottom - video_format->display_area.top;
  decoder_info.ulTargetWidth =
      video_format->display_area.right - video_format->display_area.left;
  decoder_info.ulNumDecodeSurfaces = num_decode_surfaces;
  decoder_info.ulNumOutputSurfaces = 2;
  decoder_info.ulCreationFlags = cudaVideoCreate_PreferCUVID;
  decoder_info.vidLock = nullptr;

  auto& area = decoder_info.display_area;
  area.left = video_format->display_area.left;
  area.right = video_format->display_area.right;
  area.top = video_format->display_area.top;
  area.bottom = video_format->display_area.bottom;

  CUvideodecoder raw_decoder;
  CUresult result = cuvidCreateDecoder(&raw_decoder, &decoder_info);
  TORCH_CHECK(
      result == CUDA_SUCCESS, "Failed to create NVDEC decoder: ", result);

  // Wrap in unique_ptr with custom deleter
  return UniqueCUvideodecoder(raw_decoder, CUvideoDecoderDeleter{});
}

} // namespace

BetaCudaDeviceInterface::BetaCudaDeviceInterface(const torch::Device& device)
    : DeviceInterface(device) {
  TORCH_CHECK(g_cuda_beta, "BetaCudaDeviceInterface was not registered!");
  TORCH_CHECK(
      device_.type() == torch::kCUDA, "Unsupported device: ", device_.str());

  // Initialize frame buffer for B-frame reordering
  // TODONVDEC: init size should probably be min_num_decode_surfaces from video
  // format
  frameBuffer_.resize(4);
}

BetaCudaDeviceInterface::~BetaCudaDeviceInterface() {
  // Clean up any remaining frames in the buffer
  {
    std::lock_guard<std::mutex> lock(frameBufferMutex_);
    for (auto& slot : frameBuffer_) {
      slot.occupied = false;
      slot.pts = -1;
    }
  }

  // Return decoder to cache if we have one
  if (decoder_) {
    NVDECCache::GetCache(device_.index())
        .returnDecoder(&videoFormat_, std::move(decoder_));
  }

  // Clean up video parser
  if (videoParser_) {
    cuvidDestroyVideoParser(videoParser_);
    videoParser_ = nullptr;
  }

  parserCreated_ = false;
}

std::optional<const AVCodec*> BetaCudaDeviceInterface::findCodec(
    const AVCodecID& codecId) {
  // We bypass FFmpeg codec selection entirely
  // We'll handle the codec selection in our own NVDEC initialization
  (void)codecId; // Suppress unused parameter warning
  return std::nullopt;
}

void BetaCudaDeviceInterface::initializeContext(AVCodecContext* codecContext) {
  // Don't set hw_device_ctx - we handle decoding directly with NVDEC SDK
  // Just ensure CUDA context exists for PyTorch tensors
  torch::Tensor dummyTensor = torch::empty(
      {1}, torch::TensorOptions().dtype(torch::kUInt8).device(device_));

  // Convert FFmpeg codec ID to NVDEC codec enum
  cudaVideoCodec nvCodec;
  switch (codecContext->codec_id) {
    case AV_CODEC_ID_H264:
      nvCodec = cudaVideoCodec_H264;
      break;
    default:
      TORCH_CHECK(
          false,
          "Unsupported codec for BETA CUDA interface: ",
          avcodec_get_name(codecContext->codec_id));
  }

  // TODONVDEC figure out why this is needed and where videoFormat_ is actually
  // used. Maybe this isn't needed at all since this gets overridden in
  // handleVideoSequence?
  memset(&videoFormat_, 0, sizeof(videoFormat_));
  videoFormat_.codec = nvCodec;
  videoFormat_.coded_width = 0; // Will be set when we get the first frame
  videoFormat_.coded_height = 0; // Will be set when we get the first frame
  videoFormat_.chroma_format = cudaVideoChromaFormat_420;
  videoFormat_.bit_depth_luma_minus8 = 0;
  videoFormat_.bit_depth_chroma_minus8 = 0;

  createVideoParser();
}

void BetaCudaDeviceInterface::initializeWithStream(AVStream* avStream) {
  TORCH_CHECK(avStream != nullptr, "AVStream cannot be null");
  timeBase_ = avStream->time_base;
  fallbackFrameRate_ = avStream->r_frame_rate;

  TORCH_CHECK(
      avStream->codecpar->codec_id == AV_CODEC_ID_H264,
      "Can only do H264 for now");

  const AVCodecParameters* codecpar = avStream->codecpar;
  TORCH_CHECK(codecpar != nullptr, "CodecParameters cannot be null");

  // Only initialize BSF for H264
  if (codecpar->codec_id != AV_CODEC_ID_H264) {
    return;
  }

  const AVBitStreamFilter* avBSF = av_bsf_get_by_name("h264_mp4toannexb");
  TORCH_CHECK(
      avBSF != nullptr, "Failed to find h264_mp4toannexb bitstream filter");

  AVBSFContext* avBSFContext = nullptr;
  int retVal = av_bsf_alloc(avBSF, &avBSFContext);
  TORCH_CHECK(
      retVal >= AVSUCCESS,
      "Failed to allocate bitstream filter: ",
      getFFMPEGErrorStringFromErrorCode(retVal));

  bitstreamFilter_.reset(avBSFContext);

  retVal = avcodec_parameters_copy(bitstreamFilter_->par_in, codecpar);
  TORCH_CHECK(
      retVal >= AVSUCCESS,
      "Failed to copy codec parameters: ",
      getFFMPEGErrorStringFromErrorCode(retVal));

  retVal = av_bsf_init(bitstreamFilter_.get());
  TORCH_CHECK(
      retVal == AVSUCCESS,
      "Failed to initialize bitstream filter: ",
      getFFMPEGErrorStringFromErrorCode(retVal));
}

void BetaCudaDeviceInterface::createVideoParser() {
  if (parserCreated_) {
    // TODONVDEC - is this needed?
    return;
  }

  // Set up video parser parameters
  CUVIDPARSERPARAMS parserParams = {};
  parserParams.CodecType = videoFormat_.codec;
  // Set to dummy value initially, sequence callback will update this
  // as recommended by NVDEC docs
  parserParams.ulMaxNumDecodeSurfaces = 1;
  parserParams.ulClockRate = 1000;
  parserParams.ulErrorThreshold = 0;
  parserParams.ulMaxDisplayDelay = 1;
  parserParams.pUserData = this;
  parserParams.pfnSequenceCallback = HandleVideoSequence;
  parserParams.pfnDecodePicture = HandlePictureDecode;
  parserParams.pfnDisplayPicture =
      nullptr; // Like DALI - we handle display manually

  CUresult result = cuvidCreateVideoParser(&videoParser_, &parserParams);
  TORCH_CHECK(
      result == CUDA_SUCCESS, "Failed to create video parser: ", result);

  parserCreated_ = true;
}

// This callback is called by the parser within cuvidParseVideoData, either when
// the parser encounters the start of the headers, or when "there is a change in
// the sequence" - which, I assume means a change in any one of CUVIDEOFORMAT
// fields?
int BetaCudaDeviceInterface::handleVideoSequence(CUVIDEOFORMAT* pVideoFormat) {
  TORCH_CHECK(pVideoFormat != nullptr, "Invalid video format");

  videoFormat_ = *pVideoFormat;

  if (!decoder_) {
    decoder_ = NVDECCache::GetCache(device_.index()).getDecoder(pVideoFormat);

    if (!decoder_) {
      decoder_ = createDecoder(pVideoFormat);
    }

    TORCH_CHECK(decoder_, "Failed to get or create decoder");
  }

  return static_cast<int>(pVideoFormat->min_num_decode_surfaces);
}

// Parser triggers this callback when bitstream data for one frame is ready
int BetaCudaDeviceInterface::handlePictureDecode(CUVIDPICPARAMS* pPicParams) {
  // Like DALI: if we're flushing, don't process new decode operations
  if (flush_) {
    return 0;
  }

  TORCH_CHECK(pPicParams != nullptr, "Invalid picture parameters");
  TORCH_CHECK(decoder_, "Decoder not initialized before picture decode");

  CUresult result = cuvidDecodePicture(
      static_cast<CUvideodecoder>(decoder_.get()), pPicParams);

  if (result != CUDA_SUCCESS) {
    return 0;
  }

  // Like DALI: manually create display info and handle picture display directly
  CUVIDPARSERDISPINFO dispInfo = {};
  dispInfo.picture_index = pPicParams->CurrPicIdx;
  dispInfo.progressive_frame = !pPicParams->field_pic_flag;
  dispInfo.top_field_first = pPicParams->bottom_field_flag ^ 1;
  dispInfo.repeat_first_field = 0;

  // TODONVDEC the pipe may be empty, handle that.
  TORCH_CHECK(
      !packetsPtsQueue.empty(), "PTS queue is empty when decoding a frame");

  // Simple PTS assignment: use the PTS directly from the packet queue in FIFO
  // order Each decoded frame gets the PTS from the corresponding packet in
  // decode order
  int64_t framePts = packetsPtsQueue.front();
  packetsPtsQueue.pop();

  // Set the PTS in the display info
  dispInfo.timestamp = framePts;

  // Buffer frame for B-frame reordering (like DALI)
  std::lock_guard<std::mutex> lock(frameBufferMutex_);
  FrameBufferSlot* slot = findEmptySlot();
  slot->dispInfo = dispInfo;
  slot->pts = framePts;

  slot->occupied = true;
  return 1;
}

int BetaCudaDeviceInterface::sendPacket(ReferenceAVPacket& packet) {
  if (!parserCreated_) {
    return AVERROR(EINVAL);
  }

  CUVIDSOURCEDATAPACKET cudaPacket = {};

  if (packet.get() && packet->data && packet->size > 0) {
    // Regular packet with data
    cudaPacket.payload = packet->data;
    cudaPacket.payload_size = packet->size;
    cudaPacket.flags = CUVID_PKT_TIMESTAMP;
    cudaPacket.timestamp = packet->pts;

    // Like DALI: store PTS in queue to assign to frames as they come out
    packetsPtsQueue.push(packet->pts);

  } else {
    // End of stream packet
    cudaPacket.flags = CUVID_PKT_ENDOFSTREAM;
    eofSent_ = true;
  }

  CUresult result = cuvidParseVideoData(videoParser_, &cudaPacket);
  if (result != CUDA_SUCCESS) {
    return AVERROR_EXTERNAL;
  }

  return AVSUCCESS;
}

// TODONVDEC P0: cleanup this raw pointer / reference monstruosity.
ReferenceAVPacket* BetaCudaDeviceInterface::applyBSF(
    ReferenceAVPacket& packet,
    [[maybe_unused]] AutoAVPacket& filteredAutoPacket,
    ReferenceAVPacket& filteredPacket) {
  if (!bitstreamFilter_) {
    return &packet;
  }
  int retVal = av_bsf_send_packet(bitstreamFilter_.get(), packet.get());
  TORCH_CHECK(
      retVal >= AVSUCCESS,
      "Failed to send packet to bitstream filter: ",
      getFFMPEGErrorStringFromErrorCode(retVal));

  retVal = av_bsf_receive_packet(bitstreamFilter_.get(), filteredPacket.get());
  TORCH_CHECK(
      retVal >= AVSUCCESS,
      "Failed to receive packet from bitstream filter: ",
      getFFMPEGErrorStringFromErrorCode(retVal));

  return &filteredPacket;
}

int BetaCudaDeviceInterface::receiveFrame(
    UniqueAVFrame& frame,
    int64_t desiredPts) {
  std::lock_guard<std::mutex> lock(frameBufferMutex_);

  FrameBufferSlot* slot = findFrameWithExactPts(desiredPts);
  if (slot == nullptr) {
    // TODONVDEC: Need to handle case where frame buffer is full!!!!!
    return AVERROR(EAGAIN);
  }

  CUVIDPARSERDISPINFO dispInfo = slot->dispInfo;

  slot->occupied = false;
  slot->pts = -1;

  CUdeviceptr framePtr = 0;
  unsigned int pitch = 0;
  CUVIDPROCPARAMS procParams = {};
  procParams.progressive_frame = dispInfo.progressive_frame;
  procParams.top_field_first = dispInfo.top_field_first;
  procParams.unpaired_field = dispInfo.repeat_first_field < 0;

  CUresult result = cuvidMapVideoFrame(
      static_cast<CUvideodecoder>(decoder_.get()),
      dispInfo.picture_index,
      &framePtr,
      &pitch,
      &procParams);

  if (result != CUDA_SUCCESS) {
    return AVERROR_EXTERNAL;
  }

  // Convert the NVDEC frame to AVFrame, passing the correct PTS
  frame = convertCudaFrameToAVFrame(framePtr, pitch, dispInfo, timeBase_);

  // Unmap the frame
  cuvidUnmapVideoFrame(static_cast<CUvideodecoder>(decoder_.get()), framePtr);

  return AVSUCCESS;
}

void BetaCudaDeviceInterface::flush() {
  // Set flush flag like DALI to prevent new decode operations
  flush_ = true;

  // Send EOS packet to drain decoder like DALI does
  if (parserCreated_ && !eofSent_) {
    CUVIDSOURCEDATAPACKET cudaPacket = {};
    cudaPacket.flags = CUVID_PKT_ENDOFSTREAM;
    CUresult result = cuvidParseVideoData(videoParser_, &cudaPacket);
    if (result == CUDA_SUCCESS) {
      eofSent_ = true;
    }
  }

  // Clear flush flag like DALI does
  flush_ = false;

  {
    std::lock_guard<std::mutex> lock(frameBufferMutex_);
    for (auto& slot : frameBuffer_) {
      slot.occupied = false;
      slot.pts = -1;
    }
  }

  while (!packetsPtsQueue.empty()) {
    packetsPtsQueue.pop();
  }

  // Synchronize CUDA stream to ensure all operations complete
  // TODONVDEC make sure this is syncing the right stream, not necessarily
  // stream 0
  cudaStreamSynchronize(0);

  // Reset EOF flag so we can decode more (like DALI does)
  eofSent_ = false;
}

UniqueAVFrame BetaCudaDeviceInterface::convertCudaFrameToAVFrame(
    CUdeviceptr framePtr,
    unsigned int pitch,
    const CUVIDPARSERDISPINFO& dispInfo,
    const AVRational& timeBase) {
  TORCH_CHECK(framePtr != 0, "Invalid CUDA frame pointer");

  // Get frame dimensions from video format display area (not coded dimensions)
  // This matches DALI's approach and avoids padding issues
  int width = videoFormat_.display_area.right - videoFormat_.display_area.left;
  int height = videoFormat_.display_area.bottom - videoFormat_.display_area.top;

  TORCH_CHECK(width > 0 && height > 0, "Invalid frame dimensions");
  TORCH_CHECK(
      pitch >= static_cast<unsigned int>(width), "Pitch must be >= width");

  // Allocate AVFrame
  UniqueAVFrame avFrame(av_frame_alloc());
  TORCH_CHECK(avFrame.get() != nullptr, "Failed to allocate AVFrame");

  // Set frame properties
  avFrame->width = width;
  avFrame->height = height;
  avFrame->format = AV_PIX_FMT_CUDA; // Indicate this is GPU data
  avFrame->pts =
      dispInfo.timestamp; // This PTS was set correctly by handlePictureDisplay

  // Calculate frame duration from NVDEC frame rate, fallback frame rate, and
  // stream timebase
  AVRational effectiveFrameRate = {0, 0};

  // First try NVDEC frame rate
  if (videoFormat_.frame_rate.numerator > 0 &&
      videoFormat_.frame_rate.denominator > 0) {
    effectiveFrameRate.num = videoFormat_.frame_rate.numerator;
    effectiveFrameRate.den = videoFormat_.frame_rate.denominator;
  }
  // Fallback to FFmpeg frame rate if NVDEC frame rate is unavailable
  else if (fallbackFrameRate_.num > 0 && fallbackFrameRate_.den > 0) {
    effectiveFrameRate = fallbackFrameRate_;
  }

  if (effectiveFrameRate.num > 0 && effectiveFrameRate.den > 0 &&
      timeBase.num > 0 && timeBase.den > 0) {
    // Duration in seconds = frame_rate.den / frame_rate.num
    // Duration in timebase units = (duration_seconds * timeBase.den) /
    // timeBase.num = (frame_rate.den * timeBase.den) / (frame_rate.num *
    // timeBase.num)
    setDuration(
        avFrame,
        (int64_t)((effectiveFrameRate.den * timeBase.den) /
                  (effectiveFrameRate.num * timeBase.num)));
  } else {
    setDuration(avFrame, 0); // Unknown duration
  }

  // Set color space and color range from NVDEC video format (like DALI does)
  // This is crucial for proper color conversion!

  // Map NVDEC matrix coefficients to FFmpeg color space
  switch (videoFormat_.video_signal_description.matrix_coefficients) {
    case 1: // ITU-R BT.709
      avFrame->colorspace = AVCOL_SPC_BT709;
      break;
    case 5: // ITU-R BT.470-2 System B, G (BT.601 PAL)
    case 6: // ITU-R BT.601-6 NTSC
      avFrame->colorspace = AVCOL_SPC_SMPTE170M; // BT.601
      break;
    default:
      // Default to BT.601 for unknown coefficients
      avFrame->colorspace = AVCOL_SPC_SMPTE170M;
      break;
  }

  // Set color range from full range flag
  if (videoFormat_.video_signal_description.video_full_range_flag) {
    avFrame->color_range = AVCOL_RANGE_JPEG; // Full range (0-255)
  } else {
    avFrame->color_range = AVCOL_RANGE_MPEG; // Limited range (16-235)
  }

  // For NVDEC output in NV12 format, we need to set up the data pointers
  // The framePtr points to the beginning of the NV12 data
  avFrame->data[0] = reinterpret_cast<uint8_t*>(framePtr); // Y plane
  avFrame->data[1] = reinterpret_cast<uint8_t*>(
      framePtr + (pitch * height)); // UV plane (using pitch, not width)
  avFrame->data[2] = nullptr;
  avFrame->data[3] = nullptr;

  // Set line sizes for NV12 format using the actual NVDEC pitch
  avFrame->linesize[0] = pitch; // Y plane stride (use actual pitch from NVDEC)
  avFrame->linesize[1] = pitch; // UV plane stride (use actual pitch from NVDEC)
  avFrame->linesize[2] = 0;
  avFrame->linesize[3] = 0;

  return avFrame;
}

void BetaCudaDeviceInterface::convertAVFrameToFrameOutput(
    const VideoStreamOptions& videoStreamOptions,
    const AVRational& timeBase,
    UniqueAVFrame& avFrame,
    FrameOutput& frameOutput,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  TORCH_CHECK(
      avFrame->format == AV_PIX_FMT_CUDA,
      "Expected CUDA format frame from BETA CUDA interface");

  // TODONVDEC P1: we use the 'default' cuda device interface for color
  // conversion. That's a temporary hack to make things work. we should abstract
  // the color conversion stuff separately.
  if (!defaultCudaInterface_) {
    auto cudaDevice = torch::Device(torch::kCUDA);
    defaultCudaInterface_ =
        std::unique_ptr<DeviceInterface>(createDeviceInterface(cudaDevice));
    AVCodecContext dummyCodecContext = {};
    defaultCudaInterface_->initializeContext(&dummyCodecContext);
  }

  defaultCudaInterface_->convertAVFrameToFrameOutput(
      videoStreamOptions,
      timeBase,
      avFrame,
      frameOutput,
      preAllocatedOutputTensor);
}

// Helper method to find an empty slot in frame buffer (like DALI's
// FindEmptySlot)
BetaCudaDeviceInterface::FrameBufferSlot*
BetaCudaDeviceInterface::findEmptySlot() {
  for (auto& slot : frameBuffer_) {
    if (!slot.occupied) {
      return &slot;
    }
  }
  // If no empty slots, expand buffer like DALI does
  frameBuffer_.emplace_back();
  return &frameBuffer_.back();
}

// Helper method to find frame with exact PTS match
BetaCudaDeviceInterface::FrameBufferSlot*
BetaCudaDeviceInterface::findFrameWithExactPts(int64_t desiredPts) {
  for (auto& slot : frameBuffer_) {
    if (slot.occupied && slot.pts == desiredPts) {
      return &slot;
    }
  }
  return nullptr;
}

} // namespace facebook::torchcodec

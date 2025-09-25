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
pfnSequenceCallback(void* pUserData, CUVIDEOFORMAT* videoFormat) {
  BetaCudaDeviceInterface* decoder =
      static_cast<BetaCudaDeviceInterface*>(pUserData);
  return static_cast<int>(decoder->streamPropertyChange(videoFormat));
}

static int CUDAAPI
pfnDecodePictureCallback(void* pUserData, CUVIDPICPARAMS* pPicParams) {
  BetaCudaDeviceInterface* decoder =
      static_cast<BetaCudaDeviceInterface*>(pUserData);
  return decoder->frameReadyForDecoding(pPicParams);
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

  // TODONVDEC P1: init size should probably be min_num_decode_surfaces from
  // video format
  frameBuffer_.resize(4);
}

BetaCudaDeviceInterface::~BetaCudaDeviceInterface() {
  // TODONVDEC P0: we probably need to free the frames that have been decoded by
  // NVDEC but not yet "mapped" - i.e. those that are still in frameBuffer_?

  if (decoder_) {
    NVDECCache::GetCache(device_.index())
        .returnDecoder(&videoFormat_, std::move(decoder_));
  }

  if (videoParser_) {
    // TODONVDEC P2: consider caching this? Does DALI do that?
    cuvidDestroyVideoParser(videoParser_);
    videoParser_ = nullptr;
  }
}

void BetaCudaDeviceInterface::initializeInterface(AVStream* avStream) {
  TORCH_CHECK(avStream != nullptr, "AVStream cannot be null");
  timeBase_ = avStream->time_base;
  frameRateFallback_ = avStream->r_frame_rate;

  const AVCodecParameters* codecpar = avStream->codecpar;
  TORCH_CHECK(codecpar != nullptr, "CodecParameters cannot be null");

  TORCH_CHECK(
      // TODONVDEC P0 support more
      avStream->codecpar->codec_id == AV_CODEC_ID_H264,
      "Can only do H264 for now");

  // Setup bit stream filters (BSF):
  // https://ffmpeg.org/doxygen/7.0/group__lavc__bsf.html
  // This is only needed for some formats, like H264 or HEVC.  TODONVDEC P1: For
  // now we apply BSF unconditionally, but it should be optional  and dependent
  // on codec and container.
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

  // Create parser. Default values that aren't obvious are taken from DALI.
  CUVIDPARSERPARAMS parserParams = {};
  parserParams.CodecType = cudaVideoCodec_H264;
  parserParams.ulMaxNumDecodeSurfaces = 8;
  parserParams.ulMaxDisplayDelay = 0;
  // Callback setup, all are triggered by the parser within a call
  // to cuvidParseVideoData
  parserParams.pUserData = this;
  parserParams.pfnSequenceCallback = pfnSequenceCallback;
  parserParams.pfnDecodePicture = pfnDecodePictureCallback;
  parserParams.pfnDisplayPicture = nullptr;

  CUresult result = cuvidCreateVideoParser(&videoParser_, &parserParams);
  TORCH_CHECK(
      result == CUDA_SUCCESS, "Failed to create video parser: ", result);
}

// This callback is called by the parser within cuvidParseVideoData when there
// is a change in the stream's properties (like resolution change), as specified
// by CUVIDEOFORMAT. Particularly (but not just!), this is called at the very
// start of the stream.
// TODONVDEC P1: Code below mostly assume this is called only once at the start,
// we should handle the case of multiple calls. Probably need to flush buffers,
// etc.
unsigned char BetaCudaDeviceInterface::streamPropertyChange(
    CUVIDEOFORMAT* videoFormat) {
  TORCH_CHECK(videoFormat != nullptr, "Invalid video format");

  videoFormat_ = *videoFormat;

  if (videoFormat_.min_num_decode_surfaces == 0) {
    // Same as DALI's fallback
    videoFormat_.min_num_decode_surfaces = 20;
  }

  if (!decoder_) {
    decoder_ = NVDECCache::GetCache(device_.index()).getDecoder(videoFormat);

    if (!decoder_) {
      decoder_ = createDecoder(videoFormat);
    }

    TORCH_CHECK(decoder_, "Failed to get or create decoder");
  }

  // DALI also returns min_num_decode_surfaces from this function. This
  // instructs the parser to reset its ulMaxNumDecodeSurfaces field to this
  // value.
  return videoFormat_.min_num_decode_surfaces;
}

// Moral equivalent of avcodec_send_packet(). Here, we pass the AVPacket down to
// the NVCUVID parser.
int BetaCudaDeviceInterface::sendPacket(ReferenceAVPacket& packet) {
  CUVIDSOURCEDATAPACKET cuvidPacket = {};

  if (packet.get() && packet->data && packet->size > 0) {
    // Regular packet with data
    cuvidPacket.payload = packet->data;
    cuvidPacket.payload_size = packet->size;
    cuvidPacket.flags = CUVID_PKT_TIMESTAMP;
    cuvidPacket.timestamp = packet->pts;

    // Like DALI: store packet PTS in queue to later assign to frames as they
    // come out
    packetsPtsQueue.push(packet->pts);

  } else {
    // End of stream packet
    cuvidPacket.flags = CUVID_PKT_ENDOFSTREAM;
    eofSent_ = true;
  }

  CUresult result = cuvidParseVideoData(videoParser_, &cuvidPacket);
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

// Parser triggers this callback within cuvidParseVideoData when a frame is
// ready to be decoded, i.e. the parser received all the necessary packets for a
// given frame. It means we can send that frame to be decoded by the hardware
// NVDEC decoder by calling cuvidDecodePicture which is non-blocking.
int BetaCudaDeviceInterface::frameReadyForDecoding(CUVIDPICPARAMS* pPicParams) {
  if (isFlushing_) {
    return 0;
  }

  TORCH_CHECK(pPicParams != nullptr, "Invalid picture parameters");
  TORCH_CHECK(decoder_, "Decoder not initialized before picture decode");

  // Send frame to be decoded by NVDEC - non-blocking call.
  CUresult result = cuvidDecodePicture(decoder_.get(), pPicParams);
  if (result != CUDA_SUCCESS) {
    return 0; // Yes, you're reading that right, 0 mean error.
  }

  // The frame was sent to be decoded on the NVDEC hardware. Now we store some
  // relevant info into our frame buffer so that we can retrieve the decoded
  // frame later when receiveFrame() is called.
  // Importantly we need to 'guess' the PTS of that frame. The heuristic we use
  // (like in DALI) is that the frames are ready to be decoded in the same order
  // as the packets were sent to the parser. So we assign the PTS of the frame
  // by popping the PTS of the oldest packet in our packetsPtsQueue (note:
  // oldest doesn't necessarily mean lowest PTS!).

  TORCH_CHECK(
      // TODONVDEC P0 the queue may be empty, handle that.
      !packetsPtsQueue.empty(),
      "PTS queue is empty when decoding a frame");
  int64_t guessedPts = packetsPtsQueue.front();
  packetsPtsQueue.pop();

  // Field values taken from DALI
  CUVIDPARSERDISPINFO dispInfo = {};
  dispInfo.picture_index = pPicParams->CurrPicIdx;
  dispInfo.progressive_frame = !pPicParams->field_pic_flag;
  dispInfo.top_field_first = pPicParams->bottom_field_flag ^ 1;
  dispInfo.repeat_first_field = 0;
  dispInfo.timestamp = guessedPts;

  std::lock_guard<std::mutex> lock(frameBufferMutex_);
  FrameBufferSlot* slot = findEmptySlot();
  slot->dispInfo = dispInfo;
  slot->guessedPts = guessedPts;
  slot->occupied = true;

  return 1;
}

// Moral equivalent of avcodec_receive_frame(). Here, we look for a decoded
// frame with the exact desired PTS in our frame buffer. This logic is only
// valid in exact seek_mode, for now.
int BetaCudaDeviceInterface::receiveFrame(
    UniqueAVFrame& avFrame,
    int64_t desiredPts) {
  // TODONVDEC P2 I don't think this mutex is needed, there shouldn't be
  // multi-threading *within* the same decoder/interface instance.
  std::lock_guard<std::mutex> lock(frameBufferMutex_);

  FrameBufferSlot* slot = findFrameWithExactPts(desiredPts);
  if (slot == nullptr) {
    // No frame found, instruct caller to try again later after sending more
    // packets.
    return AVERROR(EAGAIN);
  }

  slot->occupied = false;
  slot->guessedPts = -1;

  CUVIDPROCPARAMS procParams = {};
  CUVIDPARSERDISPINFO dispInfo = slot->dispInfo;
  procParams.progressive_frame = dispInfo.progressive_frame;
  procParams.top_field_first = dispInfo.top_field_first;
  procParams.unpaired_field = dispInfo.repeat_first_field < 0;
  CUdeviceptr framePtr = 0;
  unsigned int pitch = 0;

  // We know the frame we want was sent to the hardware decoder, but now we need
  // to "map" it to an "output surface" before we can use its data. This is a
  // blocking calls that waits until the frame is fully decoded and ready to be
  // used.
  CUresult result = cuvidMapVideoFrame(
      static_cast<CUvideodecoder>(decoder_.get()),
      dispInfo.picture_index,
      &framePtr,
      &pitch,
      &procParams);

  if (result != CUDA_SUCCESS) {
    return AVERROR_EXTERNAL;
  }

  avFrame = convertCudaFrameToAVFrame(framePtr, pitch, dispInfo, timeBase_);

  // Unmap the frame so that the decoder can reuse its corresponding output
  // surface. Whether this is blocking is unclear?
  cuvidUnmapVideoFrame(static_cast<CUvideodecoder>(decoder_.get()), framePtr);
  // TODONVDEC P0: Get clarity on this:
  // We assume that the framePtr is still valid after unmapping. That framePtr
  // is now part of the avFrame, which we'll return to the caller, and the
  // caller will immediately use it for color-conversion, at which point a copy
  // happens. After the copy, it doesn't matter whether framePtr is still valid.
  // And we'll return to this function (and to cuvidUnmapVideoFrame()) *after*
  // the copy is made, so there should be no risk of overwriting the data before
  // the copy.
  // Buuuut yeah, we need get more clarity on what actually happens, and on
  // what's needed. IIUC DALI makes the color-conversion copy immediately after
  // cuvidMapVideoFrame() and *before* cuvidUnmapVideoFrame() with a synchronize
  // in between. So maybe we should do the same.

  return AVSUCCESS;
}

void BetaCudaDeviceInterface::flush() {
  // Set flush flag like DALI to prevent new decode operations
  isFlushing_ = true;

  // Send EOS packet to drain decoder like DALI does
  if (!eofSent_) {
    CUVIDSOURCEDATAPACKET cuvidPacket = {};
    cuvidPacket.flags = CUVID_PKT_ENDOFSTREAM;
    CUresult result = cuvidParseVideoData(videoParser_, &cuvidPacket);
    if (result == CUDA_SUCCESS) {
      eofSent_ = true;
    }
  }

  // Clear flush flag like DALI does
  isFlushing_ = false;

  {
    std::lock_guard<std::mutex> lock(frameBufferMutex_);
    for (auto& slot : frameBuffer_) {
      slot.occupied = false;
      slot.guessedPts = -1;
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
  else if (frameRateFallback_.num > 0 && frameRateFallback_.den > 0) {
    effectiveFrameRate = frameRateFallback_;
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
    if (slot.occupied && slot.guessedPts == desiredPts) {
      return &slot;
    }
  }
  return nullptr;
}

} // namespace facebook::torchcodec

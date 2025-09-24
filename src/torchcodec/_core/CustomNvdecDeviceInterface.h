// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "src/torchcodec/_core/DeviceInterface.h"
#include "src/torchcodec/_core/Cache.h"
#include "src/torchcodec/_core/FFMPEGCommon.h"

#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>
#include <map>

#include "src/torchcodec/_core/nvcuvid_include/cuviddec.h"
#include "src/torchcodec/_core/nvcuvid_include/nvcuvid.h"

namespace facebook::torchcodec {

struct CUvideoDecoderDeleter {
  void operator()(CUvideodecoder decoder) const {
    if (decoder) {
      cuvidDestroyDecoder(decoder);
    }
  }
};

using UniqueCUvideodecoder = std::unique_ptr<void, CUvideoDecoderDeleter>;

struct NVDECCacheKey {
  cudaVideoCodec codec_type;
  unsigned width;
  unsigned height;
  cudaVideoChromaFormat chroma_format;
  unsigned int bit_depth_luma_minus8;
  unsigned char num_decode_surfaces;
  
  // TODONVDEC P2: we only implement operator< which is enough for std::map, but:
  // - we should consider using std::unordered_map
  // - we should consider a more sophisticated and potentially less strict cache key comparison logic
  bool operator<(const NVDECCacheKey& other) const {
    return std::tie(codec_type, width, height, chroma_format, bit_depth_luma_minus8, num_decode_surfaces) <
           std::tie(other.codec_type, other.width, other.height, other.chroma_format, other.bit_depth_luma_minus8, other.num_decode_surfaces);
  }
};

class NVDECCache {
 public:
  static NVDECCache& GetCache(int deviceId = -1);

  // Get decoder from cache - returns nullptr if none available
  UniqueCUvideodecoder getDecoder(const NVDECCacheKey& key);

  // Return decoder to cache - returns true if added to cache
  bool returnDecoder(const NVDECCacheKey& key, UniqueCUvideodecoder decoder);

  // Create new decoder with given parameters
  static UniqueCUvideodecoder createDecoder(CUVIDEOFORMAT* video_format);

  // Helper to create key from video format
  static NVDECCacheKey createKey(CUVIDEOFORMAT* video_format);

 private:
  NVDECCache() = default;
  ~NVDECCache() = default;

  std::map<NVDECCacheKey, UniqueCUvideodecoder> cache_;
  std::mutex cache_lock_;
  
  static constexpr int MAX_CACHE_SIZE = 20; // Much smaller, simpler cache
};


// Custom NVDEC device interface that provides direct control over NVDEC
// while keeping FFmpeg for demuxing
class CustomNvdecDeviceInterface : public DeviceInterface {
 public:
  CustomNvdecDeviceInterface(const torch::Device& device);

  virtual ~CustomNvdecDeviceInterface();

  std::optional<const AVCodec*> findCodec(const AVCodecID& codecId) override;

  void initializeContext(AVCodecContext* codecContext) override;

  // Set the timeBase for duration calculations
  void setTimeBase(const AVRational& timeBase);

  // Set the frame rate for duration calculations
  void setFrameRate(const AVRational& frameRate);

  void convertAVFrameToFrameOutput(
      const VideoStreamOptions& videoStreamOptions,
      const AVRational& timeBase,
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor =
          std::nullopt) override;

  // Extension point overrides for direct packet decoding
  bool canDecodePacketDirectly() const override {
    return true;
  }

  // Returns 0 on success, AVERROR(EAGAIN) if decoder queue full, or other AVERROR on failure
  int sendPacket(ReferenceAVPacket& packet);

  // Receive decoded frame (non-blocking) 
  // Returns 0 on success, AVERROR(EAGAIN) if no frame ready, AVERROR_EOF if end of stream,
  // or other AVERROR on failure
  int receiveFrame(UniqueAVFrame& frame, int64_t desiredPts);

  void flush();

 public:
  // NVDEC callback functions (must be public for C callbacks)
  int handleVideoSequence(CUVIDEOFORMAT* pVideoFormat);
  int handlePictureDecode(CUVIDPICPARAMS* pPicParams);

 private:
  // NVDEC decoder context and parser
  CUvideoparser videoParser_ = nullptr;
  UniqueCUvideodecoder decoder_;
  NVDECCacheKey decoderKey_;

  // Video format info
  CUVIDEOFORMAT videoFormat_;
  bool parserCreated_ = false;

  struct FrameBufferSlot {
    CUVIDPARSERDISPINFO dispInfo;
    int64_t pts;
    bool occupied = false;

    FrameBufferSlot() : pts(-1), occupied(false) {
      memset(&dispInfo, 0, sizeof(dispInfo));
    }
  };
  
  static constexpr int MAX_DECODE_SURFACES = 32; // NVDEC max
  std::vector<FrameBufferSlot> frameBuffer_;
  std::mutex frameBufferMutex_;

  std::queue<int64_t> packetsPtsQueue;


  // EOF tracking
  bool eofSent_ = false;
  
  // Flush flag to prevent decode operations during flush (like DALI's flush_)
  bool flush_ = false;
  
  // Store timeBase for duration calculations
  AVRational timeBase_ = {0, 0};
  
  // Store frame rate for duration calculations (fallback when NVDEC frame rate is unavailable)
  AVRational fallbackFrameRate_ = {0, 0};

  // Helper methods for frame reordering
  FrameBufferSlot* findEmptySlot();
  FrameBufferSlot* findFrameWithExactPts(int64_t desiredPts);

  void createVideoParser();

  // Convert CUDA frame pointer to AVFrame
  UniqueAVFrame convertCudaFrameToAVFrame(
      CUdeviceptr framePtr,
      unsigned int pitch,
      const CUVIDPARSERDISPINFO& dispInfo,
      const AVRational& timeBase);
};

} // namespace facebook::torchcodec

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "src/torchcodec/_core/DeviceInterface.h"
#include "src/torchcodec/_core/Cache.h"
#include "src/torchcodec/_core/FFMPEGCommon.h"
#include "src/torchcodec/_core/NVDECCache.h"

#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>
#include <map>

#include "src/torchcodec/_core/nvcuvid_include/cuviddec.h"
#include "src/torchcodec/_core/nvcuvid_include/nvcuvid.h"

namespace facebook::torchcodec {


// BETA CUDA device interface that provides direct control over NVDEC
// while keeping FFmpeg for demuxing
class BetaCudaDeviceInterface : public DeviceInterface {
 public:
  BetaCudaDeviceInterface(const torch::Device& device);

  virtual ~BetaCudaDeviceInterface();

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

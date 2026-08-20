// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "DeviceInterface.h"
#include "FFMPEGCommon.h"
#include "StableABICompat.h"
#include "StreamOptions.h"

namespace facebook::torchcodec {

class FORCE_PUBLIC_VISIBILITY ColorConverter {
 public:
  explicit ColorConverter(
      const StableDevice& device = StableDevice(kStableCPU),
      OutputDtypeConfig output_dtype_config = OutputDtypeConfig::UINT8);

  // `frame_device` is where the frame's samples live. It doesn't have to be
  // this converter's device: the samples are moved there first if it isn't.
  torch::stable::Tensor convert(
      const AVFrame& av_frame,
      const std::string& frame_device);

 private:
  void maybe_initialize_interface(OutputDtype output_dtype);

  // Returns the frame to color-convert, or nullptr when `av_frame` already
  // lives on our device and can be used as-is.
  UniqueAVFrame maybe_transfer(
      const AVFrame& av_frame,
      const StableDevice& frame_device);

  // An interface for a device that isn't ours, used only to pull samples off
  // it: a CPU interface can't read device memory, so the accelerator side has
  // to do the copying in both directions.
  DeviceInterface& source_interface_for(const StableDevice& device);

  std::unique_ptr<DeviceInterface> device_interface_;
  StableDevice device_;
  OutputDtypeConfig output_dtype_config_;
  std::optional<OutputDtype> initialized_output_dtype_;
  std::vector<std::pair<StableDevice, std::unique_ptr<DeviceInterface>>>
      source_interfaces_;
};

} // namespace facebook::torchcodec

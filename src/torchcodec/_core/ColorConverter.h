// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "DeviceInterface.h"
#include "FFMPEGCommon.h"
#include "StableABICompat.h"
#include "StreamOptions.h"

namespace facebook::torchcodec {

class FORCE_PUBLIC_VISIBILITY ColorConverter {
 public:
  explicit ColorConverter(
      OutputDtypeConfig output_dtype_config = OutputDtypeConfig::UINT8);

  // `device` is where the frame's samples live, and where the output lands.
  torch::stable::Tensor convert(
      const AVFrame& av_frame,
      const std::string& device);

 private:
  struct PerDevice {
    // Kept as the string we were handed, not as a StableDevice: parsing a
    // device string goes through the stable-ABI shim, and this sits on the
    // per-frame path where the string is the same every time.
    std::string device_string;
    std::unique_ptr<DeviceInterface> interface;
    std::optional<OutputDtype> initialized_output_dtype;
  };

  DeviceInterface& interface_for(
      const std::string& device,
      OutputDtype output_dtype);

  OutputDtypeConfig output_dtype_config_;
  // One interface per device we've been given a frame on. A converter has no
  // device of its own: it follows its frames, and the same one may be fed CPU
  // frames and CUDA frames in turn. There's a handful of devices at most, so a
  // linear scan beats a hash map.
  std::vector<PerDevice> interfaces_;
};

} // namespace facebook::torchcodec

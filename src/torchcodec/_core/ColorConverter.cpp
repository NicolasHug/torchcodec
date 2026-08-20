// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ColorConverter.h"

#include <algorithm>
#include <optional>
#include <vector>

#include "Frame.h"
#include "StreamOptions.h"
#include "Transform.h"

namespace facebook::torchcodec {

ColorConverter::ColorConverter(OutputDtypeConfig output_dtype_config)
    : output_dtype_config_(output_dtype_config) {}

DeviceInterface& ColorConverter::interface_for(
    const std::string& device,
    OutputDtype output_dtype) {
  auto entry = std::find_if(
      interfaces_.begin(), interfaces_.end(), [&](const PerDevice& candidate) {
        return candidate.device_string == device;
      });

  if (entry == interfaces_.end()) {
    validate_device_interface(device);
    auto interface = create_device_interface(StableDevice(device));
    STD_TORCH_CHECK(
        interface != nullptr,
        "Failed to create device interface. This should never happen, please report.");
    interfaces_.push_back(
        PerDevice{device, std::move(interface), std::nullopt});
    entry = std::prev(interfaces_.end());
  }

  // Interface initialization is done per-frame, not once and for all: with
  // AUTO, the desired output dtype is only known once we see a frame, and it
  // can differ from one frame to the next.
  if (entry->initialized_output_dtype != output_dtype) {
    VideoStreamOptions options;
    options.output_dtype = output_dtype;
    options.device = entry->interface->device();

    std::vector<std::unique_ptr<Transform>> no_transforms;
    entry->interface->initialize_color_conversion(
        options, no_transforms, /*resized_output_dims=*/std::nullopt);
    entry->initialized_output_dtype = output_dtype;
  }

  return *entry->interface;
}

torch::stable::Tensor ColorConverter::convert(
    const AVFrame& av_frame,
    const std::string& device) {
  OutputDtype output_dtype = resolve_output_dtype(
      output_dtype_config_, static_cast<AVPixelFormat>(av_frame.format));
  DeviceInterface& interface = interface_for(device, output_dtype);

  FrameOutput frame_output;
  interface.convert_av_frame_to_frame_output(
      av_frame, frame_output, std::nullopt);

  // TODO_API_BREAKDOWN PERF P2: on CPU this is a lot slower than the filter
  // graph's transpose.
  frame_output.data = rotate_hwc_tensor(
      frame_output.data,
      rotation_from_degrees(get_rotation_from_frame(av_frame)));

  return convert_to_output_dtype(frame_output.data, output_dtype);
}

} // namespace facebook::torchcodec

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

ColorConverter::ColorConverter(
    const StableDevice& device,
    OutputDtypeConfig output_dtype_config)
    : device_(device), output_dtype_config_(output_dtype_config) {
  device_interface_ = create_device_interface(device);
  STD_TORCH_CHECK(
      device_interface_ != nullptr,
      "Failed to create device interface. This should never happen, please report.");
  // An accelerator interface resolves "current device" to a concrete index at
  // construction. Take its device rather than the one we were handed, so that
  // comparing ours against a frame's doesn't report a spurious mismatch
  // between "cuda" and "cuda:0".
  device_ = device_interface_->device();
}

DeviceInterface& ColorConverter::source_interface_for(
    const StableDevice& device) {
  auto entry = std::find_if(
      source_interfaces_.begin(),
      source_interfaces_.end(),
      [&](const auto& candidate) { return candidate.first == device; });

  if (entry == source_interfaces_.end()) {
    auto interface = create_device_interface(device);
    STD_TORCH_CHECK(
        interface != nullptr,
        "Failed to create device interface. This should never happen, please report.");
    source_interfaces_.emplace_back(device, std::move(interface));
    entry = std::prev(source_interfaces_.end());
  }
  return *entry->second;
}

UniqueAVFrame ColorConverter::maybe_transfer(
    const AVFrame& av_frame,
    const StableDevice& frame_device) {
  if (frame_device == device_) {
    return nullptr;
  }

  if (frame_device.type() == kStableCPU) {
    return device_interface_->upload_frame_from_cpu(av_frame);
  }

  UniqueAVFrame on_cpu =
      source_interface_for(frame_device).download_frame_to_cpu(av_frame);
  if (device_.type() == kStableCPU) {
    return on_cpu;
  }
  // Two accelerators, or two GPUs: there's no direct path between them here, so
  // we bounce through the host. Correct, and slow enough that you'd notice.
  return device_interface_->upload_frame_from_cpu(*on_cpu);
}

void ColorConverter::maybe_initialize_interface(OutputDtype output_dtype) {
  // Interface initialization is done per-frame, not in the constructor: with
  // AUTO, the desired output dtype is only known once we see a frame, and it
  // can differ from one frame to the next.
  if (initialized_output_dtype_.has_value() &&
      *initialized_output_dtype_ == output_dtype) {
    return;
  }

  VideoStreamOptions options;
  options.output_dtype = output_dtype;
  options.device = device_;

  std::vector<std::unique_ptr<Transform>> no_transforms;
  device_interface_->initialize_color_conversion(
      options, no_transforms, /*resized_output_dims=*/std::nullopt);
  initialized_output_dtype_ = output_dtype;
}

torch::stable::Tensor ColorConverter::convert(
    const AVFrame& av_frame,
    const std::string& frame_device) {
  // `transferred` owns the moved samples for as long as it's in scope, which
  // covers the color conversion below.
  UniqueAVFrame transferred =
      maybe_transfer(av_frame, StableDevice(frame_device));
  const AVFrame& frame = transferred ? *transferred : av_frame;

  // The transfer preserves the pixel format's bit depth, so "auto" resolves the
  // same either way.
  OutputDtype output_dtype = resolve_output_dtype(
      output_dtype_config_, static_cast<AVPixelFormat>(frame.format));
  maybe_initialize_interface(output_dtype);

  FrameOutput frame_output;
  device_interface_->convert_av_frame_to_frame_output(
      frame, frame_output, std::nullopt);

  // TODO_API_BREAKDOWN PERF P2: on CPU this is a lot slower than the filter
  // graph's transpose.
  frame_output.data = rotate_hwc_tensor(
      frame_output.data, rotation_from_degrees(get_rotation_from_frame(frame)));

  return convert_to_output_dtype(frame_output.data, output_dtype);
}

} // namespace facebook::torchcodec

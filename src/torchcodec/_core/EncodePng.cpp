// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "EncodePng.h"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/util/Exception.h>

#include "StableABICompat.h"

#if !TORCHCODEC_ENABLE_PNG

namespace facebook::torchcodec {

torch::stable::Tensor encode_png(
    [[maybe_unused]] const torch::stable::Tensor& data,
    [[maybe_unused]] int64_t compression_level) {
  STD_TORCH_CHECK(
      false,
      "encode_png: torchcodec was not compiled with libpng support. "
      "Rebuild torchcodec in an environment where libpng (and its development "
      "headers) are available. If you see this error in a prebuilt wheel, "
      "please report it to the TorchCodec repo.");
}

} // namespace facebook::torchcodec

#else

#include <png.h>
#include <setjmp.h>

#include <cstdlib>
#include <cstring>
#include <optional>

namespace facebook::torchcodec {

namespace {

// Growable byte buffer that libpng writes the encoded PNG into via our
// write_callback. libpng can't hand ownership of it to a tensor, so the
// contents are copied into a freshly allocated tensor at the end.
struct MemEncode {
  char* buffer;
  size_t size;
};

struct PngErrorMgr {
  const char* last_error_message;
  jmp_buf setjmp_buffer;
};

void error_callback(png_structp png_ptr, png_const_charp error_message) {
  auto* error_ctx = static_cast<PngErrorMgr*>(png_get_error_ptr(png_ptr));
  error_ctx->last_error_message = error_message;
  longjmp(error_ctx->setjmp_buffer, 1);
}

void write_callback(png_structp png_ptr, png_bytep data, png_size_t length) {
  auto* mem = static_cast<MemEncode*>(png_get_io_ptr(png_ptr));
  size_t new_size = mem->size + length;

  if (mem->buffer) {
    mem->buffer = static_cast<char*>(realloc(mem->buffer, new_size));
  } else {
    mem->buffer = static_cast<char*>(malloc(new_size));
  }

  if (!mem->buffer) {
    png_error(png_ptr, "Write Error");
  }

  memcpy(mem->buffer + mem->size, data, length);
  mem->size += length;
}

} // namespace

torch::stable::Tensor encode_png(
    const torch::stable::Tensor& data,
    int64_t compression_level) {
  png_structp png_write = nullptr;
  png_infop info_ptr = nullptr;
  PngErrorMgr err_ptr;

  MemEncode buf_info;
  buf_info.buffer = nullptr;
  buf_info.size = 0;

  // libpng uses setjmp/longjmp for error handling. longjmp does not unwind C++
  // stack frames, so destructors of objects created after setjmp won't run. We
  // declare the tensor before setjmp with std::optional, defer its construction
  // until after, and explicitly reset it on the error path.
  std::optional<torch::stable::Tensor> input;

  if (setjmp(err_ptr.setjmp_buffer)) {
    input.reset();

    if (info_ptr != nullptr) {
      png_destroy_info_struct(png_write, &info_ptr);
    }
    if (png_write != nullptr) {
      png_destroy_write_struct(&png_write, nullptr);
    }
    if (buf_info.buffer != nullptr) {
      free(buf_info.buffer);
    }

    STD_TORCH_CHECK(false, err_ptr.last_error_message);
  }

  STD_TORCH_CHECK(
      compression_level >= 0 && compression_level <= 9,
      "Compression level should be between 0 and 9");
  STD_TORCH_CHECK(
      data.device().type() == kStableCPU, "Input tensor should be on CPU");
  STD_TORCH_CHECK(
      data.scalar_type() == kStableUInt8, "Input tensor dtype should be uint8");
  STD_TORCH_CHECK(
      data.dim() == 3, "Input data should be a 3-dimensional tensor");

  int channels = data.size(0);
  int height = data.size(1);
  int width = data.size(2);

  STD_TORCH_CHECK(
      channels == 1 || channels == 3,
      "The number of channels should be 1 or 3, got: ",
      channels);

  // libpng writes rows in HWC interleaved order, so permute from CHW.
  input = torch::stable::contiguous(stable_permute(data, {1, 2, 0}));

  png_write = png_create_write_struct(
      PNG_LIBPNG_VER_STRING, &err_ptr, error_callback, nullptr);
  info_ptr = png_create_info_struct(png_write);

  png_set_write_fn(png_write, &buf_info, write_callback, nullptr);

  auto color_type = channels == 1 ? PNG_COLOR_TYPE_GRAY : PNG_COLOR_TYPE_RGB;
  png_set_IHDR(
      png_write,
      info_ptr,
      width,
      height,
      8,
      color_type,
      PNG_INTERLACE_NONE,
      PNG_COMPRESSION_TYPE_DEFAULT,
      PNG_FILTER_TYPE_DEFAULT);

  png_set_compression_level(png_write, compression_level);

  png_write_info(png_write, info_ptr);

  auto stride = width * channels;
  auto ptr = input->const_data_ptr<uint8_t>();
  for (int y = 0; y < height; ++y) {
    png_write_row(png_write, ptr);
    ptr += stride;
  }

  png_write_end(png_write, info_ptr);

  png_destroy_write_struct(&png_write, &info_ptr);

  auto out_tensor = torch::stable::empty(
      {static_cast<int64_t>(buf_info.size)}, kStableUInt8);

  // torch can't take ownership of the malloc'd libpng buffer (no from_blob in
  // the stable ABI), so copy it into the tensor and free it.
  auto out_ptr = out_tensor.mutable_data_ptr<uint8_t>();
  std::memcpy(out_ptr, buf_info.buffer, sizeof(uint8_t) * out_tensor.numel());
  free(buf_info.buffer);

  return out_tensor;
}

} // namespace facebook::torchcodec

#endif // !TORCHCODEC_ENABLE_PNG

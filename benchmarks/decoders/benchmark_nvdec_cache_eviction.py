# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark NVDECCache LRU eviction scan time.

Generates 500 uniquely-sized videos, then decodes 3000 times (cycling through
them) so that every decoder return triggers a cache eviction after the first 20.

Usage:
    TORCHCODEC_NVDEC_CACHE_TIMING=1 python benchmarks/decoders/benchmark_nvdec_cache_eviction.py

C++ timing stats are printed to stderr at process exit.
"""

import math
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from torchcodec.decoders import set_cuda_backend, VideoDecoder

NUM_VIDEOS = 500
NUM_DECODES = 3000
VIDEO_DIR = "/tmp/nvdec_cache_benchmark_videos"


def _make_video_specs():
    """Return (width, height) tuples with unique coded dimensions.

    Both dimensions are multiples of 16 (H.264 macroblock size) so that
    coded dimensions match exactly, giving a unique NVDECCache key per video.
    We use a grid of widths x heights to stay within NVDEC resolution limits.
    """
    num_widths = math.ceil(math.sqrt(NUM_VIDEOS))
    num_heights = math.ceil(NUM_VIDEOS / num_widths)

    specs = []
    for h_idx in range(num_heights):
        for w_idx in range(num_widths):
            if len(specs) >= NUM_VIDEOS:
                return specs
            specs.append((128 + w_idx * 16, 128 + h_idx * 16))
    return specs


def _generate_video(width, height, path):
    if os.path.exists(path):
        return
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"testsrc2=size={width}x{height}:rate=1:duration=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "1",
            path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def generate_videos(specs, paths):
    os.makedirs(VIDEO_DIR, exist_ok=True)
    existing = sum(1 for p in paths if os.path.exists(p))
    if existing == len(paths):
        print(f"All {len(paths)} videos already exist, skipping generation.")
        return

    print(f"Generating {len(paths)} videos ({existing} already exist)...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = [ex.submit(_generate_video, w, h, p) for (w, h), p in zip(specs, paths)]
        for f in as_completed(futures):
            f.result()
    print(f"Done in {time.time() - t0:.1f}s")


def main():
    specs = _make_video_specs()
    paths = [os.path.join(VIDEO_DIR, f"test_{w}x{h}.mp4") for w, h in specs]
    generate_videos(specs, paths)

    print(f"\nDecoding {NUM_DECODES} videos ({NUM_VIDEOS} unique, cycling)")

    t0 = time.time()
    with set_cuda_backend("beta"):
        for i in range(NUM_DECODES):
            decoder = VideoDecoder(paths[i % NUM_VIDEOS], device="cuda")
            decoder[0]
            del decoder
    print(f"Total time: {time.time() - t0:.3f}s")


if __name__ == "__main__":
    main()

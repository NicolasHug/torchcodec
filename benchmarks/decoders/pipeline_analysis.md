# Decoding Pipeline Analysis

Benchmark results from instrumenting the torchcodec decoding pipeline on CPU and
CUDA (beta backend). All results below are for **720p libx264 testsrc videos,
GOP=50, 30fps**, on an **NVIDIA GeForce RTX 4080 Laptop GPU**.

## Instrumentation Setup

RAII-based `ScopedBenchmarkTimer` with a global `std::atomic<bool>` toggle.
Negligible overhead when disabled. Wall-clock timing via
`std::chrono::high_resolution_clock`. For CUDA, `cudaStreamSynchronize` is added
around NPP color conversion and stream sync to measure actual GPU time (otherwise
the non-blocking API calls would only measure launch overhead).

### Timer accuracy notes

| Timer | Accurate? | Why |
|---|---|---|
| demux, BSF, seek | Yes | Pure CPU work |
| decode (CPU) | Yes | `avcodec_send_packet` — CPU software decode (includes parsing) |
| YUV -> RGB (CPU) | Yes | swscale YUV→RGB — CPU work |
| map_frame (CUDA) | Yes | `cuvidMapVideoFrame` blocks until NVDEC finishes |
| YUV -> RGB (CUDA) | Yes | Includes tensor alloc + stream sync + NPP color conversion; `cudaStreamSynchronize` added when benchmarking |
| packet_parse_and_decode (CUDA) | Yes* | `cuvidParseVideoData` is CPU-side parsing; includes `nvdec_decode` callback but that's just non-blocking launch overhead |
| nvdec_decode (CUDA) | **No** | `cuvidDecodePicture` is non-blocking; real cost shows up in map_frame. **Excluded from plots** (nested inside packet_parse_and_decode) |
| decode (CUDA) | N/A | Parent timer wrapping BSF + packet_parse_and_decode. **Excluded from plots** to avoid double-counting |
| permute | N/A | `aten::permute` is a view (no kernel), timing is accurate but trivial |

---

## CPU Pipeline

### Stages

```
av_read_frame (demux)
    → avcodec_send_packet (decode) — software H.264 decode
    → avcodec_receive_frame
    → swscale/filtergraph (YUV -> RGB) — color conversion
    → permute (view op, ~free)
```

### All frames (900 frames, ~827 fps)

| Stage | ms/frame | % of total |
|---|---|---|
| decode | 0.52 | 48% |
| YUV -> RGB | 0.56 | 52% |
| demux, seek, permute | <0.01 | <1% |

Nearly a perfect **50/50 split** between software decode and color conversion.
Both are CPU-bound. Demux and permute are negligible.

### Every 50th frame (18 frames, ~294 fps)

| Stage | ms/frame | % of total |
|---|---|---|
| decode | 2.40 | 70% |
| YUV -> RGB | 1.01 | 29% |
| demux, seek | 0.04 | ~1% |

Decode now dominates at **70%**. Two reasons:

1. **I-frames are expensive.** Each requested frame lands on (or near) a
   keyframe. I-frames are fully intra-coded — no motion compensation shortcuts —
   so they cost much more to decode than P/B frames.
2. **Extra packets decoded per seek.** Despite GOP=50 matching the step size,
   the decoder still processes ~3 packets per requested frame (54 packets / 18
   frames). Seeks don't always land exactly on the keyframe; a few extra frames
   must be decoded and discarded.

YUV -> RGB jumps from 0.56 to 1.01 ms/frame (1.8x) — possibly due to the
`get_frames_at` → `getFramesAtIndices` batch code path or cache effects from the
non-sequential access pattern.

### CPU optimization opportunities

- **All frames:** Pipeline parallelism — decode frame N+1 on one thread while
  color-converting frame N on another. With a balanced 50/50 split, this gives
  close to **2x throughput**. See commit `f4460b1` for a working implementation.
- **1/N frames:** Decode dominates so pipelining helps less (~1.4x max).
  The main win is reducing wasted decode work: ensure seeks land precisely on
  keyframes, or use videos with GOP matching the sampling stride.

---

## CUDA (Beta) Pipeline

### Stages

```
av_read_frame (demux) — CPU
    → applyBSF (bitstream_filter) — CPU
    → cuvidParseVideoData (packet_parse_and_decode) — CPU, triggers callbacks
        → cuvidDecodePicture (nvdec_decode) — non-blocking GPU submit
    → cuvidMapVideoFrame (map_frame) — BLOCKS until NVDEC done
    → YUV -> RGB — tensor alloc + stream sync + NPP NV12→RGB on CUDA cores
    → permute (view op, ~free)
```

Three independent hardware units:
1. **CPU** — demux, BSF, packet parsing
2. **NVDEC** — fixed-function hardware decoder (separate from CUDA cores)
3. **CUDA cores** — NPP color conversion

### Timer hierarchy (important for interpreting results)

The C++ timers nest as follows:

```
"decode" (parent — SingleStreamDecoder.cpp, wraps sendPacket())
  ├── "bitstream_filter"  (leaf)
  └── "packet_parse_and_decode" (cuvidParseVideoData, includes callback)
        └── "nvdec_decode" (cuvidDecodePicture callback — non-blocking)
```

`decode` double-counts `bitstream_filter` + `packet_parse_and_decode`.
`packet_parse_and_decode` includes `nvdec_decode` (triggered as a synchronous
callback inside `cuvidParseVideoData`). Since `cuvidDecodePicture` is
non-blocking (just launch overhead, ~microseconds), `packet_parse_and_decode`
effectively measures **CPU-side bitstream parsing**.

For plots, the leaf categories used are: `bitstream_filter`,
`packet_parse_and_decode` (labeled "parsing"), `map_frame`,
`YUV -> RGB`, etc. Parent timers (`decode`, `nvdec_decode`) are excluded to
avoid double-counting.

### All frames (900 frames, ~1620 fps)

| Stage | ms/frame | % of leaf total |
|---|---|---|
| map_frame (NVDEC wait) | 0.534 | **86%** |
| YUV -> RGB | 0.062 | 10% |
| parsing | 0.012 | 2% |
| permute | 0.006 | 1% |
| demux | 0.005 | <1% |
| bitstream_filter, seek, unmap | <0.001 each | <1% |

**map_frame utterly dominates at 86%.** This is the blocking call that waits for
NVDEC to finish decoding. YUV -> RGB (including tensor alloc, stream sync, and
NPP color conversion) is only ~10%. The NVDEC hardware decoder is the bottleneck.

### Every 50th frame (18 frames, ~383 fps)

| Stage | ms/frame | % of leaf total |
|---|---|---|
| parsing | 1.509 | **58%** |
| map_frame (NVDEC wait) | 0.918 | 35% |
| YUV -> RGB | 0.060 | 2% |
| seek | 0.056 | 2% |
| demux | 0.058 | 2% |
| bitstream_filter, permute, unmap | <0.01 each | <1% |

The picture completely changes. **Parsing dominates at 58%.** The decoder
processes 86 packets for 18 requested frames (~4.8 packets per frame). NVDEC
must decode all intermediate frames between seeks even though only 18 are
needed, and `cuvidParseVideoData` runs on CPU for every one of those 86 packets.

`map_frame` drops from 86% to 35% — still significant (the per-frame NVDEC
blocking wait), but now the CPU-side parsing is the bigger cost.

### CUDA optimization opportunities

- **All frames:** Pipeline parallelism across 3 hardware units:
  - CPU thread: demux + BSF + `cuvidParseVideoData` (keep feeding packets)
  - NVDEC hardware: decoding frames asynchronously
  - CUDA cores: NPP color conversion on previously-decoded frames

  Submit multiple packets before calling `cuvidMapVideoFrame`, so NVDEC stays
  busy while CPU demuxes and CUDA cores color-convert. Theoretical max ~3x if
  all stages are balanced, but NVDEC at 86% is the hard ceiling — realistically
  expect to hide the remaining 14% for a modest ~1.2x gain.

- **1/N frames:** Reduce wasted decode work. With 4.8 packets per requested
  frame, most of the time is spent parsing and decoding frames that are
  discarded. Options:
  - Ensure keyframe intervals match the sampling stride
  - Improve seek precision to land exactly on keyframes
  - Consider a seek mode that skips unnecessary intermediate frames

---

## All vs Sampled: Summary

|  | CPU all | CPU 1/50 | CUDA all | CUDA 1/50 |
|---|---|---|---|---|
| FPS | ~827 | ~294 | ~1620 | ~383 |
| ms/frame | 1.21 | 3.40 | 0.62 | 2.61 |
| Bottleneck | decode 48%, YUV->RGB 52% | decode 70% | NVDEC (map_frame) 86% | parsing 58%, map_frame 35% |
| Best optimization | Pipeline decode ‖ YUV->RGB | Reduce extra packets | Pipeline CPU ‖ NVDEC ‖ CUDA | Reduce extra packets |

Note: previous CUDA FPS numbers (~807 all, ~149 sampled) were inflated by
double-counting parent timers. The corrected leaf-only totals show CUDA is
actually ~2x faster than CPU for sequential decoding.

# TorchCodec Decoding Pipeline Analysis

## CPU Decoding Pipeline

### Pipeline Stages

| Stage | What it does | Code location | Hardware |
|---|---|---|---|
| `seek` | `avformat_seek_file()` + codec flush | `SingleStreamDecoder.cpp:1272` | CPU |
| `demux` | `av_read_frame()` — reads compressed packet from container | `SingleStreamDecoder.cpp:1371-1372` | CPU |
| `send_packet` | `avcodec_send_packet()` — **software H.264/H.265 decode** | `SingleStreamDecoder.cpp:1403-1404` | CPU |
| `receive_frame` | `avcodec_receive_frame()` — retrieves decoded YUV frame | `SingleStreamDecoder.cpp:1338-1339` | CPU |
| `convert_avframe` | YUV→RGB color conversion via swscale or filtergraph | `SingleStreamDecoder.cpp:1455-1457` | CPU |
| `permute` | HWC→CHW tensor permutation | `SingleStreamDecoder.cpp:1471` | CPU |

### Time Breakdown

For testsrc-generated H.264 videos (720p and 1080p, 30fps):

- **~50%** in `send_packet` (software decode)
- **~50%** in `convert_avframe` (color conversion)
- `demux`, `seek`, `permute` are negligible

### Parallelism Opportunity: 2-Stage Pipeline

The two dominant stages (software decode and color conversion) are both CPU-bound
but independent for different frames. This enables **pipeline parallelism**:

```
Sequential:
  [decode F1][convert F1][decode F2][convert F2][decode F3][convert F3]

Pipelined (2 threads):
  Main thread:    [decode F1][decode F2 ][decode F3 ]
  Worker thread:            [convert F1][convert F2][convert F3]
```

**Implementation** (see commit `f4460b1`):
- A dedicated worker thread in `CpuDeviceInterface` runs a `colorConversionWorker()` loop
- `SingleStreamDecoder::getFramesAtIndices()` enqueues decoded AVFrames for async
  color conversion, then dequeues the previous frame's result while decoding the next
- Backpressure via `kMaxQueueDepth = 2` prevents unbounded memory growth
- With a balanced 50/50 split, this achieves close to **2x throughput**

---

## CUDA (Beta) Decoding Pipeline

### Pipeline Stages

| Stage | What it does | Code location | Hardware unit |
|---|---|---|---|
| `seek` | `avformat_seek_file()` + codec flush | `SingleStreamDecoder.cpp:1272` | CPU |
| `demux` | `av_read_frame()` — reads compressed packet | `SingleStreamDecoder.cpp:1371-1372` | CPU |
| `bitstream_filter` | Annex-B ↔ AVCC conversion via FFmpeg BSF | `BetaCudaDeviceInterface.cpp:496-498` | CPU |
| `packet_parse_and_decode` | `cuvidParseVideoData()` — submits packet to NVDEC parser | `BetaCudaDeviceInterface.cpp:507-508` | CPU → NVDEC |
| `nvdec_decode` | `cuvidDecodePicture()` — **non-blocking** NVDEC HW decode | `BetaCudaDeviceInterface.cpp:564-565` | NVDEC (fixed-function HW) |
| `map_frame` | `cuvidMapVideoFrame()` — **blocks** until decode completes | `BetaCudaDeviceInterface.cpp:624-625` | NVDEC → GPU memory |
| `unmap_frame` | `cuvidUnmapVideoFrame()` — releases output surface | `BetaCudaDeviceInterface.cpp:647-648` | GPU |
| `tensor_alloc` | Allocates empty HWC CUDA tensor | `CUDACommon.cpp` | CPU + GPU |
| `stream_sync` | Synchronizes NVDEC stream with NPP stream | `CUDACommon.cpp` | GPU |
| `color_conversion` | NPP NV12→RGB kernel | `CUDACommon.cpp` | CUDA cores |
| `permute` | HWC→CHW tensor permutation | `SingleStreamDecoder.cpp:1471` | CUDA cores |

### Current Execution Model

Everything is **serial**. For each frame:

```
CPU:          [demux][BSF][parse+submit]...........[next demux]...
NVDEC:                         [decode]
                                       ↓ map_frame blocks CPU
CUDA cores:                                       [color convert][permute]
```

The CPU sits idle while waiting for `cuvidMapVideoFrame` to return, and NVDEC sits
idle while the CPU demuxes and while CUDA cores do color conversion.

### Parallelism Opportunity: 3-Stage Pipeline

Three **independent hardware units** can run simultaneously:

1. **CPU** — demux + BSF + `cuvidParseVideoData` (packet parsing and submission)
2. **NVDEC** — fixed-function hardware decoder (separate silicon from CUDA cores)
3. **CUDA cores** — NPP color conversion + permute

The ideal pipeline overlaps all three:

```
CPU:         [demux+parse F1][demux+parse F2][demux+parse F3][demux+parse F4]...
NVDEC:                       [decode F1     ][decode F2     ][decode F3     ]...
CUDA cores:                                  [color cvt F1  ][color cvt F2  ]...
```

### Implementation Approach

1. **Decouple packet submission from frame retrieval**: Submit multiple packets via
   `cuvidParseVideoData` without immediately calling `cuvidMapVideoFrame`. This keeps
   NVDEC busy while the CPU continues demuxing.

2. **Defer `cuvidMapVideoFrame`**: Instead of blocking on map right after decode, map
   frame N only when you need its data — by then, NVDEC may have already finished.

3. **Overlap color conversion with NVDEC decode**: While CUDA cores run NPP on
   frame N, NVDEC can be decoding frame N+1. These use different hardware units and
   don't contend.

4. **Backpressure**: NVDEC has a fixed number of output surfaces (the DPB — Decoded
   Picture Buffer). The queue depth is bounded by the number of surfaces available.
   Submitting too many packets without mapping/unmapping will cause `cuvidMapVideoFrame`
   to fail.

### Key Constraint: Output Surface Management

Currently, `unmapPreviousFrame()` is called just before `cuvidMapVideoFrame()` for
the next frame (see `BetaCudaDeviceInterface.cpp:621`). This keeps at most one frame
mapped at a time. A pipelined approach would need to manage multiple mapped frames,
unmapping only after color conversion completes.

### Expected Speedup

With 3 pipeline stages on 3 hardware units, the theoretical max is **3x** (if all
stages take equal time). In practice, speedup is bounded by the slowest stage.
The dominant stages are typically `map_frame` (blocking NVDEC wait) and
`color_conversion` (NPP), so a realistic target is **1.5-2.5x** depending on
resolution and codec complexity.

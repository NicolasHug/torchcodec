"""Run benchmarks across multiple test videos for CPU and/or CUDA (beta),
then generate plots."""

import argparse
import subprocess
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torchcodec
from torchcodec._core import ops

# Leaf-level categories only (no parent timers that double-count)
# Keys are C++ timer names, values are display names for plots
CPU_LEAF_CATEGORIES = {
    "seek": "seek",
    "demux": "demux",
    "decode": "decode",
    "convert_avframe": "YUV -> RGB",
    "permute": "permute",
}

CUDA_LEAF_CATEGORIES = {
    "seek": "seek",
    "demux": "demux",
    "bitstream_filter": "BSF",
    # packet_parse_and_decode includes the nvdec_decode callback, but
    # nvdec_decode (cuvidDecodePicture) is non-blocking so its measured time is
    # just launch overhead — the real NVDEC wait is in map_frame.  We treat
    # packet_parse_and_decode as a leaf representing "CPU-side parsing + trivial
    # non-blocking decode submit".
    # We exclude "decode" (parent of bitstream_filter + packet_parse_and_decode)
    # and "nvdec_decode" (nested inside packet_parse_and_decode) to avoid
    # double-counting.
    "packet_parse_and_decode": "parsing",
    "map_frame": "map_frame",
    "unmap_frame": "unmap_frame",
    "convert_avframe": "YUV -> RGB",
    "permute": "permute",
    "decoder_creation": "decoder_creation",
    "decoder_destruction": "decoder_destruction",
}

VIDEO_DIR = "benchmarks/test_videos"

# (label, resolution WxH, duration seconds)
VIDEO_SPECS_720P = [
    ("720p_1s", "1280x720", 1),
    ("720p_5s", "1280x720", 5),
    ("720p_10s", "1280x720", 10),
    ("720p_30s", "1280x720", 30),
]

VIDEO_SPECS_1080P = [
    ("1080p_1s", "1920x1080", 1),
    ("1080p_5s", "1920x1080", 5),
    ("1080p_10s", "1920x1080", 10),
    ("1080p_30s", "1920x1080", 30),
]

NUM_RUNS = 3
OUTPUT_DIR = "benchmarks/benchmark_plots"
DEFAULT_GOP = 50
FPS = 30


def get_video_path(label, gop):
    return os.path.join(VIDEO_DIR, f"{label}_g{gop}.mp4")


def ensure_test_videos(specs, gop):
    """Generate test videos with ffmpeg if they don't already exist."""
    os.makedirs(VIDEO_DIR, exist_ok=True)
    for label, size, duration in specs:
        path = get_video_path(label, gop)
        if os.path.exists(path):
            continue
        print(f"  Generating {path}...")
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"testsrc=duration={duration}:size={size}:rate={FPS}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-g", str(gop),
            path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    print()


def _create_decoder(video_path, device, use_beta):
    if use_beta:
        with torchcodec.decoders.set_cuda_backend("beta"):
            return torchcodec.decoders.VideoDecoder(
                video_path, device=device, seek_mode="approximate"
            )
    else:
        return torchcodec.decoders.VideoDecoder(
            video_path, device=device, seek_mode="approximate"
        )


def _decode_frames(decoder, sampling, step):
    total_frames = len(decoder)
    if sampling == "all":
        for frame in decoder:
            pass
        return total_frames
    else:
        indices = list(range(0, total_frames, step))
        decoder.get_frames_at(indices)
        return len(indices)


def run_one(video_path, device, use_beta=False, sampling="all", step=50,
            warmup=False):
    """Run benchmark for a single video.

    Args:
        sampling: "all" to decode every frame, "step" to decode every `step`-th frame.
        warmup: if True, run one untimed warmup iteration first (e.g. to warm the NVDEC cache).
    """
    if warmup:
        decoder = _create_decoder(video_path, device, use_beta)
        _decode_frames(decoder, sampling, step)
        del decoder

    all_results = []
    num_frames_decoded = 0
    for _ in range(NUM_RUNS):
        ops.reset_benchmark()
        ops.enable_benchmark(True)

        decoder = _create_decoder(video_path, device, use_beta)
        num_frames_decoded = _decode_frames(decoder, sampling, step)
        del decoder

        ops.enable_benchmark(False)
        all_results.append(ops.get_benchmark_results())

    all_cats = set()
    for r in all_results:
        all_cats.update(r.keys())

    averaged = {}
    for cat in all_cats:
        vals = [r[cat] for r in all_results if cat in r]
        if vals:
            averaged[cat] = (
                sum(v[0] for v in vals) / len(vals),
                sum(v[1] for v in vals) // len(vals),
            )
    return averaged, num_frames_decoded


def _make_stacked_bar_trio(
    fig, all_results, step_results, labels, leaf_categories, cmap, step,
    value_fn, title, ylabel,
):
    """Draw 3 subplots: left = all+step side-by-side, top-right = all only, bottom-right = step only (own scale).

    leaf_categories: dict mapping cpp_timer_name -> display_name
    """
    import numpy as np
    from matplotlib.gridspec import GridSpec

    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.2, 1], hspace=0.35, wspace=0.3)
    ax_main = fig.add_subplot(gs[:, 0])
    ax_all = fig.add_subplot(gs[0, 1])
    ax_step = fig.add_subplot(gs[1, 1])

    cat_items = list(leaf_categories.items())
    x = np.arange(len(labels))
    w = 0.35

    # --- Main plot: side-by-side ---
    bottoms_all = np.zeros(len(labels))
    for ci, (cpp_name, display_name) in enumerate(cat_items):
        vals = []
        for l in labels:
            r, nf = all_results[l]
            total_ms = r[cpp_name][0] if cpp_name in r else 0
            vals.append(value_fn(total_ms, nf))
        vals = np.array(vals)
        ax_main.bar(
            x - w / 2, vals, w, bottom=bottoms_all,
            label=display_name, color=cmap[ci], edgecolor="white", linewidth=0.5,
        )
        bottoms_all += vals

    bottoms_step = np.zeros(len(labels))
    for ci, (cpp_name, display_name) in enumerate(cat_items):
        vals = []
        for l in labels:
            r, nf = step_results[l]
            total_ms = r[cpp_name][0] if cpp_name in r else 0
            vals.append(value_fn(total_ms, nf))
        vals = np.array(vals)
        ax_main.bar(
            x + w / 2, vals, w, bottom=bottoms_step,
            color=cmap[ci], edgecolor="white", linewidth=0.5,
            hatch="//", alpha=0.85,
        )
        bottoms_step += vals

    for i in range(len(labels)):
        ax_main.text(x[i] - w / 2, bottoms_all[i] + bottoms_all.max() * 0.01,
                     "all", ha="center", va="bottom", fontsize=7, fontstyle="italic")
        ax_main.text(x[i] + w / 2, bottoms_step[i] + bottoms_all.max() * 0.01,
                     f"1/{step}", ha="center", va="bottom", fontsize=7, fontstyle="italic")

    ax_main.set_xticks(x)
    ax_main.set_xticklabels(labels, rotation=30, ha="right")
    ax_main.set_ylabel(ylabel)
    ax_main.set_title("All vs Sampled", fontsize=11, fontweight="bold")
    ax_main.legend(loc="upper left", fontsize=7, ncol=1)

    # --- Top-right: all only ---
    bottoms = np.zeros(len(labels))
    for ci, (cpp_name, _) in enumerate(cat_items):
        vals = []
        for l in labels:
            r, nf = all_results[l]
            total_ms = r[cpp_name][0] if cpp_name in r else 0
            vals.append(value_fn(total_ms, nf))
        vals = np.array(vals)
        ax_all.bar(x, vals, 0.6, bottom=bottoms, color=cmap[ci], edgecolor="white", linewidth=0.5)
        bottoms += vals

    ax_all.set_xticks(x)
    ax_all.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax_all.set_ylabel(ylabel, fontsize=9)
    ax_all.set_title("All frames", fontsize=11, fontweight="bold")

    # --- Bottom-right: step only (own scale) ---
    bottoms = np.zeros(len(labels))
    for ci, (cpp_name, _) in enumerate(cat_items):
        vals = []
        for l in labels:
            r, nf = step_results[l]
            total_ms = r[cpp_name][0] if cpp_name in r else 0
            vals.append(value_fn(total_ms, nf))
        vals = np.array(vals)
        ax_step.bar(x, vals, 0.6, bottom=bottoms, color=cmap[ci], edgecolor="white", linewidth=0.5,
                    hatch="//", alpha=0.85)
        bottoms += vals

    ax_step.set_xticks(x)
    ax_step.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax_step.set_ylabel(ylabel, fontsize=9)
    ax_step.set_title(f"Every {step} frames", fontsize=11, fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold")


def make_plots(cpu_all, cpu_step, cuda_all, cuda_step, output_dir, step, labels, gop,
               use_cache=True):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f8f8",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
    })

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    gop_info = f"(GOP={gop})"
    cache_info = ", cache" if use_cache else ", no-cache"
    cache_suffix = "" if use_cache else "_no_cache"
    plot_idx = 1

    per_frame = lambda total_ms, nf: total_ms / nf if nf > 0 else 0
    total_time = lambda total_ms, nf: total_ms

    # ─── CPU breakdown: per-frame ───
    if cpu_all:
        cpu_cmap = plt.cm.Set2(np.linspace(0, 1, len(CPU_LEAF_CATEGORIES)))

        fig = plt.figure(figsize=(18, 6))
        _make_stacked_bar_trio(
            fig, cpu_all, cpu_step, labels, CPU_LEAF_CATEGORIES, cpu_cmap, step,
            per_frame, f"CPU Decode: Per-frame Time Breakdown {gop_info}", "Time per frame (ms)",
        )
        plt.tight_layout()
        path = os.path.join(output_dir, f"{plot_idx:02d}_cpu_perframe.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {path}")
        plot_idx += 1

        # ─── CPU breakdown: total time ───
        fig = plt.figure(figsize=(18, 6))
        _make_stacked_bar_trio(
            fig, cpu_all, cpu_step, labels, CPU_LEAF_CATEGORIES, cpu_cmap, step,
            total_time, f"CPU Decode: Total Time Breakdown {gop_info}", "Total time (ms)",
        )
        plt.tight_layout()
        path = os.path.join(output_dir, f"{plot_idx:02d}_cpu_total.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {path}")
        plot_idx += 1

    # ─── CUDA breakdown: per-frame ───
    if cuda_all:
        cuda_cmap = plt.cm.tab20(np.linspace(0, 1, len(CUDA_LEAF_CATEGORIES)))

        fig = plt.figure(figsize=(18, 7))
        _make_stacked_bar_trio(
            fig, cuda_all, cuda_step, labels, CUDA_LEAF_CATEGORIES, cuda_cmap, step,
            per_frame, f"CUDA (beta) Decode: Per-frame Time {gop_info}{cache_info} — {gpu_name}",
            "Time per frame (ms)",
        )
        plt.tight_layout()
        path = os.path.join(output_dir, f"{plot_idx:02d}_cuda_perframe{cache_suffix}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {path}")
        plot_idx += 1

        # ─── CUDA breakdown: total time ───
        fig = plt.figure(figsize=(18, 7))
        _make_stacked_bar_trio(
            fig, cuda_all, cuda_step, labels, CUDA_LEAF_CATEGORIES, cuda_cmap, step,
            total_time, f"CUDA (beta) Decode: Total Time {gop_info}{cache_info} — {gpu_name}",
            "Total time (ms)",
        )
        plt.tight_layout()
        path = os.path.join(output_dir, f"{plot_idx:02d}_cuda_total{cache_suffix}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {path}")
        plot_idx += 1



def _leaf_total_ms(results, leaf_categories):
    """Sum only leaf-level timer values to avoid double-counting parent timers."""
    return sum(results[k][0] for k in leaf_categories if k in results)


def main():
    parser = argparse.ArgumentParser(description="Benchmark torchcodec decoding pipeline")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "both"],
        default="both",
        help="Which device to benchmark (default: both)",
    )
    parser.add_argument(
        "--resolution",
        choices=["720p", "1080p", "all"],
        default="720p",
        help="Which resolution(s) to benchmark (default: 720p)",
    )
    parser.add_argument("--gop", type=int, default=DEFAULT_GOP,
                        help=f"GOP size (keyframe interval) for test videos (default: {DEFAULT_GOP})")
    parser.add_argument("--num-runs", type=int, default=NUM_RUNS, help="Number of runs to average")
    parser.add_argument("--step", type=int, default=50, help="Frame step for sampling comparison")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory for plots")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable NVDEC decoder cache for CUDA benchmarks (sets cache capacity to 0)")
    args = parser.parse_args()

    if args.resolution == "720p":
        specs = VIDEO_SPECS_720P
    elif args.resolution == "1080p":
        specs = VIDEO_SPECS_1080P
    else:
        specs = VIDEO_SPECS_720P + VIDEO_SPECS_1080P

    run_cpu = args.device in ("cpu", "both")
    run_cuda = args.device in ("cuda", "both")

    if run_cuda and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU only")
        run_cuda = False
        run_cpu = True

    if run_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
    print(f"Resolution: {args.resolution}")
    print(f"GOP size: {args.gop} (keyframe every {args.gop} frames)")
    print(f"Averaging over {args.num_runs} runs per video")
    print(f"Sampling comparison step: every {args.step} frames")
    if run_cuda:
        use_cache = not args.no_cache
        print(f"NVDEC cache: {'enabled (with warmup)' if use_cache else 'disabled'}")
    print()

    # Generate test videos if needed
    print("Checking test videos...")
    ensure_test_videos(specs, args.gop)

    labels = [label for label, _, _ in specs]
    videos = [(label, get_video_path(label, args.gop)) for label, _, _ in specs]

    cpu_all = {}
    cpu_step = {}
    cuda_all = {}
    cuda_step = {}

    if run_cpu:
        print("Running CPU benchmarks (all frames)...")
        for label, path in videos:
            print(f"  CPU  {label}...", end=" ", flush=True)
            results, num_frames = run_one(path, device="cpu")
            total = _leaf_total_ms(results, CPU_LEAF_CATEGORIES)
            fps = num_frames / (total / 1000) if total > 0 else 0
            print(f"{num_frames} frames, {total:.1f}ms, {fps:.0f} fps")
            cpu_all[label] = (results, num_frames)

        print(f"\nRunning CPU benchmarks (every {args.step} frames)...")
        for label, path in videos:
            print(f"  CPU  {label}...", end=" ", flush=True)
            results, num_frames = run_one(
                path, device="cpu", sampling="step", step=args.step
            )
            total = _leaf_total_ms(results, CPU_LEAF_CATEGORIES)
            fps = num_frames / (total / 1000) if total > 0 else 0
            print(f"{num_frames} frames, {total:.1f}ms, {fps:.0f} fps")
            cpu_step[label] = (results, num_frames)

    if run_cuda:
        use_cache = not args.no_cache
        if not use_cache:
            torchcodec.decoders.set_nvdec_cache_capacity(0)

        cache_label = "cache" if use_cache else "no-cache"
        print(f"\nRunning CUDA (beta) benchmarks (all frames, {cache_label})...")
        for label, path in videos:
            print(f"  CUDA {label}...", end=" ", flush=True)
            results, num_frames = run_one(
                path, device="cuda:0", use_beta=True, warmup=use_cache,
            )
            total = _leaf_total_ms(results, CUDA_LEAF_CATEGORIES)
            fps = num_frames / (total / 1000) if total > 0 else 0
            print(f"{num_frames} frames, {total:.1f}ms, {fps:.0f} fps")
            cuda_all[label] = (results, num_frames)

        print(f"\nRunning CUDA (beta) benchmarks (every {args.step} frames, {cache_label})...")
        for label, path in videos:
            print(f"  CUDA {label}...", end=" ", flush=True)
            results, num_frames = run_one(
                path, device="cuda:0", use_beta=True, sampling="step",
                step=args.step, warmup=use_cache,
            )
            total = _leaf_total_ms(results, CUDA_LEAF_CATEGORIES)
            fps = num_frames / (total / 1000) if total > 0 else 0
            print(f"{num_frames} frames, {total:.1f}ms, {fps:.0f} fps")
            cuda_step[label] = (results, num_frames)

    # --- Generate plots ---
    print(f"\nGenerating plots in {args.output_dir}/...")
    make_plots(
        cpu_all if run_cpu else None,
        cpu_step if run_cpu else None,
        cuda_all if run_cuda else None,
        cuda_step if run_cuda else None,
        args.output_dir,
        args.step,
        labels,
        args.gop,
        use_cache=not args.no_cache,
    )
    print("\nDone!")


if __name__ == "__main__":
    main()

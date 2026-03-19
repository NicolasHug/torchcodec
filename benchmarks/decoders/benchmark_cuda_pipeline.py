"""Benchmark the Beta CUDA decoding pipeline with per-stage timing.

Uses the instrumentation in torchcodec._core to measure time spent in each
stage of the CUDA decoding pipeline: decoder creation, cache operations,
demuxing, parsing, NVDEC decode, color conversion, etc.

Usage:
    python benchmarks/decoders/benchmark_cuda_pipeline.py \
        --video_path test/resources/nasa_13013.mp4
"""

import argparse
import sys

import torch

import torchcodec
from torchcodec._core import ops


ORDERED_CATEGORIES = [
    "decoder_creation",
    "cache_get",
    "cache_return",
    "seek",
    "demux",
    "bitstream_filter",
    "packet_parse_and_decode",
    "nvdec_decode",
    "map_frame",
    "unmap_frame",
    "tensor_alloc",
    "stream_sync",
    "color_conversion",
    "permute",
]


def decode_all_frames(video_path, device):
    with torchcodec.decoders.set_cuda_backend("beta"):
        decoder = torchcodec.decoders.VideoDecoder(
            video_path, device=device, seek_mode="approximate"
        )
    for frame in decoder:
        pass
    return len(decoder)


def decode_every_n(video_path, device, n):
    with torchcodec.decoders.set_cuda_backend("beta"):
        decoder = torchcodec.decoders.VideoDecoder(
            video_path, device=device, seek_mode="approximate"
        )
    num_frames = len(decoder)
    count = 0
    for i in range(0, num_frames, n):
        _ = decoder[i]
        count += 1
    return count


def decode_keyframes_only(video_path, device):
    with torchcodec.decoders.set_cuda_backend("beta"):
        decoder = torchcodec.decoders.VideoDecoder(
            video_path, device=device, seek_mode="approximate"
        )
    num_frames = len(decoder)
    count = 0
    for i in range(0, num_frames, max(1, num_frames // 20)):
        _ = decoder[i]
        count += 1
    return count


PATTERNS = {
    "all": lambda vp, dev: decode_all_frames(vp, dev),
    "every_2": lambda vp, dev: decode_every_n(vp, dev, 2),
    "every_5": lambda vp, dev: decode_every_n(vp, dev, 5),
    "every_10": lambda vp, dev: decode_every_n(vp, dev, 10),
    "keyframes_only": lambda vp, dev: decode_keyframes_only(vp, dev),
}


def run_benchmark(video_path, device, pattern, num_runs):
    all_results = []
    for run_idx in range(num_runs):
        ops.reset_benchmark()
        ops.enable_benchmark(True)

        decode_fn = PATTERNS[pattern]
        num_decoded = decode_fn(video_path, device)

        ops.enable_benchmark(False)
        results = ops.get_benchmark_results()
        all_results.append(results)

        if run_idx == 0:
            print(f"  Pattern '{pattern}': decoded {num_decoded} frames")

    # Average results across runs
    averaged = {}
    for cat in ORDERED_CATEGORIES:
        total_ms_sum = 0.0
        count_sum = 0
        num_present = 0
        for result in all_results:
            if cat in result:
                total_ms_sum += result[cat][0]
                count_sum += result[cat][1]
                num_present += 1
        if num_present > 0:
            averaged[cat] = (total_ms_sum / num_present, count_sum // num_present)

    # Also include any categories not in ORDERED_CATEGORIES
    all_cats = set()
    for result in all_results:
        all_cats.update(result.keys())
    for cat in sorted(all_cats - set(ORDERED_CATEGORIES)):
        total_ms_sum = 0.0
        count_sum = 0
        num_present = 0
        for result in all_results:
            if cat in result:
                total_ms_sum += result[cat][0]
                count_sum += result[cat][1]
                num_present += 1
        if num_present > 0:
            averaged[cat] = (total_ms_sum / num_present, count_sum // num_present)

    return averaged


def print_results_table(results_by_pattern):
    print("\n" + "=" * 90)
    print(f"{'Category':<30} {'Total (ms)':>12} {'Count':>8} {'Avg (ms)':>12}")
    print("=" * 90)

    for pattern, results in results_by_pattern.items():
        print(f"\n--- {pattern} ---")
        grand_total = 0.0
        # Print ordered categories first
        for cat in ORDERED_CATEGORIES:
            if cat in results:
                total_ms, count = results[cat]
                avg_ms = total_ms / count if count > 0 else 0
                grand_total += total_ms
                print(f"  {cat:<28} {total_ms:>12.3f} {count:>8} {avg_ms:>12.4f}")
        # Print any extra categories
        for cat in sorted(set(results.keys()) - set(ORDERED_CATEGORIES)):
            total_ms, count = results[cat]
            avg_ms = total_ms / count if count > 0 else 0
            grand_total += total_ms
            print(f"  {cat:<28} {total_ms:>12.3f} {count:>8} {avg_ms:>12.4f}")
        print(f"  {'TOTAL':<28} {grand_total:>12.3f}")


def generate_plots(results_by_pattern, output_path):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not available, skipping plot generation")
        return

    # Collect all categories that have data, in order
    active_cats = []
    for cat in ORDERED_CATEGORIES:
        for results in results_by_pattern.values():
            if cat in results:
                active_cats.append(cat)
                break
    # Add any extra categories
    all_cats = set()
    for results in results_by_pattern.values():
        all_cats.update(results.keys())
    for cat in sorted(all_cats - set(ORDERED_CATEGORIES)):
        active_cats.append(cat)

    patterns = list(results_by_pattern.keys())

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # --- Stacked bar chart ---
    ax = axes[0]
    x = np.arange(len(patterns))
    bar_width = 0.6
    bottoms = np.zeros(len(patterns))
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(active_cats), 1)))

    for cat_idx, cat in enumerate(active_cats):
        values = []
        for pattern in patterns:
            results = results_by_pattern[pattern]
            values.append(results.get(cat, (0, 0))[0])
        values = np.array(values)
        ax.bar(
            x,
            values,
            bar_width,
            bottom=bottoms,
            label=cat,
            color=colors[cat_idx],
        )
        bottoms += values

    ax.set_xlabel("Sampling Pattern")
    ax.set_ylabel("Total Time (ms)")
    ax.set_title("Time Breakdown by Decoding Stage")
    ax.set_xticks(x)
    ax.set_xticklabels(patterns, rotation=30, ha="right")
    ax.legend(loc="upper left", fontsize=7)

    # --- Pie chart for first pattern ---
    ax = axes[1]
    first_pattern = patterns[0]
    first_results = results_by_pattern[first_pattern]
    pie_labels = []
    pie_values = []
    pie_colors = []
    for cat_idx, cat in enumerate(active_cats):
        if cat in first_results and first_results[cat][0] > 0:
            pie_labels.append(cat)
            pie_values.append(first_results[cat][0])
            pie_colors.append(colors[cat_idx])

    if pie_values:
        ax.pie(
            pie_values,
            labels=pie_labels,
            colors=pie_colors,
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title(f"Time Breakdown: '{first_pattern}' pattern")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Beta CUDA decoding pipeline"
    )
    parser.add_argument("--video_path", required=True, help="Path to video file")
    parser.add_argument("--device", default="cuda:0", help="CUDA device (default: cuda:0)")
    parser.add_argument(
        "--pattern",
        default="all",
        choices=list(PATTERNS.keys()) + ["compare_all"],
        help="Sampling pattern (default: all). Use 'compare_all' to run all patterns.",
    )
    parser.add_argument(
        "--num_runs", type=int, default=3, help="Number of runs to average (default: 3)"
    )
    parser.add_argument(
        "--output", default="benchmark_results.png", help="Output plot path"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available, exiting.")
        sys.exit(1)

    print(f"Using device: {args.device}")
    print(f"Video: {args.video_path}")

    if args.pattern == "compare_all":
        patterns_to_run = list(PATTERNS.keys())
    else:
        patterns_to_run = [args.pattern]

    results_by_pattern = {}
    for pattern in patterns_to_run:
        print(f"\nRunning pattern '{pattern}' ({args.num_runs} runs)...")
        results_by_pattern[pattern] = run_benchmark(
            args.video_path, args.device, pattern, args.num_runs
        )

    print_results_table(results_by_pattern)
    generate_plots(results_by_pattern, args.output)


if __name__ == "__main__":
    main()

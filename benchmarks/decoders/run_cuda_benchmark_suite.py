"""Run benchmarks across multiple test videos for both CPU and CUDA (beta),
then generate plots."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torchcodec
from torchcodec._core import ops

# Leaf-level categories only (no parent timers that double-count)
CPU_LEAF_CATEGORIES = [
    "seek",
    "demux",
    "send_packet",
    "convert_avframe",
    "permute",
]

CUDA_LEAF_CATEGORIES = [
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
    "decoder_creation",
]

VIDEOS = [
    ("720p_1s", "benchmarks/test_videos/720p_1s.mp4"),
    ("720p_5s", "benchmarks/test_videos/720p_5s.mp4"),
    ("720p_10s", "benchmarks/test_videos/720p_10s.mp4"),
    ("720p_30s", "benchmarks/test_videos/720p_30s.mp4"),
    ("1080p_1s", "benchmarks/test_videos/1080p_1s.mp4"),
    ("1080p_5s", "benchmarks/test_videos/1080p_5s.mp4"),
    ("1080p_10s", "benchmarks/test_videos/1080p_10s.mp4"),
    ("1080p_30s", "benchmarks/test_videos/1080p_30s.mp4"),
]

NUM_RUNS = 3
OUTPUT_DIR = "benchmarks/benchmark_plots"


def run_one(video_path, device, use_beta=False):
    all_results = []
    num_frames = 0
    for _ in range(NUM_RUNS):
        ops.reset_benchmark()
        ops.enable_benchmark(True)

        if use_beta:
            with torchcodec.decoders.set_cuda_backend("beta"):
                decoder = torchcodec.decoders.VideoDecoder(
                    video_path, device=device, seek_mode="approximate"
                )
        else:
            decoder = torchcodec.decoders.VideoDecoder(
                video_path, device=device, seek_mode="approximate"
            )
        num_frames = len(decoder)
        for frame in decoder:
            pass
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
    return averaged, num_frames


def make_plots(cpu_results, cuda_results, output_dir):
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

    labels = [l for l, _ in VIDEOS]
    gpu_name = torch.cuda.get_device_name(0)

    # ─── PLOT 1: CPU vs CUDA FPS bar chart ───
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels))
    w = 0.35

    cpu_fps = []
    cuda_fps = []
    for l in labels:
        cr, cnf = cpu_results[l]
        ct = sum(v[0] for v in cr.values())
        cpu_fps.append(cnf / (ct / 1000) if ct > 0 else 0)
        gr, gnf = cuda_results[l]
        gt = sum(v[0] for v in gr.values())
        cuda_fps.append(gnf / (gt / 1000) if gt > 0 else 0)

    bars_cpu = ax.bar(x - w / 2, cpu_fps, w, label="CPU", color="#4C72B0", edgecolor="white")
    bars_cuda = ax.bar(x + w / 2, cuda_fps, w, label=f"CUDA beta ({gpu_name})", color="#DD8452", edgecolor="white")

    ax.set_ylabel("Frames per Second")
    ax.set_title("CPU vs CUDA (beta): Decode Throughput", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend(fontsize=10)

    # Add value labels
    for bar in bars_cpu:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_cuda:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "01_fps_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")

    # ─── PLOT 2: CPU stacked bar (per-frame ms) ───
    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.cm.Set2(np.linspace(0, 1, len(CPU_LEAF_CATEGORIES)))
    bottoms = np.zeros(len(labels))

    for ci, cat in enumerate(CPU_LEAF_CATEGORIES):
        vals = []
        for l in labels:
            r, nf = cpu_results[l]
            vals.append(r[cat][0] / nf if cat in r and nf > 0 else 0)
        vals = np.array(vals)
        ax.bar(x, vals, 0.6, bottom=bottoms, label=cat, color=cmap[ci], edgecolor="white", linewidth=0.5)
        bottoms += vals

    ax.set_ylabel("Time per frame (ms)")
    ax.set_title("CPU Decode: Per-frame Time Breakdown", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "02_cpu_breakdown.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")

    # ─── PLOT 3: CUDA stacked bar (per-frame ms) ───
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.tab20(np.linspace(0, 1, len(CUDA_LEAF_CATEGORIES)))
    bottoms = np.zeros(len(labels))

    for ci, cat in enumerate(CUDA_LEAF_CATEGORIES):
        vals = []
        for l in labels:
            r, nf = cuda_results[l]
            vals.append(r[cat][0] / nf if cat in r and nf > 0 else 0)
        vals = np.array(vals)
        ax.bar(x, vals, 0.6, bottom=bottoms, label=cat, color=cmap[ci], edgecolor="white", linewidth=0.5)
        bottoms += vals

    ax.set_ylabel("Time per frame (ms)")
    ax.set_title(f"CUDA (beta) Decode: Per-frame Time Breakdown — {gpu_name}", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    plt.tight_layout()
    path = os.path.join(output_dir, "03_cuda_breakdown.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")

    # ─── PLOT 4: Pie charts for 720p_30s and 1080p_30s, CPU and CUDA ───
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    for col, (res_label, res_key) in enumerate([("720p_30s", "720p_30s"), ("1080p_30s", "1080p_30s")]):
        # CPU pie
        ax = axes[0, col]
        cr, _ = cpu_results[res_key]
        cpu_vals = [(cat, cr[cat][0]) for cat in CPU_LEAF_CATEGORIES if cat in cr and cr[cat][0] > 0]
        if cpu_vals:
            pie_labels, pie_vals = zip(*cpu_vals)
            colors = plt.cm.Set2(np.linspace(0, 1, len(pie_vals)))
            wedges, texts, autotexts = ax.pie(
                pie_vals, labels=pie_labels, colors=colors,
                autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
                startangle=90, textprops={"fontsize": 9})
            for at in autotexts:
                at.set_fontsize(8)
        ax.set_title(f"CPU — {res_label}", fontsize=12, fontweight="bold")

        # CUDA pie
        ax = axes[1, col]
        gr, _ = cuda_results[res_key]
        cuda_vals = [(cat, gr[cat][0]) for cat in CUDA_LEAF_CATEGORIES if cat in gr and gr[cat][0] > 0]
        if cuda_vals:
            pie_labels, pie_vals = zip(*cuda_vals)
            colors = plt.cm.tab20(np.linspace(0, 1, len(pie_vals)))
            wedges, texts, autotexts = ax.pie(
                pie_vals, labels=pie_labels, colors=colors,
                autopct=lambda p: f"{p:.1f}%" if p > 1 else "",
                startangle=90, textprops={"fontsize": 9})
            for at in autotexts:
                at.set_fontsize(8)
        ax.set_title(f"CUDA (beta) — {res_label}", fontsize=12, fontweight="bold")

    fig.suptitle("Time Breakdown (30s videos)", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "04_pie_charts.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")

    # ─── PLOT 5: 720p vs 1080p scaling for key stages ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    durations = ["1s", "5s", "10s", "30s"]
    key_cpu_cats = ["send_packet", "convert_avframe"]
    key_cuda_cats = ["map_frame", "color_conversion", "packet_parse_and_decode"]

    # CPU scaling
    ax = axes[0]
    for cat in key_cpu_cats:
        vals_720 = []
        vals_1080 = []
        for d in durations:
            r720, nf720 = cpu_results[f"720p_{d}"]
            r1080, nf1080 = cpu_results[f"1080p_{d}"]
            vals_720.append(r720[cat][0] / nf720 if cat in r720 and nf720 > 0 else 0)
            vals_1080.append(r1080[cat][0] / nf1080 if cat in r1080 and nf1080 > 0 else 0)
        ax.plot(durations, vals_720, "o-", label=f"{cat} (720p)")
        ax.plot(durations, vals_1080, "s--", label=f"{cat} (1080p)")
    ax.set_xlabel("Video Duration")
    ax.set_ylabel("Time per frame (ms)")
    ax.set_title("CPU: Key Stages — 720p vs 1080p", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)

    # CUDA scaling
    ax = axes[1]
    for cat in key_cuda_cats:
        vals_720 = []
        vals_1080 = []
        for d in durations:
            r720, nf720 = cuda_results[f"720p_{d}"]
            r1080, nf1080 = cuda_results[f"1080p_{d}"]
            vals_720.append(r720[cat][0] / nf720 if cat in r720 and nf720 > 0 else 0)
            vals_1080.append(r1080[cat][0] / nf1080 if cat in r1080 and nf1080 > 0 else 0)
        ax.plot(durations, vals_720, "o-", label=f"{cat} (720p)")
        ax.plot(durations, vals_1080, "s--", label=f"{cat} (1080p)")
    ax.set_xlabel("Video Duration")
    ax.set_ylabel("Time per frame (ms)")
    ax.set_title(f"CUDA (beta): Key Stages — 720p vs 1080p", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "05_resolution_scaling.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")

    # ─── PLOT 6: Per-frame time comparison (CPU vs CUDA side by side) ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, res in enumerate(["720p", "1080p"]):
        ax = axes[ax_idx]
        dur_labels = ["1s", "5s", "10s", "30s"]
        x_d = np.arange(len(dur_labels))

        cpu_pf = []
        cuda_pf = []
        for d in dur_labels:
            key = f"{res}_{d}"
            cr, cnf = cpu_results[key]
            ct = sum(v[0] for v in cr.values())
            cpu_pf.append(ct / cnf if cnf > 0 else 0)
            gr, gnf = cuda_results[key]
            gt = sum(v[0] for v in gr.values())
            cuda_pf.append(gt / gnf if gnf > 0 else 0)

        bars1 = ax.bar(x_d - w / 2, cpu_pf, w, label="CPU", color="#4C72B0", edgecolor="white")
        bars2 = ax.bar(x_d + w / 2, cuda_pf, w, label="CUDA (beta)", color="#DD8452", edgecolor="white")

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

        ax.set_ylabel("Time per frame (ms)")
        ax.set_title(f"{res}: Per-frame Decode Time", fontsize=13, fontweight="bold")
        ax.set_xticks(x_d)
        ax.set_xticklabels(dur_labels)
        ax.set_xlabel("Video Duration")
        ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "06_perframe_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Averaging over {NUM_RUNS} runs per video\n")

    # --- CPU benchmarks ---
    print("Running CPU benchmarks...")
    cpu_results = {}
    for label, path in VIDEOS:
        print(f"  CPU  {label}...", end=" ", flush=True)
        results, num_frames = run_one(path, device="cpu")
        total = sum(v[0] for v in results.values())
        fps = num_frames / (total / 1000) if total > 0 else 0
        print(f"{num_frames} frames, {total:.1f}ms, {fps:.0f} fps")
        cpu_results[label] = (results, num_frames)

    # --- CUDA (beta) benchmarks ---
    print("\nRunning CUDA (beta) benchmarks...")
    cuda_results = {}
    for label, path in VIDEOS:
        print(f"  CUDA {label}...", end=" ", flush=True)
        results, num_frames = run_one(path, device="cuda:0", use_beta=True)
        total = sum(v[0] for v in results.values())
        fps = num_frames / (total / 1000) if total > 0 else 0
        print(f"{num_frames} frames, {total:.1f}ms, {fps:.0f} fps")
        cuda_results[label] = (results, num_frames)

    # --- Generate plots ---
    print(f"\nGenerating plots in {OUTPUT_DIR}/...")
    make_plots(cpu_results, cuda_results, OUTPUT_DIR)
    print("\nDone!")


if __name__ == "__main__":
    main()

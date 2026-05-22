"""Plot VPT scoreboard and nMSE diagnostics."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_METRIC_RE = re.compile(r"([A-Za-z0-9_./@]+)=(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")


def parse_train_log(log_text: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line in log_text.splitlines():
        if "[train]" not in line or "update=" not in line:
            continue
        row = {key: float(value) for key, value in _METRIC_RE.findall(line)}
        if "update" in row:
            rows.append(row)
    return rows


def plot_training_loss_vpt(history: list[dict], output_dir: Path, metric: str = "VPT80@0.25") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not history:
        raise ValueError("No training history rows available to plot.")
    updates = np.asarray([row["update"] for row in history], dtype=np.float32)
    total_loss = np.asarray([row.get("loss/total", np.nan) for row in history], dtype=np.float32)
    one_step_loss = np.asarray([row.get("loss/one_step", np.nan) for row in history], dtype=np.float32)
    rollout_loss = np.asarray([row.get("loss/rollout", np.nan) for row in history], dtype=np.float32)
    vpt = np.asarray([row.get(metric, np.nan) for row in history], dtype=np.float32)
    if not np.any(np.isfinite(total_loss)) or not np.any(np.isfinite(vpt)):
        raise ValueError(f"Training history must contain loss/total and {metric}.")

    path = output_dir / "training_loss_vpt.png"
    fig, ax_loss = plt.subplots(figsize=(7, 4))
    ax_loss.plot(updates, total_loss, label="total loss", linewidth=2)
    if np.any(np.isfinite(one_step_loss)):
        ax_loss.plot(updates, one_step_loss, label="one-step loss", linewidth=1.5, alpha=0.8)
    if np.any(np.isfinite(rollout_loss)):
        ax_loss.plot(updates, rollout_loss, label="rollout loss", linewidth=1.5, alpha=0.8)
    ax_loss.set_xlabel("Training update")
    ax_loss.set_ylabel("Training loss")
    ax_loss.grid(True, alpha=0.3)

    ax_vpt = ax_loss.twinx()
    ax_vpt.plot(updates, vpt, label=metric, color="crimson", marker="o", linewidth=2)
    ax_vpt.set_ylabel(metric)

    finite_vpt = np.isfinite(vpt)
    if np.any(finite_vpt):
        finite_indices = np.flatnonzero(finite_vpt)
        best_idx = finite_indices[int(np.nanargmax(vpt[finite_vpt]))]
        ax_vpt.scatter([updates[best_idx]], [vpt[best_idx]], s=110, color="crimson", edgecolor="black", zorder=5)
        ax_vpt.annotate(
            f"best {int(vpt[best_idx])} @ {int(updates[best_idx])}",
            xy=(updates[best_idx], vpt[best_idx]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
        )

    loss_lines, loss_labels = ax_loss.get_legend_handles_labels()
    vpt_lines, vpt_labels = ax_vpt.get_legend_handles_labels()
    ax_loss.legend(loss_lines + vpt_lines, loss_labels + vpt_labels, loc="best")
    ax_loss.set_title(f"Training loss vs {metric}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_training_loss_vpt_from_log(log_path: str | Path, output_dir: Path, metric: str = "VPT80@0.25") -> Path:
    log_text = Path(log_path).read_text(encoding="utf-8")
    rows = parse_train_log(log_text)
    if not rows:
        raise ValueError(f"No '[train] update=...' metric lines found in {log_path}.")
    return plot_training_loss_vpt(rows, output_dir, metric=metric)


def plot_survival_curve(metrics: dict, output_dir: Path, vpt_steps: np.ndarray | None = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if vpt_steps is not None:
        max_horizon = int(metrics.get("max_horizon", int(np.max(vpt_steps))))
        hs = np.arange(1, max_horizon + 1)
        rates = np.asarray([(vpt_steps >= h).mean() for h in hs], dtype=np.float32)
    else:
        pairs = []
        for key, value in metrics.items():
            if key.startswith("VPT80@"):
                pairs.append((int(value), 0.8))
            if key.startswith("VPT50@"):
                pairs.append((int(value), 0.5))
        pairs = sorted(pairs)
        hs = np.asarray([h for h, _ in pairs], dtype=np.int32)
        rates = np.asarray([rate for _, rate in pairs], dtype=np.float32)
    path = output_dir / "survival_curve.png"
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(hs, rates, marker="o", linewidth=2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("Fraction below nMSE threshold")
    ax.set_title("VPT scoreboard curve at nMSE threshold 0.25")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_horizon_rmse(metrics: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "rollout_comparison.png"
    curve = np.asarray(metrics.get("nMSE_curve", []), dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(np.arange(1, len(curve) + 1), curve, linewidth=2)
    ax.set_xlabel("Open-loop prediction horizon")
    ax.set_ylabel("rollout-average normalized MSE")
    ax.set_title("Rollout-average nMSE by horizon")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir")
    parser.add_argument("--train-log")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-metric", default="VPT80@0.25")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    if args.eval_dir is not None:
        with (Path(args.eval_dir) / "metrics.json").open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        vpt_path = Path(args.eval_dir) / "per_window_vpt_0p25.npy"
        vpt_steps = np.load(vpt_path) if vpt_path.exists() else None
        outputs["survival_curve"] = str(plot_survival_curve(metrics, output_dir, vpt_steps))
        outputs["rollout_comparison"] = str(plot_horizon_rmse(metrics, output_dir))
    if args.train_log is not None:
        outputs["training_loss_vpt"] = str(plot_training_loss_vpt_from_log(args.train_log, output_dir, metric=args.train_metric))
    if not outputs:
        raise SystemExit("Pass --eval-dir, --train-log, or both.")
    print(outputs)


if __name__ == "__main__":
    main()

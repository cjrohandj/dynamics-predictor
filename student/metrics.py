"""Student-side scoreboard metric helpers.

Official grading uses the locked implementation in `wm_hw.official_metrics`.
Students may use this file for experiments and report diagnostics.
"""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

from wm_hw.official_metrics import compute_official_metrics as compute_scoreboard_metrics


_METRIC_RE = re.compile(r"([A-Za-z0-9_./@]+)=(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")


def parse_train_log(log_text: str) -> list[dict[str, float]]:
    """Parse `[train] update=... key=value ...` lines copied from Colab."""
    rows: list[dict[str, float]] = []
    for line in log_text.splitlines():
        if "[train]" not in line or "update=" not in line:
            continue
        row = {key: float(value) for key, value in _METRIC_RE.findall(line)}
        if "update" in row:
            rows.append(row)
    return rows


def plot_curriculum_vpt_diagnostics(
    log_text: str,
    output_path: str | Path = "artifacts/student/plots/curriculum_vpt_diagnostics.png",
    *,
    metric: str = "VPT80@0.25",
) -> Path:
    """Plot VPT against loss settings and the loss-setting schedule over time.

    The left panel uses rollout horizon as the x-axis because it is the most
    interpretable curriculum knob. Point color shows rollout weight, and point
    size shows one-step weight. The right panel shows how the curriculum knobs
    evolved by training update.
    """
    rows = parse_train_log(log_text)
    if not rows:
        raise ValueError("No '[train] update=...' metric lines found in log_text.")
    if metric not in rows[0]:
        available = sorted(rows[0])
        raise KeyError(f"{metric!r} not found in parsed log. Available keys include {available[:12]}.")

    updates = np.asarray([row["update"] for row in rows], dtype=np.float32)
    vpt = np.asarray([row.get(metric, np.nan) for row in rows], dtype=np.float32)
    rollout_horizon = np.asarray([row.get("loss/rollout_horizon", np.nan) for row in rows], dtype=np.float32)
    rollout_weight = np.asarray([row.get("loss/rollout_weight", np.nan) for row in rows], dtype=np.float32)
    one_step_weight = np.asarray([row.get("loss/one_step_weight", np.nan) for row in rows], dtype=np.float32)
    progress = np.asarray([row.get("loss/curriculum_progress", np.nan) for row in rows], dtype=np.float32)

    finite = np.isfinite(vpt) & np.isfinite(rollout_horizon) & np.isfinite(rollout_weight) & np.isfinite(one_step_weight)
    if not np.any(finite):
        raise ValueError("Parsed rows do not contain VPT and curriculum loss-parameter metrics.")

    best_idx = int(np.nanargmax(vpt))
    marker_sizes = 70.0 + 180.0 * (one_step_weight - np.nanmin(one_step_weight)) / max(
        1e-6,
        float(np.nanmax(one_step_weight) - np.nanmin(one_step_weight)),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_vpt, ax_sched) = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    sc = ax_vpt.scatter(
        rollout_horizon[finite],
        vpt[finite],
        c=rollout_weight[finite],
        s=marker_sizes[finite],
        cmap="viridis",
        edgecolor="black",
        linewidth=0.4,
        alpha=0.9,
    )
    ax_vpt.plot(rollout_horizon[finite], vpt[finite], color="0.35", linewidth=1, alpha=0.45)
    ax_vpt.scatter(
        [rollout_horizon[best_idx]],
        [vpt[best_idx]],
        s=260,
        facecolors="none",
        edgecolors="crimson",
        linewidth=2.4,
        label=f"best update {int(updates[best_idx])}",
    )
    ax_vpt.set_xlabel("rollout loss horizon")
    ax_vpt.set_ylabel(metric)
    ax_vpt.set_title(f"{metric} vs rollout-loss setting")
    ax_vpt.grid(True, alpha=0.25)
    ax_vpt.legend(loc="best")
    cbar = fig.colorbar(sc, ax=ax_vpt)
    cbar.set_label("rollout weight")

    ax_sched.plot(updates, rollout_horizon, label="rollout horizon", linewidth=2.2)
    ax_sched.set_xlabel("training update")
    ax_sched.set_ylabel("rollout horizon")
    ax_sched.grid(True, alpha=0.25)
    ax_sched.set_title("Curriculum loss-parameter evolution")

    ax_weight = ax_sched.twinx()
    ax_weight.plot(updates, rollout_weight, label="rollout weight", color="tab:green", linewidth=2)
    ax_weight.plot(updates, one_step_weight, label="one-step weight", color="tab:orange", linewidth=2)
    if np.any(np.isfinite(progress)):
        ax_weight.plot(updates, progress, label="curriculum progress", color="tab:purple", linestyle="--", linewidth=1.7)
    ax_weight.set_ylabel("weights / progress")

    lines, labels = ax_sched.get_legend_handles_labels()
    lines2, labels2 = ax_weight.get_legend_handles_labels()
    ax_sched.legend(lines + lines2, labels + labels2, loc="upper left")

    fig.suptitle("Curriculum Diagnostics", fontsize=14)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_curriculum_vpt_diagnostics_from_file(
    log_path: str | Path = "artifacts/student/train.log",
    output_path: str | Path = "artifacts/student/plots/curriculum_vpt_diagnostics.png",
    *,
    metric: str = "VPT80@0.25",
) -> Path:
    """Read a saved training log file and plot curriculum/VPT diagnostics."""
    log_text = Path(log_path).read_text(encoding="utf-8")
    return plot_curriculum_vpt_diagnostics(log_text, output_path, metric=metric)


__all__ = [
    "compute_scoreboard_metrics",
    "parse_train_log",
    "plot_curriculum_vpt_diagnostics",
    "plot_curriculum_vpt_diagnostics_from_file",
]

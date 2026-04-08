#!/usr/bin/env python3
"""
scripts/plot_stm.py
Figura 5: Norma di Frobenius |STM_AAS - STM_num| vs tempo per Apophis.

CSV atteso:
    t_days, stm_error_frobenius, method
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 150, "axes.linewidth": 0.8,
    "lines.linewidth": 1.0,
})


def plot_stm_accuracy(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(3.5, 2.8), constrained_layout=True)

    for method in df["method"].unique():
        d = df[df["method"] == method].sort_values("t_days")
        ax.semilogy(d["t_days"], d["stm_error_frobenius"],
                    label=method, lw=1.0)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel(
        r"$\|\mathbf{\Phi}_{\mathrm{AAS}} - \mathbf{\Phi}_{\mathrm{num}}\|_F$"
    )
    ax.set_title("(Apophis, 30-day integration)", fontsize=8)
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    ax.legend(framealpha=0.8)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

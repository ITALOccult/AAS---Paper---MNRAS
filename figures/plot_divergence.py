#!/usr/bin/env python3
"""
scripts/plot_divergence.py
Figura 2: Divergence time τ̂_D per Ceres, 3 integratori.

CSV atteso:
    integrator, precision, t_divergence_days, perturb_index
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 150, "axes.linewidth": 0.8,
})

COLORS  = {"AAS": "#1f77b4", "SABA4": "#d62728", "RKF7(8)": "#2ca02c"}
OFFSETS = {"AAS": -0.15,     "SABA4": 0.0,       "RKF7(8)": 0.15}


def plot_divergence_time(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(3.5, 2.8), constrained_layout=True)

    for integrator in ["AAS", "SABA4", "RKF7(8)"]:
        d = df[df["integrator"] == integrator]
        if d.empty:
            continue
        x = d["perturb_index"] + OFFSETS[integrator]
        ax.scatter(x, d["t_divergence_days"] / 365.25,
                   color=COLORS[integrator], s=18,
                   label=integrator, zorder=3)

    ax.set_xlabel("Ensemble member $k$")
    ax.set_ylabel(r"$\hat{\tau}_D$ (yr)")
    ax.set_title("(Ceres, 2-yr integration)", fontsize=8)
    ax.set_xticks(range(1, 9))
    ax.grid(True, axis="y", lw=0.3, alpha=0.4)
    ax.legend(framealpha=0.8)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

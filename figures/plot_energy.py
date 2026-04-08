#!/usr/bin/env python3
"""
scripts/plot_energy.py
Figura 1: |ΔE/E| vs precision parameter per AAS, SABA4, RKF7(8).
Tre pannelli: Ceres, Apophis, Phaethon.

CSV atteso (da AstDyn benchmark_integrators):
    integrator, asteroid, precision, delta_E_over_E, n_steps, cpu_ms
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── stile AAS journals ───────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "axes.linewidth":   0.8,
    "lines.linewidth":  1.2,
})

COLORS   = {"AAS": "#1f77b4", "SABA4": "#d62728", "RKF7(8)": "#2ca02c"}
MARKERS  = {"AAS": "o",       "SABA4": "s",        "RKF7(8)": "^"}
ASTEROIDS = ["Ceres", "Apophis", "Phaethon"]
LABELS    = ["(a) Ceres ($e=0.0784$)",
             "(b) Apophis ($e=0.191$)",
             "(c) Phaethon ($e=0.890$)"]


def plot_energy_vs_precision(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6),
                             sharey=True, constrained_layout=True)

    for ax, asteroid, label in zip(axes, ASTEROIDS, LABELS):
        sub = df[df["asteroid"] == asteroid]

        for integrator in ["AAS", "SABA4", "RKF7(8)"]:
            d = sub[sub["integrator"] == integrator].sort_values("precision")
            if d.empty:
                continue
            ax.loglog(
                d["precision"], d["delta_E_over_E"],
                color=COLORS[integrator],
                marker=MARKERS[integrator],
                markersize=4,
                label=integrator,
            )

        # Guida ordine-4: slope reference
        eps = np.logspace(-6, -3, 50)
        ref = 1e-2 * eps**4
        ax.loglog(eps, ref, "k--", lw=0.7, alpha=0.5)
        ax.text(3e-5, 1.5e-14, r"$\propto\varepsilon^4$",
                fontsize=7, color="0.4")

        ax.set_xlabel(r"Precision $\varepsilon$")
        ax.set_title(label, fontsize=8, pad=3)
        ax.set_xlim(5e-7, 2e-3)
        ax.grid(True, which="both", lw=0.3, alpha=0.4)
        ax.xaxis.set_major_formatter(ticker.LogFormatterSciNotation())

    axes[0].set_ylabel(r"$|\Delta E / E|$")
    axes[0].legend(loc="upper left", framealpha=0.8)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

#!/usr/bin/env python3
"""
Legge benchmark_results/horizons_validation.csv
e genera figures/fig_horizons_validation.pdf
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

DATA    = Path("../../../examples/benchmark_results/horizons_validation.csv")
OUT     = Path("../figures/fig_horizons_validation.pdf")
TARGET  = 1.67e-11   # 2.5 μm in AU

COLORS  = {"AAS": "#1f77b4", "SABA4": "#d62728", "RKF7(8)": "#2ca02c"}
STYLES  = {"AAS": "-",       "SABA4": "--",       "RKF7(8)": "-."}

def main():
    df = pd.read_csv(DATA)
    asteroids = df["asteroid"].unique()
    ncols = 2
    nrows = (len(asteroids) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(7.0, 3.0 * nrows),
                              constrained_layout=True)
    axes = axes.flatten()

    for ax, asteroid in zip(axes, asteroids):
        sub = df[df["asteroid"] == asteroid]
        for integrator in ["AAS", "SABA4", "RKF7(8)"]:
            d = sub[sub["integrator"] == integrator].sort_values("t_days")
            if d.empty:
                continue
            ax.semilogy(d["t_days"] / 365.25,
                        d["delta_r_au"],
                        color=COLORS[integrator],
                        ls=STYLES[integrator],
                        label=integrator)

        # Linea target 2.5 μm
        ax.axhline(TARGET, color="gray", ls=":", lw=0.8)
        ax.text(0.02, TARGET * 1.5, "2.5 μm", transform=ax.get_yaxis_transform(),
                fontsize=7, color="gray")

        ax.set_xlabel("Time (yr)")
        ax.set_ylabel(r"$\delta r$ (AU)")
        ax.set_title(asteroid, fontsize=8)
        ax.grid(True, which="both", lw=0.3, alpha=0.4)
        ax.legend(framealpha=0.8)

    # Nascondi pannelli vuoti
    for ax in axes[len(asteroids):]:
        ax.set_visible(False)

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Salvato: {OUT}")

if __name__ == "__main__":
    main()

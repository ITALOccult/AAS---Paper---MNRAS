#!/usr/bin/env python3
"""
Legge benchmark_results/efficiency.csv
e genera figures/fig_efficiency.pdf
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 150, "axes.linewidth": 0.8,
})

DATA   = Path("../../../examples/benchmark_results/efficiency.csv")
OUT    = Path("../figures/fig_efficiency.pdf")
COLORS = {"AAS": "#1f77b4", "SABA4": "#d62728", "RKF7(8)": "#2ca02c"}
MARKERS= {"AAS": "o",       "SABA4": "s",        "RKF7(8)": "^"}

def main():
    df = pd.read_csv(DATA)
    asteroids = ["Phaethon", "Apophis"]
    labels = ["(a) Phaethon ($e=0.890$)", "(b) Apophis ($e=0.191$)"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0),
                              constrained_layout=True)

    for ax, asteroid, label in zip(axes, asteroids, labels):
        sub = df[df["asteroid"] == asteroid]
        for integrator in ["AAS", "SABA4", "RKF7(8)"]:
            d = sub[sub["integrator"] == integrator].sort_values("n_func_evals")
            if d.empty:
                continue
            ax.loglog(d["n_func_evals"],
                      d["delta_E_over_E"],
                      color=COLORS[integrator],
                      marker=MARKERS[integrator],
                      markersize=4,
                      label=integrator)

        ax.set_xlabel("Function evaluations")
        ax.set_ylabel(r"$|\Delta E/E|$")
        ax.set_title(label, fontsize=8)
        ax.grid(True, which="both", lw=0.3, alpha=0.4)
        ax.legend(framealpha=0.8)

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Salvato: {OUT}")

if __name__ == "__main__":
    main()

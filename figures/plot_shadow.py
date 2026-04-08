#!/usr/bin/env python3
"""
scripts/plot_shadow.py
Figura 4: H_physical e H_shadow per AAS e SABA4 su Ceres 2 anni.

CSV atteso:
    t_days, H_physical, H_shadow, delta_H_shadow_over_H0, integrator
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 150, "axes.linewidth": 0.8,
    "lines.linewidth": 0.9,
})

STYLES = {
    "AAS":    {"color": "#1f77b4", "ls": "-"},
    "SABA4":  {"color": "#d62728", "ls": "--"},
}


def plot_shadow_hamiltonian(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8),
                             sharey=False, constrained_layout=True)

    for ax, col, ylabel, title in zip(
        axes,
        ["delta_H_physical", "delta_H_shadow"],
        [r"$|\Delta H / H_0|$",
         r"$|\Delta \tilde{H} / \tilde{H}_0|$"],
        ["(a) Physical Hamiltonian", "(b) Shadow Hamiltonian"],
    ):
        for integrator, style in STYLES.items():
            d = df[df["integrator"] == integrator]
            if d.empty:
                continue
            # Usa valore assoluto per il semilogy
            yval = d[col].abs()
            ax.semilogy(d["t_days"] / 365.25, yval,
                        label=integrator, **style)

        if title == "(b) Shadow Hamiltonian":
            # Theoretical predicted shadow error O(dt^4)
            # SABA4 (dt=0.5): shadow floor ~ 5e-5
            ax.axhline(5e-5, color="black", ls=":", lw=0.7, label=r"theory $\Delta t^4$")

        ax.set_xlabel("Time (yr)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=8)
        ax.grid(True, which="both", lw=0.3, alpha=0.4)
        ax.legend(framealpha=0.8)

    fig.suptitle("Ceres — 2-year integration", fontsize=8, y=1.01)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

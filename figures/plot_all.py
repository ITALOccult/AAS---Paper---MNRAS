#!/usr/bin/env python3
"""
scripts/plot_all.py
Genera tutte le figure del paper AAS da ../data/*.csv
Output: ../figures/fig_*.pdf

Uso:
    cd scripts/
    python plot_all.py
"""

import sys
from pathlib import Path

# Assicura che gli script nella stessa directory siano importabili
sys.path.insert(0, str(Path(__file__).parent))

from plot_energy     import plot_energy_vs_precision
from plot_divergence import plot_divergence_time
from plot_steps      import plot_step_distribution
from plot_shadow     import plot_shadow_hamiltonian
from plot_stm        import plot_stm_accuracy

DATA_DIR    = Path("/Users/michelebigi/Documents/Develop/ASTDYN/IOccultLibrary/astdyn/examples/benchmark_results")
FIGURES_DIR = Path(__file__).parent.parent / "figures"

def main():
    FIGURES_DIR.mkdir(exist_ok=True)

    tasks = [
        ("Energy conservation",   plot_energy_vs_precision,
         DATA_DIR / "energy_vs_precision.csv",
         FIGURES_DIR / "fig_energy_vs_precision.pdf"),

        ("Divergence time",        plot_divergence_time,
         DATA_DIR / "divergence_time.csv",
         FIGURES_DIR / "fig_divergence_time.pdf"),

        ("Step distribution",      plot_step_distribution,
         DATA_DIR / "step_distribution.csv",
         FIGURES_DIR / "fig_step_distribution.pdf"),

        ("Shadow Hamiltonian",     plot_shadow_hamiltonian,
         DATA_DIR / "shadow_hamiltonian.csv",
         FIGURES_DIR / "fig_shadow_hamiltonian.pdf"),

        ("STM accuracy",           plot_stm_accuracy,
         DATA_DIR / "stm_accuracy.csv",
         FIGURES_DIR / "fig_stm_accuracy.pdf"),
    ]

    for label, func, data_path, out_path in tasks:
        if not data_path.exists():
            print(f"[SKIP] {label}: {data_path.name} not found")
            continue
        print(f"[PLOT] {label} → {out_path.name}")
        func(data_path, out_path)

    print("Done.")

if __name__ == "__main__":
    main()

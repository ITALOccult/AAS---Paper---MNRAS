#!/usr/bin/env python3
"""
diagnose_ias15.py
Verifica il comportamento di IAS15 su Apophis con diversi
force model e tolleranze, per capire la causa dell'energia anomala.
"""
import rebound
import numpy as np
import pandas as pd
from pathlib import Path

JD_EPOCH   = 2460310.5
DAYS_PER_YR = 365.25
GMS        = 2.9591220828559e-4

DATA = Path("/Users/michelebigi/Documents/Develop/ASTDYN/paper/AAS/AAS-Integrator-Paper/benchmark")

def get_apophis():
    df = pd.read_csv(DATA / "initial_states_mjd60310.csv")
    r  = df[df["number"] == 99942].iloc[0]
    return r[["x","y","z","vx","vy","vz"]].values.astype(float)

def helio_energy(sim):
    p0 = sim.particles[0]
    pa = sim.particles[-1]
    dx  = pa.x  - p0.x;  dy  = pa.y  - p0.y;  dz  = pa.z  - p0.z
    dvx = pa.vx - p0.vx; dvy = pa.vy - p0.vy; dvz = pa.vz - p0.vz
    r   = np.sqrt(dx**2 + dy**2 + dz**2)
    v2  = dvx**2 + dvy**2 + dvz**2
    return 0.5*v2 - GMS/r

def build_twobody(state, eps):
    sim = rebound.Simulation()
    sim.integrator = "IAS15"
    sim.ri_ias15.epsilon = eps
    sim.units = ("day","AU","Msun")
    sim.t = 0.0
    sim.add(m=1.0)
    sim.add(x=state[0],y=state[1],z=state[2],
            vx=state[3],vy=state[4],vz=state[5],m=0.0)
    sim.move_to_com()
    return sim

def build_nbody(state, eps, planets=None):
    if planets is None:
        planets = ["Sun","Mercury","Venus","Earth","Mars",
                   "Jupiter","Saturn","Uranus","Neptune"]
    sim = rebound.Simulation()
    sim.integrator = "IAS15"
    sim.ri_ias15.epsilon = eps
    sim.units = ("day","AU","Msun")
    sim.t = 0.0
    for p in planets:
        sim.add(p, date=f"JD{JD_EPOCH:.4f}")
    sim.add(x=state[0],y=state[1],z=state[2],
            vx=state[3],vy=state[4],vz=state[5],m=0.0)
    sim.move_to_com()
    return sim

def run_test(sim, T_yr, label):
    E0 = helio_energy(sim)
    sim.integrate(T_yr * DAYS_PER_YR, exact_finish_time=1)
    Ef = helio_energy(sim)
    dE = abs(Ef - E0) / abs(E0)
    print(f"  {label:<50}  dE/E = {dE:.3e}")
    return dE

if __name__ == "__main__":
    state = get_apophis()
    print("="*70)
    print("DIAGNOSTICA IAS15 — APOPHIS")
    print("="*70)

    # Test 1: two-body, varie tolleranze
    print("\n[A] TWO-BODY — effetto tolleranza (50 yr)")
    for eps in [1e-6, 1e-9, 1e-12]:
        sim = build_twobody(state, eps)
        run_test(sim, 50.0, f"two-body  eps={eps:.0e}")

    # Test 2: N-body completo, varie tolleranze
    print("\n[B] FULL N-BODY (9 pianeti) — effetto tolleranza (50 yr)")
    for eps in [1e-6, 1e-9, 1e-12]:
        sim = build_nbody(state, eps)
        run_test(sim, 50.0, f"full      eps={eps:.0e}")

    # Test 3: N-body con soli Giove+Saturno (modello intermedio)
    print("\n[C] SOL+JUP+SAT — effetto tolleranza (50 yr)")
    for eps in [1e-6, 1e-9, 1e-12]:
        sim = build_nbody(state, eps,
                          planets=["Sun","Jupiter","Saturn"])
        run_test(sim, 50.0, f"Sun+J+S   eps={eps:.0e}")

    # Test 4: verifico che uso sim.energy() vs helio_energy diano stesso risultato
    print("\n[D] CROSS-CHECK sim.energy() vs helio_energy (two-body, eps=1e-9)")
    sim = build_twobody(state, 1e-9)
    E0_helio = helio_energy(sim)
    E0_com   = sim.energy()
    sim.integrate(50*DAYS_PER_YR, exact_finish_time=1)
    Ef_helio = helio_energy(sim)
    Ef_com   = sim.energy()
    print(f"  helio:  dE/E = {abs(Ef_helio-E0_helio)/abs(E0_helio):.3e}")
    print(f"  COM:    dE/E = {abs(Ef_com  -E0_com  )/abs(E0_com  ):.3e}")

    # Test 5: reversibilità two-body (verifica che non sia bug nel test)
    print("\n[E] REVERSIBILITÀ — two-body, eps=1e-9 (50 yr)")
    sim_f = build_twobody(state, 1e-9)
    sim_f.integrate(50*DAYS_PER_YR, exact_finish_time=1)
    p0 = sim_f.particles[0]; pa = sim_f.particles[-1]
    xT = np.array([pa.x-p0.x, pa.y-p0.y, pa.z-p0.z,
                   pa.vx-p0.vx, pa.vy-p0.vy, pa.vz-p0.vz])
    xT_rev = xT.copy(); xT_rev[3:] *= -1
    sim_b = build_twobody(xT_rev, 1e-9)
    sim_b.integrate(50*DAYS_PER_YR, exact_finish_time=1)
    p0b = sim_b.particles[0]; pab = sim_b.particles[-1]
    xfin = np.array([pab.x-p0b.x, pab.y-p0b.y, pab.z-p0b.z,
                     pab.vx-p0b.vx, pab.vy-p0b.vy, pab.vz-p0b.vz])
    xfin[3:] *= -1
    eps_r = np.linalg.norm(xfin[:3]-state[:3])/np.linalg.norm(state[:3])
    eps_v = np.linalg.norm(xfin[3:]-state[3:])/np.linalg.norm(state[3:])
    print(f"  eps_r = {eps_r:.3e}  (atteso ~1e-10 per IAS15 two-body)")
    print(f"  eps_v = {eps_v:.3e}")

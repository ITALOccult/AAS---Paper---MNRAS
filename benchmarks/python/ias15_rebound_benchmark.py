#!/usr/bin/env python3
"""
benchmark_ias15.py  — versione finale

Energia: usa twobody_energy (0.5*v_rel^2 - GMS/r) — stessa
  definizione di AstDyn, comparabile tra tutti gli integratori.

Reversibilità: test TWO-BODY (solo Sole + asteroide).
  Misura la proprietà dell'integratore, non del campo perturbatore.

Lyapunov: N-body completo (8 pianeti), rinormalizzazione corretta.

Horizons residuals: N-body completo.

Uso:
    python3 benchmark_ias15.py --mode short
    python3 benchmark_ias15.py --mode long --family NEA
    python3 benchmark_ias15.py --mode long --family Trojan
    python3 benchmark_ias15.py --mode long --family Resonant
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import time as time_module

import rebound
from astroquery.jplhorizons import Horizons
from scipy import stats

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "examples" / "benchmark_results"
DATA_DIR.mkdir(parents=True, exist_ok=True)

GMS_AU3D2   = 2.9591220828559e-4
GM_J        = 9.5479190e-4
A_JUPITER   = 5.2044
MJD_EPOCH   = 60310.0
JD_EPOCH    = MJD_EPOCH + 2400000.5
D0_LYAP     = 1e-6
TAU_LYAP_YR = 10.0
DAYS_PER_YR = 365.25
INTEGRATOR  = "IAS15"
IAS15_EPS   = 1e-9

PLANET_NAMES = ["Sun", "Mercury", "Venus", "Earth", "Mars",
                "Jupiter", "Saturn", "Uranus", "Neptune"]


@dataclass
class AsteroidSpec:
    name:      str
    number:    int
    family:    str
    T_long_yr: float
    lyap_N:    int
    is_trojan: bool = False


ASTEROIDS = [
    AsteroidSpec("Apophis",   99942,  "NEA",       50.0,    5),
    AsteroidSpec("Icarus",    1566,   "NEA",       50.0,    5),
    AsteroidSpec("Phaethon",  3200,   "NEA",       50.0,    5),
    AsteroidSpec("Achilles",  588,    "Trojan",  1000.0,  100, is_trojan=True),
    AsteroidSpec("Patroclus", 617,    "Trojan",  1000.0,  100, is_trojan=True),
    AsteroidSpec("Hektor",    624,    "Trojan",  1000.0,  100, is_trojan=True),
    AsteroidSpec("Hilda",     153,    "Resonant", 500.0,   50),
    AsteroidSpec("Thule",     279,    "Resonant", 500.0,   50),
    AsteroidSpec("Griqua",    1362,   "Resonant", 500.0,   50),
    AsteroidSpec("Pluto",     134340, "TNO",    10000.0, 1000),
    AsteroidSpec("Eris",      136199, "TNO",    10000.0, 1000),
    AsteroidSpec("Sedna",     90377,  "TNO",    10000.0, 1000),
]

REV_TARGETS = {"Apophis": 50.0, "Achilles": 100.0, "Pluto": 1000.0}


# ── Energia ───────────────────────────────────────────────────────────

def twobody_energy(sim: rebound.Simulation) -> float:
    """
    Energia kepleriana due-corpo: 0.5*v_rel^2 - G*M_sun/r
    relativa al Sole (particles[0]).
    Stessa definizione di AstDyn — comparabile tra integratori.
    """
    p0  = sim.particles[0]
    pa  = sim.particles[-1]
    dx  = pa.x  - p0.x;  dy  = pa.y  - p0.y;  dz  = pa.z  - p0.z
    dvx = pa.vx - p0.vx; dvy = pa.vy - p0.vy; dvz = pa.vz - p0.vz
    r   = np.sqrt(dx**2 + dy**2 + dz**2)
    v2  = dvx**2 + dvy**2 + dvz**2
    return 0.5 * v2 - sim.G * p0.m / r


def angular_momentum_helio(x: np.ndarray) -> np.ndarray:
    return np.cross(x[:3], x[3:])


def secular_drift(t: np.ndarray,
                  y: np.ndarray) -> Tuple[float, float]:
    slope, _, r, _, _ = stats.linregress(t, y)
    return slope, r**2


def make_row(asteroid: str, test: str, T_yr: float,
             value: float, unit: str, note: str = "") -> dict:
    return {"asteroid": asteroid, "integrator": INTEGRATOR,
            "test": test, "T_yr": T_yr, "value": value,
            "unit": unit, "note": note}


# ── Simulazioni ───────────────────────────────────────────────────────

def build_sim_full(state: np.ndarray,
                   t_start: float = 0.0) -> rebound.Simulation:
    """N-body: Sole + 8 pianeti + asteroide."""
    jd  = JD_EPOCH + t_start
    sim = rebound.Simulation()
    sim.integrator = "IAS15"
    sim.ri_ias15.epsilon = IAS15_EPS
    sim.units = ("day", "AU", "Msun")
    sim.t = t_start
    for body in PLANET_NAMES:
        sim.add(body, date=f"JD{jd:.4f}")
    sim.add(x=state[0], y=state[1], z=state[2],
            vx=state[3], vy=state[4], vz=state[5], m=0.0)
    sim.move_to_com()
    return sim


def build_sim_twobody(state: np.ndarray) -> rebound.Simulation:
    """Two-body: Sole + asteroide (Kepleriano puro)."""
    sim = rebound.Simulation()
    sim.integrator = "IAS15"
    sim.ri_ias15.epsilon = IAS15_EPS
    sim.units = ("day", "AU", "Msun")
    sim.t = 0.0
    sim.add(m=1.0)
    sim.add(x=state[0], y=state[1], z=state[2],
            vx=state[3], vy=state[4], vz=state[5], m=0.0)
    sim.move_to_com()
    return sim


def get_ast_state(sim: rebound.Simulation) -> np.ndarray:
    p0 = sim.particles[0]
    pa = sim.particles[-1]
    return np.array([pa.x  - p0.x,  pa.y  - p0.y,  pa.z  - p0.z,
                     pa.vx - p0.vx, pa.vy - p0.vy, pa.vz - p0.vz])


def set_ast_state(sim: rebound.Simulation,
                  state: np.ndarray) -> None:
    p0 = sim.particles[0]
    pa = sim.particles[-1]
    pa.x  = state[0] + p0.x;  pa.y  = state[1] + p0.y
    pa.z  = state[2] + p0.z
    pa.vx = state[3] + p0.vx; pa.vy = state[4] + p0.vy
    pa.vz = state[5] + p0.vz


def fetch_state(target_id: int, jd: float) -> np.ndarray:
    obj  = Horizons(id=str(target_id), location="500@10", epochs=[jd])
    vecs = obj.vectors(refplane="ecliptic").to_pandas()
    return vecs[["x","y","z","vx","vy","vz"]].values[0]


def fetch_state_at_days(target_id: int,
                        t_days: List[float]) -> pd.DataFrame:
    jds = [JD_EPOCH + t for t in t_days]
    obj = Horizons(id=str(target_id), location="500@10", epochs=jds)
    return obj.vectors(refplane="ecliptic").to_pandas()


def get_cached_state(ast: AsteroidSpec) -> np.ndarray:
    csv = DATA_DIR / "initial_states_mjd60310.csv"
    if csv.exists():
        df  = pd.read_csv(csv)
        row = df[df["number"] == ast.number]
        if not row.empty:
            return row[["x","y","z","vx","vy","vz"]].values[0]
    state = fetch_state(ast.number, JD_EPOCH)
    row = {"asteroid": ast.name, "number": ast.number,
           "x": state[0], "y": state[1], "z": state[2],
           "vx": state[3], "vy": state[4], "vz": state[5],
           "epoch_mjd": MJD_EPOCH, "frame": "ECLIPJ2000"}
    if csv.exists():
        pd.concat([pd.read_csv(csv),
                   pd.DataFrame([row])]).to_csv(csv, index=False)
    else:
        pd.DataFrame([row]).to_csv(csv, index=False)
    return state


# ════════════════════════════════════════════════════════════════════
# SHORT-TERM
# ════════════════════════════════════════════════════════════════════

def st1_horizons(ast: AsteroidSpec,
                 state0: np.ndarray) -> List[dict]:
    """Residui vs Horizons — N-body full."""
    rows   = []
    t_days = [1.0, 5.0, 10.0, 20.0, 30.0]
    hor_df = fetch_state_at_days(ast.number, t_days)
    for force in ["two_body", "full"]:
        sim = (build_sim_twobody(state0) if force == "two_body"
               else build_sim_full(state0))
        for t, (_, hr) in zip(t_days, hor_df.iterrows()):
            sim.integrate(t, exact_finish_time=1)
            r_integ = get_ast_state(sim)[:3]
            r_hor   = np.array([hr.x, hr.y, hr.z])
            dr      = np.linalg.norm(r_integ - r_hor)
            rows.append(make_row(ast.name, f"horizons_{force}",
                                 t / DAYS_PER_YR, dr, "AU",
                                 f"t={t:.0f}d"))
    return rows


def st2_energy_30d(ast: AsteroidSpec,
                   state0: np.ndarray) -> List[dict]:
    """Energia 30 giorni — two-body + twobody_energy."""
    sim = build_sim_twobody(state0)
    E0  = twobody_energy(sim)
    sim.integrate(30.0, exact_finish_time=1)
    Ef  = twobody_energy(sim)
    dH  = abs(Ef - E0) / abs(E0)
    return [make_row(ast.name, "energy_30d",
                     30.0 / DAYS_PER_YR, dH, "dimensionless")]


def run_short_term(family_filter: str = "all") -> None:
    print("=" * 60)
    print("SHORT-TERM — IAS15/REBOUND")
    print("=" * 60)
    rows = []
    for ast in ASTEROIDS:
        if family_filter != "all" and ast.family != family_filter:
            continue
        print(f"  {ast.name}...", flush=True)
        state0 = get_cached_state(ast)
        t0     = time_module.time()
        rows  += st1_horizons(ast, state0)
        rows  += st2_energy_30d(ast, state0)
        print(f"    {time_module.time()-t0:.1f}s")
    out = DATA_DIR / "short_term_ias15.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSalvato: {out}  ({len(rows)} righe)")


# ════════════════════════════════════════════════════════════════════
# LONG-TERM
# ════════════════════════════════════════════════════════════════════

def lt1_energy(ast: AsteroidSpec,
               state0: np.ndarray) -> Tuple[List[dict], dict]:
    """
    Energia finale — two-body + twobody_energy.
    Comparabile con AstDyn.
    """
    sim  = build_sim_twobody(state0)
    E0   = twobody_energy(sim)
    t_yr = []
    dH_v = []
    step = max(1, int(ast.T_long_yr / 200))
    for yr in range(step, int(ast.T_long_yr) + 1, step):
        sim.integrate(yr * DAYS_PER_YR, exact_finish_time=1)
        dH = abs(twobody_energy(sim) - E0) / abs(E0)
        t_yr.append(yr)
        dH_v.append(dH)
    rows = [make_row(ast.name, "energy_final",
                     ast.T_long_yr, dH_v[-1], "dimensionless")]
    return rows, {"t_yr": t_yr, "dH": dH_v}


def lt2_reversibility(ast: AsteroidSpec,
                      state0: np.ndarray,
                      T_yr: float) -> List[dict]:
    """
    Reversibilità — TWO-BODY.
    Misura la proprietà dell'integratore, non del campo.
    AAS/SABA4 (simplettici): atteso < 1e-10.
    IAS15 (non simplettico): atteso ~ 1e-8 to 1e-12 su 50yr.
    """
    T_d   = T_yr * DAYS_PER_YR

    sim_f = build_sim_twobody(state0)
    sim_f.integrate(T_d, exact_finish_time=1)
    xT    = get_ast_state(sim_f)

    xT_rev = xT.copy(); xT_rev[3:] *= -1
    sim_b  = build_sim_twobody(xT_rev)
    sim_b.integrate(T_d, exact_finish_time=1)
    xfin   = get_ast_state(sim_b); xfin[3:] *= -1

    eps_r = (np.linalg.norm(xfin[:3] - state0[:3]) /
             np.linalg.norm(state0[:3]))
    eps_v = (np.linalg.norm(xfin[3:] - state0[3:]) /
             np.linalg.norm(state0[3:]))
    return [make_row(ast.name, "reversibility_r", T_yr,
                     eps_r, "dimensionless"),
            make_row(ast.name, "reversibility_v", T_yr,
                     eps_v, "dimensionless")]


def lt4_angular_momentum(ast: AsteroidSpec,
                         state0: np.ndarray) -> List[dict]:
    """Momento angolare — two-body."""
    T_d = ast.T_long_yr * DAYS_PER_YR
    sim = build_sim_twobody(state0)
    L0  = np.linalg.norm(angular_momentum_helio(state0))
    sim.integrate(T_d, exact_finish_time=1)
    LT  = np.linalg.norm(angular_momentum_helio(get_ast_state(sim)))
    dL  = abs(LT - L0) / abs(L0)
    return [make_row(ast.name, "angular_momentum_final",
                     ast.T_long_yr, dL, "dimensionless")]


def lt5_lyapunov(ast: AsteroidSpec,
                 state0: np.ndarray) -> Tuple[List[dict], List[dict]]:
    """Lyapunov — N-body full, rinormalizzazione corretta."""
    tau_d = TAU_LYAP_YR * DAYS_PER_YR
    sim   = build_sim_full(state0)
    x_p   = state0.copy(); x_p[0] += D0_LYAP
    sim_p = build_sim_full(x_p)
    lams  = []
    for i in range(ast.lyap_N):
        t_now = (i + 1) * tau_d
        sim.integrate(t_now,   exact_finish_time=1)
        sim_p.integrate(t_now, exact_finish_time=1)
        x_now  = get_ast_state(sim)
        xp_now = get_ast_state(sim_p)
        d_tau  = max(np.linalg.norm(xp_now[:3] - x_now[:3]), 1e-30)
        lam    = np.log(d_tau / D0_LYAP) / TAU_LYAP_YR
        lams.append(lam)
        delta = (xp_now - x_now) * (D0_LYAP / d_tau)
        set_ast_state(sim_p, x_now + delta)
    mLCE  = float(np.mean(lams))
    rows  = [make_row(ast.name, "lyapunov_mLCE",
                      ast.T_long_yr, mLCE, "1/yr")]
    series = [{"asteroid": ast.name, "integrator": INTEGRATOR,
               "interval_index": i + 1, "lambda_i_yr": l}
              for i, l in enumerate(lams)]
    return rows, series


def lt6_jacobi(ast: AsteroidSpec,
               state0: np.ndarray) -> List[dict]:
    """Costante di Jacobi — solo Trojani, N-body full."""
    if not ast.is_trojan:
        return []
    jup    = fetch_state(599, JD_EPOCH)
    theta0 = np.arctan2(jup[1], jup[0])
    n_J    = np.sqrt(GMS_AU3D2 / A_JUPITER**3)

    def CJ(t_day: float, xv: np.ndarray) -> float:
        th  = n_J * t_day + theta0
        c, s = np.cos(th), np.sin(th)
        xr  =  xv[0]*c + xv[1]*s
        yr  = -xv[0]*s + xv[1]*c
        zr  =  xv[2]
        vxr =  xv[3]*c + xv[4]*s + n_J * yr
        vyr = -xv[3]*s + xv[4]*c - n_J * xr
        vzr =  xv[5]
        r1  = np.linalg.norm([xr, yr, zr])
        r2  = np.linalg.norm([xr - A_JUPITER, yr, zr])
        return (n_J**2 * (xr**2 + yr**2)
                + 2 * (GMS_AU3D2 / r1 + GM_J / r2)
                - (vxr**2 + vyr**2 + vzr**2))

    T_d = ast.T_long_yr * DAYS_PER_YR
    sim = build_sim_full(state0)
    CJ0 = CJ(0.0, state0)
    sim.integrate(T_d, exact_finish_time=1)
    xT  = get_ast_state(sim)
    dCJ = abs(CJ(T_d, xT) - CJ0) / abs(CJ0)
    return [make_row(ast.name, "jacobi_final",
                     ast.T_long_yr, dCJ, "dimensionless")]


def run_long_term(family_filter: str = "all") -> None:
    print("=" * 60)
    print(f"LONG-TERM — IAS15/REBOUND  [{family_filter}]")
    print("=" * 60)
    all_rows    = []
    energy_rows = []
    lyap_rows   = []

    for ast in ASTEROIDS:
        if family_filter != "all" and ast.family != family_filter:
            continue
        print(f"\n  {ast.name} ({ast.family}, {ast.T_long_yr}yr)...",
              flush=True)
        state0 = get_cached_state(ast)
        t0     = time_module.time()

        # Energia (two-body)
        rows_e, ser_e = lt1_energy(ast, state0)
        all_rows += rows_e
        for yr, dh in zip(ser_e["t_yr"], ser_e["dH"]):
            energy_rows.append({"asteroid":   ast.name,
                                 "integrator": INTEGRATOR,
                                 "t_yr":       yr,
                                 "dH_over_H0": dh})

        # Reversibilità (two-body)
        if ast.name in REV_TARGETS:
            all_rows += lt2_reversibility(
                ast, state0, REV_TARGETS[ast.name])

        # Drift secolare (da serie energia)
        t_arr  = np.array(ser_e["t_yr"], dtype=float)
        dh_arr = np.array(ser_e["dH"],   dtype=float)
        slope, r2 = secular_drift(t_arr, dh_arr)
        all_rows.append(make_row(ast.name, "secular_slope",
                                 ast.T_long_yr, slope, "1/yr"))
        all_rows.append(make_row(ast.name, "secular_R2",
                                 ast.T_long_yr, r2, "dimensionless"))

        # Momento angolare (two-body)
        all_rows += lt4_angular_momentum(ast, state0)

        # Lyapunov (N-body)
        rows_l, ser_l = lt5_lyapunov(ast, state0)
        all_rows += rows_l
        lyap_rows += ser_l

        # Jacobi (solo Trojani, N-body)
        all_rows += lt6_jacobi(ast, state0)

        print(f"    done in {time_module.time()-t0:.1f}s")

    suffix = f"_{family_filter}" if family_filter != "all" else ""
    pd.DataFrame(all_rows).to_csv(
        DATA_DIR / f"long_term_ias15{suffix}.csv", index=False)
    pd.DataFrame(energy_rows).to_csv(
        DATA_DIR / f"energy_series_ias15{suffix}.csv", index=False)
    pd.DataFrame(lyap_rows).to_csv(
        DATA_DIR / f"lyapunov_series_ias15{suffix}.csv", index=False)
    print(f"\nCSV salvati in {DATA_DIR}")


if __name__ == "__main__":
    import rebound as _rb
    _s = _rb.Simulation()
    _s.add("Sun",  date="JD2451545.0")
    _s.add("499",  date="JD2451545.0")
    _s.integrator = "IAS15"
    _s.ri_ias15.epsilon = IAS15_EPS
    _s.integrate(365.25)
    print(f"IAS15 OK (eps={IAS15_EPS:.0e})\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        choices=["short", "long", "all"],
                        default="all")
    parser.add_argument("--family",
                        choices=["NEA", "Trojan", "Resonant",
                                 "TNO", "all"],
                        default="all")
    args = parser.parse_args()

    if args.mode in ("short", "all"):
        run_short_term(args.family)
    if args.mode in ("long", "all"):
        run_long_term(args.family)


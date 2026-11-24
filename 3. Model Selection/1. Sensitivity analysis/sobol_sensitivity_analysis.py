#!/usr/bin/env python3
"""Sobol sensitivity analysis for single-network vein-CANN candidates.

Pure-NumPy / SALib. Implements the workflow in the figure:

  1) STRETCH GRIDS + SHAPE FUNCTIONS
     - Two biaxial stretch grids (physiological + extended).
     - Build Psi (M x D) by evaluating every candidate term on each grid.

  2) SOBOL SENSITIVITY
     - Saltelli sample N coefficient vectors w.
     - Energy field on grid:    W_j = Psi . w_j
     - Scalar response:         Y_j = sqrt( (1/M) sum W_j^2 )
     - SALib computes S_i and S_Ti per term.
     - Four coefficient regimes:
         coeff_max   : bounds = +- max|w_sub * w_ann| observed across animals
         normalized  : bounds = [-1, 1] with unit-norm Psi columns (equal footing)
         raw_uniform : bounds = [-1, 1] on raw Psi (intrinsic shape)
         normalized_cohort: cohort standard-deviation bounds with normalized Psi
     - LOO across animals on a selected cohort-aware regime (per grid).

  3) RANKING
     - One CSV per (grid, regime), plus a unified ranked_terms_sobol.csv.
     - A reduced "robust" set = top-K terms that rank high on BOTH grids
       under the selected primary regime.

Outputs (per subgroup, under <out_root>/<SUBGROUP>/):
    psi_matrix_physio.npz
    psi_matrix_extended.npz
    sobol_indices_<grid>_<regime>.csv     (8 files total)
    loo_rank_stability_<grid>.csv          (2 files)
    ranked_terms_sobol.csv                 (unified ranking + category)
    sobol_vs_robust_score_comparison.csv   (optional prior-ranking comparison)
    reduced_term_set.txt                   (robust intersection)
    summary.txt
"""

from __future__ import annotations
import argparse
import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol


# ---------------------------------------------------------------------------
# Candidate basis  (mirrors Vein_polynom..v4_robust.single_network_feature_metadata)
# ---------------------------------------------------------------------------

INVARIANTS = ("I1", "I2", "I4f", "I4s", "I4el", "I4smc", "I8fs")
INVARIANT_SHAPES_7 = ("Linear", "Exp linear", "Quadratic", "Exp quadratic",
                      "xlog linear", "xlog quadratic", "Cubic")
INVARIANT_SHAPES_5 = ("Linear", "Exp linear", "Quadratic", "Exp quadratic", "Cubic")

STRAIN_TERMS_FULL = [
    ("E_θ",       (1, 0)),
    ("E_z",       (0, 1)),
    ("E_θ²",      (2, 0)),
    ("E_z²",      (0, 2)),
    ("E_θ·E_z",   (1, 1)),
    ("E_θ³",      (3, 0)),
    ("E_z³",      (0, 3)),
    ("E_θ⁴",      (4, 0)),
    ("E_z⁴",      (0, 4)),
    ("E_θ²·E_z",  (2, 1)),
    ("E_θ·E_z²",  (1, 2)),
    ("E_θ²·E_z²", (2, 2)),
]


def build_term_catalog(poly_n_terms: int = 7,
                       poly_include_linear: bool = True) -> List[Dict]:
    """Return list of dicts: {group, inv, term, key, category}."""
    rows: List[Dict] = []
    for inv in INVARIANTS:
        shapes = INVARIANT_SHAPES_5 if inv == "I8fs" else INVARIANT_SHAPES_7
        for shape in shapes:
            rows.append({
                "group": "invariant", "inv": inv, "term": shape,
                "key": f"invariant | {inv} | {shape}",
                "category": classify(group="invariant", inv=inv, term=shape),
            })
    # Strain: first n entries (with optional linear)
    strain_subset = list(STRAIN_TERMS_FULL)
    if not poly_include_linear:
        strain_subset = [t for t in strain_subset if t[1] != (1, 0) and t[1] != (0, 1)]
    # cap to poly_n_terms beyond linear
    if poly_include_linear:
        strain_subset = strain_subset[: 2 + poly_n_terms]
    else:
        strain_subset = strain_subset[: poly_n_terms]
    for name, _ in strain_subset:
        rows.append({
            "group": "strain", "inv": "strain", "term": name,
            "key": f"strain | strain | {name}",
            "category": classify(group="strain", inv="strain", term=name),
        })
    return rows


def classify(group: str, inv: str, term: str) -> str:
    g = group.lower(); iv = inv
    if g == "invariant":
        if iv in ("I1", "I2"):
            return "iso"
        if iv == "I8fs":
            return "coupling"
        return "aniso"
    if g == "strain":
        # Mixed E_theta * E_z terms couple the two directions; pure E_theta
        # or pure E_z terms are direction-aligned (aniso), same category
        # as I4theta and I4z respectively.
        if "·" in term:
            return "coupling"
        return "aniso"
    return "iso"


# ---------------------------------------------------------------------------
# Psi evaluation on a stretch grid (no recruitment — matches the figure)
# ---------------------------------------------------------------------------

def make_grid(lam_theta_min: float, lam_theta_max: float,
              lam_z_min: float,   lam_z_max: float,
              n_theta: int,        n_z: int) -> Tuple[np.ndarray, np.ndarray]:
    lt = np.linspace(lam_theta_min, lam_theta_max, n_theta)
    lz = np.linspace(lam_z_min,    lam_z_max,    n_z)
    LT, LZ = np.meshgrid(lt, lz, indexing="xy")
    return LT.ravel(), LZ.ravel()


def kinematics(lam_theta: np.ndarray, lam_z: np.ndarray) -> Dict[str, np.ndarray]:
    lam_r = 1.0 / (lam_theta * lam_z)
    C_th = lam_theta ** 2; C_z = lam_z ** 2; C_r = lam_r ** 2
    return dict(
        C_theta=C_th, C_z=C_z, C_r=C_r,
        I1=C_th + C_z + C_r,
        I2=C_th * C_z + C_th * C_r + C_z * C_r,
        E_theta=0.5 * (C_th - 1.0),
        E_z=0.5 * (C_z - 1.0),
    )


def raw_invariants(lam_theta: np.ndarray, lam_z: np.ndarray,
                   phi_f: float = np.deg2rad(35.0),
                   phi_s: float = np.deg2rad(-35.0),
                   phi_el: float = np.deg2rad(90.0),
                   phi_smc: float = 0.0,
                   kappa_f: float = 0.0,
                   kappa_s: float = 0.0) -> Dict[str, np.ndarray]:
    """Raw (un-recruited) fiber invariants with default microstructure.

    Defaults are deliberate: the figure-style Sobol asks "what is the
    intrinsic sensitivity of each shape function?" — so we use a generic
    fiber alignment, not per-animal recruitment / dispersion.

    Note: under symmetric helical mode (phi_s = -phi_f) with kappa_f =
    kappa_s, I4f and I4s are mathematically identical. This shows up in
    the correlation matrix as rho = 1.000 for every I4f-vs-I4s shape
    pair. That's a real property of the two-fiber constitutive form,
    not a numerical artifact. Setting kappa_f != kappa_s to cohort
    medians (~0.03 vs ~0.06) drops rho to 0.9999 — not enough to change
    the de-collinearization outcome. A post-recruitment Psi (per-animal
    gates stacked) is the honest fix; see docs.
    """
    C_th = lam_theta ** 2; C_z = lam_z ** 2
    trace = C_th + C_z + 1.0 / (lam_theta * lam_z) ** 2
    cf, sf = np.cos(phi_f), np.sin(phi_f)
    cs, ss = np.cos(phi_s), np.sin(phi_s)
    ce, se = np.cos(phi_el), np.sin(phi_el)
    cm, sm = np.cos(phi_smc), np.sin(phi_smc)
    fTCf = (lam_theta * cf) ** 2 + (lam_z * sf) ** 2
    sTCs = (lam_theta * cs) ** 2 + (lam_z * ss) ** 2
    I4f = kappa_f * trace + (1.0 - 3.0 * kappa_f) * fTCf
    I4s = kappa_s * trace + (1.0 - 3.0 * kappa_s) * sTCs
    I4el = (lam_theta * ce) ** 2 + (lam_z * se) ** 2
    I4smc = (lam_theta * cm) ** 2 + (lam_z * sm) ** 2
    I8fs = (lam_theta ** 2) * cf * cs + (lam_z ** 2) * sf * ss
    return dict(I4f=I4f, I4s=I4s, I4el=I4el, I4smc=I4smc, I8fs=I8fs)


def psi_value_invariant(term: str, x: np.ndarray) -> np.ndarray:
    if term == "Linear":          return x
    if term == "Quadratic":       return x ** 2
    if term == "Cubic":           return x ** 3
    if term == "Exp linear":      return np.expm1(x)
    if term == "Exp quadratic":   return np.expm1(x ** 2)
    if term == "xlog linear":
        xp = np.maximum(x, -0.999999)
        return xp * np.log1p(xp)
    if term == "xlog quadratic":
        return (x ** 2) * np.log1p(x ** 2)
    raise ValueError(f"unknown invariant term {term!r}")


_STRAIN_POW = {name: pq for name, pq in STRAIN_TERMS_FULL}


def psi_value_strain(name: str, E_theta: np.ndarray, E_z: np.ndarray) -> np.ndarray:
    p, q = _STRAIN_POW[name]
    return (E_theta ** p) * (E_z ** q)


def build_psi_matrix(terms: List[Dict],
                     lam_theta: np.ndarray, lam_z: np.ndarray) -> np.ndarray:
    """Return Psi of shape (M, D)."""
    kin = kinematics(lam_theta, lam_z)
    rfi = raw_invariants(lam_theta, lam_z)
    psi_cols = []
    for r in terms:
        if r["group"] == "invariant":
            if r["inv"] == "I1":
                x = kin["I1"] - 3.0
            elif r["inv"] == "I2":
                x = kin["I2"] - 3.0
            elif r["inv"] in ("I4f", "I4s", "I4el", "I4smc"):
                x = rfi[r["inv"]] - 1.0
            elif r["inv"] == "I8fs":
                x = rfi["I8fs"]
            else:
                raise ValueError(f"unknown invariant {r['inv']}")
            col = psi_value_invariant(r["term"], x)
        elif r["group"] == "strain":
            col = psi_value_strain(r["term"], kin["E_theta"], kin["E_z"])
        else:
            raise ValueError(f"unknown group {r['group']}")
        psi_cols.append(np.asarray(col, dtype=np.float64))
    return np.stack(psi_cols, axis=1)


# ---------------------------------------------------------------------------
# Per-animal coefficient ranges
# ---------------------------------------------------------------------------

def discover_animal_csvs(root: Path) -> List[Tuple[str, Path]]:
    """Return list of (animal_name, active_terms.csv path) under root."""
    if root is None or not root.is_dir():
        return []
    out: List[Tuple[str, Path]] = []
    for animal_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for cur, _, files in os.walk(animal_dir):
            if "active_terms.csv" in files:
                out.append((animal_dir.name, Path(cur) / "active_terms.csv"))
                break
    return out


def load_observed_coeffs(terms: List[Dict],
                         animal_csvs: List[Tuple[str, Path]]) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """Returns
        coeff_obs: (n_animals, D) signed eff_norm (w_sub * w_ann);
                   zero where the term is missing/inactive.
        animal_names: list[str] length n_animals
        df_long:   long-form DataFrame for downstream inspection.
    """
    key_to_idx = {r["key"]: i for i, r in enumerate(terms)}
    rows = []
    matrix = np.zeros((len(animal_csvs), len(terms)), dtype=np.float64)
    animals = []
    for i, (animal, csv_path) in enumerate(animal_csvs):
        animals.append(animal)
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            warnings.warn(f"could not read {csv_path}: {exc}")
            continue
        for c in ("group", "inv", "term", "w_sub", "w_ann"):
            if c not in df.columns:
                df[c] = np.nan
        df["w_sub"] = pd.to_numeric(df["w_sub"], errors="coerce").fillna(0.0)
        df["w_ann"] = pd.to_numeric(df["w_ann"], errors="coerce").fillna(0.0)
        df["eff_norm"] = df["w_sub"] * df["w_ann"]
        df["key"] = (df["group"].astype(str) + " | "
                     + df["inv"].astype(str) + " | "
                     + df["term"].astype(str))
        for _, r in df.iterrows():
            j = key_to_idx.get(r["key"])
            if j is None:
                continue
            matrix[i, j] = float(r["eff_norm"])
            rows.append({"animal": animal, "term_key": r["key"],
                         "eff_norm": float(r["eff_norm"])})
    return matrix, animals, pd.DataFrame(rows)


def coeff_max_bounds(coeff_obs: np.ndarray) -> np.ndarray:
    """+- max|eff_norm| per term, with a small floor so SALib problem is valid."""
    m = np.max(np.abs(coeff_obs), axis=0)
    floor = max(1e-6, float(np.median(m[m > 0])) * 1e-6) if np.any(m > 0) else 1e-6
    m = np.where(m > 0, m, floor)
    return np.column_stack([-m, +m])


def cohort_std_bounds(coeff_obs: np.ndarray) -> np.ndarray:
    """+- cross-animal standard deviation of eff_norm per term, with a small
    floor. This is the cohort-aware analogue of `normalized` bounds: terms with
    high variability across animals get wider intervals, terms that are
    inactive in every animal get clamped to the floor."""
    n = coeff_obs.shape[0]
    if n > 1:
        s = np.std(coeff_obs, axis=0, ddof=1)
    else:
        s = np.abs(coeff_obs.reshape(-1))
    s = np.asarray(s, dtype=np.float64)
    floor = max(1e-6, float(np.median(s[s > 0])) * 1e-6) if np.any(s > 0) else 1e-6
    s = np.where(s > 0, s, floor)
    return np.column_stack([-s, +s])


# ---------------------------------------------------------------------------
# Sobol driver
# ---------------------------------------------------------------------------

def run_sobol_once(psi: np.ndarray, bounds: np.ndarray,
                   names: List[str], N: int, seed: int,
                   normalize_columns: bool) -> Dict[str, np.ndarray]:
    """Saltelli-sample N rows, evaluate Y_j = RMS(Psi w_j), analyse."""
    D = psi.shape[1]
    problem = {"num_vars": D, "names": names,
               "bounds": bounds.tolist()}
    samples = sobol_sample.sample(
        problem, N, calc_second_order=False, scramble=True, seed=int(seed)
    )
    if normalize_columns:
        col_norm = np.linalg.norm(psi, axis=0)
        col_norm[col_norm == 0.0] = 1.0
        psi_eff = psi / col_norm
    else:
        psi_eff = psi
    # Energy field
    W = psi_eff @ samples.T               # (M, N(D+2))
    Y = np.sqrt(np.mean(W ** 2, axis=0))  # (N(D+2),)
    Si = sobol.analyze(problem, Y, calc_second_order=False,
                       conf_level=0.95, print_to_console=False, seed=seed)
    return {"S1": np.asarray(Si["S1"]),
            "S1_conf": np.asarray(Si["S1_conf"]),
            "ST": np.asarray(Si["ST"]),
            "ST_conf": np.asarray(Si["ST_conf"])}


def sobol_table(result: Dict[str, np.ndarray], terms: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame({
        "term_key":   [t["key"]      for t in terms],
        "group":      [t["group"]    for t in terms],
        "inv":        [t["inv"]      for t in terms],
        "term":       [t["term"]     for t in terms],
        "category":   [t["category"] for t in terms],
        "S1":         result["S1"],
        "S1_conf":    result["S1_conf"],
        "ST":         result["ST"],
        "ST_conf":    result["ST_conf"],
    })
    # Clip tiny negatives (Monte Carlo noise) to 0 for display
    df["S1_clipped"] = df["S1"].clip(lower=0.0)
    df["ST_clipped"] = df["ST"].clip(lower=0.0)
    df["ST_rank"] = df["ST_clipped"].rank(method="min", ascending=False).astype(int)
    df["S1_rank"] = df["S1_clipped"].rank(method="min", ascending=False).astype(int)
    return df.sort_values("ST_rank").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-subgroup driver
# ---------------------------------------------------------------------------

def process_subgroup(name: str,
                     animal_root: Optional[Path],
                     out_root: Path,
                     args: argparse.Namespace) -> Optional[Dict]:
    print(f"\n{'='*64}\n=== {name} ===\n{'='*64}")
    if not animal_root or not animal_root.is_dir():
        print(f"  [skip] root not found: {animal_root}")
        return None
    animal_csvs = discover_animal_csvs(animal_root)
    if not animal_csvs:
        print(f"  [skip] no active_terms.csv under {animal_root}")
        return None
    print(f"  found {len(animal_csvs)} animal active_terms.csv files")

    sub_out = out_root / name
    sub_out.mkdir(parents=True, exist_ok=True)

    terms = build_term_catalog(poly_n_terms=args.poly_n_terms,
                               poly_include_linear=args.poly_include_linear)
    D = len(terms)
    print(f"  candidate basis: D={D} terms "
          f"({sum(1 for t in terms if t['category']=='iso')} iso, "
          f"{sum(1 for t in terms if t['category']=='aniso')} aniso, "
          f"{sum(1 for t in terms if t['category']=='coupling')} coupling)")

    # Coeffs across animals
    coeff_obs, animals, _df_long = load_observed_coeffs(terms, animal_csvs)
    bounds_coeff_max = coeff_max_bounds(coeff_obs)

    # Grids
    grids = {
        "physio": dict(lt_min=args.phys_lam_theta_min, lt_max=args.phys_lam_theta_max,
                       lz_min=args.phys_lam_z_min,    lz_max=args.phys_lam_z_max,
                       n_t=args.phys_n_grid, n_z=args.phys_n_grid),
        "extended": dict(lt_min=args.ext_lam_theta_min, lt_max=args.ext_lam_theta_max,
                         lz_min=args.ext_lam_z_min,    lz_max=args.ext_lam_z_max,
                         n_t=args.ext_n_grid, n_z=args.ext_n_grid),
    }

    psi_by_grid: Dict[str, np.ndarray] = {}
    all_tables: Dict[Tuple[str, str], pd.DataFrame] = {}
    names_list = [r["key"] for r in terms]

    bounds_cohort_std = cohort_std_bounds(coeff_obs)
    regimes = ["coeff_max", "normalized", "raw_uniform", "normalized_cohort"]
    bounds_by_regime = {
        "coeff_max":         bounds_coeff_max,
        "normalized":        np.column_stack([-np.ones(D), np.ones(D)]),
        "raw_uniform":       np.column_stack([-np.ones(D), np.ones(D)]),
        "normalized_cohort": bounds_cohort_std,
    }
    # Regimes that normalize the Psi columns (equal footing across terms).
    normalize_columns_by_regime = {
        "coeff_max":         False,
        "normalized":        True,
        "raw_uniform":       False,
        "normalized_cohort": True,
    }

    override_path = getattr(args, "psi_matrix_override", None)
    for grid_name, g in grids.items():
        lt, lz = make_grid(g["lt_min"], g["lt_max"], g["lz_min"], g["lz_max"],
                           g["n_t"], g["n_z"])
        # Optional override: replace the physio-grid Psi with a
        # pre-computed matrix (e.g. recruitment-aware from
        # build_recruited_psi.py). Extended grid still built from raw
        # invariants because recruitment only applies where the animals
        # have data — the extended grid extrapolates well past that.
        if override_path and grid_name == "physio":
            _ov = np.load(override_path, allow_pickle=True)
            _ov_keys = [str(k) for k in _ov["term_keys"].tolist()]
            if _ov_keys != names_list:
                # Reorder to match the term catalogue's ordering.
                idx = [_ov_keys.index(k) for k in names_list]
                psi = _ov["psi"][:, idx]
            else:
                psi = _ov["psi"]
            print(f"  [override] loaded physio Psi from {override_path}")
        else:
            psi = build_psi_matrix(terms, lt, lz)
        psi_by_grid[grid_name] = psi
        np.savez_compressed(sub_out / f"psi_matrix_{grid_name}.npz",
                            psi=psi, lam_theta=lt, lam_z=lz,
                            term_keys=np.array(names_list))
        print(f"  grid {grid_name}: M={psi.shape[0]} points, Psi shape={psi.shape}")

        for regime in regimes:
            result = run_sobol_once(
                psi, bounds_by_regime[regime], names_list,
                N=args.sobol_N, seed=args.seed,
                normalize_columns=normalize_columns_by_regime[regime],
            )
            table = sobol_table(result, terms)
            table.insert(0, "grid", grid_name)
            table.insert(1, "regime", regime)
            path = sub_out / f"sobol_indices_{grid_name}_{regime}.csv"
            table.to_csv(path, index=False)
            all_tables[(grid_name, regime)] = table
            top5 = ", ".join(table["term_key"].head(5).tolist())
            print(f"    {grid_name}/{regime:<11}  top-5 by ST: {top5}")

    # --- LOO on the chosen cohort-aware regime ---
    loo_regime = args.loo_regime
    if loo_regime not in ("coeff_max", "normalized_cohort"):
        warnings.warn(
            f"--loo_regime={loo_regime!r} is not cohort-aware; LOO would be "
            "degenerate. Falling back to coeff_max for LOO."
        )
        loo_regime = "coeff_max"

    if args.loo and len(animals) > 1:
        bounds_fn = (cohort_std_bounds if loo_regime == "normalized_cohort"
                     else coeff_max_bounds)
        normalize_psi = normalize_columns_by_regime[loo_regime]
        for grid_name, psi in psi_by_grid.items():
            rows = []
            for leave_idx, leave_animal in enumerate(animals):
                mask = np.ones(len(animals), dtype=bool)
                mask[leave_idx] = False
                loo_bounds = bounds_fn(coeff_obs[mask])
                result = run_sobol_once(
                    psi, loo_bounds, names_list, N=args.sobol_N,
                    seed=args.seed, normalize_columns=normalize_psi,
                )
                tab = sobol_table(result, terms)
                for r in tab.itertuples():
                    rows.append({
                        "grid": grid_name,
                        "loo_regime": loo_regime,
                        "left_out_animal": leave_animal,
                        "term_key": r.term_key,
                        "ST_rank": int(r.ST_rank),
                        "ST_clipped": float(r.ST_clipped),
                    })
            loo_df = pd.DataFrame(rows)
            agg = (loo_df.groupby("term_key")
                   .agg(median_ST_rank=("ST_rank", "median"),
                        mean_ST_rank=("ST_rank", "mean"),
                        max_ST_rank=("ST_rank", "max"),
                        in_top_k_frac=("ST_rank",
                                       lambda s: float((s <= args.top_k).mean())))
                   .reset_index())
            (loo_df.merge(agg, on="term_key")
                  .to_csv(sub_out / f"loo_rank_stability_{grid_name}.csv",
                          index=False))
            print(f"  LOO stability written: loo_rank_stability_{grid_name}.csv "
                  f"(regime={loo_regime}, n_animals={len(animals)})")

    # --- unified ranking + comparison ---
    unified_rows = []
    for r in terms:
        rec = {"term_key": r["key"], "group": r["group"], "inv": r["inv"],
               "term": r["term"], "category": r["category"]}
        for (g, m), t in all_tables.items():
            row = t[t["term_key"] == r["key"]].iloc[0]
            rec[f"ST_{g}_{m}"]      = float(row["ST_clipped"])
            rec[f"ST_rank_{g}_{m}"] = int(row["ST_rank"])
            rec[f"S1_{g}_{m}"]      = float(row["S1_clipped"])
        unified_rows.append(rec)
    unified = pd.DataFrame(unified_rows)
    # Composite score uses only the primary regime by default; we also report
    # the mean rank across all 6 (grid, regime) combos as a secondary column.
    primary = args.primary_regime
    primary_rank_cols = [c for c in unified.columns
                         if c in (f"ST_rank_physio_{primary}",
                                  f"ST_rank_extended_{primary}")]
    all_rank_cols = [c for c in unified.columns if c.startswith("ST_rank_")]
    unified["mean_ST_rank_primary"] = unified[primary_rank_cols].mean(axis=1)
    unified["max_ST_rank_primary"]  = unified[primary_rank_cols].max(axis=1)
    unified["mean_ST_rank_all"]     = unified[all_rank_cols].mean(axis=1)
    unified["max_ST_rank_all"]      = unified[all_rank_cols].max(axis=1)
    unified = unified.sort_values("mean_ST_rank_primary").reset_index(drop=True)
    # Keep the legacy column name as an alias so older consumers don't break.
    unified["mean_ST_rank"] = unified["mean_ST_rank_primary"]
    unified["max_ST_rank"]  = unified["max_ST_rank_primary"]
    unified.to_csv(sub_out / "ranked_terms_sobol.csv", index=False)
    print(f"  wrote ranked_terms_sobol.csv ({len(unified)} rows)")

    # --- compare to existing robust_score ranking if available ---
    rb_path = (Path(args.compare_root) / name / "ranked_terms.csv") if args.compare_root else None
    if rb_path and rb_path.is_file():
        rb = pd.read_csv(rb_path)
        if "term_key" not in rb.columns:
            rb["term_key"] = (rb["group"].astype(str) + " | "
                              + rb["inv"].astype(str) + " | "
                              + rb["term"].astype(str))
        rb_subset = rb[["term_key", "rank_robust", "robust_score"]].rename(
            columns={"rank_robust": "rank_robust_score",
                     "robust_score": "robust_score_value"})
        cmp_cols = ["term_key", "category",
                    "mean_ST_rank_primary", "max_ST_rank_primary",
                    f"ST_physio_{primary}", f"ST_extended_{primary}"]
        cmp_cols = [c for c in cmp_cols if c in unified.columns]
        cmp = unified[cmp_cols].merge(rb_subset, on="term_key", how="outer")
        cmp["sobol_top10"]  = cmp["mean_ST_rank_primary"] <= 10
        cmp["robust_top10"] = cmp["rank_robust_score"].fillna(99) <= 10
        cmp["agreement_top10"] = (cmp["sobol_top10"] & cmp["robust_top10"])
        cmp.sort_values("mean_ST_rank_primary").to_csv(
            sub_out / "sobol_vs_robust_score_comparison.csv", index=False)
        overlap = int(cmp["agreement_top10"].sum())
        print(f"  comparison written: {overlap}/10 top-10 Sobol-{primary} terms also in robust_score top-10")
    else:
        cmp = None

    # --- reduced robust set: top-K of the PRIMARY regime on BOTH grids ---
    K = args.top_k
    primary = args.primary_regime
    pr_phys = set(all_tables[("physio",  primary)].head(K)["term_key"])
    pr_ext  = set(all_tables[("extended", primary)].head(K)["term_key"])
    nc_phys = set(all_tables[("physio",  "normalized_cohort")].head(K)["term_key"])
    nc_ext  = set(all_tables[("extended", "normalized_cohort")].head(K)["term_key"])
    cm_phys = set(all_tables[("physio",  "coeff_max")].head(K)["term_key"])
    cm_ext  = set(all_tables[("extended", "coeff_max")].head(K)["term_key"])
    ru_phys = set(all_tables[("physio",  "raw_uniform")].head(K)["term_key"])
    ru_ext  = set(all_tables[("extended", "raw_uniform")].head(K)["term_key"])
    robust_set = pr_phys & pr_ext
    robust_cohort = nc_phys & nc_ext            # cohort-aware robust intersection
    robust_strict = pr_phys & pr_ext & cm_phys & cm_ext & ru_phys & ru_ext & nc_phys & nc_ext
    (sub_out / "reduced_term_set.txt").write_text(
        f"# robust set: top-{K} on primary regime ({primary!r}) on BOTH grids\n"
        + "\n".join(sorted(robust_set))
        + f"\n\n# COHORT-AWARE robust set: top-{K} on normalized_cohort on BOTH grids\n"
        + "\n".join(sorted(robust_cohort))
        + f"\n\n# STRICT robust set: top-{K} on all 4 regimes x 2 grids\n"
        + "\n".join(sorted(robust_strict))
        + f"\n\n# regime-specific top-{K} (physio grid)\n"
        + f"#   {primary:<17} : {sorted(pr_phys)}\n"
        + f"#   normalized_cohort : {sorted(nc_phys)}\n"
        + f"#   coeff_max         : {sorted(cm_phys)}\n"
        + f"#   raw_uniform       : {sorted(ru_phys)}\n"
        + f"# regime-specific top-{K} (extended grid)\n"
        + f"#   {primary:<17} : {sorted(pr_ext)}\n"
        + f"#   normalized_cohort : {sorted(nc_ext)}\n"
        + f"#   coeff_max         : {sorted(cm_ext)}\n"
        + f"#   raw_uniform       : {sorted(ru_ext)}\n"
    )

    summary = [
        f"Subgroup: {name}",
        f"n_animals: {len(animals)}",
        f"animals: {', '.join(animals)}",
        f"D (candidate terms): {D}",
        f"Sobol N (Saltelli, first-order only): {args.sobol_N}",
        f"Grid points per grid: {grids['physio']['n_t']*grids['physio']['n_z']}",
        f"top_k for robust intersection: {K}",
        f"PRIMARY regime: {primary}  (used for headline ranking + robust set)",
        f"LOO regime: {loo_regime} (LOO on 'normalized' / 'raw_uniform' "
        f"would be degenerate because their bounds do not depend on the cohort)",
        "",
        f"Robust set ({primary} top-{K} on both grids): {len(robust_set)} terms",
        f"  {sorted(robust_set)}",
        "",
        f"STRICT robust set (all 4 regimes x 2 grids): {len(robust_strict)} terms",
        f"  {sorted(robust_strict)}",
        "",
        "--- top-15 by primary regime (unified across grids) ---",
    ]
    head = unified.head(15)
    for _, r in head.iterrows():
        summary.append(
            f"  {int(r['mean_ST_rank_primary']):>2}  "
            f"{r['term_key']:<40}  "
            f"category={r['category']:<5}  "
            f"ST_physio={r[f'ST_physio_{primary}']:.4f}  "
            f"ST_ext={r[f'ST_extended_{primary}']:.4f}"
        )
    (sub_out / "summary.txt").write_text("\n".join(summary) + "\n")
    print(f"  robust set ({len(robust_set)}): {sorted(robust_set)}")

    return {"subgroup": name, "robust_set": sorted(robust_set),
            "strict_set": sorted(robust_strict)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--native_root",    type=Path, default=Path("Native_ivc"))
    p.add_argument("--tevg_root",      type=Path, default=Path("TEVG"))
    p.add_argument("--saphenous_root", type=Path, default=None)
    p.add_argument(
        "--cohort", action="append", default=[], metavar="NAME=PATH",
        help=("Repeatable cohort definition. When supplied, these cohorts "
              "replace --native_root/--tevg_root/--saphenous_root. "
              "Example: --cohort ALL_VEINS=path/to/training_outputs"),
    )
    p.add_argument("--out_root",       type=Path,
                   default=Path("sobol_term_selection"))
    p.add_argument("--compare_root", type=Path, default=None,
                   help="Optional folder with a prior robust_score ranking.")
    p.add_argument("--sobol_N", type=int, default=4096,
                   help="Saltelli base sample size.")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--loo",  action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--primary_regime",
                   choices=["coeff_max", "normalized", "raw_uniform",
                            "normalized_cohort"],
                   default="normalized_cohort",
                   help="Regime used for the headline ranking + reduced-set "
                        "definition. Default 'normalized_cohort' = normalized "
                        "Psi columns + cohort-derived bounds ([-sigma,+sigma] "
                        "per term across animals). Other options: "
                        "'normalized' (no cohort info), 'coeff_max' (raw Psi + "
                        "per-cohort coeff_obs max), 'raw_uniform' (raw Psi + "
                        "[-1,1] bounds).")
    p.add_argument("--loo_regime",
                   choices=["coeff_max", "normalized_cohort"],
                   default="normalized_cohort",
                   help="Which cohort-aware regime to use for the LOO "
                        "stability check (default 'normalized_cohort'). "
                        "'normalized' / 'raw_uniform' are not LOO-meaningful "
                        "because their bounds don't depend on the cohort.")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--poly_n_terms", type=int, default=7)
    p.add_argument("--poly-include-linear", "--poly_include_linear",
                   dest="poly_include_linear",
                   action=argparse.BooleanOptionalAction, default=True)
    # Physiological grid
    p.add_argument("--phys_n_grid", type=int, default=20)
    p.add_argument("--phys_lam_theta_min", type=float, default=0.80)
    p.add_argument("--phys_lam_theta_max", type=float, default=1.60)
    p.add_argument("--phys_lam_z_min",     type=float, default=1.00)
    p.add_argument("--phys_lam_z_max",     type=float, default=1.40)
    # Extended grid
    p.add_argument("--ext_n_grid", type=int, default=20)
    p.add_argument("--ext_lam_theta_min", type=float, default=0.30)
    p.add_argument("--ext_lam_theta_max", type=float, default=2.10)
    p.add_argument("--ext_lam_z_min",     type=float, default=1.00)
    p.add_argument("--ext_lam_z_max",     type=float, default=1.60)
    p.add_argument("--psi_matrix_override", type=Path, default=None,
                   help="Path to a pre-computed physio-grid Psi matrix "
                        "(e.g. recruitment-aware output of "
                        "build_recruited_psi.py). When set, the physio-grid "
                        "Psi columns are loaded from this file (reordered "
                        "to match the internal term catalogue if needed) "
                        "instead of being rebuilt from raw invariants. "
                        "Extended grid is unaffected.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out_root.mkdir(parents=True, exist_ok=True)
    print(f"Sobol N = {args.sobol_N}")
    print(f"Output  = {args.out_root.resolve()}")

    if args.cohort:
        subgroups = []
        for item in args.cohort:
            if "=" not in item:
                raise SystemExit(
                    f"Invalid --cohort {item!r}; expected NAME=PATH"
                )
            label, root = item.split("=", 1)
            label = label.strip()
            root = root.strip()
            if not label or not root:
                raise SystemExit(
                    f"Invalid --cohort {item!r}; NAME and PATH are required"
                )
            subgroups.append((label, Path(root)))
    else:
        subgroups = [
            ("NATIVE_IVC", args.native_root),
            ("TEVG",       args.tevg_root),
        ]
        if args.saphenous_root is not None:
            subgroups.append(("SAPHENOUS", args.saphenous_root))

    summaries = []
    for sub_name, root in subgroups:
        res = process_subgroup(sub_name, root, args.out_root, args)
        if res:
            summaries.append(res)

    if summaries:
        path = args.out_root / "ALL_SUBGROUPS_robust_intersection.txt"
        lines = []
        for s in summaries:
            lines.append(f"--- {s['subgroup']} ---")
            lines.append(f"robust  : {s['robust_set']}")
            lines.append(f"strict  : {s['strict_set']}")
            lines.append("")
        path.write_text("\n".join(lines))
        print(f"\nWrote cross-subgroup summary: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Pure-NumPy NNLS refit with per-invariant Fung exponential (w2) extension.

Sibling of pure_sedf_nnls_refit_per_animal.py that generalises every "Exp
linear" / "Exp quadratic" shape from the fixed form (exp(x)-1) to the
Fung-style (exp(w2 * x) - 1) with a per-invariant curvature parameter w2.

  Original:  Psi_exp = c_j * (exp(x_inv) - 1)          # w2 == 1 implicitly
  Fung:      Psi_exp = c_j * (exp(w2_inv * x_inv) - 1) # w2 per invariant

Each invariant (I1, I2, I4f, I4s, I4el, I4smc, I8fs) carries its own
`exp_coeff` on AnimalParams. The Latin-hypercube sweep gains one extra
dimension per invariant that has an exponential shape in the kept set.
The default range is [0.3, 5.0] (linear sampling); tune with
--exp_coeff_lo / --exp_coeff_hi. Pass --no-sweep_exp_coeff to lock every
w2 at 1.0 and recover the historical behaviour of the sibling script.

All other CLI flags, output filenames, and downstream tools
(emit_smoothed_predictions.py, plot_nnls_fits_per_animal.py) are
compatible with this sibling — it emits identical directory structure.

Problem solved per (animal, k):
    minimize || sqrt(w) * (A c - y) ||_2^2   subject to c_j >= 0
where A[:, j] is the stress contribution of unit c_j on term j, and y is the
stacked measured stress vector (sigma_theta for PD rows, sigma_z for FL rows).

Stress convention (matches the training script's
stress_calc_with_recruitment_and_poly): sigma_i = 2 * dW/dC_ii, with the chain
rule through fiber recruitment dropped (dW/dI4_recruited is multiplied by the
RAW structural-tensor component dI4_raw/dC_ii). Strain-basis contribution adds
dW/dE_i (the 0.5 dC/dE factor cancels the leading 2).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.special import gammaln, erf


# ---------------------------------------------------------------------------
# Constants matching the training-code quadrature
# ---------------------------------------------------------------------------
QUAD_X = np.array([
    0.005299, 0.027712, 0.067184, 0.122299, 0.191061, 0.270991,
    0.359009, 0.451846, 0.546156, 0.638756, 0.726781, 0.807785,
], dtype=np.float64)
QUAD_W = np.array([
    0.013576, 0.031126, 0.047579, 0.062314, 0.074797, 0.084578,
    0.091322, 0.094827, 0.095038, 0.092045, 0.086079, 0.077500,
], dtype=np.float64)

FAMILY_ORDER = ("f", "s", "el", "smc")
FAMILY_TO_IDX = {n: i for i, n in enumerate(FAMILY_ORDER)}
I4_DIAGONAL_PAIR = ("I4f", "I4s")


# ---------------------------------------------------------------------------
# metrics.txt parsing (same lightweight parser as the other refit script)
# ---------------------------------------------------------------------------
_HP_PATTERN = re.compile(r"^\s*([\w\.\-]+)\s*:\s*(.*?)\s*$")
_TRUE = {"true", "on", "yes", "1"}
_FALSE = {"false", "off", "no", "0"}


def _maybe_bool(v: str):
    s = v.strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    return None


def _maybe_num(v: str):
    s = v.strip()
    try:
        if any(c in s for c in ".eE"):
            return float(s)
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return None


def parse_metrics(path: Path) -> Dict[str, object]:
    hp: Dict[str, object] = {}
    if not path.exists():
        return hp
    in_hp = False
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.rstrip()
        if line.strip().startswith("HYPERPARAMETERS"):
            in_hp = True
            continue
        if not in_hp:
            continue
        if not line.strip() or line.strip().startswith("="):
            if hp:
                break
            continue
        m = _HP_PATTERN.match(line)
        if not m:
            continue
        k = m.group(1); v = m.group(2)
        bv = _maybe_bool(v)
        if bv is not None:
            hp[k] = bv; continue
        nv = _maybe_num(v)
        if nv is not None:
            hp[k] = nv; continue
        hp[k] = v
    return hp


def parse_protocol_lists(metrics_path: Path) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """Resolve PD/FL training curve names that appear in the clean-CSV
    `source_curve` column. The "TRAINING PROTOCOLS" section in metrics.txt
    holds the resolved values (e.g., 'pd100', 'fl60') and overrides the
    symbolic hyperparameter values (e.g., 'invivo', 'p60'), which are stored
    under pd_train_protocols/fl_train_protocols."""
    pd_train_hp = fl_train_hp = None
    pd_train_resolved = fl_train_resolved = None
    if not metrics_path.exists():
        return pd_train_hp, fl_train_hp
    in_training_protocols = False
    for line in metrics_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if s.startswith("TRAINING PROTOCOLS"):
            in_training_protocols = True
            continue
        if in_training_protocols:
            if s.startswith("PD:") and "[" in s:
                pd_train_resolved = _parse_pylist(s.split(":", 1)[1])
            elif s.startswith("FL:") and "[" in s:
                fl_train_resolved = _parse_pylist(s.split(":", 1)[1])
            elif s == "" or s.startswith("="):
                if pd_train_resolved is not None and fl_train_resolved is not None:
                    in_training_protocols = False
        if s.startswith("pd_train_protocols:"):
            pd_train_hp = _parse_pylist(s.split(":", 1)[1])
        elif s.startswith("fl_train_protocols:"):
            fl_train_hp = _parse_pylist(s.split(":", 1)[1])
    return (pd_train_resolved or pd_train_hp,
            fl_train_resolved or fl_train_hp)


def _parse_pylist(s: str) -> Optional[List[str]]:
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None
    out = []
    for tok in s[1:-1].split(","):
        tok = tok.strip().strip("'").strip('"')
        if tok:
            out.append(tok)
    return out


# ---------------------------------------------------------------------------
# Parse dispersion / fiber angles from trained_weights.txt (preferred)
# or fall back to defaults.
# ---------------------------------------------------------------------------
def parse_dispersion_and_angles(folder: Path, hp: Dict[str, object]) -> Dict[str, float]:
    """Returns kappa_f, kappa_s, phi_f_rad, phi_s_rad, phi_el_rad, phi_smc_rad."""
    out = {"kappa_f": float("nan"), "kappa_s": float("nan"),
           "phi_f_rad": 0.0, "phi_s_rad": 0.0,
           "phi_el_rad": math.pi / 2, "phi_smc_rad": 0.0}
    tw_path = folder / "trained_weights.txt"
    if tw_path.exists():
        txt = tw_path.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"kappa_f\s*:\s*([\-0-9.eE]+)", txt)
        if m:
            out["kappa_f"] = float(m.group(1))
        m = re.search(r"kappa_s\s*:\s*([\-0-9.eE]+)", txt)
        if m:
            out["kappa_s"] = float(m.group(1))
        m = re.search(r"phi1\s*\(deg\)\s*:\s*([\-0-9.eE]+)", txt)
        if m:
            out["phi_f_rad"] = math.radians(float(m.group(1)))
        m = re.search(r"phi2\s*\(deg\)\s*:\s*([\-0-9.eE]+)", txt)
        if m:
            out["phi_s_rad"] = math.radians(float(m.group(1)))
        m = re.search(r"phiel\s*\(deg\)\s*:\s*([\-0-9.eE]+)", txt)
        if m:
            out["phi_el_rad"] = math.radians(float(m.group(1)))
        m = re.search(r"phismc\s*\(deg\)\s*:\s*([\-0-9.eE]+)", txt)
        if m:
            out["phi_smc_rad"] = math.radians(float(m.group(1)))
        return out

    # Fall back to defaults derived from metrics hyperparameters.
    helix_init = float(hp.get("helix_angle_init_deg", 35.0))
    out["phi_f_rad"] = math.radians(-helix_init)
    out["phi_s_rad"] = math.radians(helix_init)
    warnings.warn(
        f"trained_weights.txt missing; using helix_angle_init_deg={helix_init} "
        f"and zero kappas as a fallback. Results may differ from the original fit."
    )
    out["kappa_f"] = 0.0
    out["kappa_s"] = 0.0
    return out


# ---------------------------------------------------------------------------
# Animal folder discovery and data path resolution
# ---------------------------------------------------------------------------
def discover_animal_folders(animal_root: Path,
                            exclude_path_substrings: Optional[Sequence[str]] = None) -> List[Path]:
    """Folders containing at least learned_recruitment.csv + final_equation.txt."""
    animal_root = Path(animal_root).expanduser().resolve()
    needed = ("learned_recruitment.csv", "final_equation.txt")
    exclude_path_substrings = tuple(s for s in (exclude_path_substrings or []) if s)
    found: List[Path] = []
    seen_real_dirs = set()
    for cur, dirnames, files in os.walk(animal_root, followlinks=True):
        cur_str = str(cur)
        if any(s in cur_str for s in exclude_path_substrings):
            dirnames[:] = []
            continue
        real_cur = Path(cur).resolve()
        if real_cur in seen_real_dirs:
            dirnames[:] = []
            continue
        seen_real_dirs.add(real_cur)
        fs = set(files)
        if all(n in fs for n in needed):
            found.append(Path(cur))
    return sorted(found)


def animal_name_from_metrics(folder: Path, hp: Dict[str, object]) -> str:
    name = hp.get("report_label")
    if isinstance(name, str) and name and name.lower() != "none":
        return name
    cdr = hp.get("clean_data_root")
    if isinstance(cdr, str) and cdr and cdr.lower() != "none":
        return Path(cdr).name
    fn = hp.get("filename")
    if isinstance(fn, str) and fn:
        return fn.split("_")[0]
    parts = folder.parts
    if len(parts) >= 3 and parts[-1] == "all":
        return parts[-3].split("_")[0]
    return folder.parent.name


def load_animal_data_map(path: Optional[Path]) -> Dict[str, Path]:
    if path is None:
        return {}
    df = pd.read_csv(path)
    if "animal" not in df.columns or "clean_data_root" not in df.columns:
        raise ValueError(f"--animal_data_map must have columns animal,clean_data_root: {path}")
    return {str(r["animal"]): Path(str(r["clean_data_root"])).expanduser() for _, r in df.iterrows()}


def resolve_clean_data_root(folder: Path, hp: Dict[str, object], animal: str,
                            data_root_parent: Optional[Path],
                            animal_data_map: Dict[str, Path]) -> Path:
    if animal in animal_data_map:
        cand = animal_data_map[animal]
        if cand.is_dir():
            return cand
    cdr = hp.get("clean_data_root")
    candidates: List[Path] = []
    if isinstance(cdr, str) and cdr:
        cdr_path = Path(cdr).expanduser()
        candidates.append(cdr_path)
        if not cdr_path.is_absolute():
            if data_root_parent is not None:
                candidates.append(Path(data_root_parent).expanduser() / cdr_path)
            candidates.append(folder.parents[1] / cdr_path)
            candidates.append(Path.cwd() / cdr_path)
    if data_root_parent is not None:
        candidates.append(Path(data_root_parent).expanduser() / animal)
        if isinstance(cdr, str) and cdr:
            candidates.append(Path(data_root_parent).expanduser() / Path(cdr).name)
    for c in candidates:
        if c and Path(c).is_dir():
            return Path(c).resolve()
    raise FileNotFoundError(
        f"Could not locate clean CSV folder for '{animal}'. Tried: {[str(c) for c in candidates]}"
    )


# ---------------------------------------------------------------------------
# Standardize clean CSVs (subset of training-script logic, no TF dependency)
# ---------------------------------------------------------------------------
_PROTOCOL_ALIASES = {
    "test_case": "test_case",
    "protocol": "test_case",
    "case": "test_case",
}


def _first_col(df: pd.DataFrame, options: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for o in options:
        if o.lower() in cols_lower:
            return cols_lower[o.lower()]
    return None


def _standardize_clean_csv(df: pd.DataFrame, kind: str, source: str) -> pd.DataFrame:
    out = df.copy()
    tc = _first_col(out, ["test_case", "source_curve", "protocol", "case", "curve"])
    if tc is None:
        out["test_case"] = source
    else:
        out = out.rename(columns={tc: "test_case"})
    lt = _first_col(out, ["lambda_theta", "stretch_theta", "lam_theta"])
    lz = _first_col(out, ["lambda_z", "stretch_z", "lam_z"])
    if lt is None or lz is None:
        raise ValueError(f"{source}: missing lambda_theta/lambda_z columns. Have: {list(out.columns)}")
    out = out.rename(columns={lt: "lambda_theta", lz: "lambda_z"})
    if kind == "pd":
        st = _first_col(out, ["sigma_theta", "sigma_theta_kpa", "stress_theta", "sigma_t"])
        if st is None:
            raise ValueError(f"{source}: PD CSV missing sigma_theta column. Have: {list(out.columns)}")
        out = out.rename(columns={st: "sigma_theta"})
    else:
        sz = _first_col(out, ["sigma_z", "sigma_z_kpa", "stress_z"])
        if sz is None:
            raise ValueError(f"{source}: FL CSV missing sigma_z column. Have: {list(out.columns)}")
        out = out.rename(columns={sz: "sigma_z"})
    return out


def load_clean_csv_data(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (Fl, Pd) DataFrames matching the training-script convention."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"clean data root does not exist: {root}")
    pd_frames, fl_frames = [], []
    for cur, _, files in os.walk(root):
        if "_pre_enrichment_backups" in cur:
            continue
        for name in sorted(files):
            if not name.lower().endswith(".csv"):
                continue
            lower = name.lower()
            path = Path(cur) / name
            df = pd.read_csv(path)
            stem = Path(name).stem
            if "_pd" in lower or lower.startswith("pd"):
                pd_frames.append(_standardize_clean_csv(df, "pd", stem))
            elif "_fl" in lower or lower.startswith("fl"):
                fl_frames.append(_standardize_clean_csv(df, "fl", stem))
    if not pd_frames or not fl_frames:
        raise ValueError(f"Could not find PD and FL clean CSVs under {root}")
    return pd.concat(fl_frames, ignore_index=True), pd.concat(pd_frames, ignore_index=True)


def _protocol_tag(value: object) -> str:
    """Normalize protocol names so p10, fl10, and Mercedes_fl10 match."""
    text = str(value).strip().lower()
    m = re.search(r"(?:^|_)(?:pd|fl|p)(\d+)(?:_|$)", text)
    if m:
        return f"p{int(m.group(1))}"
    m = re.search(r"(?:invivo)([-+]?\d+)?", text)
    if m:
        return text
    return text


def _filter_protocols(df: pd.DataFrame, protocols: Optional[List[str]]) -> pd.DataFrame:
    if protocols is None:
        return df
    wanted = {_protocol_tag(p) for p in protocols}
    tags = df["test_case"].map(_protocol_tag)
    return df[tags.isin(wanted)].reset_index(drop=True)


def load_excel_data(xlsx_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Replicate the training script's processInput + getStressStretch on an
    unloading_data_*.xlsx file with sheets FL and PD. Returns (Fl, Pd) with
    columns lambda_theta, lambda_z, sigma_theta, sigma_z, test_case."""
    xlsx_path = Path(xlsx_path).expanduser().resolve()
    if not xlsx_path.is_file():
        raise FileNotFoundError(f"Excel sample file does not exist: {xlsx_path}")

    Fl = pd.read_excel(xlsx_path, sheet_name='FL')
    Pd = pd.read_excel(xlsx_path, sheet_name='PD')

    def _enrich(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Unloaded inner diameter (mm); OD and H stored in microns.
        df["ID"] = 1e-3 * (df["OD"] - 2 * df["H"])
        df["ref_vol"] = (np.pi / 4) * ((1e-3 * df["OD"]) ** 2 - df["ID"] ** 2) * df["L"]
        df["IR"] = df["ID"] / 2.0
        # Loaded outer/inner diameter under constant-volume assumption (mm).
        df["od"] = 1e-3 * df["Outer_Diameter_microm"]
        df["id"] = np.sqrt(df["od"] ** 2 - (4 * df["ref_vol"]) / (np.pi * df["Loaded_Length_mm"]))
        df["or"] = df["od"] / 2.0
        df["ir"] = df["id"] / 2.0
        df["h"] = df["or"] - df["ir"]
        # Unit conversion.
        mmHg_to_kPa = 0.133322
        grams_to_mN = 9.80665
        df["Pressure_kPa"] = mmHg_to_kPa * df["Pressure_mmHg"]
        df["Force_mN"] = grams_to_mN * df["Force_g"]
        df["Force_v_mN"] = df["Force_mN"] + np.pi * (df["ir"] ** 2) * df["Pressure_kPa"]
        # Stress / stretch.
        df["sigma_theta"] = (df["Pressure_kPa"] * df["ir"]) / df["h"]
        df["lambda_theta"] = (df["ir"] + df["h"] / 2.0) / (df["IR"] + 1e-3 * df["H"] / 2.0)
        df["sigma_z"] = df["Force_v_mN"] / (np.pi * df["h"] * (2 * df["ir"] + df["h"]))
        df["lambda_z"] = df["Loaded_Length_mm"] / df["L"]
        df["test_case"] = df["test_case"].astype(str)
        return df

    Fl = _enrich(Fl)
    Pd = _enrich(Pd)
    return Fl, Pd


def resolve_excel_sample_file(folder: Path, hp: Dict[str, object],
                              data_root_parent: Optional[Path]) -> Optional[Path]:
    """Look up the .xlsx sample file recorded in metrics.txt and return its
    absolute path, or None if it can't be located."""
    sf = hp.get("sample_file")
    if not isinstance(sf, str) or not sf:
        return None
    candidates: List[Path] = [Path(sf).expanduser()]
    if not Path(sf).is_absolute():
        if data_root_parent is not None:
            candidates.append(Path(data_root_parent).expanduser() / sf)
        # Common location of the Combined_Marisa_Divya_CANN repo on this machine.
        candidates.append(folder.parents[2] / sf)
        candidates.append(Path.cwd() / sf)
        candidates.append(Path(
            "/Users/marisabazzi/TEVG/Vein_CANN/VEIN/Sheep_degradation_study_control_veins/"
            "Combined_Marisa_Divya_CANN"
        ) / sf)
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c).resolve()
    return None


# ---------------------------------------------------------------------------
# Animal recruitment / dispersion / offsets bundle
# ---------------------------------------------------------------------------
class AnimalParams:
    """Frozen animal-specific physical parameters: recruitment per family,
    dispersion (kappa), fiber angles, and optional per-curve offsets.

    ref_lambda_theta/ref_lambda_z are the branch-shift reference stretches
    used by the trained FiberRecruitmentLayer. The training script computes
    them as median(X_train_fl[:, 0]) and median(X_train_pd[:, 1]) and stores
    them in learned_recruitment.csv. We read them from the CSV (they are the
    authoritative trained values) and only fall back to 1.0 if absent."""

    def __init__(self, recruitment_df: pd.DataFrame,
                 dispersion: Dict[str, float],
                 curve_offsets: Optional[pd.DataFrame]):
        recruitment_df = recruitment_df.set_index("family")
        recruit: Dict[str, Dict[str, float]] = {}
        for fam in FAMILY_ORDER:
            if fam not in recruitment_df.index:
                raise KeyError(f"learned_recruitment.csv missing family '{fam}'")
            row = recruitment_df.loc[fam]
            recruit[fam] = {
                "lambda_lb": float(row["lambda_lb"]),
                "lambda_ub": float(row["lambda_ub"]),
                "alpha": float(row["alpha"]),
                "beta": float(row["beta"]),
                "tau": float(row["tau"]),
                "dist_type": str(row.get("dist_type", "beta")).lower(),
                "shift_mode": str(row.get("shift_mode", "off")).lower(),
                "ref_lambda_theta": float(row.get("ref_lambda_theta", 1.0)),
                "ref_lambda_z": float(row.get("ref_lambda_z", 1.0)),
                "logn_mu": float(row["logn_mu"]) if "logn_mu" in row else float("nan"),
                "logn_sigma": float(row["logn_sigma"]) if "logn_sigma" in row else float("nan"),
            }
        self.recruit = recruit
        self.kappa_f = float(dispersion["kappa_f"])
        self.kappa_s = float(dispersion["kappa_s"])
        self.phi_f = float(dispersion["phi_f_rad"])
        self.phi_s = float(dispersion["phi_s_rad"])
        self.phi_el = float(dispersion["phi_el_rad"])
        self.phi_smc = float(dispersion["phi_smc_rad"])
        self.curve_offsets = curve_offsets  # DataFrame with columns curve_id, family, delta_lb
        if curve_offsets is None or curve_offsets.empty:
            self._curve_offset_map = {}
        else:
            self._curve_offset_map = {
                (str(r["family"]), str(r["curve_id"])): float(r["delta_lb"])
                for _, r in curve_offsets.iterrows()
            }
        # Per-invariant Fung exponent — psi_exp = exp(exp_coeff[inv] * x) - 1.
        # Default 1.0 recovers the sibling script's behaviour; LHS may sweep
        # any subset of these depending on --sweep_exp_coeff.
        self.exp_coeff = {inv: 1.0 for inv in
                          ("I1", "I2", "I4f", "I4s", "I4h", "I4d",
                           "I4el", "I4z", "I4smc", "I4theta", "I8fs")}

    def curve_offset(self, family: str, curve_id_str: str) -> float:
        return float(self._curve_offset_map.get((str(family), str(curve_id_str)), 0.0))

    def clone(self) -> "AnimalParams":
        out = object.__new__(AnimalParams)
        out.recruit = {fam: vals.copy() for fam, vals in self.recruit.items()}
        out.kappa_f = self.kappa_f
        out.kappa_s = self.kappa_s
        out.phi_f = self.phi_f
        out.phi_s = self.phi_s
        out.phi_el = self.phi_el
        out.phi_smc = self.phi_smc
        out.curve_offsets = None if self.curve_offsets is None else self.curve_offsets.copy()
        out._curve_offset_map = self._curve_offset_map.copy()
        out.exp_coeff = dict(self.exp_coeff)
        return out


def animal_params_to_dict(params: AnimalParams) -> dict:
    return {
        "recruit": {
            fam: {
                k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                for k, v in vals.items()
            }
            for fam, vals in params.recruit.items()
        },
        "kappa_f": float(params.kappa_f),
        "kappa_s": float(params.kappa_s),
        "phi_f": float(params.phi_f),
        "phi_s": float(params.phi_s),
        "phi_el": float(params.phi_el),
        "phi_smc": float(params.phi_smc),
        "curve_offsets": (
            None
            if params.curve_offsets is None
            else params.curve_offsets.to_dict(orient="records")
        ),
        "exp_coeff": dict(params.exp_coeff),
    }


def load_animal_params(folder: Path, hp: Dict[str, object]) -> AnimalParams:
    rec = pd.read_csv(folder / "learned_recruitment.csv")
    disp = parse_dispersion_and_angles(folder, hp)
    offsets_path = folder / "learned_curve_offsets.csv"
    offsets = None
    if offsets_path.exists():
        try:
            offsets = pd.read_csv(offsets_path)
        except Exception as e:
            warnings.warn(f"could not read {offsets_path}: {e}")
    return AnimalParams(rec, disp, offsets)


def _apply_initial_guess_csv(params: AnimalParams, animal: str,
                             csv_path: Path) -> bool:
    """Override AnimalParams in-place with values from cann_initial_guess.csv.

    The CSV is long-format (one row per (animal, family)) as produced by
    scripts/extract_cann_initial_guess.py. Missing columns / NaN cells are
    silently skipped so partial CSVs still work. Returns True if the animal
    was found in the CSV, False otherwise (in which case params are unchanged).

    Per-curve offsets, learned during CANN training, are *not* in the CSV -
    they remain loaded from learned_curve_offsets.csv in the animal folder."""
    df = pd.read_csv(csv_path)
    rows = df[df["animal"].astype(str) == str(animal)]
    if rows.empty:
        return False
    # Per-family recruitment
    for _, row in rows.iterrows():
        fam = str(row["family"])
        if fam not in params.recruit:
            continue
        r = params.recruit[fam]
        for k in ("lambda_lb", "lambda_ub", "alpha", "beta", "tau",
                  "logn_mu", "logn_sigma",
                  "ref_lambda_theta", "ref_lambda_z"):
            if k in row and pd.notna(row[k]):
                r[k] = float(row[k])
        for k in ("dist_type", "shift_mode"):
            if k in row and pd.notna(row[k]):
                r[k] = str(row[k])
    # Per-animal dispersion + angles - replicated across the 4 rows, take first.
    first = rows.iloc[0]
    if "kappa_f" in first and pd.notna(first["kappa_f"]):
        params.kappa_f = float(first["kappa_f"])
    if "kappa_s" in first and pd.notna(first["kappa_s"]):
        params.kappa_s = float(first["kappa_s"])
    for attr, col in (("phi_f",   "phi_f_deg"),
                      ("phi_s",   "phi_s_deg"),
                      ("phi_el",  "phi_el_deg"),
                      ("phi_smc", "phi_smc_deg")):
        if col in first and pd.notna(first[col]):
            setattr(params, attr, math.radians(float(first[col])))
    # Guardrail
    for fam in FAMILY_ORDER:
        r = params.recruit[fam]
        if r["lambda_ub"] <= r["lambda_lb"] + 1e-3:
            r["lambda_ub"] = r["lambda_lb"] + 1e-3
    return True


def _initial_guess_circ_weight(animal: str, csv_path: Path) -> Optional[float]:
    """Look up the per-animal circ_weight from the initial guess CSV, or None."""
    df = pd.read_csv(csv_path)
    rows = df[df["animal"].astype(str) == str(animal)]
    if rows.empty or "circ_weight" not in rows.columns:
        return None
    cw = rows.iloc[0]["circ_weight"]
    return float(cw) if pd.notna(cw) else None


def _apply_force_dist_type(params: AnimalParams, mode: str) -> None:
    """Override the per-family recruitment `dist_type` in place. Missing
    distribution-specific params get sensible defaults so the LHS spec has
    valid starting values to perturb from.

    modes: as_trained | all_beta | all_lognormal
    """
    if not mode or mode == "as_trained":
        return
    target = "beta" if mode == "all_beta" else \
             "lognormal" if mode == "all_lognormal" else None
    if target is None:
        raise ValueError(
            f"unknown --force_dist_type={mode!r} "
            f"(expected as_trained | all_beta | all_lognormal)"
        )
    for fam in FAMILY_ORDER:
        r = params.recruit[fam]
        r["dist_type"] = target
        if target == "beta":
            a = r.get("alpha", float("nan"))
            b = r.get("beta", float("nan"))
            if not (isinstance(a, float) and math.isfinite(a) and a > 0):
                r["alpha"] = 2.0
            if not (isinstance(b, float) and math.isfinite(b) and b > 0):
                r["beta"] = 4.0
        else:  # lognormal
            sg = r.get("logn_sigma", float("nan"))
            mu = r.get("logn_mu", float("nan"))
            if not (isinstance(sg, float) and math.isfinite(sg) and sg > 0):
                r["logn_sigma"] = 0.3
            if not (isinstance(mu, float) and math.isfinite(mu)):
                r["logn_mu"] = 0.0


def _latin_hypercube(n_samples: int, n_dim: int, rng: np.random.Generator) -> np.ndarray:
    if n_samples <= 0 or n_dim <= 0:
        return np.empty((max(n_samples, 0), max(n_dim, 0)), dtype=np.float64)
    u = np.empty((n_samples, n_dim), dtype=np.float64)
    base = np.arange(n_samples, dtype=np.float64)
    for j in range(n_dim):
        u[:, j] = (base + rng.random(n_samples)) / n_samples
        rng.shuffle(u[:, j])
    return u


def _bounded_relative_range(value: float, frac: float,
                            lo_floor: float, hi_ceiling: Optional[float] = None) -> Tuple[float, float]:
    value = float(value)
    span = abs(value) * max(frac, 0.0)
    if span <= 0:
        span = max(abs(lo_floor) * 0.05, 1e-6)
    lo = max(lo_floor, value - span)
    hi = value + span
    if hi_ceiling is not None:
        hi = min(hi, hi_ceiling)
    if hi <= lo:
        hi = lo + max(abs(lo) * 1e-3, 1e-6)
    return lo, hi


def build_param_specs(base: AnimalParams, recruitment_frac: float,
                      shape_frac: float, tau_frac: float, angle_deg: float,
                      kappa_delta: float,
                      sweep_exp_coeff: bool = True,
                      exp_coeff_lo: float = 0.3,
                      exp_coeff_hi: float = 5.0,
                      exp_coeff_invariants: Optional[List[str]] = None
                      ) -> List[Tuple[str, float, float, Callable[[AnimalParams, float], None]]]:
    """Same (name, lo, hi, setter) spec list used by LHS. Exposed so the
    scipy polish step can re-use identical bounds.

    When `sweep_exp_coeff` is True, adds one LHS dimension per invariant in
    `exp_coeff_invariants` (default: all 7 invariants) sampling `w2` in
    [exp_coeff_lo, exp_coeff_hi]. w2 governs the Fung curvature of any
    exponential shape belonging to that invariant."""
    specs: List[Tuple[str, float, float, Callable[[AnimalParams, float], None]]] = []
    for fam in FAMILY_ORDER:
        r = base.recruit[fam]
        lb_lo, lb_hi = _bounded_relative_range(r["lambda_lb"], recruitment_frac, 0.75, 1.8)
        ub_lo, ub_hi = _bounded_relative_range(r["lambda_ub"], recruitment_frac, lb_lo + 1e-3, 2.4)
        specs.append((f"{fam}.lambda_lb", lb_lo, lb_hi,
                      lambda p, v, fam=fam: p.recruit[fam].__setitem__("lambda_lb", float(v))))
        specs.append((f"{fam}.lambda_ub", ub_lo, ub_hi,
                      lambda p, v, fam=fam: p.recruit[fam].__setitem__("lambda_ub", float(v))))
        if r["dist_type"] == "beta":
            a_lo, a_hi = _bounded_relative_range(r["alpha"], shape_frac, 0.25, 25.0)
            b_lo, b_hi = _bounded_relative_range(r["beta"], shape_frac, 0.25, 25.0)
            specs.append((f"{fam}.alpha", a_lo, a_hi,
                          lambda p, v, fam=fam: p.recruit[fam].__setitem__("alpha", float(v))))
            specs.append((f"{fam}.beta", b_lo, b_hi,
                          lambda p, v, fam=fam: p.recruit[fam].__setitem__("beta", float(v))))
        elif r["dist_type"] == "lognormal":
            sg = r.get("logn_sigma", float("nan"))
            if math.isfinite(sg) and sg > 0:
                sg_lo, sg_hi = _bounded_relative_range(sg, shape_frac, 0.02, 3.0)
                specs.append((f"{fam}.logn_sigma", sg_lo, sg_hi,
                              lambda p, v, fam=fam: p.recruit[fam].__setitem__("logn_sigma", float(v))))
        tau_lo, tau_hi = _bounded_relative_range(r["tau"], tau_frac, 1e-4, 0.25)
        specs.append((f"{fam}.tau", tau_lo, tau_hi,
                      lambda p, v, fam=fam: p.recruit[fam].__setitem__("tau", float(v))))

    kf_lo = max(0.0, base.kappa_f - kappa_delta)
    kf_hi = min(1.0 / 3.0, base.kappa_f + kappa_delta)
    ks_lo = max(0.0, base.kappa_s - kappa_delta)
    ks_hi = min(1.0 / 3.0, base.kappa_s + kappa_delta)
    specs.extend([
        ("kappa_f", kf_lo, max(kf_lo + 1e-6, kf_hi), lambda p, v: setattr(p, "kappa_f", float(v))),
        ("kappa_s", ks_lo, max(ks_lo + 1e-6, ks_hi), lambda p, v: setattr(p, "kappa_s", float(v))),
    ])
    angle_rad = math.radians(max(angle_deg, 0.0))
    for attr in ("phi_f", "phi_s", "phi_el", "phi_smc"):
        center = float(getattr(base, attr))
        specs.append((attr, center - angle_rad, center + angle_rad,
                      lambda p, v, attr=attr: setattr(p, attr, float(v))))

    # Per-invariant Fung exponent (only added when sweeping is enabled).
    if sweep_exp_coeff:
        invs = exp_coeff_invariants or ("I1", "I2", "I4f", "I4s", "I4h",
                                         "I4el", "I4smc", "I8fs")
        lo = float(exp_coeff_lo)
        hi = float(max(exp_coeff_lo + 1e-6, exp_coeff_hi))
        for inv in invs:
            specs.append((
                f"exp_coeff.{inv}", lo, hi,
                lambda p, v, inv=inv: p.exp_coeff.__setitem__(inv, float(v)),
            ))
    return specs


def _extract_param_vector(params: AnimalParams,
                          specs: List[Tuple[str, float, float, Callable]]) -> np.ndarray:
    """Read current values from `params` at the positions named in `specs`."""
    out = np.empty(len(specs), dtype=np.float64)
    for i, (name, _lo, _hi, _setter) in enumerate(specs):
        if name.startswith("exp_coeff."):
            _, inv = name.split(".", 1)
            out[i] = float(params.exp_coeff.get(inv, 1.0))
        elif "." in name:
            fam, key = name.split(".", 1)
            out[i] = float(params.recruit[fam][key])
        else:
            out[i] = float(getattr(params, name))
    return out


def _apply_param_vector(base: AnimalParams,
                        specs: List[Tuple[str, float, float, Callable]],
                        x: np.ndarray) -> AnimalParams:
    """Clone `base` and apply the vector `x` via each spec's setter."""
    cand = base.clone()
    for (_name, _lo, _hi, setter), v in zip(specs, x):
        setter(cand, float(v))
    # Guardrail: keep lambda_ub strictly above lambda_lb for every family.
    for fam in FAMILY_ORDER:
        r = cand.recruit[fam]
        if r["lambda_ub"] <= r["lambda_lb"] + 1e-3:
            r["lambda_ub"] = r["lambda_lb"] + 1e-3
    return cand


def enforce_paired_diagonal_families(params: AnimalParams) -> None:
    """Tie the +/- diagonal families for a paired-diagonal ablation.

    The first diagonal family ('f') is treated as the reference. The second
    family ('s') shares its recruitment distribution, dispersion, and Fung
    curvature, and its angle is mirrored about the circumferential axis.
    """
    params.recruit["s"] = params.recruit["f"].copy()
    params.kappa_s = params.kappa_f
    params.phi_s = -float(params.phi_f)
    params.exp_coeff["I4s"] = float(params.exp_coeff.get("I4f", 1.0))
    if params._curve_offset_map:
        for (_fam, curve_id), delta in list(params._curve_offset_map.items()):
            if _fam == "f":
                params._curve_offset_map[("s", curve_id)] = float(delta)


def make_lhs_param_candidates(base: AnimalParams, n_samples: int, seed: int,
                              recruitment_frac: float, shape_frac: float,
                              tau_frac: float, angle_deg: float,
                              kappa_delta: float,
                              sweep_exp_coeff: bool = True,
                              exp_coeff_lo: float = 0.3,
                              exp_coeff_hi: float = 5.0,
                              exp_coeff_invariants: Optional[List[str]] = None
                              ) -> List[Tuple[int, str, AnimalParams, dict]]:
    """Return baseline plus Latin-hypercube perturbations around trained values."""
    candidates: List[Tuple[int, str, AnimalParams, dict]] = [(0, "baseline", base.clone(), {})]
    if n_samples <= 1:
        return candidates

    specs = build_param_specs(base, recruitment_frac=recruitment_frac,
                              shape_frac=shape_frac, tau_frac=tau_frac,
                              angle_deg=angle_deg, kappa_delta=kappa_delta,
                              sweep_exp_coeff=sweep_exp_coeff,
                              exp_coeff_lo=exp_coeff_lo,
                              exp_coeff_hi=exp_coeff_hi,
                              exp_coeff_invariants=exp_coeff_invariants)

    rng = np.random.default_rng(seed)
    unit = _latin_hypercube(n_samples - 1, len(specs), rng)
    for i in range(n_samples - 1):
        cand = base.clone()
        values = {}
        for j, (name, lo, hi, setter) in enumerate(specs):
            val = lo + unit[i, j] * (hi - lo)
            setter(cand, val)
            values[name] = float(val)
        for fam in FAMILY_ORDER:
            r = cand.recruit[fam]
            if r["lambda_ub"] <= r["lambda_lb"] + 1e-3:
                r["lambda_ub"] = r["lambda_lb"] + 1e-3
        candidates.append((i + 1, "lhs", cand, values))
    return candidates


def refine_params_local(base_params: AnimalParams,
                        specs: List[Tuple[str, float, float, Callable]],
                        objective_fn: Callable[[AnimalParams], float],
                        method: str = "L-BFGS-B",
                        maxiter: int = 100) -> Tuple[AnimalParams, float, dict]:
    """Local polish of an LHS winner.

    Wraps `objective_fn` in a vector-of-parameters signature and hands it to
    scipy.optimize.minimize. Returns the polished params, best objective,
    and a small diagnostics dict. Falls back to the input params if scipy
    is unavailable or the polish makes the fit worse.
    """
    try:
        from scipy.optimize import minimize  # local import: keep scipy optional
    except Exception as exc:
        return base_params.clone(), float("inf"), {
            "method": "unavailable", "n_iter": 0, "n_fev": 0,
            "message": f"scipy import failed: {exc}",
        }

    x0 = _extract_param_vector(base_params, specs)
    bounds = [(lo, hi) for (_n, lo, hi, _s) in specs]
    # Keep the starting vector inside the box (LHS can wander slightly).
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

    def f_of_x(x: np.ndarray) -> float:
        try:
            cand = _apply_param_vector(base_params, specs, x)
            val = float(objective_fn(cand))
        except Exception:
            return 1e12
        if not math.isfinite(val):
            return 1e12
        return val

    base_obj = f_of_x(x0)

    kwargs: Dict[str, object] = {"method": method}
    if method.upper() in {"L-BFGS-B", "TNC", "SLSQP", "POWELL", "TRUST-CONSTR"}:
        kwargs["bounds"] = bounds
        kwargs["options"] = {"maxiter": int(maxiter), "disp": False}
    else:  # Nelder-Mead handles bounds via clipping inside f_of_x
        kwargs["options"] = {"maxiter": int(maxiter), "xatol": 1e-4,
                             "fatol": 1e-6, "disp": False}

    try:
        res = minimize(f_of_x, x0, **kwargs)
    except Exception as exc:
        return base_params.clone(), base_obj, {
            "method": method, "n_iter": 0, "n_fev": 0,
            "message": f"minimize raised: {exc}",
        }

    if not math.isfinite(res.fun) or res.fun > base_obj:
        return base_params.clone(), base_obj, {
            "method": method,
            "n_iter": int(getattr(res, "nit", 0)),
            "n_fev": int(getattr(res, "nfev", 0)),
            "message": "no improvement over start; kept LHS winner",
            "improvement": 0.0,
        }
    polished = _apply_param_vector(base_params, specs, np.asarray(res.x))
    return polished, float(res.fun), {
        "method": method,
        "n_iter": int(getattr(res, "nit", 0)),
        "n_fev": int(getattr(res, "nfev", 0)),
        "message": str(getattr(res, "message", "")),
        "improvement": float(base_obj - res.fun),
    }


# ---------------------------------------------------------------------------
# Pure-NumPy invariant + kinematic helpers
# ---------------------------------------------------------------------------
def kinematics(lam_theta: np.ndarray, lam_z: np.ndarray) -> Dict[str, np.ndarray]:
    lam_r = 1.0 / (lam_theta * lam_z)
    C_theta = lam_theta ** 2
    C_z = lam_z ** 2
    C_r = lam_r ** 2
    I1 = C_theta + C_z + C_r
    I2 = C_theta * C_z + C_theta * C_r + C_z * C_r
    E_theta = 0.5 * (C_theta - 1.0)
    E_z = 0.5 * (C_z - 1.0)
    return dict(lam_r=lam_r, C_theta=C_theta, C_z=C_z, C_r=C_r,
                I1=I1, I2=I2, E_theta=E_theta, E_z=E_z)


def raw_fiber_invariants(lam_theta: np.ndarray, lam_z: np.ndarray,
                         params: AnimalParams) -> Dict[str, np.ndarray]:
    """Raw (un-recruited) I4 invariants and the structural-tensor C-components."""
    C_theta = lam_theta ** 2
    C_z = lam_z ** 2
    lam_r2 = (1.0 / (lam_theta * lam_z)) ** 2
    trace = C_theta + C_z + lam_r2

    cos_f, sin_f = math.cos(params.phi_f), math.sin(params.phi_f)
    cos_s, sin_s = math.cos(params.phi_s), math.sin(params.phi_s)
    cos_el, sin_el = math.cos(params.phi_el), math.sin(params.phi_el)
    cos_smc, sin_smc = math.cos(params.phi_smc), math.sin(params.phi_smc)

    fTCf = (lam_theta * cos_f) ** 2 + (lam_z * sin_f) ** 2
    sTCs = (lam_theta * cos_s) ** 2 + (lam_z * sin_s) ** 2
    I4f_raw = params.kappa_f * trace + (1.0 - 3.0 * params.kappa_f) * fTCf
    I4s_raw = params.kappa_s * trace + (1.0 - 3.0 * params.kappa_s) * sTCs
    I4el_raw = (lam_theta * cos_el) ** 2 + (lam_z * sin_el) ** 2
    I4smc_raw = (lam_theta * cos_smc) ** 2 + (lam_z * sin_smc) ** 2
    I8fs = (lam_theta ** 2) * cos_f * cos_s + (lam_z ** 2) * sin_f * sin_s

    # dI/dC_ii structural-tensor components (constants per family, no lambda dependence).
    Hf_tt = params.kappa_f + (1.0 - 3.0 * params.kappa_f) * cos_f ** 2
    Hf_zz = params.kappa_f + (1.0 - 3.0 * params.kappa_f) * sin_f ** 2
    Hs_tt = params.kappa_s + (1.0 - 3.0 * params.kappa_s) * cos_s ** 2
    Hs_zz = params.kappa_s + (1.0 - 3.0 * params.kappa_s) * sin_s ** 2
    Hel_tt = cos_el ** 2
    Hel_zz = sin_el ** 2
    Hsmc_tt = cos_smc ** 2
    Hsmc_zz = sin_smc ** 2

    return dict(
        I4f_raw=I4f_raw, I4s_raw=I4s_raw, I4el_raw=I4el_raw, I4smc_raw=I4smc_raw,
        I8fs=I8fs,
        Hf_tt=Hf_tt, Hf_zz=Hf_zz, Hs_tt=Hs_tt, Hs_zz=Hs_zz,
        Hel_tt=Hel_tt, Hel_zz=Hel_zz, Hsmc_tt=Hsmc_tt, Hsmc_zz=Hsmc_zz,
        I8fs_factor_tt=C_theta * cos_f * cos_s,
        I8fs_factor_zz=C_z * sin_f * sin_s,
    )


# ---------------------------------------------------------------------------
# Recruitment in pure NumPy (mirrors FiberRecruitmentLayer._recruited_invariant_v4)
# ---------------------------------------------------------------------------
def _softplus(x: np.ndarray) -> np.ndarray:
    # numerically stable softplus
    return np.where(x > 0, x + np.log1p(np.exp(-np.abs(x))), np.log1p(np.exp(x)))


def _smoothpos(x: np.ndarray, tau: float) -> np.ndarray:
    return tau * _softplus(x / tau)


def _effective_lb(params: AnimalParams, fam: str, lam_theta: np.ndarray,
                  lam_z: np.ndarray, branch_flag: np.ndarray,
                  curve_ids: np.ndarray) -> np.ndarray:
    r = params.recruit[fam]
    base_lb = r["lambda_lb"]
    base_ub = max(r["lambda_ub"], base_lb + 1e-3)
    shift = r["shift_mode"]
    if shift == "off":
        ratio = np.ones_like(lam_theta)
    else:
        # branch_flag=0 -> PD branch -> companion=lambda_z, ref=ref_lambda_z
        # branch_flag=1 -> FL branch -> companion=lambda_theta, ref=ref_lambda_theta
        companion = np.where(branch_flag < 0.5, lam_z, lam_theta)
        ref = np.where(branch_flag < 0.5, r["ref_lambda_z"], r["ref_lambda_theta"])
        ratio = companion / np.maximum(ref, 1e-6)
        if shift == "inverse":
            ratio = np.maximum(ref, 1e-6) / np.maximum(companion, 1e-6)
        ratio = np.clip(ratio, 0.5, 2.0)

    lb_eff = base_lb * ratio

    if params.curve_offsets is not None and not params.curve_offsets.empty:
        offsets_per_sample = np.array(
            [params.curve_offset(fam, str(c)) for c in curve_ids],
            dtype=np.float64,
        )
        lb_eff = lb_eff + offsets_per_sample

    lb_min = base_lb if shift == "off" else 0.5 * base_lb  # safety floor
    ub_floor = base_ub - 1e-3
    return np.minimum(np.clip(lb_eff, lb_min, base_ub - 1e-3), ub_floor)


def recruited_invariant(I4_raw: np.ndarray, params: AnimalParams, fam: str,
                        lam_theta: np.ndarray, lam_z: np.ndarray,
                        branch_flag: np.ndarray, curve_ids: np.ndarray) -> np.ndarray:
    r = params.recruit[fam]
    dist = r["dist_type"]
    if dist not in ("beta", "lognormal"):
        raise ValueError(f"Unsupported dist_type for {fam}: {dist}")

    lam_mag = np.sqrt(np.maximum(I4_raw, 1e-12))
    lb_eff = _effective_lb(params, fam, lam_theta, lam_z, branch_flag, curve_ids)
    ub = max(r["lambda_ub"], lb_eff.max() + 1e-3) if lb_eff.size else r["lambda_ub"]
    ub_arr = np.full_like(lb_eff, r["lambda_ub"])
    ub_arr = np.maximum(ub_arr, lb_eff + 1e-3)
    span = np.maximum(ub_arr - lb_eff, 1e-3)
    tau = float(r["tau"])

    u = QUAD_X.reshape(1, -1)        # (1, nq)
    w_gl = QUAD_W.reshape(1, -1)
    lam_s = lb_eff[:, None] + span[:, None] * u   # (N, nq)

    if dist == "beta":
        a = float(r["alpha"]); b = float(r["beta"])
        u_safe = np.clip(u, 1e-6, 1 - 1e-6)
        logB = gammaln(a) + gammaln(b) - gammaln(a + b)
        log_pdf = (a - 1.0) * np.log(u_safe) + (b - 1.0) * np.log(1 - u_safe) - logB
        pdf_u = np.exp(log_pdf)
        quad_weight = pdf_u * w_gl
    else:
        mu = float(r["logn_mu"]); sg = float(r["logn_sigma"])
        if not (math.isfinite(mu) and math.isfinite(sg) and sg > 0):
            raise ValueError(
                f"Lognormal recruitment requested for family {fam!r} but "
                f"logn_mu/logn_sigma are not finite. Provide --animal_data_map "
                f"or re-run training to populate these fields."
            )
        y = np.maximum(lam_s - lb_eff[:, None], 1e-12)
        base = np.exp(-0.5 * ((np.log(y) - mu) / sg) ** 2) / (y * sg * math.sqrt(2 * math.pi))
        yub = np.maximum(ub_arr[:, None] - lb_eff[:, None], 1e-12)
        Za = 0.5 * (1.0 + erf((np.log(yub) - mu) / (sg * math.sqrt(2.0))))
        pdf_lam = base / np.maximum(Za, 1e-9)
        quad_weight = pdf_lam * span[:, None] * w_gl

    activation = _smoothpos((lam_mag[:, None] / lam_s) - 1.0, tau)
    expected_activation = np.sum(activation * quad_weight, axis=1)
    recruited_lambda = 1.0 + expected_activation
    return recruited_lambda ** 2


# ---------------------------------------------------------------------------
# psi basis: closed-form expressions and analytic derivatives
# ---------------------------------------------------------------------------
def psi_and_derivative_invariant(term_name: str, x: np.ndarray,
                                 w2: float = 1.0
                                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (psi(x), d psi / dx) for invariant-shift argument x.

    `w2` scales the argument inside the exponential shapes so the full
    contribution is c * (exp(w2 * x) - 1) — the Fung-style strain-stiffening
    curvature parameter. Defaults to 1.0, which reproduces the historical
    sibling-script behaviour for every non-exp caller. Linear / Quadratic /
    Cubic / xlog shapes are unaffected."""
    if term_name == "Linear":
        return x, np.ones_like(x)
    if term_name == "Quadratic":
        return x ** 2, 2.0 * x
    if term_name == "Cubic":
        return x ** 3, 3.0 * x ** 2
    if term_name == "Exp linear":
        z = np.clip(w2 * x, -80.0, 80.0)
        ez = np.exp(z)
        return np.expm1(z), w2 * ez
    if term_name == "Exp quadratic":
        z = np.clip(w2 * x ** 2, -80.0, 80.0)
        ez = np.exp(z)
        return np.expm1(z), 2.0 * w2 * x * ez
    if term_name == "Exp cubic":
        z = np.clip(w2 * x ** 3, -80.0, 80.0)
        ez = np.exp(z)
        return np.expm1(z), 3.0 * w2 * x ** 2 * ez
    if term_name == "xlog linear":
        # psi = x * log(1+x); d psi/dx = log(1+x) + x/(1+x)
        # The training-code activation clamps x>=0 via log1p; we mirror that for
        # the psi value but compute the derivative analytically.
        xp = np.maximum(x, -0.999999)
        return xp * np.log1p(xp), np.log1p(xp) + xp / (1.0 + xp)
    if term_name == "xlog quadratic":
        x2 = x ** 2
        return x2 * np.log1p(x2), 2.0 * x * np.log1p(x2) + 2.0 * x ** 3 / (1.0 + x2)
    raise ValueError(f"Unknown invariant term name: {term_name!r}")


_STRAIN_DISPLAY_TO_POW = {
    "E_θ":     (1, 0),
    "E_z":     (0, 1),
    "E_θ²":    (2, 0),
    "E_z²":    (0, 2),
    "E_θ·E_z": (1, 1),
    "E_θ³":    (3, 0),
    "E_z³":    (0, 3),
    "E_θ⁴":    (4, 0),
    "E_z⁴":    (0, 4),
    "E_θ²·E_z":  (2, 1),
    "E_θ·E_z²":  (1, 2),
    "E_θ²·E_z²": (2, 2),
}


def psi_and_derivatives_strain(term_name: str, E_theta: np.ndarray, E_z: np.ndarray,
                               scale: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (psi, d psi / dE_theta, d psi / dE_z) where psi = (E_th^p * E_z^q) / scale."""
    if term_name not in _STRAIN_DISPLAY_TO_POW:
        raise ValueError(f"Unknown strain term: {term_name!r}")
    p, q = _STRAIN_DISPLAY_TO_POW[term_name]
    base = (E_theta ** p) * (E_z ** q)
    psi = base / max(scale, 1e-12)
    if p == 0:
        dpsi_dEt = np.zeros_like(E_theta)
    else:
        dpsi_dEt = p * (E_theta ** (p - 1)) * (E_z ** q) / max(scale, 1e-12)
    if q == 0:
        dpsi_dEz = np.zeros_like(E_z)
    else:
        dpsi_dEz = q * (E_theta ** p) * (E_z ** (q - 1)) / max(scale, 1e-12)
    return psi, dpsi_dEt, dpsi_dEz


# ---------------------------------------------------------------------------
# Strain basis scale recovery
# ---------------------------------------------------------------------------
def compute_strain_basis_scales(E_th: np.ndarray, E_z: np.ndarray) -> Dict[str, float]:
    """Max|E_th^p * E_z^q| over the training set; matches training script logic."""
    scales: Dict[str, float] = {}
    for name, (p, q) in _STRAIN_DISPLAY_TO_POW.items():
        if E_th.size == 0:
            scales[name] = 1.0; continue
        vals = (E_th ** p) * (E_z ** q)
        s = float(np.max(np.abs(vals))) if vals.size else 1.0
        if not math.isfinite(s) or s <= 0:
            s = 1.0
        scales[name] = max(s, 1e-6)
    return scales


# ---------------------------------------------------------------------------
# Build stress-contribution design column for one selected term
# ---------------------------------------------------------------------------
def stress_column_for_term(group: str, inv: str, term: str,
                           strain_scale: float,
                           kin_pd: Dict[str, np.ndarray], rfi_pd: Dict[str, np.ndarray],
                           recruited_pd: Dict[str, np.ndarray],
                           kin_fl: Dict[str, np.ndarray], rfi_fl: Dict[str, np.ndarray],
                           recruited_fl: Dict[str, np.ndarray],
                           derivative_mode: str = "analytic",
                           fd_eps: float = 1e-5,
                           w2: float = 1.0,
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (sigma_theta_contrib_PD, sigma_z_contrib_FL) for unit c_j on this term.

    Stress convention follows the training-code
    stress_calc_with_recruitment_and_poly:
      sigma_theta_inv_j = 2 * dpsi_j/dI_a * (dI_a_raw / dC_theta)
        - for I1: dI1/dC_theta = 1
        - for I2: dI2/dC_theta = I1 - C_theta
        - for I4_a: H_a_tt (raw structural-tensor component)
        - for I8fs: C_theta * cos(phi_f) * cos(phi_s)
      sigma_theta_strain_j = dpsi_j/dE_theta
    and analogously for sigma_z with C_theta -> C_z (and *_tt -> *_zz).
    """
    def _eval_invariant(arg_pd: np.ndarray, arg_fl: np.ndarray):
        if derivative_mode == "analytic":
            psi_pd, dpsi_pd = psi_and_derivative_invariant(term, arg_pd, w2=w2)
            psi_fl, dpsi_fl = psi_and_derivative_invariant(term, arg_fl, w2=w2)
        else:
            # central FD on the SAME argument (matches analytic in the limit)
            psi_pd, _ = psi_and_derivative_invariant(term, arg_pd, w2=w2)
            psi_fl, _ = psi_and_derivative_invariant(term, arg_fl, w2=w2)
            ep_pd, _ = psi_and_derivative_invariant(term, arg_pd + fd_eps, w2=w2)
            em_pd, _ = psi_and_derivative_invariant(term, arg_pd - fd_eps, w2=w2)
            ep_fl, _ = psi_and_derivative_invariant(term, arg_fl + fd_eps, w2=w2)
            em_fl, _ = psi_and_derivative_invariant(term, arg_fl - fd_eps, w2=w2)
            dpsi_pd = (ep_pd - em_pd) / (2.0 * fd_eps)
            dpsi_fl = (ep_fl - em_fl) / (2.0 * fd_eps)
        return dpsi_pd, dpsi_fl

    if group == "invariant":
        if inv == "I1":
            arg_pd = kin_pd["I1"] - 3.0
            arg_fl = kin_fl["I1"] - 3.0
            dpsi_pd, dpsi_fl = _eval_invariant(arg_pd, arg_fl)
            return 2.0 * dpsi_pd * 1.0, 2.0 * dpsi_fl * 1.0  # dI1/dC_ii = 1
        if inv == "I2":
            arg_pd = kin_pd["I2"] - 3.0
            arg_fl = kin_fl["I2"] - 3.0
            dpsi_pd, dpsi_fl = _eval_invariant(arg_pd, arg_fl)
            return (2.0 * dpsi_pd * (kin_pd["I1"] - kin_pd["C_theta"]),
                    2.0 * dpsi_fl * (kin_fl["I1"] - kin_fl["C_z"]))
        if inv in ("I4f", "I4s", "I4h", "I4d",
                   "I4el", "I4z", "I4smc", "I4theta"):
            _ALIAS = {"I4h": "I4f", "I4d": "I4f",
                      "I4z": "I4el", "I4theta": "I4smc"}
            key = _ALIAS.get(inv, inv)
            arg_pd = recruited_pd[key] - 1.0
            arg_fl = recruited_fl[key] - 1.0
            dpsi_pd, dpsi_fl = _eval_invariant(arg_pd, arg_fl)
            tt_map = {"I4f": "Hf_tt", "I4s": "Hs_tt",
                      "I4el": "Hel_tt", "I4smc": "Hsmc_tt"}
            zz_map = {"I4f": "Hf_zz", "I4s": "Hs_zz",
                      "I4el": "Hel_zz", "I4smc": "Hsmc_zz"}
            tt_key = tt_map[key]
            zz_key = zz_map[key]
            return 2.0 * dpsi_pd * rfi_pd[tt_key], 2.0 * dpsi_fl * rfi_fl[zz_key]
        if inv == "I4diag_pair":
            arg_f_pd = recruited_pd["I4f"] - 1.0
            arg_f_fl = recruited_fl["I4f"] - 1.0
            arg_s_pd = recruited_pd["I4s"] - 1.0
            arg_s_fl = recruited_fl["I4s"] - 1.0
            dpsi_f_pd, dpsi_f_fl = _eval_invariant(arg_f_pd, arg_f_fl)
            dpsi_s_pd, dpsi_s_fl = _eval_invariant(arg_s_pd, arg_s_fl)
            col_pd = (
                2.0 * dpsi_f_pd * rfi_pd["Hf_tt"]
                + 2.0 * dpsi_s_pd * rfi_pd["Hs_tt"]
            )
            col_fl = (
                2.0 * dpsi_f_fl * rfi_fl["Hf_zz"]
                + 2.0 * dpsi_s_fl * rfi_fl["Hs_zz"]
            )
            return col_pd, col_fl
        if inv == "I8fs":
            if term in ("xlog linear", "xlog quadratic"):
                raise ValueError(f"I8fs does not support xlog terms (got {term!r}).")
            arg_pd = rfi_pd["I8fs"]
            arg_fl = rfi_fl["I8fs"]
            dpsi_pd, dpsi_fl = _eval_invariant(arg_pd, arg_fl)
            return (2.0 * dpsi_pd * rfi_pd["I8fs_factor_tt"],
                    2.0 * dpsi_fl * rfi_fl["I8fs_factor_zz"])
        raise ValueError(f"Unknown invariant key: {inv!r}")

    if group == "strain":
        _, dpsi_dEt_pd, _ = psi_and_derivatives_strain(term, kin_pd["E_theta"], kin_pd["E_z"], strain_scale)
        _, _, dpsi_dEz_fl = psi_and_derivatives_strain(term, kin_fl["E_theta"], kin_fl["E_z"], strain_scale)
        # sigma_theta_strain = 2 * dW/dC_theta_strain = 2 * 0.5 * dW/dE_theta = dW/dE_theta.
        return dpsi_dEt_pd, dpsi_dEz_fl

    raise ValueError(f"Unknown term group: {group!r}")


# ---------------------------------------------------------------------------
# Ranked-terms loading
# ---------------------------------------------------------------------------
def load_ranked_terms(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"group", "inv", "term"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ranked_terms.csv missing columns: {missing}")
    return df


def select_top_k(ranked: pd.DataFrame, k: int) -> pd.DataFrame:
    if "rank_robust" in ranked.columns:
        out = ranked.sort_values("rank_robust").head(k).reset_index(drop=True)
    elif "robust_score" in ranked.columns:
        out = ranked.sort_values("robust_score", ascending=False).head(k).reset_index(drop=True)
    else:
        out = ranked.head(k).reset_index(drop=True)
    out["selected_rank"] = np.arange(1, len(out) + 1)
    return out


def get_strain_scale(row: pd.Series, recomputed_scales: Dict[str, float]) -> float:
    if row["group"] != "strain":
        return 1.0
    ms = row.get("median_scale", None)
    if ms is not None and not pd.isna(ms) and float(ms) > 0:
        return float(ms)
    s = recomputed_scales.get(row["term"], 1.0)
    if s <= 0 or not math.isfinite(s):
        warnings.warn(f"Strain scale missing or invalid for {row['term']!r}; using 1.0")
        return 1.0
    return s


# ---------------------------------------------------------------------------
# NNLS + metrics
# ---------------------------------------------------------------------------
def weighted_nnls(A: np.ndarray, y: np.ndarray, sample_weight: np.ndarray,
                  normalize_columns: bool) -> Tuple[np.ndarray, np.ndarray, bool]:
    if not np.all(np.isfinite(A)):
        bad = np.where(~np.isfinite(A))
        raise FloatingPointError(
            f"Design matrix contains non-finite entries at rows={bad[0][:5]}, cols={bad[1][:5]}"
        )
    if not np.all(np.isfinite(y)):
        raise FloatingPointError("Target vector contains non-finite entries")
    sqrt_w = np.sqrt(np.asarray(sample_weight, dtype=np.float64))
    Aw = A * sqrt_w[:, None]
    yw = y * sqrt_w

    col_norm = np.linalg.norm(Aw, axis=0)
    success = True
    # Treat columns whose contribution is below numerical precision as zero so
    # NNLS doesn't assign astronomical coefficients to silent terms.
    zero_col_threshold = 1e-10 * max(np.linalg.norm(yw), 1.0)
    near_zero = col_norm < zero_col_threshold
    if normalize_columns:
        scale = col_norm.copy()
        scale[near_zero] = 1.0
        Aw_s = Aw / scale
        Aw_s[:, near_zero] = 0.0
        try:
            coef_s, _ = nnls(Aw_s, yw, maxiter=200 * Aw_s.shape[1] + 100)
        except RuntimeError as e:
            warnings.warn(f"NNLS failed: {e}")
            coef_s = np.zeros(Aw_s.shape[1]); success = False
        coef = coef_s / scale
    else:
        try:
            coef, _ = nnls(Aw, yw, maxiter=200 * Aw.shape[1] + 100)
        except RuntimeError as e:
            warnings.warn(f"NNLS failed: {e}")
            coef = np.zeros(Aw.shape[1]); success = False
    assert np.all(coef >= -1e-12), f"NNLS produced negatives: {coef}"
    return np.where(coef < 0, 0.0, coef), col_norm, success


def _i4_diagonal_pair_key(row: pd.Series) -> Optional[Tuple[str, str]]:
    if str(row.get("group", "")).lower() != "invariant":
        return None
    inv = str(row.get("inv", "")).strip()
    if inv not in I4_DIAGONAL_PAIR:
        return None
    return ("I4d_pair", str(row.get("term", "")).strip())


def weighted_nnls_with_i4_diagonal_tie(
        A: np.ndarray, y: np.ndarray, sample_weight: np.ndarray,
        normalize_columns: bool, selected: pd.DataFrame,
        pair_diagonal_families: bool) -> Tuple[np.ndarray, np.ndarray, bool, int]:
    if not pair_diagonal_families:
        coef, col_norm, success = weighted_nnls(A, y, sample_weight, normalize_columns)
        return coef, col_norm, success, A.shape[1]

    groups: List[List[int]] = []
    paired: Dict[Tuple[str, str], int] = {}
    for idx, row in selected.reset_index(drop=True).iterrows():
        key = _i4_diagonal_pair_key(row)
        if key is None:
            groups.append([int(idx)])
            continue
        if key in paired:
            groups[paired[key]].append(int(idx))
        else:
            paired[key] = len(groups)
            groups.append([int(idx)])

    if all(len(g) == 1 for g in groups):
        coef, col_norm, success = weighted_nnls(A, y, sample_weight, normalize_columns)
        return coef, col_norm, success, A.shape[1]

    A_solve = np.column_stack([A[:, cols].sum(axis=1) for cols in groups])
    coef_solve, _solve_norm, success = weighted_nnls(
        A_solve, y, sample_weight, normalize_columns,
    )
    coef = np.zeros(A.shape[1], dtype=np.float64)
    for c, cols in zip(coef_solve, groups):
        coef[cols] = float(c)

    sqrt_w = np.sqrt(np.asarray(sample_weight, dtype=np.float64))
    col_norm = np.linalg.norm(A * sqrt_w[:, None], axis=0)
    return coef, col_norm, success, A_solve.shape[1]


def fit_metrics(y_pd: np.ndarray, yhat_pd: np.ndarray, y_fl: np.ndarray, yhat_fl: np.ndarray,
                w_pd: np.ndarray, w_fl: np.ndarray, n_params: int) -> Dict[str, float]:
    def _rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2))) if a.size else float("nan")

    def _r2(a, b):
        if a.size == 0:
            return float("nan")
        ss_res = float(np.sum((a - b) ** 2))
        ss_tot = float(np.sum((a - np.mean(a)) ** 2))
        return float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot

    rmse_t = _rmse(y_pd, yhat_pd); rmse_z = _rmse(y_fl, yhat_fl)
    r2_t = _r2(y_pd, yhat_pd); r2_z = _r2(y_fl, yhat_fl)
    y_all = np.concatenate([y_pd, y_fl]); yhat_all = np.concatenate([yhat_pd, yhat_fl])
    w_all = np.concatenate([w_pd, w_fl])
    combined = _rmse(y_all, yhat_all)
    if y_all.size:
        wsum = float(np.sum(w_all))
        wrmse = float(np.sqrt(np.sum(w_all * (y_all - yhat_all) ** 2)
                              / max(wsum, 1e-12)))
        # Weighted R^2: uses the same per-sample weights the NNLS solve does.
        # SS_res_w = sum w (y - yhat)^2;  SS_tot_w = sum w (y - ybar_w)^2 with
        # ybar_w = weighted mean of y. This is the R^2 that matches the loss.
        y_mean_w = float(np.sum(w_all * y_all) / max(wsum, 1e-12))
        ss_res_w = float(np.sum(w_all * (y_all - yhat_all) ** 2))
        ss_tot_w = float(np.sum(w_all * (y_all - y_mean_w) ** 2))
        r2_weighted = float("nan") if ss_tot_w <= 0 else 1.0 - ss_res_w / ss_tot_w
    else:
        wrmse = float("nan")
        r2_weighted = float("nan")
    # Unweighted overall R^2 too, for anyone still reporting it alongside the
    # per-direction R^2.
    r2_overall = _r2(y_all, yhat_all) if y_all.size else float("nan")
    rss = float(np.sum((y_all - yhat_all) ** 2))
    wrss = float(np.sum(w_all * (y_all - yhat_all) ** 2))
    n = y_all.size
    eps = 1e-12
    aic = bic = float("nan")
    if n > 0 and n_params > 0:
        aic = n * math.log(max(rss, eps) / n) + 2 * n_params
        bic = n * math.log(max(rss, eps) / n) + n_params * math.log(max(n, 1))
    return dict(
        RMSE_theta=rmse_t, RMSE_z=rmse_z, R2_theta=r2_t, R2_z=r2_z,
        R2_overall=r2_overall, R2_weighted=r2_weighted,
        combined_RMSE=combined, weighted_RMSE=wrmse,
        RSS=rss, weighted_RSS=wrss, AIC=aic, BIC=bic,
    )


# ---------------------------------------------------------------------------
# Per-animal driver
# ---------------------------------------------------------------------------
def process_animal(folder: Path, animal: str, ranked: pd.DataFrame, ks: List[int],
                   clean_data_root: Path, out_root: Path,
                   circ_weight: float, normalize_columns: bool,
                   derivative_mode: str, fd_eps: float, make_plots: bool,
                   validate_original_terms: bool,
                   lhs_samples: int = 1, lhs_seed: int = 12345,
                   lhs_recruitment_frac: float = 0.20,
                   lhs_shape_frac: float = 0.50,
                   lhs_tau_frac: float = 0.50,
                   lhs_angle_deg: float = 15.0,
                   lhs_kappa_delta: float = 0.05,
                   lhs_objective: str = "weighted_RMSE",
                   refine_top_k: int = 0,
                   refine_method: str = "L-BFGS-B",
                   refine_maxiter: int = 100,
                   force_dist_type: str = "as_trained",
                   initial_guess_csv: Optional[Path] = None,
                   sweep_exp_coeff: bool = True,
                   exp_coeff_lo: float = 0.3,
                   exp_coeff_hi: float = 5.0,
                   exp_coeff_invariants: Optional[List[str]] = None,
                   no_recruit_families: Optional[List[str]] = None,
                   pair_diagonal_families: bool = False) -> Tuple[List[dict], List[dict], List[dict]]:
    # Families for which the recruitment integral is skipped. Under phi_el=90
    # and kappa_el=0, bypassing recruitment for 'el' makes I4el = lambda_z^2,
    # equivalent to the unrecruited raw axial strain.
    _no_recruit_set = set(no_recruit_families or [])
    print(f"\n=== animal: {animal} ===")
    print(f"  result folder: {folder}")
    print(f"  data folder:   {clean_data_root}")
    if _no_recruit_set:
        print(f"  no_recruit_families: {sorted(_no_recruit_set)}")
    if pair_diagonal_families:
        print("  pair_diagonal_families: tying f/s recruitment, kappa, angle, and coefficient basis")
    metrics_path = folder / "metrics.txt"
    hp = parse_metrics(metrics_path)
    pd_protocols, fl_protocols = parse_protocol_lists(metrics_path)

    params = load_animal_params(folder, hp)
    if pair_diagonal_families:
        enforce_paired_diagonal_families(params)
    if initial_guess_csv is not None:
        try:
            found = _apply_initial_guess_csv(params, animal,
                                             Path(initial_guess_csv))
        except FileNotFoundError:
            warnings.warn(f"initial_guess_csv not found: {initial_guess_csv}")
            found = False
        if found:
            print(f"  initial guess: applied from {initial_guess_csv}")
        else:
            print(f"  initial guess: {animal!r} not found in "
                  f"{initial_guess_csv} - using trained-folder values")
    if pair_diagonal_families:
        enforce_paired_diagonal_families(params)
    if force_dist_type and force_dist_type != "as_trained":
        _apply_force_dist_type(params, force_dist_type)
        print(f"  force_dist_type: {force_dist_type} "
              f"({', '.join(fam + ':' + params.recruit[fam]['dist_type'] for fam in FAMILY_ORDER)})")
    # Restrict the swept exp_coeff invariants to those that actually have an
    # exponential shape in the kept term set. This keeps the LHS dimension
    # count minimal (and readable) — no dead dimensions for invariants whose
    # shape is Linear/Quadratic/Cubic/xlog.
    active_exp_invs = None
    if sweep_exp_coeff:
        active_exp_invs = sorted({
            str(row["inv"]) for _, row in ranked.iterrows()
            if str(row.get("group", "")).lower() == "invariant"
            and str(row.get("term", "")).lower().startswith("exp")
        })
        if not active_exp_invs:
            print("  sweep_exp_coeff: on, but no exponential shape in the "
                  "kept set - all w2 stay locked at 1.0")
        else:
            print(f"  sweep_exp_coeff: on for {active_exp_invs} "
                  f"(range [{exp_coeff_lo}, {exp_coeff_hi}])")
        if exp_coeff_invariants is not None:
            # Intersect with user-specified list if given
            active_exp_invs = [i for i in active_exp_invs
                                if i in exp_coeff_invariants]

    param_candidates = make_lhs_param_candidates(
        params, n_samples=max(1, int(lhs_samples)), seed=int(lhs_seed),
        recruitment_frac=lhs_recruitment_frac, shape_frac=lhs_shape_frac,
        tau_frac=lhs_tau_frac, angle_deg=lhs_angle_deg, kappa_delta=lhs_kappa_delta,
        sweep_exp_coeff=sweep_exp_coeff and bool(active_exp_invs),
        exp_coeff_lo=exp_coeff_lo, exp_coeff_hi=exp_coeff_hi,
        exp_coeff_invariants=active_exp_invs,
    )
    # Reusable spec list for the optional scipy polish (same bounds as LHS).
    refine_specs = build_param_specs(
        params, recruitment_frac=lhs_recruitment_frac, shape_frac=lhs_shape_frac,
        tau_frac=lhs_tau_frac, angle_deg=lhs_angle_deg, kappa_delta=lhs_kappa_delta,
        sweep_exp_coeff=sweep_exp_coeff and bool(active_exp_invs),
        exp_coeff_lo=exp_coeff_lo, exp_coeff_hi=exp_coeff_hi,
        exp_coeff_invariants=active_exp_invs,
    )

    input_mode = str(hp.get("input_mode", "clean_csv")).lower()
    if input_mode == "excel":
        Fl, Pd = load_excel_data(clean_data_root)
        print(f"  loaded Excel sample file: {clean_data_root}")
    else:
        Fl, Pd = load_clean_csv_data(clean_data_root)
    Pd = _filter_protocols(Pd, pd_protocols)
    Fl = _filter_protocols(Fl, fl_protocols)

    lam_theta_pd = Pd["lambda_theta"].to_numpy(dtype=np.float64)
    lam_z_pd = Pd["lambda_z"].to_numpy(dtype=np.float64)
    sigma_theta_obs = Pd["sigma_theta"].to_numpy(dtype=np.float64)
    cid_pd = Pd["test_case"].astype(str).to_numpy()

    lam_theta_fl = Fl["lambda_theta"].to_numpy(dtype=np.float64)
    lam_z_fl = Fl["lambda_z"].to_numpy(dtype=np.float64)
    sigma_z_obs = Fl["sigma_z"].to_numpy(dtype=np.float64)
    cid_fl = Fl["test_case"].astype(str).to_numpy()

    n_pd = lam_theta_pd.size
    n_fl = lam_theta_fl.size
    # Training-time stress normalization: the network was trained against
    # y_true * stress_scale, so it outputs y / stress_scale. To match observed
    # kPa, design-column contributions must be multiplied by these factors.
    theta_max = float(np.max(np.abs(sigma_theta_obs))) if n_pd else 1.0
    z_max = float(np.max(np.abs(sigma_z_obs))) if n_fl else 1.0
    print(f"  n_PD: {n_pd}  n_FL: {n_fl}  circ_weight: {circ_weight}")
    print(f"  stress_scale (theta_max, z_max) = ({theta_max:.3f}, {z_max:.3f}) kPa")

    branch_pd = np.zeros(n_pd, dtype=np.float64)
    branch_fl = np.ones(n_fl, dtype=np.float64)

    kin_pd = kinematics(lam_theta_pd, lam_z_pd)
    kin_fl = kinematics(lam_theta_fl, lam_z_fl)
    # Recover training-time strain scales from the union of PD+FL training data
    E_th_train = np.concatenate([kin_pd["E_theta"], kin_fl["E_theta"]])
    E_z_train = np.concatenate([kin_pd["E_z"], kin_fl["E_z"]])
    recomputed_scales = compute_strain_basis_scales(E_th_train, E_z_train)

    coef_rows: List[dict] = []
    metric_rows: List[dict] = []
    pred_rows: List[dict] = []

    def _eval_candidate(eval_params: AnimalParams, selected: pd.DataFrame):
        if pair_diagonal_families:
            enforce_paired_diagonal_families(eval_params)
        rfi_pd = raw_fiber_invariants(lam_theta_pd, lam_z_pd, eval_params)
        rfi_fl = raw_fiber_invariants(lam_theta_fl, lam_z_fl, eval_params)
        recruited_pd: Dict[str, np.ndarray] = {}
        recruited_fl: Dict[str, np.ndarray] = {}
        for raw_key, fam, inv_key in (
            ("I4f_raw", "f", "I4f"), ("I4s_raw", "s", "I4s"),
            ("I4el_raw", "el", "I4el"), ("I4smc_raw", "smc", "I4smc"),
        ):
            if fam in _no_recruit_set:
                # Skip recruitment for this family (typically 'el' — elastin is
                # biologically always active, so recruitment kinetics don't apply).
                # Under phi_el=90 and kappa_el=0 this makes I4el = lambda_z^2.
                recruited_pd[inv_key] = rfi_pd[raw_key]
                recruited_fl[inv_key] = rfi_fl[raw_key]
            else:
                recruited_pd[inv_key] = recruited_invariant(
                    rfi_pd[raw_key], eval_params, fam,
                    lam_theta_pd, lam_z_pd, branch_pd, cid_pd,
                )
                recruited_fl[inv_key] = recruited_invariant(
                    rfi_fl[raw_key], eval_params, fam,
                    lam_theta_fl, lam_z_fl, branch_fl, cid_fl,
                )

        A_pd = np.zeros((n_pd, len(selected)), dtype=np.float64)
        A_fl = np.zeros((n_fl, len(selected)), dtype=np.float64)
        for j, row in selected.iterrows():
            # Look up the per-invariant Fung exponent (defaults to 1.0 for
            # strain terms and for any invariant the sweep isn't touching).
            w2_j = 1.0
            inv_j = str(row["inv"])
            if str(row["group"]).lower() == "invariant":
                w2_j = float(eval_params.exp_coeff.get(inv_j, 1.0))
            inv_for_stress = (
                "I4diag_pair"
                if pair_diagonal_families and inv_j in I4_DIAGONAL_PAIR
                else inv_j
            )
            col_pd, col_fl = stress_column_for_term(
                row["group"], inv_for_stress, row["term"], float(row["scale"]),
                kin_pd, rfi_pd, recruited_pd,
                kin_fl, rfi_fl, recruited_fl,
                derivative_mode=derivative_mode, fd_eps=fd_eps,
                w2=w2_j,
            )
            A_pd[:, j] = col_pd
            A_fl[:, j] = col_fl

        A_pd *= theta_max
        A_fl *= z_max
        A = np.vstack([A_pd, A_fl])
        y = np.concatenate([sigma_theta_obs, sigma_z_obs])
        sw = np.concatenate([np.full(n_pd, circ_weight), np.full(n_fl, 1.0)])

        if not np.all(np.isfinite(A)):
            bad = np.where(~np.isfinite(A))
            term_bad = selected.iloc[int(bad[1][0])]["term_key"] if bad[1].size else "?"
            raise FloatingPointError(
                f"animal={animal}: non-finite design entry at term {term_bad}"
            )

        cond = float(np.linalg.cond(A)) if min(A.shape) > 0 else float("nan")
        rank = int(np.linalg.matrix_rank(A)) if min(A.shape) > 0 else 0
        coef, col_norm, success, n_effective_params = \
            weighted_nnls_with_i4_diagonal_tie(
                A, y, sw, normalize_columns, selected,
                pair_diagonal_families=pair_diagonal_families,
            )
        y_pred = A @ coef
        yhat_pd = y_pred[:n_pd]
        yhat_fl = y_pred[n_pd:]
        fm = fit_metrics(sigma_theta_obs, yhat_pd, sigma_z_obs, yhat_fl,
                         np.full(n_pd, circ_weight), np.full(n_fl, 1.0),
                         n_params=n_effective_params)
        return {
            "A": A, "coef": coef, "col_norm": col_norm, "success": success,
            "yhat_pd": yhat_pd, "yhat_fl": yhat_fl, "fm": fm,
            "cond": cond, "rank": rank,
            "rfi_pd": rfi_pd, "rfi_fl": rfi_fl,
            "recruited_pd": recruited_pd, "recruited_fl": recruited_fl,
        }

    for k in ks:
        selected = select_top_k(ranked, k).copy()
        # Strain scale per term
        selected["scale"] = [get_strain_scale(r, recomputed_scales) for _, r in selected.iterrows()]
        selected["term_key"] = [
            f"{g} | {i} | {t}" for g, i, t in
            zip(selected["group"], selected["inv"], selected["term"])
        ]

        trial_rows = []
        best_trial = None
        best_key = float("inf")
        for cand_id, cand_kind, cand_params, cand_values in param_candidates:
            try:
                trial = _eval_candidate(cand_params, selected)
                fm = trial["fm"]
                objective_value = float(fm.get(lhs_objective, float("nan")))
                if lhs_objective in {"AIC", "BIC"} and not math.isfinite(objective_value):
                    objective_value = float("inf")
                if not math.isfinite(objective_value):
                    objective_value = float("inf")
                trial_rows.append({
                    "animal": animal, "k": k, "lhs_candidate": cand_id,
                    "lhs_kind": cand_kind, "lhs_objective": lhs_objective,
                    "lhs_objective_value": objective_value,
                    "success": bool(trial["success"]),
                    "condition_number": trial["cond"], "matrix_rank": trial["rank"],
                    **fm, **cand_values,
                })
                if objective_value < best_key:
                    best_key = objective_value
                    best_trial = (cand_id, cand_kind, cand_params, cand_values, trial)
            except Exception as e:
                trial_rows.append({
                    "animal": animal, "k": k, "lhs_candidate": cand_id,
                    "lhs_kind": cand_kind, "lhs_objective": lhs_objective,
                    "lhs_objective_value": float("inf"),
                    "success": False, "error": f"{type(e).__name__}: {e}",
                    **cand_values,
                })
        if best_trial is None:
            raise RuntimeError(f"animal={animal} k={k}: all LHS candidates failed")

        # -------- Optional scipy polish on top of the LHS winner ----------
        # Runs scipy.optimize on the top-K LHS candidates within the same
        # nonlinear-parameter box; keeps the polished result only if it
        # improves the objective. Rows recording each polish attempt are
        # appended to trial_rows so the audit trail stays complete.
        polish_records: List[dict] = []
        if refine_top_k and refine_top_k > 0:
            def _obj_of(cand_params: AnimalParams) -> float:
                trial = _eval_candidate(cand_params, selected)
                val = float(trial["fm"].get(lhs_objective, float("nan")))
                if lhs_objective in {"AIC", "BIC"} and not math.isfinite(val):
                    return float("inf")
                return val if math.isfinite(val) else float("inf")

            # Rank trials by objective ascending; scipy.minimize starts from
            # each of the top-K. Trials that already failed are skipped.
            ranked_trials = sorted(
                (t for t in trial_rows if math.isfinite(t.get("lhs_objective_value", float("inf")))),
                key=lambda t: t["lhs_objective_value"],
            )[: max(1, int(refine_top_k))]
            id_to_params = {
                cid: (kind, prm) for (cid, kind, prm, _vals) in param_candidates
            }

            best_polish_key = float(best_trial[4]["fm"].get(lhs_objective, float("inf")))
            best_polish_trial = best_trial
            best_polish_source: Optional[int] = None
            for tr in ranked_trials:
                seed_id = int(tr["lhs_candidate"])
                seed_kind, seed_params = id_to_params.get(seed_id, (None, None))
                if seed_params is None:
                    continue
                pol_params, pol_obj, pol_info = refine_params_local(
                    seed_params, refine_specs, _obj_of,
                    method=refine_method, maxiter=refine_maxiter,
                )
                polish_records.append({
                    "animal": animal, "k": k, "seed_lhs_candidate": seed_id,
                    "seed_lhs_kind": seed_kind,
                    "seed_objective_value": tr["lhs_objective_value"],
                    "polished_objective_value": pol_obj,
                    "improvement": tr["lhs_objective_value"] - pol_obj,
                    "method": pol_info["method"],
                    "n_iter": pol_info.get("n_iter", 0),
                    "n_fev": pol_info.get("n_fev", 0),
                    "message": pol_info.get("message", ""),
                })
                if pol_obj < best_polish_key:
                    # Re-evaluate to get the full fit dict for the polished params.
                    pol_trial = _eval_candidate(pol_params, selected)
                    best_polish_key = pol_obj
                    best_polish_trial = (
                        seed_id, "polished", pol_params, {}, pol_trial,
                    )
                    best_polish_source = seed_id
            if best_polish_source is not None:
                print(f"    polished (seed lhs={best_polish_source}): "
                      f"{lhs_objective}={best_polish_key:.4g} "
                      f"(was {float(best_trial[4]['fm'].get(lhs_objective, float('nan'))):.4g})")
                best_trial = best_polish_trial
                best_key = best_polish_key

        lhs_candidate, lhs_kind, best_params, lhs_values, fit = best_trial
        A = fit["A"]
        coef = fit["coef"]
        col_norm = fit["col_norm"]
        success = fit["success"]
        yhat_pd = fit["yhat_pd"]
        yhat_fl = fit["yhat_fl"]
        fm = fit["fm"]
        cond = fit["cond"]
        rank = fit["rank"]

        print(f"  k={k:<3}  cond={cond:.3e}  success={success}  "
              f"RMSE_t={fm['RMSE_theta']:.4g} RMSE_z={fm['RMSE_z']:.4g}  "
              f"R2_t={fm['R2_theta']:.4f} R2_z={fm['R2_z']:.4f}  "
              f"combined_RMSE={fm['combined_RMSE']:.4g}  AIC={fm['AIC']:.4g} BIC={fm['BIC']:.4g}  "
              f"lhs_candidate={lhs_candidate} ({lhs_objective}={best_key:.4g})")
        for _, r in selected.iterrows():
            print(f"      - [{r['group']}] {r['inv']}::{r['term']}  scale={r['scale']:.4g}")

        k_dir = out_root / animal / f"k{k}"
        k_dir.mkdir(parents=True, exist_ok=True)
        if trial_rows:
            pd.DataFrame(trial_rows).to_csv(k_dir / "lhs_candidate_metrics.csv", index=False)
        if polish_records:
            pd.DataFrame(polish_records).to_csv(
                k_dir / "polish_records.csv", index=False,
            )

        coef_df = pd.DataFrame({
            "animal": animal, "k": k,
            "lhs_candidate": lhs_candidate, "lhs_kind": lhs_kind,
            "selected_rank": selected["selected_rank"],
            "group": selected["group"], "inv": selected["inv"], "term": selected["term"],
            "coefficient": coef, "coefficient_nonnegative": True,
            "scale": selected["scale"], "term_key": selected["term_key"],
            "column_norm": col_norm,
        })
        coef_df.to_csv(k_dir / "pure_nnls_coefficients.csv", index=False)
        coef_rows.extend(coef_df.to_dict("records"))

        m_row = {
            "animal": animal, "k": k, "n_terms": int(len(selected)),
            "n_PD": int(n_pd), "n_FL": int(n_fl), "n_total": int(n_pd + n_fl),
            **fm,
            "condition_number": cond, "matrix_rank": rank, "success": bool(success),
            "lhs_samples": int(lhs_samples), "lhs_candidate": int(lhs_candidate),
            "lhs_kind": lhs_kind, "lhs_objective": lhs_objective,
            "lhs_objective_value": float(best_key),
        }
        pd.DataFrame([m_row]).to_csv(k_dir / "pure_nnls_metrics.csv", index=False)
        metric_rows.append(m_row)

        dms = pd.DataFrame({
            "animal": animal, "k": k,
            "group": selected["group"], "inv": selected["inv"], "term": selected["term"],
            "column_norm": col_norm,
            "column_min": A.min(axis=0) if A.size else np.array([]),
            "column_max": A.max(axis=0) if A.size else np.array([]),
            "column_mean": A.mean(axis=0) if A.size else np.array([]),
            "column_std": A.std(axis=0) if A.size else np.array([]),
            "column_mean_abs": np.mean(np.abs(A), axis=0) if A.size else np.array([]),
            "column_rms": np.sqrt(np.mean(A ** 2, axis=0)) if A.size else np.array([]),
            "column_peak_abs": np.max(np.abs(A), axis=0) if A.size else np.array([]),
        })
        dms.to_csv(k_dir / "design_matrix_summary.csv", index=False)

        preds = pd.concat([
            pd.DataFrame({
                "animal": animal, "k": k, "branch": "PD", "curve_id": cid_pd,
                "lambda_theta": lam_theta_pd, "lambda_z": lam_z_pd,
                "sigma_exp_kPa": sigma_theta_obs, "sigma_pred_kPa": yhat_pd,
                "residual_kPa": sigma_theta_obs - yhat_pd,
            }),
            pd.DataFrame({
                "animal": animal, "k": k, "branch": "FL", "curve_id": cid_fl,
                "lambda_theta": lam_theta_fl, "lambda_z": lam_z_fl,
                "sigma_exp_kPa": sigma_z_obs, "sigma_pred_kPa": yhat_fl,
                "residual_kPa": sigma_z_obs - yhat_fl,
            }),
        ], ignore_index=True)
        preds.to_csv(k_dir / "predictions.csv", index=False)
        pred_rows.extend(preds.to_dict("records"))

        import json as _json
        _state = {
            "animal": animal,
            "k": int(k),
            "lhs_candidate": int(lhs_candidate),
            "lhs_kind": lhs_kind,
            "no_recruit_families": sorted(_no_recruit_set),
            "pair_diagonal_families": bool(pair_diagonal_families),
            "selected_terms": [
                {"group": str(r["group"]), "inv": str(r["inv"]),
                 "term": str(r["term"]), "term_key": str(r["term_key"]),
                 "scale": float(r.get("scale", 1.0))}
                for _, r in selected.iterrows()
            ],
            "coefficients": [
                {"term_key": str(tk), "coefficient": float(c),
                 "col_norm": float(cn)}
                for tk, c, cn in zip(selected["term_key"], coef, col_norm)
            ],
            "animal_params": animal_params_to_dict(best_params),
            "stress_scale": {"theta_max": float(theta_max),
                             "z_max": float(z_max)},
        }
        (k_dir / "selected_nonlinear_parameters.json").write_text(
            _json.dumps(_state, indent=2, default=str)
        )

        # Equation text
        lines = [
            f"# Pure-NumPy NNLS refit for {animal}, k={k}",
            "# regression = weighted NNLS",
            "# coefficient_nonnegative = True",
            "# no_neural_network_used = True",
            "# no_tensorflow_used = True",
            f"# circ_weight = {circ_weight}",
            f"# derivative_mode = {derivative_mode}",
            f"# lhs_samples = {lhs_samples}",
            f"# lhs_selected_candidate = {lhs_candidate} ({lhs_kind})",
            f"# lhs_objective = {lhs_objective}",
            "# Recruitment, dispersion, fiber angles, and per-curve offsets are",
            "# fixed at the selected candidate values. Candidate 0 is the",
            "# animal-specific trained value parsed from learned_recruitment.csv,",
            "# learned_curve_offsets.csv, and trained_weights.txt.",
            "",
            f"W_{animal}_k{k}(lambda_theta, lambda_z) =",
        ]
        for i, (c, row) in enumerate(zip(coef, selected.itertuples())):
            sep = "      " if i == 0 else "    + "
            lines.append(f"{sep}{c:+.6e} * psi[{row.group}, {row.inv}, {row.term}]")
        lines.append("")
        lines.append("# Fixed recruitment parameters (per family):")
        for fam, p in best_params.recruit.items():
            lines.append(f"#   {fam}: lambda_lb={p['lambda_lb']:.6g} lambda_ub={p['lambda_ub']:.6g} "
                         f"alpha={p['alpha']:.6g} beta={p['beta']:.6g} tau={p['tau']:.6g} "
                         f"dist={p['dist_type']} shift={p['shift_mode']}")
        lines.append(f"# Fixed dispersion: kappa_f={best_params.kappa_f:.6g}  kappa_s={best_params.kappa_s:.6g}")
        lines.append(f"# Fixed angles (deg): phi_f={math.degrees(best_params.phi_f):.3f}  "
                     f"phi_s={math.degrees(best_params.phi_s):.3f}  "
                     f"phi_el={math.degrees(best_params.phi_el):.3f}  "
                     f"phi_smc={math.degrees(best_params.phi_smc):.3f}")
        lines.append("")
        lines.append("# Metrics:")
        for k_m, v_m in fm.items():
            lines.append(f"#   {k_m} = {v_m:.6g}")
        lines.append(f"#   condition_number = {cond:.6g}")
        lines.append(f"#   matrix_rank      = {rank}")
        (k_dir / "final_equation_pure_nnls.txt").write_text("\n".join(lines))

        if make_plots:
            _save_plots(k_dir, sigma_theta_obs, yhat_pd, sigma_z_obs, yhat_fl, k)
            _save_stress_stretch_plot(
                k_dir, animal, k,
                lam_theta_pd=lam_theta_pd, sigma_theta_obs=sigma_theta_obs,
                sigma_theta_pred=yhat_pd, cid_pd=cid_pd,
                lam_z_fl=lam_z_fl, sigma_z_obs=sigma_z_obs,
                sigma_z_pred=yhat_fl, cid_fl=cid_fl,
                r2_theta=fm.get("R2_theta"), r2_z=fm.get("R2_z"),
            )

    if validate_original_terms:
        try:
            rfi_pd = raw_fiber_invariants(lam_theta_pd, lam_z_pd, params)
            rfi_fl = raw_fiber_invariants(lam_theta_fl, lam_z_fl, params)
            recruited_pd: Dict[str, np.ndarray] = {}
            recruited_fl: Dict[str, np.ndarray] = {}
            for raw_key, fam, inv_key in (
                ("I4f_raw", "f", "I4f"), ("I4s_raw", "s", "I4s"),
                ("I4el_raw", "el", "I4el"), ("I4smc_raw", "smc", "I4smc"),
            ):
                if fam in _no_recruit_set:
                    recruited_pd[inv_key] = rfi_pd[raw_key]
                    recruited_fl[inv_key] = rfi_fl[raw_key]
                else:
                    recruited_pd[inv_key] = recruited_invariant(
                        rfi_pd[raw_key], params, fam, lam_theta_pd, lam_z_pd, branch_pd, cid_pd,
                    )
                    recruited_fl[inv_key] = recruited_invariant(
                        rfi_fl[raw_key], params, fam, lam_theta_fl, lam_z_fl, branch_fl, cid_fl,
                    )
            _validate_original(folder, animal, params, kin_pd, rfi_pd, recruited_pd,
                               kin_fl, rfi_fl, recruited_fl, recomputed_scales,
                               sigma_theta_obs, sigma_z_obs, out_root / animal,
                               theta_max=theta_max, z_max=z_max)
        except Exception as e:
            warnings.warn(f"validate_original_terms failed for {animal}: {e}")

    return coef_rows, metric_rows, pred_rows


def _validate_original(folder: Path, animal: str, params: AnimalParams,
                       kin_pd, rfi_pd, recruited_pd, kin_fl, rfi_fl, recruited_fl,
                       recomputed_scales, sigma_theta_obs, sigma_z_obs, out_dir: Path,
                       theta_max: float = 1.0, z_max: float = 1.0):
    """Evaluate the animal's original active terms using c_j = w_sub * w_ann
    (i.e., eff_norm), apply the training-time stress_scale, and compare against
    the measured stresses. Residual norm should be on the order of the
    original RMSE if the pure-NumPy chain is consistent with the trained model."""
    at_path = folder / "active_terms.csv"
    if not at_path.exists():
        warnings.warn(f"{at_path} not found; skipping validate_original_terms.")
        return
    at = pd.read_csv(at_path)
    # The correct coefficient is w_sub * w_ann (== eff_norm). active_terms.csv's
    # 'importance' column is eff_norm * mean|raw_basis|/scale for strain terms,
    # which is a ranking quantity, not the SEDF coefficient.
    if not {"w_sub", "w_ann"}.issubset(at.columns):
        warnings.warn("active_terms.csv missing w_sub/w_ann columns; skipping check.")
        return
    sigma_t = np.zeros(kin_pd["I1"].shape, dtype=np.float64)
    sigma_z = np.zeros(kin_fl["I1"].shape, dtype=np.float64)
    for _, row in at.iterrows():
        coef = float(row["w_sub"]) * float(row["w_ann"])
        if abs(coef) < 1e-12:
            continue
        scale = 1.0
        if row["group"] == "strain":
            scale = recomputed_scales.get(row["term"], 1.0)
        col_pd, col_fl = stress_column_for_term(
            row["group"], row["inv"], row["term"], scale,
            kin_pd, rfi_pd, recruited_pd, kin_fl, rfi_fl, recruited_fl,
            derivative_mode="analytic", fd_eps=1e-5,
        )
        sigma_t += coef * col_pd
        sigma_z += coef * col_fl
    sigma_t *= theta_max
    sigma_z *= z_max

    df = pd.DataFrame({
        "branch": ["PD"] * sigma_t.size + ["FL"] * sigma_z.size,
        "sigma_obs_kPa": np.concatenate([sigma_theta_obs, sigma_z_obs]),
        "sigma_recon_kPa": np.concatenate([sigma_t, sigma_z]),
        "residual_kPa": np.concatenate([sigma_theta_obs - sigma_t, sigma_z_obs - sigma_z]),
    })
    df.to_csv(out_dir / "original_term_reconstruction_check.csv", index=False)


def _save_stress_stretch_plot(k_dir: Path, animal: str, k: int,
                              lam_theta_pd: np.ndarray, sigma_theta_obs: np.ndarray,
                              sigma_theta_pred: np.ndarray, cid_pd: np.ndarray,
                              lam_z_fl: np.ndarray, sigma_z_obs: np.ndarray,
                              sigma_z_pred: np.ndarray, cid_fl: np.ndarray,
                              r2_theta: Optional[float] = None,
                              r2_z: Optional[float] = None,
                              label_fontsize: int = 24,
                              tick_fontsize: int = 18,
                              title_fontsize: int = 20,
                              data_marker_size: float = 4.5,
                              model_linewidth: float = 2.5,
                              n_ticks: int = 3) -> None:
    """Two-panel publication figure in the same style as the CANN training
    code's save_publication_figure: black filled dots for measurements, red
    dashed line for the NNLS fit, serif mathtext labels, 3 ticks per axis."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except Exception as e:
        warnings.warn(f"matplotlib unavailable: {e}"); return

    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['font.family'] = 'serif'

    fig, (ax_pd, ax_fl) = plt.subplots(1, 2, figsize=(12, 5))

    def _plot_branch(ax, lam, obs, pred, cids, missing_label):
        if lam.size == 0:
            ax.text(0.5, 0.5, f"no {missing_label} data",
                    transform=ax.transAxes, ha='center', va='center')
            return
        for u in np.unique(cids):
            mask = cids == u
            if not mask.any():
                continue
            order = np.argsort(lam[mask])
            lam_s = lam[mask][order]
            obs_s = obs[mask][order]
            pred_s = pred[mask][order]
            ax.plot(lam_s, obs_s,
                    linestyle='None', marker='o',
                    markersize=data_marker_size,
                    markerfacecolor='black', markeredgecolor='black',
                    alpha=0.85)
            ax.plot(lam_s, pred_s,
                    linestyle='--', color='#d6224d',
                    linewidth=model_linewidth)

    _plot_branch(ax_pd, lam_theta_pd, sigma_theta_obs, sigma_theta_pred, cid_pd, "theta")
    _plot_branch(ax_fl, lam_z_fl, sigma_z_obs, sigma_z_pred, cid_fl, "z")

    ax_pd.set_xlabel(r'$\lambda_{\theta}$', fontsize=label_fontsize)
    ax_pd.set_ylabel(r'$\sigma_{\theta}$ [kPa]', fontsize=label_fontsize)
    ax_fl.set_xlabel(r'$\lambda_{z}$', fontsize=label_fontsize)
    ax_fl.set_ylabel(r'$\sigma_{z}$ [kPa]', fontsize=label_fontsize)

    for ax in (ax_pd, ax_fl):
        ax.xaxis.set_major_locator(MaxNLocator(int(n_ticks)))
        ax.yaxis.set_major_locator(MaxNLocator(int(n_ticks)))
        ax.tick_params(axis='both', labelsize=tick_fontsize, width=1.2, length=5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    title_parts = []
    if animal:
        title_parts.append(f"{animal} (k={k})")
    r2_pieces = []
    if r2_theta is not None and np.isfinite(r2_theta):
        r2_pieces.append(rf"$R^{{2}}_{{\theta}}={r2_theta:.3f}$")
    if r2_z is not None and np.isfinite(r2_z):
        r2_pieces.append(rf"$R^{{2}}_{{z}}={r2_z:.3f}$")
    if r2_pieces:
        title_parts.append("   ".join(r2_pieces))
    if title_parts:
        fig.suptitle("   ".join(title_parts), fontsize=title_fontsize, y=0.98)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(k_dir / "sigma_vs_lambda.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(k_dir / "sigma_vs_lambda.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def _save_plots(k_dir: Path, y_pd, yhat_pd, y_fl, yhat_fl, k: int):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        warnings.warn(f"matplotlib unavailable, skipping plots: {e}")
        return
    if y_pd.size:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(y_pd, yhat_pd, s=8)
        lim = [0.0, float(max(np.max(y_pd), np.max(yhat_pd), 1e-9))]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel("sigma_theta observed (kPa)"); ax.set_ylabel("sigma_theta predicted (kPa)")
        ax.set_title(f"k={k}")
        fig.tight_layout(); fig.savefig(k_dir / "observed_vs_predicted_theta.pdf"); plt.close(fig)
    if y_fl.size:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(y_fl, yhat_fl, s=8)
        lim = [0.0, float(max(np.max(y_fl), np.max(yhat_fl), 1e-9))]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel("sigma_z observed (kPa)"); ax.set_ylabel("sigma_z predicted (kPa)")
        ax.set_title(f"k={k}")
        fig.tight_layout(); fig.savefig(k_dir / "observed_vs_predicted_z.pdf"); plt.close(fig)


# ---------------------------------------------------------------------------
# Master summaries
# ---------------------------------------------------------------------------
def write_master_summaries(out_root: Path, all_metrics: List[dict], all_coefs: List[dict],
                           all_fixed: List[dict]) -> None:
    m_df = pd.DataFrame(all_metrics)
    c_df = pd.DataFrame(all_coefs)
    if not c_df.empty:
        c_df["coefficient_abs"] = c_df["coefficient"].abs()
        c_df.to_csv(out_root / "pure_nnls_coefficients_all_animals.csv", index=False)
        if not m_df.empty:
            sel = (c_df.groupby(["animal", "k"])["term_key"]
                   .apply(lambda s: "; ".join(s)).reset_index()
                   .rename(columns={"term_key": "selected_terms"}))
            m_df = m_df.merge(sel, on=["animal", "k"], how="left")

    cols = ["animal", "k", "n_terms", "selected_terms",
            "RMSE_theta", "RMSE_z", "R2_theta", "R2_z",
            "combined_RMSE", "weighted_RMSE", "AIC", "BIC",
            "condition_number", "matrix_rank", "success",
            "lhs_samples", "lhs_candidate", "lhs_kind",
            "lhs_objective", "lhs_objective_value"]
    for c in cols:
        if c not in m_df.columns:
            m_df[c] = pd.NA
    if not m_df.empty:
        m_df[cols].to_csv(out_root / "pure_nnls_summary_all_animals.csv", index=False)

        # Model selection per animal
        sel_rows = []
        for animal, sub in m_df.groupby("animal"):
            sub = sub.dropna(subset=["combined_RMSE"])
            if sub.empty:
                continue
            best_rmse = sub.loc[sub["combined_RMSE"].idxmin()]
            best_aic = sub.loc[sub["AIC"].idxmin()] if "AIC" in sub else best_rmse
            best_bic = sub.loc[sub["BIC"].idxmin()] if "BIC" in sub else best_rmse
            thr = float(best_rmse["combined_RMSE"]) * 1.05
            within = sub[sub["combined_RMSE"] <= thr]
            sel_rows.append({
                "animal": animal,
                "best_k_by_combined_RMSE": int(best_rmse["k"]),
                "best_k_by_AIC": int(best_aic["k"]),
                "best_k_by_BIC": int(best_bic["k"]),
                "smallest_k_within_5_percent_of_best_RMSE": int(within["k"].min()) if not within.empty else int(best_rmse["k"]),
            })
        pd.DataFrame(sel_rows).to_csv(out_root / "pure_nnls_model_selection.csv", index=False)

        agg = m_df.groupby("k").agg(
            n_animals=("animal", "nunique"),
            mean_RMSE_theta=("RMSE_theta", "mean"),
            median_RMSE_theta=("RMSE_theta", "median"),
            mean_RMSE_z=("RMSE_z", "mean"),
            median_RMSE_z=("RMSE_z", "median"),
            mean_combined_RMSE=("combined_RMSE", "mean"),
            median_combined_RMSE=("combined_RMSE", "median"),
            mean_R2_theta=("R2_theta", "mean"),
            median_R2_theta=("R2_theta", "median"),
            mean_R2_z=("R2_z", "mean"),
            median_R2_z=("R2_z", "median"),
            mean_R2_weighted=("R2_weighted", "mean"),
            median_R2_weighted=("R2_weighted", "median"),
            mean_R2_overall=("R2_overall", "mean"),
            median_R2_overall=("R2_overall", "median"),
            mean_AIC=("AIC", "mean"),
            median_AIC=("AIC", "median"),
            mean_BIC=("BIC", "mean"),
            median_BIC=("BIC", "median"),
        ).reset_index()
        agg.to_csv(out_root / "pure_nnls_summary_by_k.csv", index=False)

    if all_fixed:
        pd.DataFrame(all_fixed).to_csv(
            out_root / "fixed_recruitment_dispersion_by_animal.csv", index=False,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-animal pure-NumPy weighted NNLS refit of selected SEDF coefficients.",
    )
    p.add_argument("--animal_root", type=Path, required=True)
    p.add_argument("--data_root_parent", type=Path, default=None)
    p.add_argument("--ranked_terms", type=Path, required=True)
    p.add_argument("--topk_terms", type=Path, default=None)
    p.add_argument("--out_root", type=Path, required=True)
    p.add_argument("--ks", type=int, nargs="+", default=[5, 8, 10, 12, 15])
    p.add_argument("--circ_weight", type=float, default=4.0)
    p.add_argument("--normalize_columns", type=lambda v: str(v).lower() not in {"0", "false", "no", "off"},
                   default=True)
    p.add_argument("--derivative_mode", choices=["analytic", "finite_difference"],
                   default="finite_difference")
    p.add_argument("--fd_eps", type=float, default=1e-5)
    p.add_argument("--animal_data_map", type=Path, default=None)
    p.add_argument("--exclude_path_substrings", type=str, nargs="*", default=None,
                   help="Skip discovered animal folders whose path contains any "
                        "of these substrings. Useful for excluding nested "
                        "summary roots such as _sobol_roots.")
    p.add_argument("--validate_original_terms", action="store_true")
    p.add_argument("--make_plots", action="store_true")
    p.add_argument("--lhs_samples", type=int, default=1,
                   help="Total parameter candidates to test per animal/k. Candidate 0 is the trained baseline.")
    p.add_argument("--lhs_seed", type=int, default=12345)
    p.add_argument("--lhs_recruitment_frac", type=float, default=0.20,
                   help="Relative half-width for lambda_lb/lambda_ub sampling around trained values.")
    p.add_argument("--lhs_shape_frac", type=float, default=0.50,
                   help="Relative half-width for beta/lognormal shape parameters.")
    p.add_argument("--lhs_tau_frac", type=float, default=0.50,
                   help="Relative half-width for recruitment smoothness tau.")
    p.add_argument("--lhs_angle_deg", type=float, default=15.0,
                   help="Absolute half-width in degrees for fiber-angle sampling.")
    p.add_argument("--lhs_kappa_delta", type=float, default=0.05,
                   help="Absolute half-width for kappa_f/kappa_s sampling, clipped to [0, 1/3].")
    p.add_argument("--lhs_objective", choices=["weighted_RMSE", "combined_RMSE", "AIC", "BIC"],
                   default="weighted_RMSE")
    p.add_argument("--refine_top_k", type=int, default=0,
                   help="If > 0, run scipy.optimize.minimize on each of the "
                        "top-K LHS candidates (ranked by --lhs_objective) to "
                        "polish the nonlinear parameters. Same bounds as LHS. "
                        "0 disables the polish stage.")
    p.add_argument("--refine_method", type=str, default="L-BFGS-B",
                   choices=["L-BFGS-B", "Nelder-Mead", "Powell", "TNC"],
                   help="scipy.optimize.minimize method used by --refine_top_k.")
    p.add_argument("--refine_maxiter", type=int, default=100,
                   help="Max iterations per scipy polish call.")
    p.add_argument("--initial_guess_csv", type=Path, default=None,
                   help="Optional path to cann_initial_guess.csv (long format, "
                        "one row per (animal, family)). When set, per-family "
                        "recruitment values and per-animal kappa / fiber "
                        "angles are read from the CSV instead of the trained "
                        "CANN folder. Per-curve offsets and data metadata are "
                        "still read from the animal folder. Generate the CSV "
                        "with scripts/extract_cann_initial_guess.py.")
    p.add_argument("--force_dist_type", type=str, default="as_trained",
                   choices=["as_trained", "all_beta", "all_lognormal"],
                   help="Override the per-family recruitment distribution. "
                        "'as_trained' (default) uses whatever the CANN chose "
                        "per family. 'all_beta' / 'all_lognormal' forces every "
                        "fiber family to that distribution; missing shape "
                        "params fall back to safe defaults (alpha=2,beta=4 for "
                        "beta; logn_mu=0,logn_sigma=0.3 for lognormal) and the "
                        "LHS then perturbs them.")
    # -- Fung exponential extension --------------------------------------
    p.add_argument("--sweep_exp_coeff",
                   action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Sweep the per-invariant Fung exponent w2 as an extra "
                        "LHS dimension for every invariant that has an "
                        "exponential shape in the kept term set. Default on. "
                        "Use --no-sweep_exp_coeff to lock every w2 at 1.0 "
                        "(recovering the sibling script's behaviour).")
    p.add_argument("--exp_coeff_lo", type=float, default=0.3,
                   help="Lower bound on w2 during LHS sampling.")
    p.add_argument("--exp_coeff_hi", type=float, default=5.0,
                   help="Upper bound on w2 during LHS sampling.")
    p.add_argument("--exp_coeff_invariants", type=str, nargs="*", default=None,
                   help="Optional whitelist of invariants to sweep w2 on "
                        "(e.g. --exp_coeff_invariants I4el I4smc). Default is "
                        "any invariant that has an exp shape in the kept set.")
    p.add_argument("--no_recruit_families", type=str, nargs="*", default=None,
                   help="Family keys ('f','s','el','smc') whose recruitment "
                        "integral is skipped (raw invariant used directly). "
                        "Typical: --no_recruit_families el  (elastin is "
                        "biologically always active; under phi_el=90, "
                        "kappa_el=0 this makes I4el = lambda_z^2).")
    p.add_argument("--pair_diagonal_families", action="store_true",
                   help="Tie the two diagonal families as a +/- phi pair and "
                        "use a combined I4diag_pair basis column for their "
                        "shared NNLS coefficient.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out_root = Path(args.out_root).expanduser().resolve()
    args.out_root.mkdir(parents=True, exist_ok=True)

    ranked = load_ranked_terms(args.ranked_terms)
    animal_data_map = load_animal_data_map(args.animal_data_map)

    folders = discover_animal_folders(
        args.animal_root,
        exclude_path_substrings=args.exclude_path_substrings,
    )
    if not folders:
        raise SystemExit(f"No valid animal folders under {args.animal_root}")
    print(f"Discovered {len(folders)} animal folder(s).")

    all_metrics: List[dict] = []
    all_coefs: List[dict] = []
    all_fixed: List[dict] = []

    for folder in folders:
        hp = parse_metrics(folder / "metrics.txt")
        animal = animal_name_from_metrics(folder, hp)
        # circ_weight precedence: CLI wins if user changed it; otherwise use
        # (a) the initial-guess CSV if provided and it has this animal, or
        # (b) the source metrics.txt value.
        cw = args.circ_weight
        if cw == 4.0:  # default sentinel - user did not override on CLI
            cw_from_csv = None
            if args.initial_guess_csv is not None:
                try:
                    cw_from_csv = _initial_guess_circ_weight(
                        animal, Path(args.initial_guess_csv))
                except FileNotFoundError:
                    pass
            if cw_from_csv is not None:
                cw = cw_from_csv
            else:
                mw = hp.get("circ_weight")
                if isinstance(mw, (int, float)):
                    cw = float(mw)
        input_mode = str(hp.get("input_mode", "clean_csv")).lower()
        if input_mode == "excel":
            data_root = resolve_excel_sample_file(folder, hp, args.data_root_parent)
            if data_root is None:
                warnings.warn(
                    f"Animal '{animal}' uses input_mode=excel but the sample_file "
                    f"{hp.get('sample_file')!r} could not be located. "
                    f"Pass --data_root_parent or --animal_data_map to override."
                )
                continue
        else:
            try:
                data_root = resolve_clean_data_root(
                    folder, hp, animal,
                    data_root_parent=args.data_root_parent,
                    animal_data_map=animal_data_map,
                )
            except FileNotFoundError as e:
                warnings.warn(str(e)); continue
        try:
            coefs, mets, _preds = process_animal(
                folder=folder, animal=animal, ranked=ranked, ks=sorted(set(args.ks)),
                clean_data_root=data_root, out_root=args.out_root,
                circ_weight=cw, normalize_columns=args.normalize_columns,
                derivative_mode=args.derivative_mode, fd_eps=args.fd_eps,
                make_plots=args.make_plots,
                validate_original_terms=args.validate_original_terms,
                lhs_samples=args.lhs_samples, lhs_seed=args.lhs_seed,
                lhs_recruitment_frac=args.lhs_recruitment_frac,
                lhs_shape_frac=args.lhs_shape_frac,
                lhs_tau_frac=args.lhs_tau_frac,
                lhs_angle_deg=args.lhs_angle_deg,
                lhs_kappa_delta=args.lhs_kappa_delta,
                lhs_objective=args.lhs_objective,
                refine_top_k=args.refine_top_k,
                refine_method=args.refine_method,
                refine_maxiter=args.refine_maxiter,
                force_dist_type=args.force_dist_type,
                initial_guess_csv=args.initial_guess_csv,
                sweep_exp_coeff=bool(args.sweep_exp_coeff),
                exp_coeff_lo=float(args.exp_coeff_lo),
                exp_coeff_hi=float(args.exp_coeff_hi),
                exp_coeff_invariants=args.exp_coeff_invariants,
                no_recruit_families=args.no_recruit_families,
                pair_diagonal_families=bool(args.pair_diagonal_families),
            )
        except Exception as e:
            warnings.warn(f"Animal {animal} failed: {type(e).__name__}: {e}")
            continue
        all_coefs.extend(coefs)
        all_metrics.extend(mets)

        # Per-animal fixed parameters row
        params = load_animal_params(folder, hp)
        for fam in FAMILY_ORDER:
            r = params.recruit[fam]
            all_fixed.append({
                "animal": animal, "family": fam,
                "lambda_lb": r["lambda_lb"], "lambda_ub": r["lambda_ub"],
                "alpha": r["alpha"], "beta": r["beta"], "tau": r["tau"],
                "dist_type": r["dist_type"], "shift_mode": r["shift_mode"],
                "ref_lambda_theta": r["ref_lambda_theta"], "ref_lambda_z": r["ref_lambda_z"],
                "kappa_f": params.kappa_f, "kappa_s": params.kappa_s,
                "phi_f_deg": math.degrees(params.phi_f),
                "phi_s_deg": math.degrees(params.phi_s),
                "phi_el_deg": math.degrees(params.phi_el),
                "phi_smc_deg": math.degrees(params.phi_smc),
            })

    write_master_summaries(args.out_root, all_metrics, all_coefs, all_fixed)
    print(f"\nDone. Outputs under {args.out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

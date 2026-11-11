# VeinCANN: hyperelastic constitutive model discovery for venous tissue.
#
# Input: an Excel workbook with FL (force-length) and PD (pressure-diameter)
# sheets containing experimental measurements and unloaded specimen geometry.
# Stresses and mid-wall stretches are calculated before fitting.
#
# Model: one shared strain-energy function combining invariant-based candidates
# and Green-strain polynomial terms. Fiber families include trainable helical
# directions, dispersion, and optional recruitment with per-curve offsets.
# The I8 cross-fiber subnet omits logarithmic terms because I8 can be negative.
#
# Training supports stress or reconstructed-load objectives, robust residuals,
# curve and tail weighting, learning-rate schedules, independent restarts,
# curriculum training, pruning/refitting, and checkpoint-based fine-tuning.
#
# Outputs include the fitted energy equation, model weights, fit metrics,
# recruitment parameters, diagnostic tables, and stress-stretch figures.
# See README.md in this folder for input requirements and command-line options.

#%% Load Python Packages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tensorflow as tf
# Use only the available CPU devices for all TensorFlow operations.
tf.config.set_visible_devices(tf.config.list_physical_devices("CPU"))
import tensorflow.keras.backend as K
import tensorflow.keras as keras
import pickle
import json
import pandas as pd
import os
import copy
import argparse
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from colorsys import rgb_to_hls
import matplotlib.colors as mcolors

# Check Versions
print('Numpy: ' + np.__version__)
print('Tensorflow: ' + tf.__version__)

# %% Excel sample data (override with --sample_file or SAMPLE_FILE).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_FILE = os.environ.get(
    "SAMPLE_FILE",
    os.path.join(
        SCRIPT_DIR, "..", "1. Sample data and pre-processing", "sample data",
        "JV_VG_106_19Jun20_unloading_data.xlsx",
    ),
)
DEFAULT_SEED = int(os.environ.get("FIT_SEED", "45"))

def _geometry_column_as_mm(df, col):
    values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError(f"obj6 geometry column {col!r} has no finite values.")
    value = float(np.nanmedian(values))
    # Excel inputs store OD/H in microns and L in mm.
    if value > 20.0:
        value *= 1e-3
    return value

def infer_obj6_reference_geometry(Pd, Fl):
    missing = [col for col in ("OD", "H", "L") if col not in Pd.columns and col not in Fl.columns]
    if missing:
        raise ValueError(
            "obj6 requires unloaded OD/H/L geometry to reconstruct pressure and force. "
            f"Missing Excel geometry columns: {missing}. Include OD, H, and L."
        )

    source = Pd if all(col in Pd.columns for col in ("OD", "H", "L")) else Fl
    od_mm = _geometry_column_as_mm(source, "OD")
    h_mm = _geometry_column_as_mm(source, "H")
    l_mm = _geometry_column_as_mm(source, "L")

    if not (od_mm > 0 and h_mm > 0 and l_mm > 0 and od_mm > 2.0 * h_mm):
        raise ValueError(
            "Invalid obj6 geometry: expected OD > 2*H and positive OD/H/L, got "
            f"OD={od_mm:.6g} mm, H={h_mm:.6g} mm, L={l_mm:.6g} mm."
        )

    ir_mm = 0.5 * (od_mm - 2.0 * h_mm)
    r_mid_mm = ir_mm + 0.5 * h_mm
    geom = {
        "OD_mm": od_mm,
        "H_mm": h_mm,
        "L_mm": l_mm,
        "IR_mm": ir_mm,
        "R_mid_mm": r_mid_mm,
    }
    print("\nobj6 reference geometry:")
    for key, value in geom.items():
        print(f"  {key}: {value:.6f}")
    return geom

def obj6_loaded_geometry(X, geom):
    X = np.asarray(X, dtype=np.float32)
    if X.size == 0:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty, empty

    lambda_theta = X[:, 0].astype(np.float64)
    lambda_z = X[:, 1].astype(np.float64)
    h = geom["H_mm"] / (lambda_theta * lambda_z)
    r_mid = geom["R_mid_mm"] * lambda_theta
    ir = r_mid - 0.5 * h
    outer = r_mid + 0.5 * h

    if np.any(~np.isfinite(ir)) or np.any(~np.isfinite(outer)) or np.any(ir <= 0.0):
        raise ValueError("Invalid obj6 reconstructed loaded geometry: non-finite or non-positive inner radius.")

    area = np.pi * (outer**2 - ir**2)
    return (
        h.astype(np.float32),
        ir.astype(np.float32),
        outer.astype(np.float32),
        area.astype(np.float32),
    )

def make_obj6_targets(X_pd, y_theta, X_fl, y_z, geom):
    h_pd, ir_pd, _, _ = obj6_loaded_geometry(X_pd, geom)
    _, _, _, area_fl = obj6_loaded_geometry(X_fl, geom)

    pressure_factor_pd = h_pd / ir_pd if len(h_pd) else np.array([], dtype=np.float32)
    force_factor_fl = area_fl if len(area_fl) else np.array([], dtype=np.float32)

    pi_exp = y_theta.astype(np.float32) * pressure_factor_pd
    fv_exp = y_z.astype(np.float32) * force_factor_fl

    pi_scale = float(np.mean(pi_exp)) if len(pi_exp) else 1.0
    fv_scale = float(np.mean(fv_exp)) if len(fv_exp) else 1.0
    pi_scale = max(abs(pi_scale), 1e-8)
    fv_scale = max(abs(fv_scale), 1e-8)

    return {
        "pi_exp": pi_exp.astype(np.float32),
        "pressure_factor_pd": pressure_factor_pd.astype(np.float32),
        "pi_scale": pi_scale,
        "fv_exp": fv_exp.astype(np.float32),
        "force_factor_fl": force_factor_fl.astype(np.float32),
        "fv_scale": fv_scale,
    }

def pack_obj6_y(target, factor, scale):
    if len(target) == 0:
        return np.empty((0, 3), dtype=np.float32)
    return np.column_stack([
        target.astype(np.float32),
        factor.astype(np.float32),
        np.full_like(target, scale, dtype=np.float32),
    ]).astype(np.float32)

def obj6_loss(stress_scale):
    stress_scale = tf.constant(float(stress_scale), dtype=tf.float32)

    def loss(y_true, y_pred):
        target_load = y_true[:, 0:1]
        stress_to_load = y_true[:, 1:2]
        load_scale = tf.maximum(tf.abs(y_true[:, 2:3]), tf.constant(1e-8, dtype=tf.float32))
        pred_load = y_pred * stress_scale * stress_to_load
        return tf.square((pred_load - target_load) / load_scale)

    return loss

def stress_hybrid_log_loss(stress_scale, log_stress_scale=1.0, log_loss_weight=0.5):
    stress_scale = tf.constant(float(stress_scale), dtype=tf.float32)
    log_stress_scale = tf.constant(max(float(log_stress_scale), 1e-8), dtype=tf.float32)
    log_loss_weight = tf.constant(float(np.clip(log_loss_weight, 0.0, 1.0)), dtype=tf.float32)

    def loss(y_true, y_pred):
        mse_part = tf.square(y_pred - y_true)
        true_stress = y_true * stress_scale
        pred_stress = y_pred * stress_scale
        log_true = tf.math.log1p(tf.nn.relu(true_stress) / log_stress_scale)
        log_pred = tf.math.log1p(tf.nn.relu(pred_stress) / log_stress_scale)
        log_part = tf.square(log_pred - log_true)
        return (1.0 - log_loss_weight) * mse_part + log_loss_weight * log_part

    return loss

def make_tail_sample_weights(X, y, base_weight=1.0, stress_scale=1.0,
                             tail_weight=0.0, stress_frac=0.0,
                             stretch_threshold=0.0, stretch_index=0,
                             width=0.05):
    if len(y) == 0:
        return np.array([], dtype=np.float32)

    weights = base_weight * np.ones_like(y, dtype=np.float32)
    if tail_weight <= 0.0:
        return weights

    width = max(float(width), 1e-6)
    trigger = np.zeros_like(y, dtype=np.float32)

    if stress_frac and stress_frac > 0.0 and stress_scale > 0.0:
        stress_level = np.abs(y.astype(np.float32)) / float(stress_scale)
        trigger = np.maximum(
            trigger,
            1.0 / (1.0 + np.exp(-(stress_level - float(stress_frac)) / width)),
        )

    if stretch_threshold and stretch_threshold > 0.0 and len(X) > 0:
        stretch_level = X[:, int(stretch_index)].astype(np.float32)
        trigger = np.maximum(
            trigger,
            1.0 / (1.0 + np.exp(-(stretch_level - float(stretch_threshold)) / width)),
        )

    return (weights * (1.0 + float(tail_weight) * trigger)).astype(np.float32)


# =============================================================================
# v4 helpers: recruitment presets, robust loss, equal-curve weights, tail gain,
# lognormal PDF over slack stretch, curve-id encoding, post-training diagnostics
# =============================================================================

RECRUIT_PRESETS = {
    "early":  dict(init_threshold=1.05, recruit_lb_min=0.90,
                   recruit_lb_max=1.35, recruit_ub_max=1.60),
    "medium": dict(init_threshold=1.20, recruit_lb_min=0.90,
                   recruit_lb_max=1.60, recruit_ub_max=1.90),
    "late":   dict(init_threshold=1.40, recruit_lb_min=1.00,
                   recruit_lb_max=1.90, recruit_ub_max=2.30),
}


def apply_recruit_preset(args):
    """If --recruit_preset is one of the named presets, overwrite the corresponding
    recruitment CLI args with the preset values *unless* the user set them
    explicitly. We detect 'explicitly set' by comparing against the v3 defaults
    that lived on the parser. With ArgumentDefaultsHelpFormatter we can't easily
    tell user-set vs default-set, so the convention here is: the preset wins
    unless preset == 'custom'."""
    preset = (getattr(args, "recruit_preset", "custom") or "custom").lower()
    if preset == "custom" or preset not in RECRUIT_PRESETS:
        return args
    p = RECRUIT_PRESETS[preset]
    args.init_threshold = float(p["init_threshold"])
    args.recruit_lb_min = float(p["recruit_lb_min"])
    args.recruit_lb_max = float(p["recruit_lb_max"])
    args.recruit_ub_max = float(p["recruit_ub_max"])
    return args


def robust_residual(residual, kind="mse", delta=0.5):
    """Per-sample robust loss term. Returns a tensor with the same leading shape
    as residual; caller is responsible for sample-weighted reduction."""
    kind = (kind or "mse").lower()
    if kind == "huber":
        a = tf.abs(residual)
        d = tf.constant(float(delta), dtype=tf.float32)
        return tf.where(a < d, 0.5 * a * a, d * (a - 0.5 * d))
    if kind == "charbonnier":
        eps = tf.constant(1e-3, dtype=tf.float32)
        return tf.sqrt(residual * residual + eps * eps) - eps
    return tf.square(residual)


def make_robust_stress_loss(stress_scale, robust_kind="mse", robust_delta=0.5):
    """Direct stress objective wrapped with optional Huber/Charbonnier residual."""
    stress_scale = tf.constant(float(stress_scale), dtype=tf.float32)
    def loss(y_true, y_pred):
        # y_true and y_pred live in normalized stress space.
        return robust_residual(y_pred - y_true, kind=robust_kind, delta=robust_delta)
    return loss


def make_robust_hybrid_log_loss(stress_scale, log_stress_scale=1.0,
                                log_loss_weight=0.5,
                                robust_kind="mse", robust_delta=0.5):
    """Hybrid normalized-stress + log-stress residuals, with optional robust kernel
    on the linear part. The log part stays as squared residual because log1p
    already compresses tails."""
    stress_scale = tf.constant(float(stress_scale), dtype=tf.float32)
    log_scale = tf.constant(max(float(log_stress_scale), 1e-8), dtype=tf.float32)
    w = tf.constant(float(np.clip(log_loss_weight, 0.0, 1.0)), dtype=tf.float32)
    def loss(y_true, y_pred):
        mse_part = robust_residual(y_pred - y_true, kind=robust_kind, delta=robust_delta)
        true_stress = y_true * stress_scale
        pred_stress = y_pred * stress_scale
        log_true = tf.math.log1p(tf.nn.relu(true_stress) / log_scale)
        log_pred = tf.math.log1p(tf.nn.relu(pred_stress) / log_scale)
        return (1.0 - w) * mse_part + w * tf.square(log_pred - log_true)
    return loss


def make_robust_obj6_loss(stress_scale, robust_kind="mse", robust_delta=0.5):
    """Obj6 reconstructed-load loss with optional robust kernel."""
    stress_scale = tf.constant(float(stress_scale), dtype=tf.float32)
    def loss(y_true, y_pred):
        target_load = y_true[:, 0:1]
        stress_to_load = y_true[:, 1:2]
        load_scale = tf.maximum(tf.abs(y_true[:, 2:3]),
                                tf.constant(1e-8, dtype=tf.float32))
        pred_load = y_pred * stress_scale * stress_to_load
        residual = (pred_load - target_load) / load_scale
        return robust_residual(residual, kind=robust_kind, delta=robust_delta)
    return loss


def equal_curve_sample_weights(curve_ids, base_weight=1.0):
    """Return per-sample weights so every distinct curve_id sums to base_weight,
    matching Base_Poly_new_updated.build_dataset_with_weights logic. The mean
    of returned weights equals base_weight, which keeps loss magnitudes
    interpretable across runs."""
    curve_ids = np.asarray(curve_ids)
    if curve_ids.size == 0:
        return np.array([], dtype=np.float32)
    weights = np.zeros(curve_ids.shape[0], dtype=np.float64)
    uniq = np.unique(curve_ids)
    for u in uniq:
        mask = (curve_ids == u)
        n = int(mask.sum())
        if n > 0:
            weights[mask] = 1.0 / n
    # Rescale so mean == base_weight.
    mean = float(weights.mean()) if weights.size else 0.0
    if mean > 0:
        weights = weights * (float(base_weight) / mean)
    return weights.astype(np.float32)


def make_polynomial_tail_weights(X, base_weight=1.0, tail_gain=0.0,
                                 tail_power=2.0, tail_stretch=0.0,
                                 stretch_index=0):
    """Polynomial tail weighting (separate from the v3 sigmoid-tail weighting).
    For circumferential PD: stretch_index=0 picks lambda_theta. For axial FL:
    stretch_index=1 picks lambda_z. The weight is:
        w_i = base_weight * (1 + tail_gain * max(lambda - tail_stretch, 0)^tail_power).
    """
    if len(X) == 0:
        return np.array([], dtype=np.float32)
    weights = base_weight * np.ones(X.shape[0], dtype=np.float64)
    if tail_gain <= 0.0 or tail_stretch <= 0.0:
        return weights.astype(np.float32)
    lam = X[:, int(stretch_index)].astype(np.float64)
    excess = np.maximum(lam - float(tail_stretch), 0.0)
    weights = weights * (1.0 + float(tail_gain) * np.power(excess, float(tail_power)))
    return weights.astype(np.float32)


def encode_curve_ids(curve_id_strings, vocabulary=None):
    """Encode string curve ids to int32. Returns (ids_int32, vocabulary list).
    Unknown ids resolve to 0 (the first id in the vocabulary)."""
    if vocabulary is None:
        vocabulary = sorted({str(c) for c in curve_id_strings})
    table = {name: i for i, name in enumerate(vocabulary)}
    ids = np.array([table.get(str(c), 0) for c in curve_id_strings], dtype=np.int32)
    return ids, vocabulary


def shape_diagnostics_per_curve(X, y_true, y_pred, curve_ids, stretch_index):
    """Returns a list of per-curve dicts with negative-slope-count, min slope,
    roughness (sum of squared second differences), and total negative descent.
    X may be the full (N,2) array; we sort by X[:, stretch_index] within each
    curve before computing finite differences."""
    rows = []
    if len(y_pred) == 0:
        return rows
    X = np.asarray(X)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    curve_ids = np.asarray(curve_ids)
    uniq = np.unique(curve_ids)
    for u in uniq:
        mask = (curve_ids == u)
        if mask.sum() < 3:
            continue
        lam = X[mask, int(stretch_index)]
        order = np.argsort(lam)
        lam = lam[order]
        yp = y_pred[mask][order]
        dy = np.diff(yp)
        dlam = np.diff(lam)
        slopes = np.where(np.abs(dlam) > 1e-12, dy / np.maximum(np.abs(dlam), 1e-12), 0.0)
        neg_count = int(np.sum(slopes < 0))
        min_slope = float(slopes.min()) if slopes.size else 0.0
        neg_descent = float(np.sum(np.minimum(slopes, 0.0) * np.abs(dlam[:slopes.size])))
        d2 = np.diff(slopes) if slopes.size > 1 else np.array([0.0])
        roughness = float(np.mean(d2 * d2))
        rows.append(dict(
            curve_id=str(u),
            n=int(mask.sum()),
            neg_slope_count=neg_count,
            min_slope=min_slope,
            total_neg_descent=neg_descent,
            roughness=roughness,
        ))
    return rows


def count_effective_parameters(model_nn, Psi_model, importance,
                               equation_thresh=1e-3,
                               per_curve_offsets_on=False,
                               n_curves=1):
    """Effective k for AIC/BIC. Counts:
      - surviving candidate terms (importance > equation_thresh; pruned terms
        do not count, which matches the L1 + nonneg implicit sparsity);
      - recruitment params per family: lambda_lb_raw, lambda_ub_raw, beta_a_raw,
        beta_b_raw, tau_raw. For dist_type='lognormal' add logn_mu_raw and
        logn_sig_raw. Times 4 families;
      - dispersion: kappa_f, kappa_s, plus 1 (symmetric) or 2 (independent)
        fiber angle params when train_fiber_angles is True;
      - per-curve recruitment offsets: n_curves * 4 families when enabled.
    Returns a dict with the breakdown for the metrics report.
    """
    k_terms = sum(1 for r in importance if r['importance'] > equation_thresh)

    rec_layer = next((l for l in model_nn.layers
                      if isinstance(l, FiberRecruitmentLayer)), None)
    if rec_layer is None:
        k_recruit = 0
        dist_type = "n/a"
    else:
        per_family = 5  # lb, ub, alpha, beta, tau
        if rec_layer.dist_type == 'lognormal':
            per_family += 2  # mu, sigma
        k_recruit = per_family * 4
        dist_type = rec_layer.dist_type

    h_layer = next((l for l in model_nn.layers
                    if isinstance(l, H_Layer_kappas)), None)
    if h_layer is None:
        k_disp = 0
    else:
        k_disp = 2  # kappa_f, kappa_s
        if h_layer.train_fiber_angles:
            k_disp += 1 if h_layer.helical_mode == "symmetric" else 2

    k_offsets = int(n_curves) * 4 if per_curve_offsets_on else 0
    k_total = int(k_terms + k_recruit + k_disp + k_offsets)
    return dict(
        k_terms=int(k_terms),
        k_recruit=int(k_recruit),
        k_disp=int(k_disp),
        k_offsets=int(k_offsets),
        k_total=k_total,
        dist_type=dist_type,
    )


def aic_bic_from_rss(rss, n, k):
    """Standard nonlinear-regression AIC/BIC under Gaussian residuals.
        AIC = n * ln(RSS / n) + 2 * k
        BIC = n * ln(RSS / n) + k * ln(n)
    Returns (AIC, BIC). Both nan when n or rss are invalid."""
    if not np.isfinite(rss) or n <= 0 or rss <= 0:
        return float('nan'), float('nan')
    mse = float(rss) / float(n)
    aic = float(n) * float(np.log(mse)) + 2.0 * float(k)
    bic = float(n) * float(np.log(mse)) + float(k) * float(np.log(n))
    return float(aic), float(bic)


def compute_aic_bic_v4(model_nn, Psi_model, importance, args, curve_vocabulary,
                       X_train_pd, y_train_theta, cid_train_pd,
                       X_train_fl, y_train_z, cid_train_fl):
    """AIC/BIC for the joint fit. Predictions go through the full v4 model
    (including per-curve offsets and branch shift) so the residuals match what
    the plots show. Returns a dict with per-branch (PD, FL) and pooled values
    plus the parameter-count breakdown."""
    rss_pd, n_pd = 0.0, 0
    rss_fl, n_fl = 0.0, 0
    if X_train_pd.size:
        bf = np.zeros((X_train_pd.shape[0], 1), dtype=np.float32)
        if cid_train_pd is not None:
            cid_int, _ = encode_curve_ids(cid_train_pd, curve_vocabulary)
        else:
            cid_int = np.zeros((X_train_pd.shape[0],), dtype=np.int32)
        preds, _ = model_nn.predict(
            [X_train_pd[:, 0:1], X_train_pd[:, 1:2], bf, cid_int.reshape(-1, 1)],
            verbose=0,
        )
        preds = preds.reshape(-1) * float(model_nn.theta_max)
        resid_pd = preds.astype(np.float64) - y_train_theta.astype(np.float64)
        rss_pd = float(np.sum(resid_pd * resid_pd))
        n_pd = int(y_train_theta.shape[0])
    if X_train_fl.size:
        bf = np.ones((X_train_fl.shape[0], 1), dtype=np.float32)
        if cid_train_fl is not None:
            cid_int, _ = encode_curve_ids(cid_train_fl, curve_vocabulary)
        else:
            cid_int = np.zeros((X_train_fl.shape[0],), dtype=np.int32)
        _, preds = model_nn.predict(
            [X_train_fl[:, 0:1], X_train_fl[:, 1:2], bf, cid_int.reshape(-1, 1)],
            verbose=0,
        )
        preds = preds.reshape(-1) * float(model_nn.z_max)
        resid_fl = preds.astype(np.float64) - y_train_z.astype(np.float64)
        rss_fl = float(np.sum(resid_fl * resid_fl))
        n_fl = int(y_train_z.shape[0])

    pcounts = count_effective_parameters(
        model_nn, Psi_model, importance,
        equation_thresh=getattr(args, 'equation_thresh', 1e-3),
        per_curve_offsets_on=(getattr(args, 'per_curve_recruit', 'off') == 'on'),
        n_curves=max(1, len(curve_vocabulary)),
    )
    k = pcounts['k_total']

    aic_pd, bic_pd = aic_bic_from_rss(rss_pd, n_pd, k)
    aic_fl, bic_fl = aic_bic_from_rss(rss_fl, n_fl, k)
    aic_all, bic_all = aic_bic_from_rss(rss_pd + rss_fl, n_pd + n_fl, k)

    return {
        'AIC_PD': aic_pd, 'BIC_PD': bic_pd, 'n_PD': n_pd, 'RSS_PD': rss_pd,
        'AIC_FL': aic_fl, 'BIC_FL': bic_fl, 'n_FL': n_fl, 'RSS_FL': rss_fl,
        'AIC_pooled': aic_all, 'BIC_pooled': bic_all,
        'n_total': n_pd + n_fl,
        'k_total': k,
        'k_terms_active': pcounts['k_terms'],
        'k_recruit': pcounts['k_recruit'],
        'k_disp': pcounts['k_disp'],
        'k_offsets': pcounts['k_offsets'],
        'aic_dist_type': pcounts['dist_type'],
    }


# %% Load and convert single sample data from Excel file
def processInput(file_path):
    """
    Process input data similar to artery code.
    Calculates geometric dimensions and forces. Ensures consistent units.
    """
    print("\n" + "="*60)
    print("LOADING AND PROCESSING DATA")
    print("="*60)
    
    # Load both sheets
    Fl = pd.read_excel(file_path, sheet_name='FL')
    Pd = pd.read_excel(file_path, sheet_name='PD')
    
    print(f"Loaded FL sheet: {len(Fl)} rows")
    print(f"Loaded PD sheet: {len(Pd)} rows")
    
    # Get metadata (H, L, OD are in columns)
    # Check units and handle NaN
    H = Fl['H'].iloc[0]  # Wall thickness (assume microns)
    L = Fl['L'].iloc[0]  # Unloaded length (mm)
    OD = Fl['OD'].iloc[0]  # Outer diameter (microns)
    PHI = Fl['PHI'].iloc[0] if 'PHI' in Fl.columns else 0.0  # Opening angle (degrees)
    
    if np.isnan(PHI):
        print(f"  WARNING: Opening angle (PHI) is NaN. Using default value of 0 degrees.")
        PHI = 0.0
    
    print(f"\nMetadata:")
    print(f"  H (thickness): {H} microns")
    print(f"  L (length): {L} mm")
    print(f"  OD (outer diameter): {OD} microns")
    print(f"  PHI (opening angle): {PHI} degrees")
    
    # Unloaded Inner Diameter (mm) - converting from microns
    Fl["ID"] = 1e-3 * (Fl["OD"] - 2 * Fl["H"])
    Pd["ID"] = 1e-3 * (Pd["OD"] - 2 * Pd["H"])
    
    # Unloaded Volume (mm^3)
    Fl["ref_vol"] = (np.pi / 4) * ((1e-3 * Fl["OD"])**2 - Fl["ID"]**2) * Fl["L"]
    Pd["ref_vol"] = (np.pi / 4) * ((1e-3 * Pd["OD"])**2 - Pd["ID"]**2) * Pd["L"]
    
    # Unloaded Outer and Inner Radii (mm)
    Fl["OR"] = 1e-3 * Fl["OD"] / 2
    Fl["IR"] = Fl["ID"] / 2
    Pd["OR"] = 1e-3 * Pd["OD"] / 2
    Pd["IR"] = Pd["ID"] / 2
    
    # Loaded Outer Diameter (mm)
    Fl["od"] = 1e-3 * Fl["Outer_Diameter_microm"]
    Pd["od"] = 1e-3 * Pd["Outer_Diameter_microm"]
    
    # DEBUG
    under_sqrt_fl = Fl["od"]**2 - ((4 * Fl["ref_vol"]) / (np.pi * Fl["Loaded_Length_mm"]))
    under_sqrt_pd = Pd["od"]**2 - ((4 * Pd["ref_vol"]) / (np.pi * Pd["Loaded_Length_mm"]))

    print(f"FL sqrt argument range: [{under_sqrt_fl.min():.6f}, {under_sqrt_fl.max():.6f}]")
    print(f"PD sqrt argument range: [{under_sqrt_pd.min():.6f}, {under_sqrt_pd.max():.6f}]")

    # Loaded Inner Diameter (mm), using constant volume assumption
    Fl["id"] = np.sqrt(Fl["od"]**2 - ((4 * Fl["ref_vol"]) / (np.pi * Fl["Loaded_Length_mm"])))
    Pd["id"] = np.sqrt(Pd["od"]**2 - ((4 * Pd["ref_vol"]) / (np.pi * Pd["Loaded_Length_mm"])))
    
    # Loaded Outer and Inner Radii (mm)
    Fl["or"] = Fl["od"] / 2
    Fl["ir"] = Fl["id"] / 2
    Pd["or"] = Pd["od"] / 2
    Pd["ir"] = Pd["id"] / 2
    
    # Loaded Thickness (mm)
    Fl["h"] = Fl["or"] - Fl["ir"]
    Pd["h"] = Pd["or"] - Pd["ir"]
    
    # Change units for loads
    mmHg_to_kPa = 0.133322
    grams_to_mN = 9.80665
    
    # Pressure
    Fl["Pressure_kPa"] = mmHg_to_kPa * Fl["Pressure_mmHg"]
    Pd["Pressure_kPa"] = mmHg_to_kPa * Pd["Pressure_mmHg"]
    
    # Force measured by transducer
    Fl["Force_mN"] = grams_to_mN * Fl["Force_g"]
    Pd["Force_mN"] = grams_to_mN * Pd["Force_g"]
    
    # CRITICAL: Force acting on the vessel (includes pressure contribution)
    Fl["Force_v_mN"] = Fl["Force_mN"] + np.pi * (Fl["ir"]**2) * Fl["Pressure_kPa"]
    Pd["Force_v_mN"] = Pd["Force_mN"] + np.pi * (Pd["ir"]**2) * Pd["Pressure_kPa"]
    
    print(f"\nGeometry validation:")
    print(f"  FL - IR range: [{Fl['IR'].min():.4f}, {Fl['IR'].max():.4f}] mm")
    print(f"  PD - IR range: [{Pd['IR'].min():.4f}, {Pd['IR'].max():.4f}] mm")
    print(f"  FL - ir range: [{Fl['ir'].min():.4f}, {Fl['ir'].max():.4f}] mm")
    print(f"  PD - ir range: [{Pd['ir'].min():.4f}, {Pd['ir'].max():.4f}] mm")
    
    return Fl, Pd

def getStressStretch(Pd, Fl):
    """
    Calculates stress and stretch for all dataframes and adds these columns to the dataframes.
    """
    # Circumferential stress (thin-wall approximation)
    Fl["sigma_theta"] = (Fl["Pressure_kPa"] * Fl["ir"]) / Fl["h"]
    Pd["sigma_theta"] = (Pd["Pressure_kPa"] * Pd["ir"]) / Pd["h"]
    
    # Circumferential stretch at mid-wall: Ferruzzi et al. (2013), Eq. (4).
    Fl["lambda_theta"] = (Fl["ir"] + Fl["h"] / 2) / (Fl["IR"] + 1e-3 * Fl["H"] / 2)
    Pd["lambda_theta"] = (Pd["ir"] + Pd["h"] / 2) / (Pd["IR"] + 1e-3 * Pd["H"] / 2)
    
    # Axial stress (includes pressure contribution through Force_v_mN)
    Fl["sigma_z"] = Fl["Force_v_mN"] / (np.pi * Fl["h"] * (2 * Fl["ir"] + Fl["h"]))
    Pd["sigma_z"] = Pd["Force_v_mN"] / (np.pi * Pd["h"] * (2 * Pd["ir"] + Pd["h"]))
    
    # Axial stretch
    Fl["lambda_z"] = Fl["Loaded_Length_mm"] / Fl["L"]
    Pd["lambda_z"] = Pd["Loaded_Length_mm"] / Pd["L"]
    
    print("\n" + "="*60)
    print("STRESS AND STRETCH RANGES:")
    print("="*60)
    print(f"FL data:")
    print(f"  lambda_theta: [{Fl['lambda_theta'].min():.3f}, {Fl['lambda_theta'].max():.3f}]")
    print(f"  lambda_z: [{Fl['lambda_z'].min():.3f}, {Fl['lambda_z'].max():.3f}]")
    print(f"  sigma_z: [{Fl['sigma_z'].min():.3f}, {Fl['sigma_z'].max():.3f}] kPa")
    print(f"PD data:")
    print(f"  lambda_theta: [{Pd['lambda_theta'].min():.3f}, {Pd['lambda_theta'].max():.3f}]")
    print(f"  lambda_z: [{Pd['lambda_z'].min():.3f}, {Pd['lambda_z'].max():.3f}]")
    print(f"  sigma_theta: [{Pd['sigma_theta'].min():.3f}, {Pd['sigma_theta'].max():.3f}] kPa")
    print("="*60 + "\n")

def traindata(mode, Fl, Pd):
    """
    Returns data in CANN style: each row has both stretches [lambda_theta, lambda_z],
    with separate outputs for each stress component.
    
    Returns
    -------
    X_pd : np.array (N_pd, 2) - [lambda_theta, lambda_z] during Pd tests
    y_theta : np.array (N_pd,) - measured sigma_theta
    X_fl : np.array (N_fl, 2) - [lambda_theta, lambda_z] during Fl tests  
    y_z : np.array (N_fl,) - measured sigma_z
    """
    if mode == 'all':
        # Pd data: have lambda_theta and sigma_theta
        lambda_theta_pd = Pd["lambda_theta"].values
        lambda_z_pd = Pd["lambda_z"].values  # actual axial stretch during Pd test
        sigma_theta_pd = Pd["sigma_theta"].values
        
        # Fl data: have lambda_z and sigma_z
        lambda_theta_fl = Fl["lambda_theta"].values  # actual circumferential stretch during Fl test
        lambda_z_fl = Fl["lambda_z"].values
        sigma_z_fl = Fl["sigma_z"].values
        
        # Stack into (N, 2) format
        X_pd = np.column_stack([lambda_theta_pd, lambda_z_pd])
        X_fl = np.column_stack([lambda_theta_fl, lambda_z_fl])
        
        return X_pd, sigma_theta_pd, X_fl, sigma_z_fl
    else:
        raise ValueError(f"Only 'all' mode supported, got {mode}")

# %% Split data into train/validation sets
def split_train_val(Fl, Pd,
                    pd_train=['invivo'],
                    fl_train=['p60', 'p100']):
    """
    Split protocols into training and validation sets based on test case names.
    """
    # Get all unique test cases
    pd_all_cases = Pd['test_case'].unique()
    fl_all_cases = Fl['test_case'].unique()
    
    # Determine validation lists
    pd_val = [t for t in pd_all_cases if t not in pd_train]
    fl_val = [t for t in fl_all_cases if t not in fl_train]
    
    def collect(df, keys, use_theta=True):
        """Collect data from specified test cases. Returns (X, y, curve_id_strings)."""
        if len(keys) == 0:
            return (np.empty((0,2), dtype=np.float32),
                    np.empty((0,), dtype=np.float32),
                    np.empty((0,), dtype=object))

        mask = df['test_case'].isin(keys)
        if not mask.any():
            return (np.empty((0,2), dtype=np.float32),
                    np.empty((0,), dtype=np.float32),
                    np.empty((0,), dtype=object))

        data = df[mask]
        X = np.column_stack([data['lambda_theta'].values, data['lambda_z'].values])
        if use_theta:
            y = data['sigma_theta'].values
        else:
            y = data['sigma_z'].values
        cids = data['test_case'].astype(str).values
        return X.astype(np.float32), y.astype(np.float32), cids

    X_train_pd, y_train_theta, cid_train_pd = collect(Pd, pd_train, use_theta=True)
    X_val_pd,   y_val_theta,   cid_val_pd   = collect(Pd, pd_val,   use_theta=True)
    X_train_fl, y_train_z,     cid_train_fl = collect(Fl, fl_train, use_theta=False)
    X_val_fl,   y_val_z,       cid_val_fl   = collect(Fl, fl_val,   use_theta=False)
    
    print("\n" + "="*60)
    print("DATA SPLIT INFORMATION:")
    print("="*60)
    print(f"PD Training protocols: {pd_train} ({len(y_train_theta)} points)")
    print(f"PD Validation protocols: {list(pd_val)} ({len(y_val_theta)} points)")
    print(f"FL Training protocols: {fl_train} ({len(y_train_z)} points)")
    print(f"FL Validation protocols: {list(fl_val)} ({len(y_val_z)} points)")
    print("="*60 + "\n")
    
    return (X_train_pd, y_train_theta, X_train_fl, y_train_z,
            X_val_pd, y_val_theta, X_val_fl, y_val_z,
            cid_train_pd, cid_val_pd, cid_train_fl, cid_val_fl)

# %% Model architecture components
initializer_exp = tf.keras.initializers.RandomUniform(minval=0., maxval=0.1, seed=DEFAULT_SEED)
initializer_1 = 'glorot_normal'

def set_random_seed(seed):
    """Set repeatable seeds for the fit and rebuild seeded initializers."""
    global initializer_exp
    if seed is None:
        return
    np.random.seed(seed)
    tf.random.set_seed(seed)
    initializer_exp = tf.keras.initializers.RandomUniform(minval=0., maxval=0.1, seed=seed)

def regularize(reg, pen):
    if reg == 'L2':
        return keras.regularizers.l2(pen)
    if reg == 'L1':
        return keras.regularizers.l1(pen)

def activation_Exp(x):
    return 1.0*(tf.math.exp(x)-1.0)

def activation_ln(x):
    return -1.0*tf.math.log(1.0 - (x))

def SingleInvNet(I1_ref, idi, reg, pen, nonneg=True):
    constraint = keras.constraints.NonNeg() if nonneg else None
    I1_sq = keras.layers.Lambda(lambda x: tf.math.square(x), name='inv'+str(idi)+'_square')(I1_ref)
    I1_cu = keras.layers.Lambda(lambda x: tf.math.pow(x, 3), name='inv'+str(idi)+'_cubic')(I1_ref)
    linear1 = keras.layers.Dense(1,kernel_initializer=initializer_1,kernel_constraint=constraint,
                                 kernel_regularizer=regularize(reg, pen),
                                 use_bias=False, activation=None,name='w'+str(idi)+'_1')(I1_ref)
    exp_linear1 = keras.layers.Dense(1,kernel_initializer=initializer_exp,kernel_constraint=constraint,
                                 kernel_regularizer=regularize(reg, pen),
                                 use_bias=False, activation=activation_Exp,name='w'+str(idi)+'_2')(I1_ref)
    quad1 = keras.layers.Dense(1,kernel_initializer=initializer_1,kernel_constraint=constraint,
                                 kernel_regularizer=regularize(reg, pen),
                                 use_bias=False, activation=None,name='w'+str(idi)+'_3')(I1_sq)
    exp_quad1 = keras.layers.Dense(1,kernel_initializer=initializer_exp,kernel_constraint=constraint,
                                 kernel_regularizer=regularize(reg, pen),
                                 use_bias=False, activation=activation_Exp,name='w'+str(idi)+'_4')(I1_sq)
    xlog_linear1 = keras.layers.Dense(1, kernel_initializer=initializer_exp, kernel_constraint=constraint,
                                   kernel_regularizer=regularize(reg, pen),
                                   use_bias=False, activation=lambda x: x * tf.math.log1p(x),
                                   name='w'+str(idi)+'_5')(I1_ref)
    xlog_quad1   = keras.layers.Dense(1, kernel_initializer=initializer_exp, kernel_constraint=constraint,
                                    kernel_regularizer=regularize(reg, pen),
                                    use_bias=False, activation=lambda x: x * tf.math.log1p(x),
                                    name='w'+str(idi)+'_6')(I1_sq)
    cubic1 = keras.layers.Dense(1,kernel_initializer=initializer_1,kernel_constraint=constraint,
                                kernel_regularizer=regularize(reg, pen),
                                use_bias=False, activation=None,name='w'+str(idi)+'_7')(I1_cu)
    collect = [linear1, exp_linear1, quad1, exp_quad1, xlog_linear1, xlog_quad1, cubic1]
    collect_out = tf.keras.layers.concatenate(collect, axis=1)
    return collect_out

def I8Net(I8_in, idi, reg, pen, nonneg=True):
    """5-term network for I8: linear, exp-linear, quadratic, exp-quadratic, cubic (polynomial).
    No xlog or exp-cubic terms — I8 is signed so xlog (which clamps to x>=0) would discard
    physically meaningful negative values.
    When nonneg=True, I8 kernels use the same NonNeg constraint as the other invariant
    subnetworks."""
    constraint = keras.constraints.NonNeg() if nonneg else None
    I8_sq = keras.layers.Lambda(lambda x: tf.math.square(x), name='i8_square')(I8_in)
    I8_cu = keras.layers.Lambda(lambda x: tf.math.pow(x, 3), name='i8_cubic')(I8_in)
    linear1 = keras.layers.Dense(1, kernel_initializer=initializer_1,
                                 kernel_constraint=constraint,
                                 kernel_regularizer=regularize(reg, pen),
                                 use_bias=False, activation=None,
                                 name='w'+str(idi)+'_1')(I8_in)
    exp_lin1 = keras.layers.Dense(1, kernel_initializer=initializer_exp,
                                  kernel_constraint=constraint,
                                  kernel_regularizer=regularize(reg, pen),
                                  use_bias=False, activation=activation_Exp,
                                  name='w'+str(idi)+'_2')(I8_in)
    quad1 = keras.layers.Dense(1, kernel_initializer=initializer_1,
                               kernel_constraint=constraint,
                               kernel_regularizer=regularize(reg, pen),
                               use_bias=False, activation=None,
                               name='w'+str(idi)+'_3')(I8_sq)
    exp_quad1 = keras.layers.Dense(1, kernel_initializer=initializer_exp,
                                   kernel_constraint=constraint,
                                   kernel_regularizer=regularize(reg, pen),
                                   use_bias=False, activation=activation_Exp,
                                   name='w'+str(idi)+'_4')(I8_sq)
    cubic1 = keras.layers.Dense(1, kernel_initializer=initializer_1,
                                kernel_constraint=constraint,
                                kernel_regularizer=regularize(reg, pen),
                                use_bias=False, activation=None,
                                name='w'+str(idi)+'_5')(I8_cu)
    collect = [linear1, exp_lin1, quad1, exp_quad1, cubic1]
    return tf.keras.layers.concatenate(collect, axis=1)

STRAIN_TERM_NAMES_10 = [
    "E_theta",
    "E_z",
    "E_theta^2",
    "E_z^2",
    "E_theta*E_z",
    "E_theta^3",
    "E_z^3",
    "E_theta^4",
    "E_z^4",
    "E_theta^2*E_z",
    "E_theta*E_z^2",
    "E_theta^2*E_z^2",
]

STRAIN_TERM_DISPLAY_10 = [
    "E_θ",
    "E_z",
    "E_θ²",
    "E_z²",
    "E_θ·E_z",
    "E_θ³",
    "E_z³",
    "E_θ⁴",
    "E_z⁴",
    "E_θ²·E_z",
    "E_θ·E_z²",
    "E_θ²·E_z²",
]


def strain_basis_tensors(E_theta, E_z, n_terms=5, include_linear=True, basis_scales=None):
    basis = []
    names = []
    display = []

    if include_linear:
        basis.extend([E_theta, E_z])
        names.extend(STRAIN_TERM_NAMES_10[:2])
        display.extend(STRAIN_TERM_DISPLAY_10[:2])

    poly_basis = [
        E_theta**2,
        E_z**2,
        E_theta * E_z,
        E_theta**3,
        E_z**3,
    ]
    poly_names = STRAIN_TERM_NAMES_10[2:7]
    poly_display = STRAIN_TERM_DISPLAY_10[2:7]

    if n_terms >= 7:
        poly_basis.extend([E_theta**4, E_z**4])
        poly_names.extend(STRAIN_TERM_NAMES_10[7:9])
        poly_display.extend(STRAIN_TERM_DISPLAY_10[7:9])

    if n_terms >= 10:
        poly_basis.extend([
            E_theta**2 * E_z,
            E_theta * E_z**2,
            E_theta**2 * E_z**2,
        ])
        poly_names.extend(STRAIN_TERM_NAMES_10[9:12])
        poly_display.extend(STRAIN_TERM_DISPLAY_10[9:12])

    basis.extend(poly_basis[:n_terms])
    names.extend(poly_names[:n_terms])
    display.extend(poly_display[:n_terms])
    if basis_scales is not None:
        if len(basis_scales) != len(basis):
            raise ValueError(f"Expected {len(basis)} strain basis scales, got {len(basis_scales)}.")
        basis = [term / float(scale) for term, scale in zip(basis, basis_scales)]
    return basis, names, display


def strain_basis_values(lambda_pairs, n_terms=5, include_linear=True):
    lambda_pairs = np.asarray(lambda_pairs, dtype=np.float64)
    if lambda_pairs.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    E_theta = 0.5 * (np.square(lambda_pairs[:, 0:1]) - 1.0)
    E_z = 0.5 * (np.square(lambda_pairs[:, 1:2]) - 1.0)

    basis = []
    if include_linear:
        basis.extend([E_theta, E_z])
    poly_basis = [
        E_theta**2,
        E_z**2,
        E_theta * E_z,
        E_theta**3,
        E_z**3,
    ]
    if n_terms >= 7:
        poly_basis.extend([E_theta**4, E_z**4])
    if n_terms >= 10:
        poly_basis.extend([
            E_theta**2 * E_z,
            E_theta * E_z**2,
            E_theta**2 * E_z**2,
        ])
    basis.extend(poly_basis[:n_terms])
    return np.concatenate(basis, axis=1) if basis else np.empty((len(lambda_pairs), 0), dtype=np.float64)


def compute_strain_basis_scales(X_train_pd, X_train_fl, n_terms=5, include_linear=True,
                                mode="auto", floor=1e-6):
    values_pd = np.asarray(X_train_pd, dtype=np.float64)
    values_fl = np.asarray(X_train_fl, dtype=np.float64)
    combined = np.concatenate(
        [arr for arr in (values_pd, values_fl) if arr.size > 0],
        axis=0,
    ) if (values_pd.size > 0 or values_fl.size > 0) else np.empty((0, 2), dtype=np.float64)

    n_total = (2 if include_linear else 0) + int(n_terms)
    if mode == "none" or combined.size == 0:
        return np.ones(n_total, dtype=np.float64)
    if mode != "auto":
        raise ValueError(f"strain basis scale mode must be 'auto' or 'none', got {mode!r}.")

    basis = strain_basis_values(combined, n_terms=n_terms, include_linear=include_linear)
    scales = np.max(np.abs(basis), axis=0) if basis.size else np.ones(n_total, dtype=np.float64)
    scales = np.where(np.isfinite(scales), scales, 1.0)
    scales = np.maximum(scales, float(floor))
    return scales.astype(np.float64)


def StrainBasisNet(E_theta, E_z, idi, reg, pen, nonneg=True, n_terms=5, include_linear=True,
                   basis_scales=None):
    """Green-strain basis candidates, each with a subnet weight before shared wx2."""
    constraint = keras.constraints.NonNeg() if nonneg else None
    basis, _, _ = strain_basis_tensors(
        E_theta,
        E_z,
        n_terms=n_terms,
        include_linear=include_linear,
        basis_scales=basis_scales,
    )
    collect = []
    for k, term in enumerate(basis):
        collect.append(
            keras.layers.Dense(
                1,
                kernel_initializer=initializer_1,
                kernel_constraint=constraint,
                kernel_regularizer=regularize(reg, pen),
                use_bias=False,
                activation=None,
                name='w'+str(idi)+'_'+str(k+1),
            )(term)
        )
    return tf.keras.layers.concatenate(collect, axis=1), len(collect)


def single_network_feature_metadata(poly_n_terms=5, poly_include_linear=True, strain_basis_scales=None):
    invariants = ['I1', 'I2', 'I4f', 'I4s', 'I4el', 'I4smc', 'I8fs']
    inv_n_terms = {inv: 7 for inv in invariants}
    inv_n_terms['I8fs'] = 5
    term_names_7 = ["Linear", "Exp linear", "Quadratic", "Exp quadratic",
                    "xlog linear", "xlog quadratic", "Cubic"]
    term_names_5 = ["Linear", "Exp linear", "Quadratic", "Exp quadratic", "Cubic"]

    rows = []
    pos = 0
    for inv in invariants:
        names = term_names_5 if inv == 'I8fs' else term_names_7
        for k, name in enumerate(names):
            rows.append({
                'group': 'invariant',
                'inv': inv,
                'term': name,
                'idx': pos,
                'weight_idx': pos,
            })
            pos += 1

    strain_names = []
    strain_display = []
    if poly_include_linear:
        strain_names.extend(STRAIN_TERM_NAMES_10[:2])
        strain_display.extend(STRAIN_TERM_DISPLAY_10[:2])
    strain_names.extend(STRAIN_TERM_NAMES_10[2:2 + poly_n_terms])
    strain_display.extend(STRAIN_TERM_DISPLAY_10[2:2 + poly_n_terms])
    if strain_basis_scales is None:
        strain_basis_scales = np.ones(len(strain_names), dtype=np.float64)
    for k, (name, display) in enumerate(zip(strain_names, strain_display)):
        rows.append({
            'group': 'strain',
            'inv': 'strain',
            'term': display,
            'raw_term': name,
            'scale': float(strain_basis_scales[k]),
            'idx': pos,
            'weight_idx': pos,
        })
        pos += 1
    return rows


def StrainEnergyCANN_single_network(reg, pen, nonneg=True, poly_n_terms=5, poly_include_linear=True,
                                    strain_basis_scales=None, strain_basis_scale_mode="none"):
    I1_in = tf.keras.Input(shape=(1,), name='I1')
    I2_in = tf.keras.Input(shape=(1,), name='I2')
    I4f_in = tf.keras.Input(shape=(1,), name='I4f')
    I4s_in = tf.keras.Input(shape=(1,), name='I4s')
    I4el_in = tf.keras.Input(shape=(1,), name='I4el')
    I4smc_in = tf.keras.Input(shape=(1,), name='I4smc')
    I8fs_in = tf.keras.Input(shape=(1,), name='I8fs')
    Etheta_in = tf.keras.Input(shape=(1,), name='E_theta')
    Ez_in = tf.keras.Input(shape=(1,), name='E_z')

    I1_ref = keras.layers.Lambda(lambda x: (x-3.0))(I1_in)
    I2_ref = keras.layers.Lambda(lambda x: (x-3.0))(I2_in)
    I4f_ref = keras.layers.Lambda(lambda x: (x-1.0))(I4f_in)
    I4s_ref = keras.layers.Lambda(lambda x: (x-1.0))(I4s_in)
    I4el_ref = keras.layers.Lambda(lambda x: (x-1.0))(I4el_in)
    I4smc_ref = keras.layers.Lambda(lambda x: (x-1.0))(I4smc_in)
    # I8fs fed raw — no reference shift, can be negative

    I1_out = SingleInvNet(I1_ref, 0, reg, pen, nonneg=nonneg)
    terms = int(I1_out.shape[-1])
    I2_out = SingleInvNet(I2_ref, terms, reg, pen, nonneg=nonneg)
    I4f_out = SingleInvNet(I4f_ref, 2*terms, reg, pen, nonneg=nonneg)
    I4s_out = SingleInvNet(I4s_ref, 3*terms, reg, pen, nonneg=nonneg)
    I4el_out = SingleInvNet(I4el_ref, 4*terms, reg, pen, nonneg=nonneg)
    I4smc_out = SingleInvNet(I4smc_ref, 5*terms, reg, pen, nonneg=nonneg)
    I8fs_out = I8Net(I8fs_in, 6*terms, reg, pen, nonneg=nonneg)
    strain_start = 6*terms + int(I8fs_out.shape[-1])
    Strain_out, strain_terms = StrainBasisNet(
        Etheta_in, Ez_in, strain_start, reg, pen,
        nonneg=nonneg,
        n_terms=poly_n_terms,
        include_linear=poly_include_linear,
        basis_scales=strain_basis_scales,
    )

    ALL_I_out = [I1_out, I2_out, I4f_out, I4s_out, I4el_out, I4smc_out, I8fs_out, Strain_out]
    ALL_I_out = tf.keras.layers.concatenate(ALL_I_out, axis=1)

    final_constraint = keras.constraints.NonNeg() if nonneg else None
    W_ANN = keras.layers.Dense(1,kernel_initializer='glorot_normal',kernel_constraint=final_constraint,
                               kernel_regularizer=regularize(reg, pen),
                           use_bias=False, activation=None,name='wx2')(ALL_I_out)
    Psi_model = keras.models.Model(
        inputs=[I1_in, I2_in, I4f_in, I4s_in, I4el_in, I4smc_in, I8fs_in, Etheta_in, Ez_in],
        outputs=[W_ANN],
        name='Psi_single',
    )
    Psi_model.strain_basis_scales = np.asarray(
        strain_basis_scales if strain_basis_scales is not None else np.ones(strain_terms),
        dtype=np.float64,
    )
    Psi_model.strain_basis_scale_mode = strain_basis_scale_mode

    total_terms = 6 * terms + int(I8fs_out.shape[-1]) + strain_terms
    return Psi_model, int(total_terms)


def StrainEnergyCANN_invariant(reg, pen, nonneg=True):
    return StrainEnergyCANN_single_network(
        reg, pen, nonneg=nonneg, poly_n_terms=5, poly_include_linear=False
    )

def initialize_i8fs_terms(Psi_model, i8fs_init=0.0):
    """
    Give the I8fs quadratic terms a small nonzero start so the cross-fiber
    contribution has a gradient path under nonnegative constraints.
    """
    if i8fs_init is None or i8fs_init <= 0.0:
        return

    seeded = []
    for layer_name in ("w42_3", "w42_4"):
        try:
            layer = Psi_model.get_layer(layer_name)
        except ValueError:
            continue
        weights = layer.get_weights()
        if weights:
            weights[0][...] = float(i8fs_init)
            layer.set_weights(weights)
            seeded.append(layer_name)

    try:
        wx2 = Psi_model.get_layer("wx2")
        weights = wx2.get_weights()
        if weights:
            kernel = weights[0]
            i8_start = 6 * 7
            for offset in (2, 3):
                kernel[i8_start + offset, :] = np.maximum(kernel[i8_start + offset, :], float(i8fs_init))
            wx2.set_weights(weights)
            seeded.append("wx2[I8fs quadratic rows]")
    except ValueError:
        pass

    if seeded:
        print(f"Initialized I8fs cross terms to {i8fs_init:g}: {', '.join(seeded)}")

def myGradient(a, b):
    grads = K.gradients(a, b)
    if isinstance(grads, (list, tuple)):
        return grads[0]
    return grads

class NonNegMaxConstraint(tf.keras.constraints.Constraint):
    def __init__(self, max_value=1.0/3.0):
        self.max_value = max_value
    def __call__(self, w):
        return tf.clip_by_value(w, 0.0, self.max_value)

class H_Layer_kappas(keras.layers.Layer):
    def __init__(self, nameU, setAl, init,
                 helical_mode="independent",
                 helix_angle_init_deg=35.0,
                 helix_angle_min_deg=5.0,
                 helix_angle_max_deg=85.0,
                 train_fiber_angles=True):
        super(H_Layer_kappas, self).__init__()
        self.nameU = nameU
        self.setAl = setAl
        self.init = init
        self.helical_mode = helical_mode
        self.helix_angle_init_deg = float(helix_angle_init_deg)
        self.helix_angle_min_deg = float(helix_angle_min_deg)
        self.helix_angle_max_deg = float(helix_angle_max_deg)
        self.train_fiber_angles = train_fiber_angles
        if self.helical_mode not in ("independent", "symmetric"):
            raise ValueError(f"helical_mode must be 'independent' or 'symmetric', got {helical_mode!r}")
        if not (0.0 <= self.helix_angle_min_deg < self.helix_angle_max_deg <= 90.0):
            raise ValueError("Expected 0 <= helix_angle_min_deg < helix_angle_max_deg <= 90.")

        angle_frac = np.clip(
            (self.helix_angle_init_deg - self.helix_angle_min_deg)
            / (self.helix_angle_max_deg - self.helix_angle_min_deg),
            1e-4,
            1.0 - 1e-4,
        )
        angle_raw_init = float(np.log(angle_frac / (1.0 - angle_frac)))

        self.kappa_f = self.add_weight(name="kappa_f", initializer=tf.keras.initializers.Constant(0.0), constraint=NonNegMaxConstraint(1.0/3.0), dtype=tf.float32, trainable=True)
        self.kappa_s = self.add_weight(name="kappa_s", initializer=tf.keras.initializers.Constant(0.0), constraint=NonNegMaxConstraint(1.0/3.0), dtype=tf.float32, trainable=True)
        if self.helical_mode == "symmetric":
            self.helix_angle_raw = self.add_weight(
                name="helix_angle_raw",
                initializer=tf.keras.initializers.Constant(angle_raw_init),
                dtype=tf.float32,
                trainable=train_fiber_angles,
            )
            self.f_dir = None
            self.s_dir = None
        else:
            angle_rad = np.deg2rad(self.helix_angle_init_deg)
            self.f_dir = self.add_weight(name="phi1", initializer=tf.keras.initializers.Constant(-angle_rad), dtype=tf.float32, trainable=train_fiber_angles)
            self.s_dir = self.add_weight(name="phi2", initializer=tf.keras.initializers.Constant(angle_rad), dtype=tf.float32, trainable=train_fiber_angles)
            self.helix_angle_raw = None
        self.elas_dir = self.add_weight(name="elas", initializer=tf.keras.initializers.Constant(np.pi/2), dtype=tf.float32, trainable=False)
        self.smc_dir = self.add_weight(name="smc", initializer=tf.keras.initializers.Constant(0.0), dtype=tf.float32, trainable=False)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "nameU": self.nameU,
            "setAl": self.setAl,
            "init": self.init,
            "helical_mode": self.helical_mode,
            "helix_angle_init_deg": self.helix_angle_init_deg,
            "helix_angle_min_deg": self.helix_angle_min_deg,
            "helix_angle_max_deg": self.helix_angle_max_deg,
            "train_fiber_angles": self.train_fiber_angles,
        })
        return config

    def helical_dirs(self):
        if self.helical_mode == "symmetric":
            min_rad = tf.constant(np.deg2rad(self.helix_angle_min_deg), dtype=tf.float32)
            max_rad = tf.constant(np.deg2rad(self.helix_angle_max_deg), dtype=tf.float32)
            alpha = min_rad + (max_rad - min_rad) * tf.sigmoid(self.helix_angle_raw)
            return alpha, -alpha
        return self.f_dir, self.s_dir

    def helical_dirs_numpy(self):
        f_dir, s_dir = self.helical_dirs()
        return float(f_dir.numpy()), float(s_dir.numpy())

    def call(self, lam):
        lam_theta = lam[:, 0:1]
        lam_z = lam[:, 1:2]
        lam_r = 1 / (lam_theta * lam_z)
        f_dir, s_dir = self.helical_dirs()

        fTCf = (lam_theta * tf.cos(f_dir))**2 + (lam_z * tf.sin(f_dir))**2
        sTCs = (lam_theta * tf.cos(s_dir))**2 + (lam_z * tf.sin(s_dir))**2

        I4f = self.kappa_f * (lam_theta**2 + lam_z**2 + lam_r**2) + (1.0 - 3.0 * self.kappa_f) * fTCf
        I4s = self.kappa_s * (lam_theta**2 + lam_z**2 + lam_r**2) + (1.0 - 3.0 * self.kappa_s) * sTCs

        I4el = (lam_theta * tf.cos(self.elas_dir))**2 + (lam_z * tf.sin(self.elas_dir))**2
        I4smc = (lam_theta * tf.cos(self.smc_dir))**2 + (lam_z * tf.sin(self.smc_dir))**2

        I8fs = (lam_theta**2 * tf.cos(f_dir) * tf.cos(s_dir) +
                lam_z**2    * tf.sin(f_dir) * tf.sin(s_dir))

        out = tf.concat([tf.fill(tf.shape(lam_theta), self.kappa_f),
                         tf.fill(tf.shape(lam_theta), self.kappa_s),
                         tf.fill(tf.shape(lam_theta), f_dir),
                         tf.fill(tf.shape(lam_theta), s_dir),
                         tf.fill(tf.shape(lam_theta), self.elas_dir),
                         tf.fill(tf.shape(lam_theta), self.smc_dir),
                         I4f, I4s, I4el, I4smc, I8fs], axis=1)
        return out

class MinStretchConstraint(tf.keras.constraints.Constraint):
    """Constrains weight to be >= 1.0 (valid stretch values only)"""
    def __init__(self, min_value=1.0):
        self.min_value = min_value
    
    def __call__(self, w):
        return tf.maximum(w, self.min_value)
    
    def get_config(self):
        return {'min_value': self.min_value}

class FiberRecruitmentLayer(keras.layers.Layer):
    """
    Applies recruitment to all 4 fiber invariants. v4 supports:
      - dist_type: "beta" (default, identical to v3) or "lognormal" (shifted truncated
        lognormal on y = lam_s - lambda_lb, integrated over (0, lambda_ub - lambda_lb])
      - shift_mode: "off" (default), "direct" (lambda_lb_eff = lambda_lb * companion/ref),
        or "inverse" (lambda_lb_eff = lambda_lb * ref/companion). The companion stretch
        is lambda_z when branch_flag==0 (PD) and lambda_theta when branch_flag==1 (FL),
        mirroring Base_Poly_new_updated.effective_lambda_lb.
      - per_curve_offsets: optional bounded delta_lb per curve_id, learned via
        curve_offset_max * tanh(embedding); regularized with L2 = curve_offset_l2.
    """
    def __init__(self,
                 init_threshold=1.0,
                 init_sharpness=10.0,
                 lb_min=0.90,
                 lb_max=1.60,
                 ub_max=1.80,
                 trainable_threshold=True,
                 trainable_sharpness=True,
                 dist_type="beta",
                 shift_mode="off",
                 ref_lambda_theta=1.0,
                 ref_lambda_z=1.0,
                 per_curve_offsets=False,
                 n_curves=1,
                 curve_offset_max=0.30,
                 curve_offset_l2=1e-3):
        super(FiberRecruitmentLayer, self).__init__()
        self.init_threshold = init_threshold
        self.init_sharpness = init_sharpness
        self.trainable_threshold = trainable_threshold
        self.trainable_sharpness = trainable_sharpness
        self.family_names = ("f", "s", "el", "smc")
        self.lb_min = float(lb_min)
        self.lb_max = float(lb_max)
        self.ub_max = float(ub_max)
        dist_type = (dist_type or "beta").lower()
        shift_mode = (shift_mode or "off").lower()
        if dist_type not in ("beta", "lognormal"):
            raise ValueError(f"dist_type must be 'beta' or 'lognormal', got {dist_type!r}")
        if shift_mode not in ("off", "direct", "inverse"):
            raise ValueError(f"shift_mode must be 'off', 'direct' or 'inverse', got {shift_mode!r}")
        self.dist_type = dist_type
        self.shift_mode = shift_mode
        self.ref_lambda_theta = float(ref_lambda_theta)
        self.ref_lambda_z = float(ref_lambda_z)
        self.per_curve_offsets = bool(per_curve_offsets)
        self.n_curves = max(1, int(n_curves))
        self.curve_offset_max = float(curve_offset_max)
        self.curve_offset_l2 = float(curve_offset_l2)
        if not (self.lb_min < self.lb_max < self.ub_max):
            raise ValueError(
                "Fiber recruitment bounds must satisfy "
                f"recruit_lb_min < recruit_lb_max < recruit_ub_max; got "
                f"{self.lb_min}, {self.lb_max}, {self.ub_max}."
            )
        self.tau_min = 1e-2
        self.tau_max = 0.15
        lb_frac = np.clip((float(init_threshold) - self.lb_min) / (self.lb_max - self.lb_min), 1e-4, 1.0 - 1e-4)
        lb_raw_init = float(np.log(lb_frac / (1.0 - lb_frac)))

        self.lambda_lb_raw = self.add_weight(
            name='lambda_lb_raw',
            shape=(4,),
            initializer=tf.keras.initializers.Constant(lb_raw_init),
            dtype=tf.float32,
            trainable=self.trainable_threshold
        )
        self.lambda_ub_raw = self.add_weight(
            name='lambda_ub_raw',
            shape=(4,),
            initializer=tf.keras.initializers.Constant(1.0),
            dtype=tf.float32,
            trainable=self.trainable_threshold
        )
        self.beta_a_raw = self.add_weight(
            name='beta_a_raw',
            shape=(4,),
            initializer=tf.keras.initializers.Constant(0.2),
            dtype=tf.float32,
            trainable=self.trainable_sharpness
        )
        self.beta_b_raw = self.add_weight(
            name='beta_b_raw',
            shape=(4,),
            initializer=tf.keras.initializers.Constant(0.2),
            dtype=tf.float32,
            trainable=self.trainable_sharpness
        )
        self.tau_raw = self.add_weight(
            name='tau_raw',
            shape=(4,),
            initializer=tf.keras.initializers.Constant(-4.0),
            dtype=tf.float32,
            trainable=self.trainable_sharpness
        )

        # Lognormal parameters (used only when dist_type=='lognormal'). mu lives in
        # [-3, +1] via sigmoid; sigma in [0.05, 0.60] via sigmoid. Same bounds as
        # Base_Poly_new_updated.logn_mu / logn_sig so values stay interpretable.
        self.logn_mu_raw = self.add_weight(
            name='logn_mu_raw', shape=(4,),
            initializer=tf.keras.initializers.Constant(0.0),
            dtype=tf.float32,
            trainable=self.trainable_sharpness and (self.dist_type == "lognormal"),
        )
        self.logn_sig_raw = self.add_weight(
            name='logn_sig_raw', shape=(4,),
            initializer=tf.keras.initializers.Constant(0.0),
            dtype=tf.float32,
            trainable=self.trainable_sharpness and (self.dist_type == "lognormal"),
        )

        # Per-curve recruitment offsets. delta_lb(curve, family) =
        #   curve_offset_max * tanh(curve_offset_raw[curve, family]). L2 regularizer
        # applies to the raw values.
        if self.per_curve_offsets:
            reg = keras.regularizers.l2(self.curve_offset_l2)
            self.curve_offset_raw = self.add_weight(
                name='curve_offset_raw',
                shape=(self.n_curves, 4),
                initializer=tf.keras.initializers.Zeros(),
                dtype=tf.float32,
                trainable=True,
                regularizer=reg,
            )
        else:
            self.curve_offset_raw = None

        _x, _w = np.polynomial.legendre.leggauss(12)
        self.quad_x = tf.constant(0.5 * (_x + 1.0), dtype=tf.float32)   # nodes on [0,1]
        self.quad_w = tf.constant(0.5 * _w, dtype=tf.float32)           # weights on [0,1], sum=1
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "init_threshold": self.init_threshold,
            "init_sharpness": self.init_sharpness,
            "lb_min": self.lb_min,
            "lb_max": self.lb_max,
            "ub_max": self.ub_max,
            "trainable_threshold": self.trainable_threshold,
            "trainable_sharpness": self.trainable_sharpness,
            "dist_type": self.dist_type,
            "shift_mode": self.shift_mode,
            "ref_lambda_theta": self.ref_lambda_theta,
            "ref_lambda_z": self.ref_lambda_z,
            "per_curve_offsets": self.per_curve_offsets,
            "n_curves": self.n_curves,
            "curve_offset_max": self.curve_offset_max,
            "curve_offset_l2": self.curve_offset_l2,
        })
        return config

    # ---- v4 lognormal helpers (used when dist_type=='lognormal') ----
    def logn_mu(self):
        return -3.0 + 4.0 * tf.sigmoid(self.logn_mu_raw)
    def logn_sig(self):
        return 0.05 + 0.55 * tf.sigmoid(self.logn_sig_raw)
    def curve_offsets(self):
        """Returns a (n_curves, 4) tensor of bounded lambda_lb offsets, or None."""
        if self.curve_offset_raw is None:
            return None
        return self.curve_offset_max * tf.tanh(self.curve_offset_raw)

    def lambda_lb(self):
        return self.lb_min + (self.lb_max - self.lb_min) * tf.sigmoid(self.lambda_lb_raw)

    def lambda_ub(self):
        lb = self.lambda_lb()
        return lb + (self.ub_max - lb) * tf.sigmoid(self.lambda_ub_raw)

    def beta_ab(self):
        alpha = 0.2 + 7.8 * tf.sigmoid(self.beta_a_raw)
        beta = 0.2 + 7.8 * tf.sigmoid(self.beta_b_raw)
        return alpha, beta

    def tau(self):
        return self.tau_min + (self.tau_max - self.tau_min) * tf.sigmoid(self.tau_raw)

    @staticmethod
    def _smoothpos(x, tau):
        return tf.nn.softplus(x / tau) * tau

    def _effective_lb(self, family_idx, lambda_theta, lambda_z, branch_flag, curve_id):
        """Compute lambda_lb_eff per sample for one family, applying optional
        branch-aware shift and per-curve offset. Returns a (batch, 1) tensor."""
        base_lb = self.lambda_lb()[family_idx]  # scalar tensor
        base_ub = tf.maximum(self.lambda_ub()[family_idx], base_lb + 1e-3)

        # Branch-aware shift: choose companion stretch per sample.
        if self.shift_mode == "off" or lambda_theta is None or lambda_z is None:
            shift = tf.ones_like(lambda_theta) if lambda_theta is not None else tf.constant([[1.0]], dtype=tf.float32)
            lb_eff = base_lb * shift
        else:
            branch = tf.cast(branch_flag, tf.float32) if branch_flag is not None else tf.zeros_like(lambda_theta)
            # branch==0 (PD/theta): companion = lambda_z, ref = ref_lambda_z.
            # branch==1 (FL/z):     companion = lambda_theta, ref = ref_lambda_theta.
            companion = tf.where(branch < 0.5, lambda_z, lambda_theta)
            ref = tf.where(
                branch < 0.5,
                tf.constant(self.ref_lambda_z, dtype=tf.float32) * tf.ones_like(companion),
                tf.constant(self.ref_lambda_theta, dtype=tf.float32) * tf.ones_like(companion),
            )
            ratio = companion / tf.maximum(ref, 1e-6)
            if self.shift_mode == "inverse":
                ratio = tf.maximum(ref, 1e-6) / tf.maximum(companion, 1e-6)
            # Bound the multiplicative factor so a wild prestretch can't yank lb
            # outside the [lb_min, ub_max] band.
            ratio = tf.clip_by_value(ratio, 0.5, 2.0)
            lb_eff = base_lb * ratio

        # Per-curve additive offset on lambda_lb.
        offs = self.curve_offsets()
        if offs is not None and curve_id is not None:
            # gather offs[curve_id, family_idx] per sample
            cid = tf.cast(tf.reshape(curve_id, [-1]), tf.int32)
            cid = tf.clip_by_value(cid, 0, self.n_curves - 1)
            offset_per_sample = tf.gather(offs[:, family_idx], cid)
            lb_eff = lb_eff + tf.reshape(offset_per_sample, tf.shape(lb_eff))

        # Final clamp so lb stays inside the legal band and below ub.
        lb_eff = tf.clip_by_value(lb_eff, self.lb_min, self.ub_max - 1e-3)
        lb_eff = tf.minimum(lb_eff, base_ub - 1e-3)
        return lb_eff, base_ub

    def _recruited_invariant_v4(self, I4, family_idx,
                                 lambda_theta, lambda_z, branch_flag, curve_id):
        """Generalized version of _beta_recruited_invariant. Handles per-sample
        lambda_lb (branch-aware + per-curve) and dist_type in {beta, lognormal}."""
        lambda_mag = tf.sqrt(tf.maximum(I4, 1e-12))
        lb_eff, ub = self._effective_lb(family_idx, lambda_theta, lambda_z,
                                        branch_flag, curve_id)
        # Broadcast scalars to per-sample shapes for the quadrature.
        ub_b = tf.broadcast_to(ub * tf.ones_like(lb_eff), tf.shape(lb_eff))
        span = tf.maximum(ub_b - lb_eff, 1e-3)
        tau = self.tau()[family_idx]

        u = tf.reshape(self.quad_x, (1, -1))   # (1, n_quad)
        w_gl = tf.reshape(self.quad_w, (1, -1))
        # lam_s shape: (batch, n_quad)
        lam_s = lb_eff + span * u

        if self.dist_type == "beta":
            alpha, beta = self.beta_ab()
            a = alpha[family_idx]
            b = beta[family_idx]
            u_safe = tf.clip_by_value(u, 1e-6, 1.0 - 1e-6)
            logB = tf.math.lgamma(a) + tf.math.lgamma(b) - tf.math.lgamma(a + b)
            pdf_u = tf.exp((a - 1.0) * tf.math.log(u_safe) +
                           (b - 1.0) * tf.math.log(1.0 - u_safe) - logB)
            # pdf in u-space; the change-of-variable jacobian (1/span) and the
            # subsequent multiplication by span cancel, so quad_weight = pdf_u * w_gl.
            quad_weight = pdf_u * w_gl
        else:  # lognormal: shifted truncated lognormal on y = lam_s - lb_eff
            mu = self.logn_mu()[family_idx]
            sg = self.logn_sig()[family_idx]
            y = tf.maximum(lam_s - lb_eff, 1e-12)
            base = tf.exp(-0.5 * tf.square((tf.math.log(y) - mu) / sg)) / (
                y * sg * tf.constant(np.sqrt(2.0 * np.pi), dtype=tf.float32)
            )
            yub = tf.maximum(ub_b - lb_eff, 1e-12)
            Za = 0.5 * (1.0 + tf.math.erf(
                (tf.math.log(yub) - mu) / (sg * tf.constant(np.sqrt(2.0), dtype=tf.float32))
            ))
            pdf_lam = base / tf.maximum(Za, 1e-9)  # shape (batch, n_quad), in lam_s space
            # Gauss-Legendre weight on u maps to lam-space via *span.
            quad_weight = pdf_lam * span * w_gl

        recruited_activation = self._smoothpos((lambda_mag / lam_s) - 1.0, tau)
        expected_activation = tf.reduce_sum(
            recruited_activation * quad_weight, axis=1, keepdims=True
        )
        recruited_lambda = 1.0 + expected_activation
        return recruited_lambda ** 2

    # Backwards-compat alias used by plot_recruitment_curves on a v3-style call site.
    def _beta_recruited_invariant(self, I4, family_idx):
        return self._recruited_invariant_v4(I4, family_idx, None, None, None, None)

    def call(self, inputs):
        # v4 accepts 4 invariants and (optionally) lambda_theta, lambda_z, branch_flag,
        # curve_id. Caller decides whether to pass the optional ones; layer falls back
        # to the v3 behaviour when they are omitted.
        I4f, I4s, I4el, I4smc = inputs[0], inputs[1], inputs[2], inputs[3]
        lambda_theta = inputs[4] if len(inputs) > 4 else None
        lambda_z = inputs[5] if len(inputs) > 5 else None
        branch_flag = inputs[6] if len(inputs) > 6 else None
        curve_id = inputs[7] if len(inputs) > 7 else None

        I4f_recruited = self._recruited_invariant_v4(
            I4f, 0, lambda_theta, lambda_z, branch_flag, curve_id)
        I4s_recruited = self._recruited_invariant_v4(
            I4s, 1, lambda_theta, lambda_z, branch_flag, curve_id)
        I4el_recruited = self._recruited_invariant_v4(
            I4el, 2, lambda_theta, lambda_z, branch_flag, curve_id)
        I4smc_recruited = self._recruited_invariant_v4(
            I4smc, 3, lambda_theta, lambda_z, branch_flag, curve_id)

        lb = self.lambda_lb()
        ub = self.lambda_ub()
        alpha, beta = self.beta_ab()
        tau = self.tau()
        return (I4f_recruited, I4s_recruited, I4el_recruited, I4smc_recruited,
                lb, ub, alpha, beta, tau)


# Stress calculation with strain-basis derivatives from the shared CANN energy.
def stress_calc_with_recruitment_and_poly(inputs):
    """
    Updated stress calculation including Green-strain basis contributions.
    """
    (dWdI1, dWdI2, dWdI4f, dWdI4s, dWdI4el, dWdI4smc,
     Lambda_theta, Lambda_z, kf, ks, phi1, phi2, phiel, phismc,
     dWpolyC_theta, dWpolyC_z, dWdI8fs) = inputs
    
    two = tf.constant(2.0, dtype='float32')

    lambda_r = 1.0 / (Lambda_theta * Lambda_z)
    C_theta = Lambda_theta**2
    C_z = Lambda_z**2
    C_r = lambda_r**2
    I1 = C_theta + C_z + C_r

    # Fiber direction components
    f_theta = tf.cos(phi1)
    f_z = tf.sin(phi1)
    s_theta = tf.cos(phi2)
    s_z = tf.sin(phi2)
    el_theta = tf.cos(phiel)
    el_z = tf.sin(phiel)
    smc_theta = tf.cos(phismc)
    smc_z = tf.sin(phismc)

    factor1 = (1.0 - 3.0 * kf)
    factor2 = (1.0 - 3.0 * ks)

    # Structural tensors
    Hf_tt = kf + factor1 * f_theta**2
    Hf_zz = kf + factor1 * f_z**2
    Hs_tt = ks + factor2 * s_theta**2
    Hs_zz = ks + factor2 * s_z**2
    Hel_tt = el_theta**2
    Hel_zz = el_z**2
    Hsmc_tt = smc_theta**2
    Hsmc_zz = smc_z**2

    # Invariant-based stress (from fiber and isotropic terms)
    stress_theta_inv = two * ((dWdI1 + I1 * dWdI2) - dWdI2 * C_theta + 
                              dWdI4f * Hf_tt + dWdI4s * Hs_tt + 
                              dWdI4el * Hel_tt + dWdI4smc * Hsmc_tt +
                              dWdI8fs * C_theta * tf.cos(phi1) * tf.cos(phi2))
    stress_z_inv = two * ((dWdI1 + I1 * dWdI2) - dWdI2 * C_z + 
                          dWdI4f * Hf_zz + dWdI4s * Hs_zz + 
                          dWdI4el * Hel_zz + dWdI4smc * Hsmc_zz +
                          dWdI8fs * C_z * tf.sin(phi1) * tf.sin(phi2))
    
    # Strain-basis stress contribution from the same Psi_single energy.
    stress_theta_poly = two * dWpolyC_theta
    stress_z_poly = two * dWpolyC_z
    
    # Total stress
    stress_theta = stress_theta_inv + stress_theta_poly
    stress_z = stress_z_inv + stress_z_poly
    
    return stress_theta, stress_z


# Modified model architecture with a single invariant + strain CANN energy network
def model_architecture_with_recruitment_and_poly(Psi_model,
                                                 init_threshold=1.0,
                                                 init_sharpness=10.0,
                                                 recruit_lb_min=0.90,
                                                 recruit_lb_max=1.60,
                                                 recruit_ub_max=1.80,
                                                 helical_mode="independent",
                                                 helix_angle_init_deg=35.0,
                                                 helix_angle_min_deg=5.0,
                                                 helix_angle_max_deg=85.0,
                                                 train_fiber_angles=True,
                                                 use_recruitment=True,
                                                 use_poly=True,
                                                 poly_n_terms=5,
                                                 poly_trainable=False,
                                                 poly_regularizer=0.1,
                                                 poly_nonneg=True,
                                                 poly_include_linear=False,
                                                 # v4 additions
                                                 recruit_dist="beta",
                                                 recruit_shift_mode="off",
                                                 ref_lambda_theta=1.0,
                                                 ref_lambda_z=1.0,
                                                 per_curve_offsets=False,
                                                 n_curves=1,
                                                 curve_offset_max=0.30,
                                                 curve_offset_l2=1e-3):
    """Model architecture with one shared CANN energy over invariant and strain bases.

    v4 adds two optional auxiliary inputs:
      - branch_flag (int per sample; 0 = PD/theta branch, 1 = FL/z branch)
      - curve_id    (int per sample; index into the per-curve offset table)
    These are consumed only by FiberRecruitmentLayer; the rest of the graph is
    identical to v3.
    """
    Lambda_theta = keras.layers.Input(shape=(1,), name='Lambda_theta')
    Lambda_z = keras.layers.Input(shape=(1,), name='Lambda_z')
    branch_flag = keras.layers.Input(shape=(1,), name='branch_flag', dtype='float32')
    curve_id = keras.layers.Input(shape=(1,), name='curve_id', dtype='int32')

    Lam = keras.layers.Concatenate(axis=1)([Lambda_theta, Lambda_z])
    H_out = H_Layer_kappas(
        'kappa',
        True,
        0.05,
        helical_mode=helical_mode,
        helix_angle_init_deg=helix_angle_init_deg,
        helix_angle_min_deg=helix_angle_min_deg,
        helix_angle_max_deg=helix_angle_max_deg,
        train_fiber_angles=train_fiber_angles,
    )(Lam)

    kf = H_out[:, 0:1]
    ks = H_out[:, 1:2]
    phi1 = H_out[:, 2:3]
    phi2 = H_out[:, 3:4]
    phiel = H_out[:, 4:5]
    phismc = H_out[:, 5:6]
    I4f_H = H_out[:, 6:7]
    I4s_H = H_out[:, 7:8]
    I4el_H = H_out[:, 8:9]
    I4smc_H = H_out[:, 9:10]
    I8fs_H = H_out[:, 10:11]

    # Compute invariants
    I1 = keras.layers.Lambda(lambda x: x[0]**2 + x[1]**2 + 1.0/(x[0]*x[1])**2)([Lambda_theta, Lambda_z])
    I2 = keras.layers.Lambda(lambda x: (x[0]**2)*(x[1]**2) + 1.0/x[0]**2 + 1.0/x[1]**2)([Lambda_theta, Lambda_z])
    E_theta = keras.layers.Lambda(lambda x: 0.5 * (tf.square(x) - 1.0), name='E_theta_from_lambda')(Lambda_theta)
    E_z = keras.layers.Lambda(lambda x: 0.5 * (tf.square(x) - 1.0), name='E_z_from_lambda')(Lambda_z)
    
    # Fiber recruitment (v4 supports lognormal + branch-aware shift + per-curve offsets)
    if use_recruitment:
        recruitment_layer = FiberRecruitmentLayer(
            init_threshold=init_threshold,
            init_sharpness=init_sharpness,
            lb_min=recruit_lb_min,
            lb_max=recruit_lb_max,
            ub_max=recruit_ub_max,
            trainable_threshold=True,
            trainable_sharpness=True,
            dist_type=recruit_dist,
            shift_mode=recruit_shift_mode,
            ref_lambda_theta=ref_lambda_theta,
            ref_lambda_z=ref_lambda_z,
            per_curve_offsets=per_curve_offsets,
            n_curves=n_curves,
            curve_offset_max=curve_offset_max,
            curve_offset_l2=curve_offset_l2,
        )
        recruit_inputs = [I4f_H, I4s_H, I4el_H, I4smc_H,
                          Lambda_theta, Lambda_z, branch_flag, curve_id]
        (I4f_gated, I4s_gated, I4el_gated, I4smc_gated,
         lambda_lb, lambda_ub, beta_alpha, beta_beta, tau_rec) = recruitment_layer(recruit_inputs)
        
        I4f = I4f_gated
        I4s = I4s_gated
        I4el = I4el_gated
        I4smc = I4smc_gated
    else:
        I4f = I4f_H
        I4s = I4s_H
        I4el = I4el_H
        I4smc = I4smc_H

    # Keep I8fs_H on the gradient tape by wrapping it in an identity layer.
    # Without this, K.gradients(W_total, I8fs_H) returns None because I8fs_H
    # is a bare slice of H_out that bypasses FiberRecruitmentLayer, so Keras
    # cannot trace the path W_total → I8fs_H at graph-build time.
    I8fs_tape = keras.layers.Lambda(lambda x: tf.identity(x), name='I8fs_tape')(I8fs_H)

    # Single shared energy: invariant features and Green-strain features enter
    # Psi_model together and compete in one final wx2 layer.
    W_total = Psi_model([I1, I2, I4f, I4s, I4el, I4smc, I8fs_tape, E_theta, E_z])[0]

    # Compute derivatives for stress calculation
    dWdI1 = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]), output_shape=(1,))([W_total, I1])
    dWdI2 = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]), output_shape=(1,))([W_total, I2])
    dWdI4f = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]), output_shape=(1,))([W_total, I4f])
    dWdI4s = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]), output_shape=(1,))([W_total, I4s])
    dWdI4el = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]), output_shape=(1,))([W_total, I4el])
    dWdI4smc = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]), output_shape=(1,))([W_total, I4smc])
    dWdI8fs  = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]), output_shape=(1,))([W_total, I8fs_tape])
    dWdE_theta = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]), output_shape=(1,))([W_total, E_theta])
    dWdE_z = keras.layers.Lambda(lambda x: myGradient(x[0], x[1]), output_shape=(1,))([W_total, E_z])
    dWpolyC_theta = keras.layers.Lambda(lambda x: 0.5 * x, output_shape=(1,), name='dWstrain_dCtheta')(dWdE_theta)
    dWpolyC_z = keras.layers.Lambda(lambda x: 0.5 * x, output_shape=(1,), name='dWstrain_dCz')(dWdE_z)

    # Compute stresses
    Stress_theta, Stress_z = keras.layers.Lambda(
        lambda x: stress_calc_with_recruitment_and_poly(x),
        output_shape=[(1,), (1,)]
    )([dWdI1, dWdI2, dWdI4f, dWdI4s, dWdI4el, dWdI4smc, 
       Lambda_theta, Lambda_z, kf, ks, phi1, phi2, phiel, phismc,
       dWpolyC_theta, dWpolyC_z, dWdI8fs])

    model_nn = keras.models.Model(
        inputs=[Lambda_theta, Lambda_z, branch_flag, curve_id],
        outputs=[Stress_theta, Stress_z],
    )
    model_nn.single_network_poly_n_terms = poly_n_terms
    model_nn.single_network_poly_include_linear = poly_include_linear
    return model_nn, Psi_model


def _zeros_branch_curve(n):
    """Convenience: build default branch_flag/curve_id arrays for v3-style call
    sites that only pass [lambda_theta, lambda_z]. branch_flag=0 disables the
    branch shift (when shift_mode=='off' it doesn't matter anyway); curve_id=0
    selects the first per-curve offset (so per_curve_offset effect is the
    smallest learned offset for that curve)."""
    bf = np.zeros((n, 1), dtype=np.float32)
    cid = np.zeros((n, 1), dtype=np.int32)
    return bf, cid


def predict_stress(model_nn, X, branch_flag=None, curve_id=None):
    """Wrapper around model.predict() that supplies sensible defaults for the
    two new inputs. X is (N, 2) with columns [lambda_theta, lambda_z]."""
    n = int(X.shape[0])
    if branch_flag is None:
        branch_flag = np.zeros((n, 1), dtype=np.float32)
    if curve_id is None:
        curve_id = np.zeros((n, 1), dtype=np.int32)
    return model_nn.predict(
        [X[:, 0:1].astype(np.float32), X[:, 1:2].astype(np.float32),
         branch_flag.astype(np.float32), curve_id.astype(np.int32)],
        verbose=0,
    )


# Updated save function to include polynomial parameters
def save_weights_with_recruitment_and_poly(model, Psi_model, outdir, filename='weights.txt'):
    """Save all weights including polynomial parameters"""
    os.makedirs(outdir, exist_ok=True)
    filepath = os.path.join(outdir, filename)

    with open(filepath, 'w') as f:
        f.write("="*60 + "\n")
        f.write("VEIN CANN MODEL WEIGHTS WITH RECRUITMENT AND SINGLE-NETWORK STRAIN TERMS\n")
        f.write("="*60 + "\n\n")

        # Fiber recruitment parameters
        for layer in model.layers:
            if isinstance(layer, FiberRecruitmentLayer):
                f.write("FIBER RECRUITMENT PARAMETERS (BETA SLACK-STRETCH DISTRIBUTION)\n")
                f.write("-"*60 + "\n")
                lb = layer.lambda_lb().numpy()
                ub = layer.lambda_ub().numpy()
                alpha, beta = layer.beta_ab()
                tau = layer.tau().numpy()
                alpha = alpha.numpy()
                beta = beta.numpy()
                for i, family in enumerate(layer.family_names):
                    f.write(
                        f"{family:>4s}: lambda_lb={lb[i]:.6f}, lambda_ub={ub[i]:.6f}, "
                        f"alpha={alpha[i]:.6f}, beta={beta[i]:.6f}, tau={tau[i]:.6f}\n"
                    )
                f.write("\n")

        weights = Psi_model.get_weights()
        weight_names = [w.name for w in Psi_model.weights]
        wx2_weights = weights[-1].flatten()
        wx2_name = weight_names[-1]

        # Green-strain candidate parameters
        f.write("="*60 + "\n")
        f.write("GREEN-STRAIN CANDIDATES INSIDE SHARED PSI NETWORK\n")
        f.write("-"*60 + "\n")
        poly_n_terms = getattr(model, 'single_network_poly_n_terms', 5)
        poly_include_linear = getattr(model, 'single_network_poly_include_linear', True)
        strain_scales = getattr(Psi_model, 'strain_basis_scales', None)
        rows = single_network_feature_metadata(
            poly_n_terms=poly_n_terms,
            poly_include_linear=poly_include_linear,
            strain_basis_scales=strain_scales,
        )
        strain_rows = [r for r in rows if r['group'] == 'strain']
        f.write(f"Polynomial basis terms requested: {poly_n_terms}\n")
        f.write(f"Include linear strain terms: {poly_include_linear}\n")
        f.write(f"Strain basis scale mode: {getattr(Psi_model, 'strain_basis_scale_mode', 'none')}\n")
        f.write("All rows below feed the same final wx2 layer as invariant candidates.\n\n")
        for s in strain_rows:
            sub_w = float(weights[s['weight_idx']].flatten()[0])
            ann_w = float(wx2_weights[s['idx']])
            scale = float(s.get('scale', 1.0))
            eff_norm = sub_w * ann_w
            eff_raw = eff_norm / scale
            f.write(f"  [{s['idx']}] {s['term']:<16s} "
                    f"scale={scale:.6e} w_sub={sub_w:.6e} {wx2_name}={ann_w:.6e} "
                    f"eff_norm={eff_norm:.6e} eff_raw={eff_raw:.6e}\n")

        # Fiber dispersion parameters
        f.write("\n" + "="*60 + "\n")
        f.write("FIBER DISPERSION PARAMETERS\n")
        f.write("-"*60 + "\n")
        for layer in model.layers:
            if isinstance(layer, H_Layer_kappas):
                phi1, phi2 = layer.helical_dirs_numpy()
                f.write(f"kappa_f: {float(layer.kappa_f.numpy()):.6f}\n")
                f.write(f"kappa_s: {float(layer.kappa_s.numpy()):.6f}\n")
                f.write(f"phi1 (deg): {phi1 * 180/np.pi:.2f}°\n")
                f.write(f"phi2 (deg): {phi2 * 180/np.pi:.2f}°\n")
                f.write(f"phiel (deg): {float(layer.elas_dir.numpy()) * 180/np.pi:.2f}°\n")
                f.write(f"phismc (deg): {float(layer.smc_dir.numpy()) * 180/np.pi:.2f}°\n")

        # Psi model weights
        f.write("\n" + "="*60 + "\n")
        f.write("PSI MODEL WEIGHTS (Invariant-based)\n")
        f.write("="*60 + "\n\n")

        invariants = ['I1', 'I2', 'I4f', 'I4s', 'I4el', 'I4smc', 'I8fs']
        inv_n_terms = {inv: 7 for inv in invariants}
        inv_n_terms['I8fs'] = 5  # I8Net: lin/quad/cubic + exp lin/quad (no xlog, no exp-cubic)
        all_meanings = {
            'default': ["Linear", "Exp linear", "Quadratic", "Exp quadratic", "xlog linear", "xlog quadratic",
                        "Cubic"],
            'I8fs':    ["Linear", "Exp linear", "Quadratic", "Exp quadratic", "Cubic"],
        }
        idx = 0
        for inv in invariants:
            f.write(f"{inv} weights:\n")
            f.write("-"*40 + "\n")
            meanings = all_meanings.get(inv, all_meanings['default'])
            for j in range(inv_n_terms[inv]):
                if idx >= len(weights):
                    break
                f.write(f"Layer {j+1} ({meanings[j]}): {weight_names[idx]}\n")
                f.write(np.array2string(weights[idx], precision=6, suppress_small=True))
                f.write("\n")
                idx += 1
            f.write("\n")

        f.write("="*60 + "\n")
        f.write("FINAL W_ANN LAYER (combined weights across all invariant sub-networks)\n")
        f.write("="*60 + "\n\n")
        w_ann_weights = wx2_weights
        f.write(f"{weight_names[-1]} (shape {weights[-1].shape}):\n\n")

        current_group = None
        for row_meta in rows:
            group_name = row_meta['inv'] if row_meta['group'] == 'invariant' else 'strain'
            if group_name != current_group:
                current_group = group_name
                f.write(f"  {current_group} contributions:\n")
            f.write(f"    [{row_meta['idx']}] ({row_meta['term']:16s}): {w_ann_weights[row_meta['idx']]:.6f}\n")
        f.write(f"\n  Total W_ANN input rows: {len(rows)}\n")

    print(f"Weights saved: {filepath}")


def Compile_and_fit_joint(model_nn,
                          X_train_pd, y_train_theta, X_train_fl, y_train_z,
                          X_val_pd, y_val_theta, X_val_fl, y_val_z,
                          epochs, batch_size,
                          obj6_geom,
                          objective='obj6',
                          circ_weight=1.0,
                          lr=0.01,
                          lr_schedule='plateau',
                          lr_decay=0.5,
                          lr_patience=200,
                          lr_min=1e-5,
                          tail_weight=0.0,
                          tail_stress_frac=0.0,
                          theta_tail_stretch=0.0,
                          z_tail_stretch=0.0,
                          tail_width=0.05,
                          log_stress_scale=1.0,
                          log_loss_weight=0.5,
                          verbose=1,
                          # v4 additions
                          cid_train_pd=None, cid_val_pd=None,
                          cid_train_fl=None, cid_val_fl=None,
                          curve_vocabulary=None,
                          curve_weighting="point",
                          robust_kind="mse",
                          robust_delta=0.5,
                          theta_tail_gain=0.0, theta_tail_power=2.0, theta_tail_stretch_v4=None,
                          z_tail_gain=0.0, z_tail_power=2.0, z_tail_stretch_v4=None,
                          early_stop_patience=1000):
    objective = (objective or 'obj6').lower()
    if objective not in ('obj6', 'stress', 'stress_hybrid_log'):
        raise ValueError(f"objective must be 'obj6', 'stress', or 'stress_hybrid_log', got {objective!r}.")
    if objective == 'obj6' and obj6_geom is None:
        raise ValueError("obj6 requires unloaded OD/H/L geometry to reconstruct pressure and force.")
    
    X_train_pd = X_train_pd.astype(np.float32)
    X_train_fl = X_train_fl.astype(np.float32)
    y_train_theta = y_train_theta.astype(np.float32)
    y_train_z = y_train_z.astype(np.float32)
    
    theta_max = np.max(np.abs(y_train_theta)) if len(y_train_theta) > 0 else 1.0
    z_max = np.max(np.abs(y_train_z)) if len(y_train_z) > 0 else 1.0
    
    print(f"\nNormalization factors:")
    print(f"  Theta max: {theta_max:.4f}")
    print(f"  Z max: {z_max:.4f}")
    
    train_obj6 = None
    if objective == 'obj6':
        train_obj6 = make_obj6_targets(X_train_pd, y_train_theta, X_train_fl, y_train_z, obj6_geom)
        print(f"  obj6 pressure mean scale: {train_obj6['pi_scale']:.4f} kPa")
        print(f"  obj6 vessel-force mean scale: {train_obj6['fv_scale']:.4f} mN")
    else:
        if objective == 'stress_hybrid_log':
            print(
                "  objective: hybrid normalized stress + log stress residuals "
                f"(log_weight={log_loss_weight:g}, log_scale={log_stress_scale:g} kPa)"
            )
        else:
            print("  objective: direct normalized stress residuals")
    
    X_train_combined = np.concatenate([X_train_pd, X_train_fl], axis=0) if (len(X_train_pd) > 0 or len(X_train_fl) > 0) else np.empty((0,2), dtype=np.float32)
    
    if objective == 'obj6':
        y_theta_pd_obj6 = pack_obj6_y(
            train_obj6["pi_exp"],
            train_obj6["pressure_factor_pd"],
            train_obj6["pi_scale"],
        )
        y_z_fl_obj6 = pack_obj6_y(
            train_obj6["fv_exp"],
            train_obj6["force_factor_fl"],
            train_obj6["fv_scale"],
        )

        y_train_theta_combined = np.concatenate([
            y_theta_pd_obj6,
            np.zeros((len(y_train_z), 3), dtype=np.float32)
        ], axis=0) if (len(y_train_theta) > 0 or len(y_train_z) > 0) else np.empty((0, 3), dtype=np.float32)
        
        y_train_z_combined = np.concatenate([
            np.zeros((len(y_train_theta), 3), dtype=np.float32),
            y_z_fl_obj6
        ], axis=0) if (len(y_train_theta) > 0 or len(y_train_z) > 0) else np.empty((0, 3), dtype=np.float32)
    else:
        y_train_theta_combined = np.concatenate([
            y_train_theta / theta_max,
            np.zeros_like(y_train_z)
        ], axis=0).astype(np.float32).reshape(-1, 1) if (len(y_train_theta) > 0 or len(y_train_z) > 0) else np.empty((0, 1), dtype=np.float32)
        y_train_z_combined = np.concatenate([
            np.zeros_like(y_train_theta),
            y_train_z / z_max
        ], axis=0).astype(np.float32).reshape(-1, 1) if (len(y_train_theta) > 0 or len(y_train_z) > 0) else np.empty((0, 1), dtype=np.float32)
    
    print(f"  Circumferential loss weight: {circ_weight:.4f}")

    # v4: optional equal-per-curve base weighting (each PD/FL protocol contributes
    # equally, regardless of point density).
    cw = (curve_weighting or "point").lower()
    if cw == "equal" and cid_train_pd is not None and len(y_train_theta) > 0:
        theta_pd_base = equal_curve_sample_weights(cid_train_pd, base_weight=circ_weight)
    else:
        theta_pd_base = circ_weight * np.ones_like(y_train_theta, dtype=np.float32)
    if cw == "equal" and cid_train_fl is not None and len(y_train_z) > 0:
        z_fl_base = equal_curve_sample_weights(cid_train_fl, base_weight=1.0)
    else:
        z_fl_base = np.ones_like(y_train_z, dtype=np.float32)

    # v3 sigmoid-tail weighting (kept for backwards compatibility).
    theta_pd_weights = make_tail_sample_weights(
        X_train_pd, y_train_theta,
        base_weight=1.0,
        stress_scale=theta_max,
        tail_weight=tail_weight,
        stress_frac=tail_stress_frac,
        stretch_threshold=theta_tail_stretch,
        stretch_index=0,
        width=tail_width,
    ) * theta_pd_base
    z_fl_weights = make_tail_sample_weights(
        X_train_fl, y_train_z,
        base_weight=1.0,
        stress_scale=z_max,
        tail_weight=tail_weight,
        stress_frac=tail_stress_frac,
        stretch_threshold=z_tail_stretch,
        stretch_index=1,
        width=tail_width,
    ) * z_fl_base

    # v4 polynomial tail gain (multiplicative on top of the v3 weights).
    # By default it uses the requested --theta_tail_stretch / --z_tail_stretch
    # CLI flags. The old *_v4 names remain as explicit override aliases.
    theta_tail_stretch_poly = (
        theta_tail_stretch if theta_tail_stretch_v4 is None
        else theta_tail_stretch_v4
    )
    z_tail_stretch_poly = (
        z_tail_stretch if z_tail_stretch_v4 is None
        else z_tail_stretch_v4
    )
    theta_pd_weights = theta_pd_weights * make_polynomial_tail_weights(
        X_train_pd, base_weight=1.0,
        tail_gain=theta_tail_gain, tail_power=theta_tail_power,
        tail_stretch=theta_tail_stretch_poly, stretch_index=0,
    )
    z_fl_weights = z_fl_weights * make_polynomial_tail_weights(
        X_train_fl, base_weight=1.0,
        tail_gain=z_tail_gain, tail_power=z_tail_power,
        tail_stretch=z_tail_stretch_poly, stretch_index=1,
    )

    if tail_weight > 0.0:
        print(
            f"  Tail/elbow weighting: weight={tail_weight:g}, stress_frac={tail_stress_frac:g}, "
            f"theta_tail_stretch={theta_tail_stretch:g}, z_tail_stretch={z_tail_stretch:g}, width={tail_width:g}"
        )
        if len(theta_pd_weights):
            print(f"    theta sample weights: min={theta_pd_weights.min():.3f}, max={theta_pd_weights.max():.3f}")
        if len(z_fl_weights):
            print(f"    z sample weights: min={z_fl_weights.min():.3f}, max={z_fl_weights.max():.3f}")

    sample_weight_theta_train = np.concatenate([
        theta_pd_weights if len(y_train_theta) > 0 else np.array([]),
        np.zeros_like(y_train_z) if len(y_train_z) > 0 else np.array([])
    ], axis=0)

    sample_weight_z_train = np.concatenate([
        np.zeros_like(y_train_theta) if len(y_train_theta) > 0 else np.array([]),
        z_fl_weights if len(y_train_z) > 0 else np.array([])
    ], axis=0)
    
    has_validation = (len(X_val_pd) > 0 or len(X_val_fl) > 0)
    
    if has_validation:
        X_val_pd = X_val_pd.astype(np.float32)
        X_val_fl = X_val_fl.astype(np.float32)
        y_val_theta = y_val_theta.astype(np.float32)
        y_val_z = y_val_z.astype(np.float32)
        val_obj6 = make_obj6_targets(X_val_pd, y_val_theta, X_val_fl, y_val_z, obj6_geom) if objective == 'obj6' else None
        
        X_val_combined = np.concatenate([X_val_pd, X_val_fl], axis=0) if (len(X_val_pd) > 0 or len(X_val_fl) > 0) else np.empty((0,2), dtype=np.float32)

        if objective == 'obj6':
            y_val_theta_pd_obj6 = pack_obj6_y(
                val_obj6["pi_exp"],
                val_obj6["pressure_factor_pd"],
                val_obj6["pi_scale"],
            )
            y_val_z_fl_obj6 = pack_obj6_y(
                val_obj6["fv_exp"],
                val_obj6["force_factor_fl"],
                val_obj6["fv_scale"],
            )
            
            y_val_theta_combined = np.concatenate([
                y_val_theta_pd_obj6,
                np.zeros((len(y_val_z), 3), dtype=np.float32)
            ], axis=0) if (len(y_val_theta) > 0 or len(y_val_z) > 0) else np.empty((0, 3), dtype=np.float32)
            
            y_val_z_combined = np.concatenate([
                np.zeros((len(y_val_theta), 3), dtype=np.float32),
                y_val_z_fl_obj6
            ], axis=0) if (len(y_val_theta) > 0 or len(y_val_z) > 0) else np.empty((0, 3), dtype=np.float32)
        else:
            y_val_theta_combined = np.concatenate([
                y_val_theta / theta_max,
                np.zeros_like(y_val_z)
            ], axis=0).astype(np.float32).reshape(-1, 1) if (len(y_val_theta) > 0 or len(y_val_z) > 0) else np.empty((0, 1), dtype=np.float32)
            y_val_z_combined = np.concatenate([
                np.zeros_like(y_val_theta),
                y_val_z / z_max
            ], axis=0).astype(np.float32).reshape(-1, 1) if (len(y_val_theta) > 0 or len(y_val_z) > 0) else np.empty((0, 1), dtype=np.float32)
        
        if cw == "equal" and cid_val_pd is not None and len(y_val_theta) > 0:
            theta_val_base = equal_curve_sample_weights(cid_val_pd, base_weight=circ_weight)
        else:
            theta_val_base = circ_weight * np.ones_like(y_val_theta, dtype=np.float32)
        if cw == "equal" and cid_val_fl is not None and len(y_val_z) > 0:
            z_val_base = equal_curve_sample_weights(cid_val_fl, base_weight=1.0)
        else:
            z_val_base = np.ones_like(y_val_z, dtype=np.float32)
        theta_val_weights = make_tail_sample_weights(
            X_val_pd, y_val_theta,
            base_weight=1.0,
            stress_scale=theta_max,
            tail_weight=tail_weight,
            stress_frac=tail_stress_frac,
            stretch_threshold=theta_tail_stretch,
            stretch_index=0,
            width=tail_width,
        ) * theta_val_base
        z_val_weights = make_tail_sample_weights(
            X_val_fl, y_val_z,
            base_weight=1.0,
            stress_scale=z_max,
            tail_weight=tail_weight,
            stress_frac=tail_stress_frac,
            stretch_threshold=z_tail_stretch,
            stretch_index=1,
            width=tail_width,
        ) * z_val_base
        theta_val_weights = theta_val_weights * make_polynomial_tail_weights(
            X_val_pd, base_weight=1.0,
            tail_gain=theta_tail_gain, tail_power=theta_tail_power,
            tail_stretch=theta_tail_stretch_poly, stretch_index=0,
        )
        z_val_weights = z_val_weights * make_polynomial_tail_weights(
            X_val_fl, base_weight=1.0,
            tail_gain=z_tail_gain, tail_power=z_tail_power,
            tail_stretch=z_tail_stretch_poly, stretch_index=1,
        )

        sample_weight_theta_val = np.concatenate([
            theta_val_weights if len(y_val_theta) > 0 else np.array([]),
            np.zeros_like(y_val_z) if len(y_val_z) > 0 else np.array([])
        ], axis=0)
        
        sample_weight_z_val = np.concatenate([
            np.zeros_like(y_val_theta) if len(y_val_theta) > 0 else np.array([]),
            z_val_weights if len(y_val_z) > 0 else np.array([])
        ], axis=0)
        
        X_val_theta_combined = X_val_combined[:, 0:1] if X_val_combined.size else np.empty((0,1), dtype=np.float32)
        X_val_z_combined = X_val_combined[:, 1:2] if X_val_combined.size else np.empty((0,1), dtype=np.float32)
        
        validation_data = (
            [X_val_theta_combined, X_val_z_combined],
            [y_val_theta_combined, y_val_z_combined],
            [sample_weight_theta_val, sample_weight_z_val]
        )
        monitor_metric = "val_loss"
    else:
        validation_data = None
        monitor_metric = "loss"
    
    # Build optimizer with optional scheduled learning rate.
    schedule_kind = (lr_schedule or 'none').lower()
    print(f"  LR schedule: {schedule_kind} (initial lr={lr:g})")
    if schedule_kind == 'exponential':
        decay_steps = max(1, epochs // 10)
        optimizer_lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=decay_steps,
            decay_rate=lr_decay,
            staircase=False,
        )
    elif schedule_kind == 'cosine':
        optimizer_lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=max(1, epochs),
            alpha=lr_min / max(lr, 1e-12),
        )
    else:
        optimizer_lr = lr

    # v4: optional robust kernel around any of the three objectives. When
    # robust_kind == 'mse' this matches v3 exactly.
    if objective == 'obj6':
        losses = [
            make_robust_obj6_loss(theta_max, robust_kind=robust_kind, robust_delta=robust_delta),
            make_robust_obj6_loss(z_max,    robust_kind=robust_kind, robust_delta=robust_delta),
        ]
    elif objective == 'stress_hybrid_log':
        losses = [
            make_robust_hybrid_log_loss(theta_max, log_stress_scale=log_stress_scale,
                                        log_loss_weight=log_loss_weight,
                                        robust_kind=robust_kind, robust_delta=robust_delta),
            make_robust_hybrid_log_loss(z_max,    log_stress_scale=log_stress_scale,
                                        log_loss_weight=log_loss_weight,
                                        robust_kind=robust_kind, robust_delta=robust_delta),
        ]
    else:
        losses = [
            make_robust_stress_loss(theta_max, robust_kind=robust_kind, robust_delta=robust_delta),
            make_robust_stress_loss(z_max,    robust_kind=robust_kind, robust_delta=robust_delta),
        ]
    print(f"  Robust kernel: {robust_kind} (delta={robust_delta:g}); curve_weighting={cw}")

    model_nn.compile(
        loss=losses,
        optimizer=tf.optimizers.Adam(learning_rate=optimizer_lr)
    )

    callbacks = [keras.callbacks.EarlyStopping(
        monitor=monitor_metric, min_delta=0.0001,
        patience=int(early_stop_patience), restore_best_weights=True
    )]
    if schedule_kind == 'plateau':
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor=monitor_metric,
            factor=lr_decay,
            patience=lr_patience,
            min_lr=lr_min,
            verbose=verbose,
        ))

    X_theta_train_combined = X_train_combined[:, 0:1] if X_train_combined.size else np.empty((0,1), dtype=np.float32)
    X_z_train_combined = X_train_combined[:, 1:2] if X_train_combined.size else np.empty((0,1), dtype=np.float32)

    # v4: build branch_flag and curve_id arrays aligned with X_train_combined order
    # (PD samples first, then FL samples).
    n_pd = int(X_train_pd.shape[0])
    n_fl = int(X_train_fl.shape[0])
    branch_train = np.concatenate([
        np.zeros((n_pd, 1), dtype=np.float32),
        np.ones((n_fl, 1), dtype=np.float32),
    ], axis=0) if (n_pd + n_fl) > 0 else np.empty((0,1), dtype=np.float32)
    if curve_vocabulary is not None and (cid_train_pd is not None or cid_train_fl is not None):
        cid_pd_int = encode_curve_ids(
            cid_train_pd if cid_train_pd is not None else np.array([], dtype=object),
            curve_vocabulary)[0] if n_pd > 0 else np.empty((0,), dtype=np.int32)
        cid_fl_int = encode_curve_ids(
            cid_train_fl if cid_train_fl is not None else np.array([], dtype=object),
            curve_vocabulary)[0] if n_fl > 0 else np.empty((0,), dtype=np.int32)
        curve_train = np.concatenate(
            [cid_pd_int.reshape(-1,1), cid_fl_int.reshape(-1,1)], axis=0
        ).astype(np.int32)
    else:
        curve_train = np.zeros((n_pd + n_fl, 1), dtype=np.int32)

    if has_validation:
        n_vpd = int(X_val_pd.shape[0])
        n_vfl = int(X_val_fl.shape[0])
        branch_val = np.concatenate([
            np.zeros((n_vpd, 1), dtype=np.float32),
            np.ones((n_vfl, 1), dtype=np.float32),
        ], axis=0) if (n_vpd + n_vfl) > 0 else np.empty((0,1), dtype=np.float32)
        if curve_vocabulary is not None and (cid_val_pd is not None or cid_val_fl is not None):
            cid_vpd = encode_curve_ids(
                cid_val_pd if cid_val_pd is not None else np.array([], dtype=object),
                curve_vocabulary)[0] if n_vpd > 0 else np.empty((0,), dtype=np.int32)
            cid_vfl = encode_curve_ids(
                cid_val_fl if cid_val_fl is not None else np.array([], dtype=object),
                curve_vocabulary)[0] if n_vfl > 0 else np.empty((0,), dtype=np.int32)
            curve_val = np.concatenate(
                [cid_vpd.reshape(-1,1), cid_vfl.reshape(-1,1)], axis=0
            ).astype(np.int32)
        else:
            curve_val = np.zeros((n_vpd + n_vfl, 1), dtype=np.int32)
        # Rebuild validation_data to include the two extra inputs.
        X_val_theta_combined, X_val_z_combined = validation_data[0]
        validation_data = (
            [X_val_theta_combined, X_val_z_combined, branch_val, curve_val],
            validation_data[1],
            validation_data[2],
        )

    print(
        f"\nStarting fit: epochs={epochs}, batch_size={batch_size}, "
        f"monitor={monitor_metric}, verbose={verbose}",
        flush=True,
    )

    history = model_nn.fit(
        x=[X_theta_train_combined, X_z_train_combined, branch_train, curve_train],
        y=[y_train_theta_combined, y_train_z_combined],
        sample_weight=[sample_weight_theta_train, sample_weight_z_train],
        validation_data=validation_data,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        callbacks=callbacks,
        verbose=verbose
    )
    
    model_nn.theta_max = theta_max
    model_nn.z_max = z_max
    model_nn.objective = objective
    model_nn.obj6_geom = obj6_geom
    if train_obj6 is not None:
        model_nn.obj6_pressure_scale = train_obj6["pi_scale"]
        model_nn.obj6_force_scale = train_obj6["fv_scale"]
    
    return model_nn, history


# =============================================================================
# v4 curriculum / prune-refit / restarts driver
# =============================================================================

# Layer-name patterns used by SingleInvNet / I8Net / StrainBasisNet.
# Per SingleInvNet: w{idi}_1 linear, _2 exp-lin, _3 quad, _4 exp-quad, _5 xlog-lin,
# _6 xlog-quad, _7 cubic. For I8Net: _1 lin, _2 exp-lin, _3 quad, _4 exp-quad, _5 cubic.
# We mark a term "high order" (frozen in phase 1) if it is exp/xlog/cubic or signed
# (anything in the I8fs net).
_I8FS_INV_INDEX = 6  # I1=0, I2=1, I4f=2, I4s=3, I4el=4, I4smc=5, I8fs=6

def _phase1_freeze_pattern(layer_name):
    """Return True if a layer should be frozen during Phase 1."""
    if not layer_name.startswith('w'):
        return False
    # Anything ending in _2, _4, _5, _6 across SingleInvNet IDs 0..5 (the 6 invariant
    # subnets) is exp/xlog -> high-order -> freeze. _7 is cubic -> freeze. Same for
    # all I8fs (signed) terms (idi == 6 * terms = 42 base; we freeze entire block).
    # The strain basis starts at higher idi values; treat it as Phase-2 only too,
    # since polynomial cross terms can compete with low-order isotropic terms.
    return True  # default: freeze everything except the explicit phase-1 list below


def _layer_phase1_allowed(layer_name):
    """Phase-1 keeps only low-order invariant + linear/quad strain terms trainable.
    Specifically: linear (_1) and quadratic (_3) for invariant subnets I1..I4smc;
    plus the StrainBasis linear E_theta/E_z terms. Everything else is frozen until
    Phase 2."""
    if not layer_name.startswith('w'):
        return True  # leave non-w layers (recruitment, kappa, wx2) trainable
    # Parse out the subnet id and term index from name 'w{idi}_{k}'.
    try:
        body = layer_name[1:]
        idi_str, k_str = body.split('_')
        idi = int(idi_str)
        k = int(k_str)
    except ValueError:
        return True
    # Invariant subnets occupy idi 0, 7, 14, 21, 28, 35 (steps of 7) for the 6 nets.
    # I8fs sits at idi = 6 * 7 = 42 (5 terms, indices _1.._5).
    # StrainBasisNet starts after I8fs.
    n_inv_blocks = 6
    inv_terms = 7
    i8_start = n_inv_blocks * inv_terms  # 42
    i8_terms = 5
    strain_start = i8_start + i8_terms   # 47
    if idi < i8_start:
        # one of the 6 standard invariant subnets
        # phase 1: only k in {1 (linear), 3 (quadratic)}
        return k in (1, 3)
    if idi < strain_start:
        # I8fs subnet entirely frozen in phase 1 (signed cross-fiber term)
        return False
    # StrainBasisNet: in phase 1 keep only the linear E_theta and E_z terms,
    # which are k == 1 and k == 2 when include_linear=True.
    return k in (1, 2)


def _apply_phase_freezing(model_nn, phase):
    """Freeze/unfreeze invariant/strain subnet layers per phase."""
    counts = {"frozen": 0, "trainable": 0}
    for layer in model_nn.layers:
        if not layer.weights:
            continue
        name = layer.name
        if name.startswith('w') and '_' in name:
            allow = True if phase >= 2 else _layer_phase1_allowed(name)
            layer.trainable = bool(allow)
            counts["trainable" if layer.trainable else "frozen"] += 1
    return counts


def _compute_term_importance(model_nn, Psi_model, X_train_pd, X_train_fl,
                             curve_vocabulary=None,
                             cid_train_pd=None, cid_train_fl=None):
    """Effective per-term contribution = |w_sub * w_ann| * mean(|basis(x)|) over
    training X. Returns a list of dicts compatible with active_terms.csv."""
    weights = Psi_model.get_weights()
    wx2 = weights[-1].flatten()
    rows = single_network_feature_metadata(
        poly_n_terms=getattr(model_nn, 'single_network_poly_n_terms', 5),
        poly_include_linear=getattr(model_nn, 'single_network_poly_include_linear', True),
        strain_basis_scales=getattr(Psi_model, 'strain_basis_scales', None),
    )
    # Build mean|basis| over the training X for the StrainBasisNet inputs (Green strain).
    if X_train_pd.shape[0] + X_train_fl.shape[0] > 0:
        Xall = np.concatenate([X_train_pd, X_train_fl], axis=0)
    else:
        Xall = np.empty((0, 2), dtype=np.float32)
    strain_basis = strain_basis_values(
        Xall,
        n_terms=getattr(model_nn, 'single_network_poly_n_terms', 5),
        include_linear=getattr(model_nn, 'single_network_poly_include_linear', True),
    )
    strain_mean_abs = np.mean(np.abs(strain_basis), axis=0) if strain_basis.size else np.array([])

    # For invariant rows we don't have direct access to candidate values without
    # re-running the model. Approximate by treating |w_sub * w_ann| as importance
    # (this is also what save_final_equation_report uses as "eff_norm").
    importance = []
    strain_idx = 0
    for row in rows:
        idx = row['idx']
        ann_w = float(wx2[idx])
        sub_w = float(weights[row['weight_idx']].flatten()[0])
        eff = abs(sub_w * ann_w)
        if row['group'] == 'strain' and strain_mean_abs.size > strain_idx:
            scale = float(row.get('scale', 1.0))
            # eff_raw * mean|basis_raw|
            eff = eff * float(strain_mean_abs[strain_idx]) / max(scale, 1e-12)
            strain_idx += 1
        importance.append(dict(
            group=row['group'],
            inv=row['inv'],
            term=row['term'],
            idx=idx,
            weight_idx=row['weight_idx'],
            w_sub=sub_w,
            w_ann=ann_w,
            importance=eff,
        ))
    return importance


def _apply_prune(Psi_model, importance, threshold):
    """Zero rows of wx2 whose importance falls below threshold and freeze them."""
    weights = Psi_model.get_weights()
    wx2 = weights[-1]
    pruned = 0
    for row in importance:
        if row['importance'] < threshold:
            wx2[row['idx'], 0] = 0.0
            pruned += 1
    weights[-1] = wx2
    Psi_model.set_weights(weights)
    return pruned


def train_with_strategy(model_nn, Psi_model,
                        X_train_pd, y_train_theta, X_train_fl, y_train_z,
                        X_val_pd, y_val_theta, X_val_fl, y_val_z,
                        obj6_geom, args,
                        cid_train_pd, cid_val_pd, cid_train_fl, cid_val_fl,
                        curve_vocabulary):
    """Run the requested training strategy and return (model_nn, history_list,
    importance_list). `history_list` is a list of one Keras history per phase
    so we can plot the full training trajectory afterwards."""
    histories = []
    importance = []
    mode = (args.training_mode or 'normal').lower()

    fit_kwargs = dict(
        obj6_geom=obj6_geom,
        objective=args.objective,
        circ_weight=args.circ_weight,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        lr_decay=args.lr_decay,
        lr_patience=args.lr_patience,
        lr_min=args.lr_min,
        tail_weight=args.tail_weight,
        tail_stress_frac=args.tail_stress_frac,
        theta_tail_stretch=args.theta_tail_stretch,
        z_tail_stretch=args.z_tail_stretch,
        tail_width=args.tail_width,
        log_stress_scale=args.log_stress_scale,
        log_loss_weight=args.log_loss_weight,
        verbose=args.verbose,
        cid_train_pd=cid_train_pd, cid_val_pd=cid_val_pd,
        cid_train_fl=cid_train_fl, cid_val_fl=cid_val_fl,
        curve_vocabulary=curve_vocabulary,
        curve_weighting=args.curve_weighting,
        robust_kind=args.robust,
        robust_delta=args.robust_delta,
        theta_tail_gain=args.theta_tail_gain,
        theta_tail_power=args.theta_tail_power,
        theta_tail_stretch_v4=args.theta_tail_stretch_v4,
        z_tail_gain=args.z_tail_gain,
        z_tail_power=args.z_tail_power,
        z_tail_stretch_v4=args.z_tail_stretch_v4,
        early_stop_patience=args.early_stop_patience,
    )

    def _fit_phase(label, epochs):
        print(f"\n[{label}] starting fit ({epochs} epochs)")
        m, h = Compile_and_fit_joint(
            model_nn,
            X_train_pd, y_train_theta, X_train_fl, y_train_z,
            X_val_pd, y_val_theta, X_val_fl, y_val_z,
            epochs=epochs, batch_size=args.batch_size,
            **fit_kwargs,
        )
        histories.append((label, h.history))
        return m

    if mode == 'normal':
        _fit_phase("normal", args.epochs)
        return model_nn, histories, importance

    # Curriculum: phase 1 (low-order only) -> phase 2 (full library, higher reg on
    # exp/xlog) -> optional phase 3 (prune + refit).
    if mode in ('curriculum', 'curriculum_prune_refit'):
        c = _apply_phase_freezing(model_nn, phase=1)
        print(f"  Phase 1 freezing: {c['trainable']} trainable / {c['frozen']} frozen w-layers")
        _fit_phase("phase1", max(int(args.epochs_phase1), 1))

        c = _apply_phase_freezing(model_nn, phase=2)
        print(f"  Phase 2 unfreezing: {c['trainable']} trainable / {c['frozen']} frozen w-layers")
        _fit_phase("phase2", max(int(args.epochs_phase2), 1))

        if mode == 'curriculum_prune_refit' and args.epochs_refit > 0:
            importance = _compute_term_importance(
                model_nn, Psi_model, X_train_pd, X_train_fl,
                curve_vocabulary=curve_vocabulary,
                cid_train_pd=cid_train_pd, cid_train_fl=cid_train_fl,
            )
            pruned = _apply_prune(Psi_model, importance, args.prune_threshold)
            print(f"  Phase 3 prune: removed {pruned} terms below threshold "
                  f"{args.prune_threshold:g}; refitting for {args.epochs_refit} epochs")
            _fit_phase("phase3_refit", max(int(args.epochs_refit), 1))

    return model_nn, histories, importance


def merge_histories(histories):
    """Concatenate per-phase Keras histories into a single dict for plotting."""
    out = {}
    for label, h in histories:
        for k, v in h.items():
            out.setdefault(k, []).extend(v)
    return out


def plotLoss(axe, history, has_validation=True):
    axe.plot(history['loss'], label='Training Loss')
    if has_validation and 'val_loss' in history:
        axe.plot(history['val_loss'], label='Validation Loss')
    axe.set_yscale('log')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)

def plot_raw_data(Fl, Pd, save_path=None):
    """Plot raw stress-stretch data before model fitting - simplified style"""
    
    # Create simple plots
    plt.figure()
    plt.plot(Pd["lambda_theta"], Pd["sigma_theta"], color='red', marker='o', linestyle='', markersize=4)
    plt.xlabel('Stretch')
    plt.ylabel('Stress (kPa)')
    plt.title('Vein sample circumferential stress-stretch')
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'raw_circumferential.pdf'))
        plt.savefig(os.path.join(save_path, 'raw_circumferential.png'))
    plt.show()

    plt.figure()
    plt.plot(Fl["lambda_z"], Fl["sigma_z"], color='red', marker='o', linestyle='', markersize=4)
    plt.xlabel('Stretch')
    plt.ylabel('Stress (kPa)')
    plt.title('Vein sample axial stress-stretch')
    if save_path:
        plt.savefig(os.path.join(save_path, 'raw_axial.pdf'))
        plt.savefig(os.path.join(save_path, 'raw_axial.png'))
    plt.show()



def evaluate_on_dataset(model_nn, X_pd, X_fl, y_theta, y_z, dataset_name="", save_path=None,
                        cid_pd=None, cid_fl=None, curve_vocabulary=None):
    """Evaluate model and create plots for a dataset (training or validation).

    v4 note: pass cid_pd / cid_fl (per-sample curve-id strings) and curve_vocabulary
    so the per-curve recruitment offsets line up. If either is missing we fall back
    to curve_id=0, which is *only* correct when --per_curve_recruit is off."""
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    results = {}
    all_errors = []  # Collect all errors for RMS calculation
    y_true_pool = []  # Collect raw true values (theta + z) for overall R^2
    y_pred_pool = []  # Collect raw predicted values (theta + z) for overall R^2

    if len(X_pd) > 0:
        bf_pd = np.zeros((X_pd.shape[0], 1), dtype=np.float32)
        if cid_pd is not None and curve_vocabulary is not None:
            cid_pd_int, _ = encode_curve_ids(cid_pd, curve_vocabulary)
            cid_pd_arr = cid_pd_int.reshape(-1, 1).astype(np.int32)
        else:
            cid_pd_arr = np.zeros((X_pd.shape[0], 1), dtype=np.int32)
        Stress_preds_pd = model_nn.predict(
            [X_pd[:, 0:1], X_pd[:, 1:2], bf_pd, cid_pd_arr], verbose=0)
        Stress_theta_pred_norm = Stress_preds_pd[0].reshape(-1)
        
        # DENORMALIZE predictions (multiply by max)
        if hasattr(model_nn, 'theta_max'):
            Stress_theta_pred = Stress_theta_pred_norm * model_nn.theta_max
        else:
            Stress_theta_pred = Stress_theta_pred_norm
        
        # Calculate normalized errors for theta
        errors_theta = (Stress_theta_pred - y_theta) / np.mean(y_theta)
        all_errors.extend(errors_theta)
        y_true_pool.append(np.asarray(y_theta).reshape(-1))
        y_pred_pool.append(np.asarray(Stress_theta_pred).reshape(-1))
            
        R2_theta = r2_score(y_theta, Stress_theta_pred)
        rmse_theta = np.sqrt(mean_squared_error(y_theta, Stress_theta_pred))
        results['R2_theta'] = R2_theta
        results['RMSE_theta'] = rmse_theta
        print(f"{dataset_name} - R² Circ: {R2_theta:.4f}, RMSE Circ: {rmse_theta:.4f}")
        
        # Plot circumferential
        plt.figure(figsize=[8, 6])
        plt.plot(X_pd[:, 0], y_theta, 'ro', label='Data', markersize=4)
        plt.plot(X_pd[:, 0], Stress_theta_pred, 'bo', label='Model', markersize=3)
        plt.xlabel('Circumferential Stretch')
        plt.ylabel('Stress (kPa)')
        plt.title(f'{dataset_name} - Circumferential')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f'{dataset_name}_circumferential.pdf'), dpi=300)
            plt.savefig(os.path.join(save_path, f'{dataset_name}_circumferential.png'), dpi=300)
        plt.show()
    
    if len(X_fl) > 0:
        bf_fl = np.ones((X_fl.shape[0], 1), dtype=np.float32)
        if cid_fl is not None and curve_vocabulary is not None:
            cid_fl_int, _ = encode_curve_ids(cid_fl, curve_vocabulary)
            cid_fl_arr = cid_fl_int.reshape(-1, 1).astype(np.int32)
        else:
            cid_fl_arr = np.zeros((X_fl.shape[0], 1), dtype=np.int32)
        Stress_preds_fl = model_nn.predict(
            [X_fl[:, 0:1], X_fl[:, 1:2], bf_fl, cid_fl_arr], verbose=0)
        Stress_z_pred_norm = Stress_preds_fl[1].reshape(-1)
        
        # DENORMALIZE predictions (multiply by max)
        if hasattr(model_nn, 'z_max'):
            Stress_z_pred = Stress_z_pred_norm * model_nn.z_max
        else:
            Stress_z_pred = Stress_z_pred_norm
        
        # Calculate normalized errors for z
        errors_z = (Stress_z_pred - y_z) / np.mean(y_z)
        all_errors.extend(errors_z)
        y_true_pool.append(np.asarray(y_z).reshape(-1))
        y_pred_pool.append(np.asarray(Stress_z_pred).reshape(-1))
            
        R2_z = r2_score(y_z, Stress_z_pred)
        rmse_z = np.sqrt(mean_squared_error(y_z, Stress_z_pred))
        results['R2_z'] = R2_z
        results['RMSE_z'] = rmse_z
        print(f"{dataset_name} - R² Axial: {R2_z:.4f}, RMSE Axial: {rmse_z:.4f}")
        
        # Plot axial
        plt.figure(figsize=[8, 6])
        plt.plot(X_fl[:, 1], y_z, 'ro', label='Data', markersize=4)
        plt.plot(X_fl[:, 1], Stress_z_pred, 'bo', label='Model', markersize=3)
        plt.xlabel('Axial Stretch')
        plt.ylabel('Stress (kPa)')
        plt.title(f'{dataset_name} - Axial')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, f'{dataset_name}_axial.pdf'), dpi=300)
            plt.savefig(os.path.join(save_path, f'{dataset_name}_axial.png'), dpi=300)
        plt.show()
    
    # Calculate combined RMS (matching MATLAB: norm(e,2)/sqrt(length(e)))
    if len(all_errors) > 0:
        all_errors = np.array(all_errors)
        rms = np.linalg.norm(all_errors, 2) / np.sqrt(len(all_errors))
        results['RMS'] = rms
        print(f"{dataset_name} - RMS: {rms:.4f}")

    # Overall R^2 / RMSE pooling circumferential + axial stress together.
    # This is valid here because both channels are the same physical quantity
    # (stress in kPa), so a single grand mean / SS_tot is meaningful. This is
    # NOT a simple average of R2_theta and R2_z -- it is 1 - SS_res/SS_tot
    # computed on the pooled data, so it is implicitly weighted by how many
    # points (and how much variance) each direction contributes.
    if len(y_true_pool) > 0:
        y_true_all = np.concatenate(y_true_pool)
        y_pred_all = np.concatenate(y_pred_pool)
        R2_overall = r2_score(y_true_all, y_pred_all)
        rmse_overall = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
        results['R2_overall'] = R2_overall
        results['RMSE_overall'] = rmse_overall
        print(f"{dataset_name} - R² Overall: {R2_overall:.4f}, RMSE Overall: {rmse_overall:.4f}")

    return results


def _publication_protocol_colors(n_curves, branch_label):
    """Return paired experimental/model colors for one publication panel."""
    n_curves = max(1, int(n_curves))
    if n_curves == 1:
        greys = [0.12]
        reds = [0.78]
    else:
        greys = np.linspace(0.12, 0.62, n_curves)
        reds = np.linspace(0.92, 0.48, n_curves)

    # PD and FL use separate but matching shade scales so each panel is readable
    # on its own, while paired data/model colors stay aligned by protocol order.
    if str(branch_label).upper() == 'FL':
        greys = np.clip(greys + 0.03, 0.0, 0.68)
        reds = np.clip(reds - 0.04, 0.38, 1.0)

    data_colors = [mcolors.to_hex((g, g, g)) for g in greys]
    model_colors = [mcolors.to_hex(plt.cm.Reds(r)) for r in reds]
    return data_colors, model_colors


def _publication_legend_label(curve_id, branch_label, animal_name):
    """Format curve labels for publication legends without changing curve ids."""
    curve_id = str(curve_id)
    if str(branch_label).upper() != 'FL':
        return curve_id

    animal_key = str(animal_name or '').lower().replace(' ', '').replace('_', '')
    if animal_key in ('vein1', 'vein6'):
        pressure_map = {
            'p10': '5 mmHg',
            'p60': '10 mmHg',
            'p100': '15 mmHg',
            'p140': '30 mmHg',
        }
    elif animal_key == 'vein4':
        pressure_map = {
            'p10': '5 mmHg',
            'p60': '10 mmHg',
            'p100': '5 mmHg',
            'p140': '10 mmHg',
        }
    elif animal_key in ('jugular106', 'jugular142', 'jugular143', '106', '142', '143'):
        pressure_map = {
            'p10': '2 mmHg',
            'p60': '6 mmHg',
            'p100': '10 mmHg',
            'p140': '30 mmHg',
        }
    elif 'dodge' in animal_key:
        pressure_map = {
            'p10': '2 mmHg',
            'p60': '9 mmHg',
            'p100': '10 mmHg',
            'p140': '30 mmHg',
        }
    elif 'lexus' in animal_key:
        pressure_map = {
            'p10': '2 mmHg',
            'p60': '8 mmHg',
            'p100': '12 mmHg',
            'p140': '30 mmHg',
        }
    elif 'mercedes' in animal_key:
        pressure_map = {
            'p10': '2 mmHg',
            'p60': '10 mmHg',
            'p100': '15 mmHg',
            'p140': '30 mmHg',
        }
    else:
        pressure_map = {}
    return pressure_map.get(curve_id, curve_id)


def save_publication_figure(model_nn, X_pd, y_theta, cid_pd_strings,
                            X_fl, y_z, cid_fl_strings, curve_vocabulary,
                            animal_name="",
                            r2_theta=None, r2_z=None, r2_overall=None,
                            save_path=None,
                            output_prefix='publication_figure',
                            curve_color_mode='single',
                            show_legend=False,
                            legend_loc='upper left',
                            label_fontsize=24,
                            tick_fontsize=18,
                            title_fontsize=20,
                            data_marker_size=4.5,
                            model_linewidth=2.5,
                            n_ticks=3,
                            n_smooth=300):
    """Two-panel publication figure: PD (sigma_theta vs lambda_theta) and FL
    (sigma_z vs lambda_z). Experimental data are drawn at measured stretches.
    Model fits are drawn as smooth protocol-wise curves rather than connected
    predictions at noisy measured stretch pairs.

    Smooth-curve convention mirrors the quadratic/Fung plotting script:
      * PD/theta: sweep lambda_theta; hold lambda_z at that protocol's mean.
      * FL/z: sweep lambda_z; use a degree-2 smooth fit for
        lambda_theta(lambda_z) within the protocol, with lower-degree fallbacks.

    Title shows the animal name and the per-direction R^2 values, matching the
    style of typical Vein CANN publication figures.

    Saves <save_path>/<output_prefix>.pdf and .png. Returns the matplotlib
    Figure (caller closes)."""
    from matplotlib.ticker import MaxNLocator
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple
    import matplotlib as _mpl

    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # Use Computer Modern mathtext so the labels look LaTeX-like by default.
    # No system LaTeX needed; this is matplotlib's bundled mathtext renderer.
    _mpl.rcParams['mathtext.fontset'] = 'cm'
    _mpl.rcParams['font.family'] = 'serif'

    fig, (ax_pd, ax_fl) = plt.subplots(1, 2, figsize=(12, 5))

    # Rows for the companion CSV. Data rows and smooth model rows are both saved;
    # row_type distinguishes measured points from the smooth plotted model curve.
    csv_rows = []

    def _smooth_companion(primary, companion, n_grid, max_bins=35):
        """Return a smooth companion stretch along the measured protocol path.

        A global polynomial can leave the local measured loading envelope and
        distort a 2D model's apparent stress-stretch shape. Instead, smooth the
        measured companion path with primary-axis bins and interpolate robust
        bin medians onto the plotting grid.
        """
        primary = np.asarray(primary, dtype=np.float64)
        companion = np.asarray(companion, dtype=np.float64)
        finite = np.isfinite(primary) & np.isfinite(companion)
        primary = primary[finite]
        companion = companion[finite]
        if primary.size == 0:
            grid = np.linspace(0.0, 1.0, int(n_grid))
            return grid.astype(np.float32), np.ones_like(grid, dtype=np.float32)
        if np.nanmax(primary) > np.nanmin(primary):
            grid = np.linspace(np.nanmin(primary), np.nanmax(primary), int(n_grid))
        else:
            grid = np.full(int(n_grid), float(primary[0]))
        order = np.argsort(primary)
        primary_sorted = primary[order]
        companion_sorted = companion[order]
        unique_primary = np.unique(primary_sorted)
        if unique_primary.size < 2:
            smooth_companion = np.full_like(grid, float(np.nanmedian(companion_sorted)))
            return grid.astype(np.float32), smooth_companion.astype(np.float32)

        n_bins = min(int(max_bins), max(3, int(np.sqrt(primary_sorted.size))))
        edges = np.linspace(primary_sorted.min(), primary_sorted.max(), n_bins + 1)
        centers = []
        medians = []
        for i in range(n_bins):
            if i == n_bins - 1:
                mask = (primary_sorted >= edges[i]) & (primary_sorted <= edges[i + 1])
            else:
                mask = (primary_sorted >= edges[i]) & (primary_sorted < edges[i + 1])
            if not mask.any():
                continue
            centers.append(float(np.nanmedian(primary_sorted[mask])))
            medians.append(float(np.nanmedian(companion_sorted[mask])))

        centers = np.asarray(centers, dtype=np.float64)
        medians = np.asarray(medians, dtype=np.float64)
        if centers.size < 2:
            unique_primary, inverse = np.unique(primary_sorted, return_inverse=True)
            medians = np.array([
                np.nanmedian(companion_sorted[inverse == j])
                for j in range(unique_primary.size)
            ], dtype=np.float64)
            centers = unique_primary

        center_order = np.argsort(centers)
        centers = centers[center_order]
        medians = medians[center_order]
        centers, unique_idx = np.unique(centers, return_index=True)
        medians = medians[unique_idx]
        smooth_companion = np.interp(grid, centers, medians)
        smooth_companion = np.clip(
            smooth_companion,
            float(np.nanmin(companion_sorted)),
            float(np.nanmax(companion_sorted)),
        )
        return grid.astype(np.float32), smooth_companion.astype(np.float32)

    def _plot_branch(ax, X, y_data, cid_strings, branch_flag, stretch_idx,
                     model_max, theta_or_z, branch_label):
        if X is None or len(X) == 0:
            ax.text(0.5, 0.5, f"no {theta_or_z} data",
                    transform=ax.transAxes, ha='center', va='center')
            return

        X = np.asarray(X, dtype=np.float32)
        y_data = np.asarray(y_data, dtype=np.float32)
        n = X.shape[0]
        cid_strings_arr = np.asarray(cid_strings) if cid_strings is not None \
            else np.array(['default'] * n, dtype=object)
        if cid_strings_arr.shape[0] != n:
            cid_strings_arr = np.array(['default'] * n, dtype=object)

        unique_curves = np.unique(cid_strings_arr)
        data_colors, model_colors = _publication_protocol_colors(
            len(unique_curves), branch_label
        )
        legend_handles = []
        legend_labels = []
        model_traces = []

        for curve_i, u in enumerate(unique_curves):
            mask = (cid_strings_arr == u)
            if mask.sum() == 0:
                continue
            if curve_color_mode == 'protocol':
                data_color = data_colors[curve_i]
                model_color = model_colors[curve_i]
            else:
                data_color = 'black'
                model_color = '#d6224d'

            # Experimental points: sort only for cleaner CSV output; do not connect.
            lam_data = X[mask, int(stretch_idx)]
            order = np.argsort(lam_data)
            X_data_sorted = X[mask][order]
            y_sorted = y_data[mask][order]
            ax.plot(X_data_sorted[:, int(stretch_idx)], y_sorted,
                    linestyle='None', marker='o',
                    markersize=data_marker_size,
                    markerfacecolor=data_color, markeredgecolor=data_color,
                    alpha=0.85)
            for row_i in range(X_data_sorted.shape[0]):
                csv_rows.append({
                    'row_type': 'data',
                    'branch': branch_label,
                    'curve_id': str(u),
                    'lambda_theta': float(X_data_sorted[row_i, 0]),
                    'lambda_z': float(X_data_sorted[row_i, 1]),
                    'sigma_exp_kPa': float(y_sorted[row_i]),
                    'sigma_model_kPa': np.nan,
                })

            # Smooth model curve, one per protocol.
            X_curve = X[mask]
            if theta_or_z == 'theta':
                # PD: sweep lambda_theta and hold lambda_z at the protocol mean.
                lt = X_curve[:, 0]
                if np.nanmax(lt) > np.nanmin(lt):
                    lt_grid = np.linspace(np.nanmin(lt), np.nanmax(lt), int(n_smooth))
                else:
                    lt_grid = np.full(int(n_smooth), float(lt[0]))
                lz_grid = np.full(int(n_smooth), float(np.nanmean(X_curve[:, 1])))
                X_smooth = np.column_stack([lt_grid, lz_grid]).astype(np.float32)
                plot_x = lt_grid.astype(np.float32)
            else:
                # FL: sweep lambda_z and smooth the experimental lambda_theta(lambda_z)
                # coupling within this protocol.
                lz_grid, lt_grid = _smooth_companion(
                    primary=X_curve[:, 1], companion=X_curve[:, 0], n_grid=n_smooth
                )
                X_smooth = np.column_stack([lt_grid, lz_grid]).astype(np.float32)
                plot_x = lz_grid.astype(np.float32)

            bf = np.full((X_smooth.shape[0], 1), float(branch_flag), dtype=np.float32)
            if curve_vocabulary is not None:
                cid_int, _ = encode_curve_ids(np.array([str(u)] * X_smooth.shape[0], dtype=object),
                                              curve_vocabulary)
            else:
                cid_int = np.zeros((X_smooth.shape[0],), dtype=np.int32)
            preds_th, preds_z = model_nn.predict(
                [X_smooth[:, 0:1], X_smooth[:, 1:2],
                 bf, cid_int.reshape(-1, 1).astype(np.int32)],
                verbose=0,
            )
            preds = (preds_th if theta_or_z == 'theta' else preds_z).reshape(-1)
            preds = preds * float(model_max)

            model_traces.append((plot_x, preds, model_color))

            if show_legend:
                data_handle = Line2D(
                    [0], [0], linestyle='None', marker='o',
                    markersize=data_marker_size,
                    markerfacecolor=data_color, markeredgecolor=data_color,
                    alpha=0.85,
                )
                model_handle = Line2D(
                    [0], [0], linestyle='--', color=model_color,
                    linewidth=model_linewidth,
                )
                legend_handles.append((data_handle, model_handle))
                legend_labels.append(
                    _publication_legend_label(u, branch_label, animal_name)
                )

            for row_i in range(X_smooth.shape[0]):
                csv_rows.append({
                    'row_type': 'model_smooth',
                    'branch': branch_label,
                    'curve_id': str(u),
                    'lambda_theta': float(X_smooth[row_i, 0]),
                    'lambda_z': float(X_smooth[row_i, 1]),
                    'sigma_exp_kPa': np.nan,
                    'sigma_model_kPa': float(preds[row_i]),
                })

        for plot_x, preds, model_color in model_traces:
            ax.plot(plot_x, preds,
                    linestyle='--', color=model_color,
                    linewidth=model_linewidth,
                    zorder=4)

        if show_legend and legend_handles:
            ax.legend(
                legend_handles, legend_labels,
                handler_map={tuple: HandlerTuple(ndivide=None)},
                loc=legend_loc,
                fontsize=max(8, int(tick_fontsize * 0.55)),
                frameon=True,
                framealpha=0.78,
                facecolor='white',
                edgecolor='none',
                borderpad=0.35,
                handlelength=2.3,
                handletextpad=0.6,
                labelspacing=0.35,
            )

    # PD panel
    _plot_branch(ax_pd, X_pd, y_theta, cid_pd_strings,
                 branch_flag=0.0, stretch_idx=0,
                 model_max=getattr(model_nn, 'theta_max', 1.0),
                 theta_or_z='theta', branch_label='PD')
    # FL panel
    _plot_branch(ax_fl, X_fl, y_z, cid_fl_strings,
                 branch_flag=1.0, stretch_idx=1,
                 model_max=getattr(model_nn, 'z_max', 1.0),
                 theta_or_z='z', branch_label='FL')

    # Axis labels (mathtext)
    ax_pd.set_xlabel(r'$\lambda_{\theta}$', fontsize=label_fontsize)
    ax_pd.set_ylabel(r'$\sigma_{\theta}$ [kPa]', fontsize=label_fontsize)
    ax_fl.set_xlabel(r'$\lambda_{z}$', fontsize=label_fontsize)
    ax_fl.set_ylabel(r'$\sigma_{z}$ [kPa]', fontsize=label_fontsize)

    # 3-tick locator and tick label size on both panels.
    for ax in (ax_pd, ax_fl):
        ax.xaxis.set_major_locator(MaxNLocator(int(n_ticks)))
        ax.yaxis.set_major_locator(MaxNLocator(int(n_ticks)))
        ax.tick_params(axis='both', labelsize=tick_fontsize, width=1.2, length=5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    # Title: <animal>: <R^2_theta>  <R^2_z>
    title_parts = []
    if animal_name:
        title_parts.append(str(animal_name))
    r2_pieces = []
    if r2_theta is not None and np.isfinite(r2_theta):
        r2_pieces.append(rf"$R^{{2}}_{{\theta}}={r2_theta:.3f}$")
    if r2_z is not None and np.isfinite(r2_z):
        r2_pieces.append(rf"$R^{{2}}_{{z}}={r2_z:.3f}$")
    if r2_overall is not None and np.isfinite(r2_overall):
        r2_pieces.append(rf"$R^{{2}}_{{overall}}={r2_overall:.3f}$")
    if r2_pieces:
        title_parts.append("   ".join(r2_pieces))
    full_title = "   ".join(title_parts)
    if full_title:
        fig.suptitle(full_title, fontsize=title_fontsize, y=0.98)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        pdf_path = os.path.join(save_path, f'{output_prefix}.pdf')
        png_path = os.path.join(save_path, f'{output_prefix}.png')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"Publication figure saved:")
        print(f"  {pdf_path}")
        print(f"  {png_path}")

        # Companion CSV with every plotted stress/strain point. Smooth model rows
        # are labeled separately from experimental data rows.
        if csv_rows:
            csv_path = os.path.join(save_path, f'{output_prefix}_data.csv')
            df_fig = pd.DataFrame(csv_rows,
                columns=['row_type', 'branch', 'curve_id', 'lambda_theta', 'lambda_z',
                         'sigma_exp_kPa', 'sigma_model_kPa'])
            df_fig.to_csv(csv_path, index=False)
            print(f"  {csv_path}")
    return fig


def plot_recruitment_curves(model, save_path=None, strain_range=(0, 0.5)):
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    recruitment_layer = None
    for layer in model.layers:
        if isinstance(layer, FiberRecruitmentLayer):
            recruitment_layer = layer
            break

    if recruitment_layer is None:
        print("No FiberRecruitmentLayer found in model!")
        return
    
    strain_min, strain_max = strain_range
    strains = np.linspace(strain_min, strain_max, 500)
    stretches = 1.0 + strains
    I4_grid = tf.constant((stretches**2).reshape(-1, 1), dtype=tf.float32)
    lb = recruitment_layer.lambda_lb().numpy()
    ub = recruitment_layer.lambda_ub().numpy()
    alpha_tf, beta_tf = recruitment_layer.beta_ab()
    alpha = alpha_tf.numpy()
    beta = beta_tf.numpy()
    tau = recruitment_layer.tau().numpy()
    
    fiber_families = ['f', 's', 'el', 'smc']
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    labels = ['Fiber family f', 'Fiber family s', 'Elastin', 'SMC']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (family, color, label) in enumerate(zip(fiber_families, colors, labels)):
        ax = axes[idx]
        recruited_I4 = recruitment_layer._beta_recruited_invariant(I4_grid, idx).numpy().reshape(-1)
        recruited_stretches = np.sqrt(np.maximum(recruited_I4, 0.0))
        
        ax.plot(strains * 100, recruited_stretches, color=color, linewidth=2.5, label=label)
        ax.plot(strains * 100, stretches, 'k--', linewidth=1.5, alpha=0.5, label='No recruitment (1:1)')
        
        lb_strain = (lb[idx] - 1.0) * 100
        ub_strain = (ub[idx] - 1.0) * 100
        ax.axvline(lb_strain, color=color, linestyle=':', linewidth=2, alpha=0.7,
                   label=f'λ_lb ({lb[idx]:.3f})')
        ax.axvline(ub_strain, color=color, linestyle='-.', linewidth=1.5, alpha=0.5,
                   label=f'λ_ub ({ub[idx]:.3f})')
        
        ax.set_xlabel('Applied Strain (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Recruited Stretch λ', fontsize=11, fontweight='bold')
        ax.set_title(f'{label}\n(α={alpha[idx]:.2f}, β={beta[idx]:.2f}, τ={tau[idx]:.3f})', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=9)
        ax.set_xlim(strain_min * 100, strain_max * 100)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'recruitment_curves.pdf'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_path, 'recruitment_curves.png'), dpi=300, bbox_inches='tight')
        print(f"Recruitment curves saved to {save_path}")
    
    plt.show()
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for idx, (family, color, label) in enumerate(zip(fiber_families, colors, labels)):
        recruited_I4 = recruitment_layer._beta_recruited_invariant(I4_grid, idx).numpy().reshape(-1)
        recruited_stretches = np.sqrt(np.maximum(recruited_I4, 0.0))
        
        recruitment_fraction = np.where(stretches > 1.0, 
                                       (recruited_stretches - 1.0) / (stretches - 1.0),
                                       0.0)
        
        ax.plot(strains * 100, recruitment_fraction, color=color, linewidth=2.5, 
                label=f'{label} (λ_lb={lb[idx]:.3f}, λ_ub={ub[idx]:.3f}, α={alpha[idx]:.2f}, β={beta[idx]:.2f})')
    
    ax.axhline(1.0, color='k', linestyle='--', linewidth=1.5, alpha=0.5, label='Full recruitment')
    ax.set_xlabel('Applied Strain (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Recruitment Fraction', fontsize=12, fontweight='bold')
    ax.set_title('Fiber Recruitment vs Strain (All Families)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    ax.set_xlim(strain_min * 100, strain_max * 100)
    ax.set_ylim(-0.05, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'recruitment_combined.pdf'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_path, 'recruitment_combined.png'), dpi=300, bbox_inches='tight')
    
    plt.show()
    
    print("\n" + "="*70)
    print("FIBER BETA RECRUITMENT SUMMARY")
    print("="*70)
    for idx, (family, label) in enumerate(zip(fiber_families, labels)):
        lb_strain = (lb[idx] - 1.0) * 100
        ub_strain = (ub[idx] - 1.0) * 100
        print(
            f"{label:20s}: λ_lb={lb[idx]:.4f} ({lb_strain:5.2f}% strain), "
            f"λ_ub={ub[idx]:.4f} ({ub_strain:5.2f}% strain), "
            f"α={alpha[idx]:.3f}, β={beta[idx]:.3f}, τ={tau[idx]:.4f}"
        )
    print("="*70 + "\n")

def save_model_predictions_for_plotting(model_nn, Fl, Pd, save_folder, prefix="CANN_model",
                                        curve_vocabulary=None):
    """
    Saves model predictions in the SAME STRUCTURE as Jay/Sokolis code outputs.
    This allows direct compatibility with your existing multi-model plotting script.
    """
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)

    grams_to_mN = 9.81

    required_cols = ["od", "Pressure_kPa", "Force_v_mN"]
    missing_pd = [c for c in required_cols if c not in Pd.columns]
    missing_fl = [c for c in required_cols if c not in Fl.columns]
    if missing_pd or missing_fl:
        print(
            "Skipping save_model_predictions_for_plotting: input lacks "
            f"load columns (Pd missing {missing_pd}, Fl missing {missing_fl})."
        )
        return

    os.makedirs(save_folder, exist_ok=True)

    # -----------------------
    #  PD: Circumferential tests
    # -----------------------
    X_pd = np.column_stack([Pd["lambda_theta"], Pd["lambda_z"]])
    bf_pd = np.zeros((X_pd.shape[0], 1), dtype=np.float32)
    if curve_vocabulary is not None and "test_case" in Pd.columns:
        cid_pd_int, _ = encode_curve_ids(Pd["test_case"].astype(str).values, curve_vocabulary)
        cid_pd = cid_pd_int.reshape(-1, 1).astype(np.int32)
    else:
        cid_pd = np.zeros((X_pd.shape[0], 1), dtype=np.int32)
    pred_theta_norm, _ = model_nn.predict(
        [X_pd[:,0:1], X_pd[:,1:2], bf_pd, cid_pd], verbose=0
    )
    pred_theta = pred_theta_norm.reshape(-1) * model_nn.theta_max

    # Save PD loads (pressure–force)
    df_pd_load = pd.DataFrame({
        "od_exp": Pd["od"],
        "Pi_exp_kPa": Pd["Pressure_kPa"],
        "Ft_exp": Pd["Force_v_mN"] / grams_to_mN,
        "od_th": Pd["od"],
        "Pi_th_kPa": Pd["Pressure_kPa"],
        "Ft_th": pred_theta / grams_to_mN
    })
    df_pd_load.to_excel(os.path.join(save_folder, f"{prefix}_loads_Pd.xlsx"),
                        index=False, header=False)

    # Save PD stress–stretch
    df_pd_stress = pd.DataFrame({
        "lambdaq_exp": Pd["lambda_theta"],
        "tqq_exp": Pd["sigma_theta"],
        "tzz_exp": np.zeros_like(Pd["sigma_theta"]),
        "lambdaq_th": Pd["lambda_theta"],
        "tqq_th": pred_theta,
        "tzz_th": np.zeros_like(pred_theta)
    })
    df_pd_stress.to_excel(os.path.join(save_folder, f"{prefix}_stress_Pd.xlsx"),
                          index=False, header=False)

    # -----------------------
    #  FL: Axial tests
    # -----------------------
    X_fl = np.column_stack([Fl["lambda_theta"], Fl["lambda_z"]])
    bf_fl = np.ones((X_fl.shape[0], 1), dtype=np.float32)
    if curve_vocabulary is not None and "test_case" in Fl.columns:
        cid_fl_int, _ = encode_curve_ids(Fl["test_case"].astype(str).values, curve_vocabulary)
        cid_fl = cid_fl_int.reshape(-1, 1).astype(np.int32)
    else:
        cid_fl = np.zeros((X_fl.shape[0], 1), dtype=np.int32)
    _, pred_z_norm = model_nn.predict(
        [X_fl[:,0:1], X_fl[:,1:2], bf_fl, cid_fl], verbose=0
    )
    pred_z = pred_z_norm.reshape(-1) * model_nn.z_max

    # Save FL loads (stretch–force)
    df_fl_load = pd.DataFrame({
        "od_exp": Fl["od"],
        "Ft_exp": Fl["Force_v_mN"] / grams_to_mN,
        "lambdaz_exp": Fl["lambda_z"],
        "od_th": Fl["od"],
        "Ft_th": pred_z / grams_to_mN,
        "lambdaz_th": Fl["lambda_z"]
    })
    df_fl_load.to_excel(os.path.join(save_folder, f"{prefix}_loads_Fl.xlsx"),
                        index=False, header=False)

    # Save FL stress–stretch
    df_fl_stress = pd.DataFrame({
        "lambdaz_exp": Fl["lambda_z"],
        "lambdaq_exp": Fl["lambda_theta"],
        "tzz_exp": Fl["sigma_z"],
        "lambdaz_th": Fl["lambda_z"],
        "lambdaq_th": Fl["lambda_theta"],
        "tzz_th": pred_z
    })
    df_fl_stress.to_excel(os.path.join(save_folder, f"{prefix}_stress_Fl.xlsx"),
                          index=False, header=False)

    print(f"\nModel prediction files saved to:\n  {save_folder}\n")


def save_final_equation_report(model_nn, Psi_model, outdir, threshold=0.001,
                               sample_name='', args=None, filename='final_equation.txt'):
    """
    Write a final-equation report for the unified invariant + strain network.
    A candidate survives when |W_ANN[row] * w_subnet| exceeds `threshold`.
    For normalized strain bases, this is the coefficient on the normalized basis;
    the raw Green-strain coefficient is also reported as eff_raw = eff_norm / scale.
    """
    os.makedirs(outdir, exist_ok=True)
    filepath = os.path.join(outdir, filename)

    psi_weights = Psi_model.get_weights()
    w_ann = psi_weights[-1].flatten()  # shape (total_terms,)
    poly_n_terms = getattr(args, 'poly_n_terms', 5) if args is not None else 5
    poly_include_linear = getattr(args, 'poly_include_linear', True) if args is not None else True
    strain_scales = getattr(Psi_model, 'strain_basis_scales', None)
    rows = single_network_feature_metadata(
        poly_n_terms=poly_n_terms,
        poly_include_linear=poly_include_linear,
        strain_basis_scales=strain_scales,
    )

    survivors = []
    for row in rows:
        idx = row['idx']
        sub_w = float(psi_weights[row['weight_idx']].flatten()[0])
        ann_w = float(w_ann[idx])
        scale = float(row.get('scale', 1.0))
        eff = sub_w * ann_w
        eff_raw = eff / scale
        if abs(eff) > threshold:
            survivor = dict(row)
            survivor.update({'w_sub': sub_w, 'w_ann': ann_w, 'eff': eff, 'eff_raw': eff_raw})
            survivors.append(survivor)

    survivors_inv = [s for s in survivors if s['group'] == 'invariant']
    survivors_strain = [s for s in survivors if s['group'] == 'strain']

    h_layer = next((l for l in model_nn.layers if isinstance(l, H_Layer_kappas)), None)
    recruit_layer = next((l for l in model_nn.layers if isinstance(l, FiberRecruitmentLayer)), None)

    with open(filepath, 'w') as f:
        title = f"FINAL EQUATION FOR {sample_name}" if sample_name else "FINAL EQUATION"
        f.write(title + "\n")
        f.write("=" * 60 + "\n\n")

        objective_label = getattr(args, 'objective', 'obj6') if args is not None else getattr(model_nn, 'objective', 'obj6')
        f.write("Model: vein 4-fiber + I8 + Green-strain single-network CANN (v3 obj6)\n")
        if objective_label == 'stress_hybrid_log':
            f.write("  objective      = hybrid normalized stress + log stress residuals\n")
        elif objective_label == 'stress':
            f.write("  objective      = direct normalized stress residuals\n")
        else:
            f.write("  objective      = obj6 reconstructed pressure/force loads\n")
        if args is not None:
            f.write(f"  use_recruitment = {args.use_recruitment}\n")
            f.write("  energy_topology = single shared wx2 over invariant + strain candidates\n")
            f.write(f"  strain_poly_n_terms = {args.poly_n_terms}\n")
            f.write(f"  strain_include_linear = {getattr(args, 'poly_include_linear', False)}\n")
            f.write(f"  strain_basis_scale = {getattr(args, 'strain_basis_scale', 'none')}\n")
            f.write(f"  deprecated use_poly / poly_trainable / poly_regularizer = "
                    f"{args.use_poly} / {args.poly_trainable} / {args.poly_regularizer}\n")
            f.write(f"  nonneg_weights  = {args.nonneg_weights}\n")
            f.write(f"  circ_weight     = {args.circ_weight}\n")
            f.write(f"  i8fs_init       = {args.i8fs_init}\n")
            f.write(f"  recruit_lb_min  = {args.recruit_lb_min}\n")
            f.write(f"  recruit_lb_max  = {args.recruit_lb_max}\n")
            f.write(f"  recruit_ub_max  = {args.recruit_ub_max}\n")
            f.write(f"  helical_mode    = {args.helical_mode}\n")
            f.write(f"  helix_angle_init_deg = {args.helix_angle_init_deg}\n")
            f.write(f"  helix_angle_min_deg  = {args.helix_angle_min_deg}\n")
            f.write(f"  helix_angle_max_deg  = {args.helix_angle_max_deg}\n")
            f.write(f"  train_fiber_angles   = {args.train_fiber_angles}\n")
            f.write(f"  reg / pen       = {args.reg} / {args.pen}\n")
        f.write(f"  survive threshold (|normalized eff weight|) = {threshold:g}\n\n")

        f.write("=" * 60 + "\n")
        f.write("SURVIVING SINGLE-NETWORK CANDIDATES\n")
        f.write("=" * 60 + "\n")
        if not survivors:
            f.write("  (none above threshold)\n")
        else:
            f.write(f"  {'group':<10} {'basis':<10} {'term':<16} {'idx':>4}  "
                    f"{'scale':>11} {'w_sub':>13} {'W_ANN':>13} {'eff_norm':>13} {'eff_raw':>13}\n")
            f.write("  " + "-" * 119 + "\n")
            for s in survivors:
                scale = float(s.get('scale', 1.0))
                f.write(f"  {s['group']:<10} {s['inv']:<10} {s['term']:<16} {s['idx']:>4}  "
                        f"{scale:>11.4e} {s['w_sub']:>13.6e} {s['w_ann']:>13.6e} "
                        f"{s['eff']:>13.6e} {s['eff_raw']:>13.6e}\n")
        f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("SURVIVING INVARIANT TERMS\n")
        f.write("=" * 60 + "\n")
        if not survivors_inv:
            f.write("  (none above threshold)\n")
        else:
            for s in survivors_inv:
                f.write(f"  {s['inv']:<6} {s['term']:<16} eff_norm={s['eff']:>13.6e}\n")
        f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("SURVIVING GREEN-STRAIN TERMS\n")
        f.write("=" * 60 + "\n")
        if not survivors_strain:
            f.write("  (none above threshold)\n")
        else:
            for s in survivors_strain:
                f.write(f"  {s['term']:<16} eff_norm={s['eff']:>13.6e} "
                        f"eff_raw={s['eff_raw']:>13.6e} scale={float(s.get('scale', 1.0)):>11.4e}\n")
        f.write("\n")

        if h_layer is not None:
            f.write("=" * 60 + "\n")
            f.write("FIBER DIRECTION + DISPERSION\n")
            f.write("=" * 60 + "\n")
            phi_f, phi_s = h_layer.helical_dirs_numpy()
            f.write(f"  kappa_f      : {float(h_layer.kappa_f.numpy()):.6f}\n")
            f.write(f"  kappa_s      : {float(h_layer.kappa_s.numpy()):.6f}\n")
            f.write(f"  phi_f (deg)  : {phi_f * 180/np.pi:.3f}\n")
            f.write(f"  phi_s (deg)  : {phi_s * 180/np.pi:.3f}\n")
            f.write(f"  phi_el (deg) : {float(h_layer.elas_dir.numpy()) * 180/np.pi:.3f}\n")
            f.write(f"  phi_smc (deg): {float(h_layer.smc_dir.numpy()) * 180/np.pi:.3f}\n\n")

        if recruit_layer is not None:
            f.write("=" * 60 + "\n")
            f.write("FIBER RECRUITMENT (Beta slack-stretch)\n")
            f.write("=" * 60 + "\n")
            lb = recruit_layer.lambda_lb().numpy()
            ub = recruit_layer.lambda_ub().numpy()
            alpha, beta = recruit_layer.beta_ab()
            tau = recruit_layer.tau().numpy()
            alpha = alpha.numpy()
            beta = beta.numpy()
            for i, family in enumerate(recruit_layer.family_names):
                f.write(f"  {family:>4s}: lb={lb[i]:.4f} ub={ub[i]:.4f} "
                        f"alpha={alpha[i]:.4f} beta={beta[i]:.4f} tau={tau[i]:.4f}\n")
            f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("FINAL EQUATION\n")
        f.write("=" * 60 + "\n")
        f.write("W_total = Psi_single(invariants, E_θ, E_z)\n\n")
        f.write("  W_total = Σ_j (W_ANN[j] · w_j) · normalized_basis_j\n")
        f.write("  For Green-strain terms: normalized_basis_j = raw_basis_j / scale_j.\n")
        f.write("  All invariant and Green-strain candidates share one wx2 layer.\n\n")

        inv_groups = {}
        for s in survivors_inv:
            inv_groups.setdefault(s['inv'], []).append(f"{s['term']}={s['eff']:.4e}")
        if inv_groups:
            f.write("Surviving invariant contributions:\n")
            for inv, terms in inv_groups.items():
                f.write(f"  {inv}: " + ", ".join(terms) + "\n")
        else:
            f.write("Surviving invariant contributions: (none)\n")
        f.write("\n")
        if survivors_strain:
            strain_set = ", ".join(s['term'] for s in survivors_strain)
            f.write(f"Surviving Green-strain basis: {{{strain_set}}}\n")
        else:
            f.write("Surviving Green-strain basis: {∅}\n")

    print(f"Final equation report saved: {filepath}")
    return filepath


#%% MAIN EXECUTION

def makeDIR(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _print_existing_columns(label, df, preferred_cols):
    cols = [c for c in preferred_cols if c in df.columns]
    print(f"\nSample {label} data:")
    print(df[cols].head() if cols else df.head())


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("on", "true", "1", "yes", "y"):
        return True
    if s in ("off", "false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def parse_args(argv=None):
    """Hyperparameters for the v3 obj6 vein 4-fiber CANN run."""
    p = argparse.ArgumentParser(
        description="Vein 4-fiber CANN with one shared invariant + Green-strain network (v3 obj6 reconstructed loads).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- I/O ---
    p.add_argument("--sample_file", default=SAMPLE_FILE,
                   help="Excel workbook with FL and PD sheets (the only supported input format).")
    p.add_argument("--path_root",
                   default="./Vein_i8_v3_results/results_single_sample_i8_obj6_single_network_Lexus/",
                   help="Root directory for run outputs.")
    p.add_argument("--filename", default="VeinCANN_combined_SingleSample",
                   help="Run sub-folder name.")
    p.add_argument("--folder_name", default=None,
                   help="Inner folder name (defaults to autogenerated with epochs/batch/pen).")

    # --- Training ---
    p.add_argument("--no_train", action="store_true",
                   help="Skip training (kept for symmetry; v3 only supports train=True).")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--reg", choices=["L1", "L2"], default="L1")
    p.add_argument("--pen", type=float, default=1e-5,
                   help="Regularization penalty.")
    p.add_argument("--objective", choices=["obj6", "stress", "stress_hybrid_log"], default="stress",
                   help="Training objective: reconstructed pressure/force loads, direct normalized stress, or hybrid stress+log stress.")
    p.add_argument("--log_stress_scale", type=float, default=1.0,
                   help="Physical kPa scale s0 used by stress_hybrid_log: log1p(stress / s0).")
    p.add_argument("--log_loss_weight", type=float, default=0.5,
                   help="Fraction of stress_hybrid_log loss assigned to the log-stress term.")
    p.add_argument("--circ_weight", type=float, default=3.5,
                   help="Loss weight on the circumferential (PD) samples vs. axial (FL).")
    p.add_argument("--tail_weight", type=float, default=0.0,
                   help="Extra sample-weight multiplier for high-stress/high-stretch elbow/tail points.")
    p.add_argument("--tail_stress_frac", type=float, default=0.0,
                   help="Activate tail weighting once |stress|/max_stress exceeds this fraction (0 disables).")
    p.add_argument("--theta_tail_stretch", type=float, default=0.0,
                   help="Activate circumferential tail weighting once lambda_theta exceeds this value (0 disables).")
    p.add_argument("--z_tail_stretch", type=float, default=0.0,
                   help="Activate axial tail weighting once lambda_z exceeds this value (0 disables).")
    p.add_argument("--tail_width", type=float, default=0.05,
                   help="Sigmoid width for smooth tail/elbow sample weighting.")
    p.add_argument("--lr", type=float, default=0.01,
                   help="Adam initial learning rate.")
    p.add_argument("--lr_schedule", choices=["none", "plateau", "exponential", "cosine"],
                   default="plateau",
                   help="LR schedule: 'plateau' (ReduceLROnPlateau, adaptive), "
                        "'exponential' (decays each step), 'cosine' (CosineDecay from lr to lr_min), "
                        "'none' (constant lr).")
    p.add_argument("--lr_decay", type=float, default=0.2,
                   help="Factor used by plateau / exponential schedules "
                        "(plateau: new_lr = lr * factor; exponential: decay_rate).")
    p.add_argument("--lr_patience", type=int, default=200,
                   help="Epochs without improvement before plateau drops the LR.")
    p.add_argument("--lr_min", type=float, default=1e-5,
                   help="Lower bound on the learning rate (plateau / cosine).")

    # --- Excel protocol splits ---
    p.add_argument("--pd_train_protocols", nargs="+",
                   default=["invivo-5%", "invivo", "invivo+5%"],
                   help="Test cases (PD) used for training.")
    p.add_argument("--fl_train_protocols", nargs="+",
                   default=["p10", "p60", "p100", "p140"],
                   help="Test cases (FL) used for training.")

    # --- Fiber angle knobs ---
    p.add_argument("--helical_mode", choices=["independent", "symmetric"], default="independent",
                   help="Fiber angle parameterization. 'symmetric' uses one bounded angle pair +/- alpha.")
    p.add_argument("--helix_angle_init_deg", type=float, default=35.0,
                   help="Initial helical fiber angle magnitude in degrees.")
    p.add_argument("--helix_angle_min_deg", type=float, default=5.0,
                   help="Minimum helical fiber angle magnitude in degrees for symmetric mode.")
    p.add_argument("--helix_angle_max_deg", type=float, default=85.0,
                   help="Maximum helical fiber angle magnitude in degrees for symmetric mode.")
    p.add_argument("--train_fiber_angles", type=_str2bool, default=True,
                   help="Allow helical fiber angle parameters to train (on/off).")

    # --- Fiber recruitment ---
    # Preset vs custom: --recruit_preset (early/medium/late) overwrites
    # --init_threshold and --recruit_lb_min/_max/_ub_max at runtime.
    # Use --recruit_preset custom to control those values directly via CLI.
    p.add_argument("--recruit_preset", choices=["early", "medium", "late", "custom"],
                   default="medium",
                   help="Shorthand to set init_threshold + lb/ub bounds together. "
                        "early=(1.05, 0.90-1.35, ub=1.60), medium=(1.20, 0.90-1.60, ub=1.90), "
                        "late=(1.40, 1.00-1.90, ub=2.30). Overrides --init_threshold and "
                        "--recruit_lb_min/_max/_ub_max unless set to 'custom'.")
    p.add_argument("--init_threshold", type=float, default=1.4,
                   help="Initial lambda_lb for FiberRecruitmentLayer. Only used when "
                        "--recruit_preset custom; ignored otherwise (preset overwrites it).")
    p.add_argument("--init_sharpness", type=float, default=5.0,
                   help="Initial sharpness (tau) of the softplus recruitment transition.")
    p.add_argument("--recruit_lb_min", type=float, default=0.90,
                   help="Hard lower bound for learned lambda_lb. Only used when "
                        "--recruit_preset custom; ignored otherwise.")
    p.add_argument("--recruit_lb_max", type=float, default=2.0,
                   help="Hard upper bound for learned lambda_lb. Only used when "
                        "--recruit_preset custom; ignored otherwise.")
    p.add_argument("--recruit_ub_max", type=float, default=2.3,
                   help="Hard upper bound for learned lambda_ub. Only used when "
                        "--recruit_preset custom; ignored otherwise.")
    p.add_argument("--use_recruitment", type=_str2bool, default=True,
                   help="Enable the FiberRecruitmentLayer (on/off).")
    p.add_argument("--recruit_dist", choices=["beta", "lognormal"], default="lognormal",
                   help="Slack-stretch distribution across the fiber population: "
                        "beta (compact support, flexible shape) or "
                        "shifted truncated lognormal (right-skewed, harder floor at lambda_lb).")
    p.add_argument("--recruit_shift", choices=["on", "off"], default="on",
                   help="Enable branch-aware lambda_lb shift driven by the held "
                        "companion stretch (PD samples shift with lambda_z, FL with "
                        "lambda_theta).")
    p.add_argument("--shift_mode", choices=["direct", "inverse"], default="inverse",
                   help="When --recruit_shift on: 'direct' = lambda_lb * "
                        "companion/ref (higher companion → later recruitment), 'inverse' = lambda_lb "
                        "* ref/companion (higher companion → earlier recruitment).")
    p.add_argument("--per_curve_recruit", choices=["on", "off"], default="on",
                   help="Enable per-curve learned bounded lambda_lb offsets.")
    p.add_argument("--curve_offset_max", type=float, default=0.30,
                   help="Max |delta lambda_lb| per curve when --per_curve_recruit on.")
    p.add_argument("--curve_offset_l2", type=float, default=5e-4,
                   help="L2 regularizer on the raw per-curve offset weights.")

    # --- Strain energy network knobs ---
    p.add_argument("--use_poly", type=_str2bool, default=True,
                   help="Deprecated compatibility flag. Ignored: Green-strain candidates are always part of the single shared network.")
    p.add_argument("--poly_n_terms", type=int, choices=[5, 7, 10], default=7,
                   help="Number of polynomial basis terms.")
    p.add_argument("--poly_trainable", type=_str2bool, default=True,
                   help="Deprecated compatibility flag. Ignored: strain candidates are always trainable through the shared network.")
    p.add_argument("--poly_regularizer", type=float, default=0.001,
                   help="Deprecated compatibility flag. Ignored: use --reg/--pen for shared candidate regularization.")
    p.add_argument("--poly_include_linear", type=_str2bool, default=True,
                   help="Add linear E_theta and E_z basis terms to the polynomial layer (on/off). "
                        "Linear I4f/I4s/I4el/I4smc terms already exist as the first weight per family "
                        "inside each SingleInvNet, so they are always trainable.")
    p.add_argument("--strain_basis_scale", choices=["auto", "none"], default="auto",
                   help="Scale Green-strain basis columns before the shared wx2 layer. "
                        "'auto' divides each strain candidate by its training max-abs value.")
    p.add_argument("--strain_scale_floor", type=float, default=1e-6,
                   help="Minimum denominator for auto strain-basis scaling.")
    p.add_argument("--nonneg_weights", type=_str2bool, default=True,
                   help="Constrain invariant-subnet, I8fs, W_ANN, and polynomial weights to be >= 0 (on/off).")
    p.add_argument("--i8fs_init", type=float, default=0.1,
                   help="Initial value for I8fs quadratic and exp-quadratic cross-fiber terms.")

    # --- Reporting / misc ---
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="Random seed for repeatable initialization.")
    p.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1,
                   help="Keras fit verbosity: 0=silent, 1=progress bar, 2=one line per epoch.")
    p.add_argument("--equation_thresh", type=float, default=0.001,
                   help="|effective weight| threshold for listing a term as surviving in the final-equation report.")
    p.add_argument("--report_label", default=None,
                   help="Label used in the final-equation report title (defaults to the data source folder/file name).")

    # --- Loss / weighting ---
    p.add_argument("--curve_weighting", choices=["equal", "point"], default="point",
                   help="'equal' = each physical PD/FL curve contributes equally; "
                        "'point' = each sample contributes equally (v3 default).")
    p.add_argument("--robust", choices=["mse", "huber", "charbonnier"], default="mse",
                   help="Residual kernel for the stress/obj6 objective.")
    p.add_argument("--robust_delta", type=float, default=0.5,
                   help="Huber transition / Charbonnier eps stand-in.")
    p.add_argument("--theta_tail_gain", type=float, default=2.0,
                   help="Polynomial tail gain on PD residuals: w *= 1 + gain * "
                        "max(lambda_theta - theta_tail_stretch, 0)^power.")
    p.add_argument("--theta_tail_power", type=float, default=4.0)
    p.add_argument("--theta_tail_stretch_v4", type=float, default=None,
                   help="Deprecated alias for the v4 polynomial PD tail onset. "
                        "If set, overrides --theta_tail_stretch for the "
                        "polynomial tail only.")
    p.add_argument("--z_tail_gain", type=float, default=1.5)
    p.add_argument("--z_tail_power", type=float, default=4.0)
    p.add_argument("--z_tail_stretch_v4", type=float, default=1.40,
                   help="Deprecated alias for the v4 polynomial FL tail onset. "
                        "If set, overrides --z_tail_stretch for the polynomial "
                        "tail only.")

    p.add_argument("--training_mode", choices=["normal", "curriculum",
                                               "curriculum_prune_refit"],
                   default="normal",
                   help="Training strategy. curriculum: low-order terms only "
                        "first, then full library. curriculum_prune_refit: also "
                        "prunes weak terms and refits.")
    p.add_argument("--epochs_phase1", type=int, default=1500)
    p.add_argument("--epochs_phase2", type=int, default=3000)
    p.add_argument("--epochs_refit", type=int, default=1000)
    p.add_argument("--prune_threshold", type=float, default=1e-3,
                   help="Effective |w_sub * w_ann| (scaled by mean |basis|) below "
                        "which a candidate is zeroed at the start of phase 3.")
    p.add_argument("--early_stop_patience", type=int, default=1000,
                   help="Epochs of no improvement before EarlyStopping restores "
                        "best weights.")

    p.add_argument("--n_restarts", type=int, default=1,
                   help="Number of independent restarts (best by val_loss when "
                        "available, otherwise by training weighted loss).")

    # ---------------------------------------------------------------------------
    # Fine-tuning / warm start
    # ---------------------------------------------------------------------------
    # Two-stage workflow: pre-train on smoothed/preprocessed data to find the
    # symbolic equation form, then fine-tune on the original noisy measurements
    # with --init_from <weights.h5> --freeze_pretrained on. The smoothed-data
    # equation is preserved; only recruitment + per-curve offsets + helix angles
    # adapt. See the README/comments at the top of this file for the typical
    # invocation pattern.
    p.add_argument("--init_from", default=None,
                   help="Path to a final_weights.h5 file produced by a prior run. "
                        "Loaded before training; enables warm-start fine-tuning.")
    p.add_argument("--freeze_pretrained", choices=["on", "off"], default="off",
                   help="When --init_from is set: freeze the invariant subnet "
                        "weights (w*_*) and the final wx2 layer. Recruitment, "
                        "per-curve offsets, helix angles, and kappa stay trainable.")
    p.add_argument("--finetune", choices=["on", "off"], default="off",
                   help="Convenience flag: when on, sets sensible fine-tune "
                        "defaults (lr=1e-3, n_restarts=1, training_mode=normal, "
                        "epochs=500) unless the user explicitly overrode them.")

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args = apply_recruit_preset(args)

    # --finetune on: apply sane defaults for warm-start fine-tuning. Only
    # override values that match the parser default (so an explicit user
    # value always wins).
    if getattr(args, "finetune", "off") == "on":
        finetune_overrides = []
        if abs(args.lr - 0.01) < 1e-12:
            args.lr = 1e-3
            finetune_overrides.append("lr 0.01 -> 1e-3")
        if args.n_restarts == 1:
            pass  # already 1; no change
        elif args.n_restarts == 5:
            args.n_restarts = 1
            finetune_overrides.append("n_restarts 5 -> 1")
        if args.training_mode == "curriculum_prune_refit":
            args.training_mode = "normal"
            finetune_overrides.append("training_mode curriculum_prune_refit -> normal")
        if args.epochs == 1000:
            args.epochs = 500
            finetune_overrides.append("epochs 1000 -> 500")
        if finetune_overrides:
            print(f"[finetune] overrides: {', '.join(finetune_overrides)}")
        if not args.init_from:
            print("[finetune] WARNING: --finetune on but --init_from is not set. "
                  "This is just a low-LR random-init run, not a warm-start.")

    set_random_seed(args.seed)

    train = not args.no_train
    epochs = args.epochs
    batch_size = args.batch_size
    reg = args.reg
    pen = args.pen
    circ_weight = args.circ_weight
    modelFit_mode_all = ["all"]

    pd_train_protocols = list(args.pd_train_protocols)
    fl_train_protocols = list(args.fl_train_protocols)

    filename = args.filename
    folder_name = args.folder_name or (
        f"single_network_obj6_train_{epochs}_{batch_size}_{reg}{pen}_"
        f"poly{args.poly_n_terms}_scale-{args.strain_basis_scale}_all_protocols_train"
    )
    path_root = args.path_root
    path2saveResults_0 = os.path.join(path_root, filename, folder_name)
    makeDIR(path2saveResults_0)

    print("="*60)
    print("RUN HYPERPARAMETERS")
    print("="*60)
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print(f"  output_dir: {path2saveResults_0}")
    print("="*60 + "\n")

    # --- Load Excel measurements and calculate mid-wall stretches ---
    Fl, Pd = processInput(args.sample_file)
    getStressStretch(Pd, Fl)

    # obj6 needs OD/H/L geometry to reconstruct pressure and force; skip the
    # inference (and the obj6 objective itself) when geometry is missing. For
    # the stress / stress_hybrid_log objectives obj6_geom is unused.
    obj6_geom = None
    try:
        obj6_geom = infer_obj6_reference_geometry(Pd, Fl)
    except ValueError as exc:
        if args.objective == "obj6":
            raise
        print(f"\n[obj6 geometry inference skipped: {exc}]")

    print("\nSample Pd data:")
    _print_existing_columns("Pd", Pd, ["IR", "ir", "H", "h", "lambda_theta", "lambda_z", "sigma_theta", "test_case"])
    _print_existing_columns("Fl", Fl, ["IR", "ir", "H", "h", "lambda_theta", "lambda_z", "sigma_z", "test_case"])

    # Plot raw data before training
    print("\n" + "="*60)
    print("PLOTTING RAW DATA")
    print("="*60)
    plot_raw_data(Fl, Pd, save_path=path2saveResults_0)

    # Split into train/validation (v4: also returns per-sample curve ids)
    (X_train_pd, y_train_theta, X_train_fl, y_train_z,
     X_val_pd, y_val_theta, X_val_fl, y_val_z,
     cid_train_pd, cid_val_pd, cid_train_fl, cid_val_fl) = split_train_val(
        Fl, Pd,
        pd_train=pd_train_protocols,
        fl_train=fl_train_protocols,
    )

    # Build vocabulary spanning train + validation so curve_ids are stable.
    all_cids = sorted(set(
        list(map(str, cid_train_pd)) + list(map(str, cid_val_pd)) +
        list(map(str, cid_train_fl)) + list(map(str, cid_val_fl))
    ))
    curve_vocabulary = all_cids if all_cids else ["default"]
    print(f"\nCurve vocabulary ({len(curve_vocabulary)}): {curve_vocabulary}")

    # ref_lambda_z / ref_lambda_theta = training-data medians, used by the
    # branch-aware shift. Falls back to 1.0 when the corresponding branch is empty.
    ref_lambda_z_val = float(np.median(X_train_pd[:, 1])) if X_train_pd.size else 1.0
    ref_lambda_th_val = float(np.median(X_train_fl[:, 0])) if X_train_fl.size else 1.0
    print(f"Branch-shift reference stretches: ref_lambda_z={ref_lambda_z_val:.4f} "
          f"(PD median), ref_lambda_theta={ref_lambda_th_val:.4f} (FL median)")
    strain_basis_scales = compute_strain_basis_scales(
        X_train_pd,
        X_train_fl,
        n_terms=args.poly_n_terms,
        include_linear=args.poly_include_linear,
        mode=args.strain_basis_scale,
        floor=args.strain_scale_floor,
    )
    print("\nSingle-network strain basis scaling:")
    print(f"  mode: {args.strain_basis_scale}")
    print("  scales: " + ", ".join(f"{v:.6g}" for v in strain_basis_scales))

    # Training Loop
    """
    CHOOSING NUMBER OF TERMS:
    ------------------------
    poly_n_terms = 5:  E_θ², E_z², E_θ·E_z, E_θ³, E_z³
                       (Basic - recommended to start)
    poly_n_terms = 7:  + E_θ⁴, E_z⁴
                       (Adds quartic terms for more flexibility)
    poly_n_terms = 10: + E_θ²·E_z, E_θ·E_z², E_θ²·E_z²
                       (Full set - only if simpler models fail)
    """

    model_nn = None
    Psi_model = None
    path2saveResults = path2saveResults_0
    _v4_aic_bic = None  # set inside the loop after CSV outputs

    for count, modelFit_mode in enumerate(modelFit_mode_all, 1):
        print('='*40)
        print(f"Run {count} / {len(modelFit_mode_all)}")
        print(f"Training mode: {modelFit_mode}")
        print('='*40)

        path2saveResults = os.path.join(path2saveResults_0, modelFit_mode)
        makeDIR(path2saveResults)

        # Build model (v4: pass curve vocabulary + ref stretches + recruitment flags)
        def _build_model():
            Psi_model_local, _terms = StrainEnergyCANN_single_network(
                reg, pen,
                nonneg=args.nonneg_weights,
                poly_n_terms=args.poly_n_terms,
                poly_include_linear=args.poly_include_linear,
                strain_basis_scales=strain_basis_scales,
                strain_basis_scale_mode=args.strain_basis_scale,
            )
            model_local, Psi_model_local = model_architecture_with_recruitment_and_poly(
                Psi_model_local,
                init_threshold=args.init_threshold,
                init_sharpness=args.init_sharpness,
                recruit_lb_min=args.recruit_lb_min,
                recruit_lb_max=args.recruit_lb_max,
                recruit_ub_max=args.recruit_ub_max,
                helical_mode=args.helical_mode,
                helix_angle_init_deg=args.helix_angle_init_deg,
                helix_angle_min_deg=args.helix_angle_min_deg,
                helix_angle_max_deg=args.helix_angle_max_deg,
                train_fiber_angles=args.train_fiber_angles,
                use_recruitment=args.use_recruitment,
                use_poly=args.use_poly,
                poly_n_terms=args.poly_n_terms,
                poly_trainable=args.poly_trainable,
                poly_regularizer=args.poly_regularizer,
                poly_nonneg=args.nonneg_weights,
                poly_include_linear=args.poly_include_linear,
                recruit_dist=args.recruit_dist,
                recruit_shift_mode=("off" if args.recruit_shift == "off" else args.shift_mode),
                ref_lambda_theta=ref_lambda_th_val,
                ref_lambda_z=ref_lambda_z_val,
                per_curve_offsets=(args.per_curve_recruit == "on"),
                n_curves=max(1, len(curve_vocabulary)),
                curve_offset_max=args.curve_offset_max,
                curve_offset_l2=args.curve_offset_l2,
            )
            initialize_i8fs_terms(Psi_model_local, args.i8fs_init)
            return model_local, Psi_model_local

        if not train:
            raise RuntimeError("train=False is disabled because checkpoint loading is intentionally removed.")

        print("Training on single sample data...")
        print(f"Training - Pd: {X_train_pd.shape[0]} points, Fl: {X_train_fl.shape[0]} points")
        print(f"Validation - Pd: {X_val_pd.shape[0]} points, Fl: {X_val_fl.shape[0]} points")

        # Restart loop: keep the best model by validation loss (or training
        # weighted loss when no validation is available).
        n_restarts = max(1, int(args.n_restarts))
        best_score = float('inf')
        best_pack = None
        for restart in range(n_restarts):
            seed_r = int(args.seed) + restart
            print(f"\n[restart {restart+1}/{n_restarts}] seed={seed_r}")
            set_random_seed(seed_r)
            model_nn, Psi_model = _build_model()

            # ----- v4 fine-tune: load pretrained weights + optional freeze -----
            if args.init_from:
                # Accept either the Keras 3 convention (final_weights.weights.h5)
                # or the legacy plain .h5. If the user pointed at the legacy name
                # but the new file exists, prefer the new one silently.
                resolved_init = args.init_from
                if not os.path.isfile(resolved_init):
                    alt = None
                    if resolved_init.endswith('.weights.h5'):
                        alt = resolved_init[:-len('.weights.h5')] + '.h5'
                    elif resolved_init.endswith('.h5'):
                        alt = resolved_init[:-len('.h5')] + '.weights.h5'
                    if alt and os.path.isfile(alt):
                        print(f"[init_from] {resolved_init} missing, falling back to {alt}")
                        resolved_init = alt
                if not os.path.isfile(resolved_init):
                    raise FileNotFoundError(
                        f"--init_from path does not exist: {args.init_from}"
                    )
                try:
                    model_nn.load_weights(resolved_init)
                    print(f"[init_from] loaded weights from {resolved_init}")
                except Exception as exc:
                    print(f"[init_from] FAILED to load: {exc}")
                    print("[init_from] hint: pretrained vocabulary may differ from "
                          "current run; the per-curve offset embedding shape is "
                          "(n_curves, 4). Check curve_vocabulary.json next to the "
                          "weights file.")
                    raise

                # Sanity check the curve vocabulary if it was saved alongside.
                vocab_sidecar = os.path.join(
                    os.path.dirname(resolved_init), "curve_vocabulary.json"
                )
                if os.path.isfile(vocab_sidecar):
                    try:
                        with open(vocab_sidecar, "r") as f:
                            pretrained_vocab = json.load(f)
                        if pretrained_vocab != curve_vocabulary:
                            print("[init_from] WARNING: pretrained vocabulary "
                                  f"differs from current run.")
                            print(f"  pretrained: {pretrained_vocab}")
                            print(f"  current:    {curve_vocabulary}")
                            if len(pretrained_vocab) == len(curve_vocabulary):
                                print("  same length -> offsets transferred slot-by-slot")
                            else:
                                print("  DIFFERENT length -> offsets transferred up to "
                                      "min length; extras will start at zero / random")
                    except Exception as exc:
                        print(f"[init_from] could not read vocab sidecar: {exc}")

                if args.freeze_pretrained == "on":
                    frozen, kept = 0, 0
                    frozen_names = []
                    for layer in model_nn.layers:
                        if not layer.weights:
                            continue
                        name = layer.name
                        # Invariant subnets are named w<id>_<term> (e.g. w0_1, w14_3).
                        # The final shared layer is wx2. Freeze both: that locks the
                        # symbolic equation form learned on the smoothed data.
                        if name == "wx2" or (name.startswith("w") and "_" in name):
                            layer.trainable = False
                            frozen += 1
                            frozen_names.append(name)
                        else:
                            kept += 1
                    print(f"[freeze_pretrained] frozen {frozen} layers, "
                          f"kept {kept} trainable (recruitment + helix + offsets)")
                    if frozen <= 0:
                        print("[freeze_pretrained] WARNING: no layers matched the "
                              "freeze pattern; nothing was frozen.")

                # Ensure curriculum freezing won't override our pretrained freeze.
                if args.freeze_pretrained == "on" and args.training_mode != "normal":
                    print(f"[freeze_pretrained] forcing training_mode "
                          f"{args.training_mode} -> normal so phase freezing does "
                          "not unfreeze the pretrained candidate library")
                    args.training_mode = "normal"

            model_nn, histories, importance = train_with_strategy(
                model_nn, Psi_model,
                X_train_pd, y_train_theta, X_train_fl, y_train_z,
                X_val_pd, y_val_theta, X_val_fl, y_val_z,
                obj6_geom, args,
                cid_train_pd, cid_val_pd, cid_train_fl, cid_val_fl,
                curve_vocabulary,
            )

            merged = merge_histories(histories)
            score_key = 'val_loss' if 'val_loss' in merged else 'loss'
            score = float(min(merged.get(score_key, [float('inf')])))
            print(f"[restart {restart+1}] best {score_key} = {score:.6g}")
            if score < best_score:
                best_score = score
                best_pack = (model_nn, Psi_model, merged, importance, seed_r)

        model_nn, Psi_model, merged_history, importance, best_seed = best_pack
        print(f"\nBest restart seed: {best_seed}, score={best_score:.6g}")

        has_validation = (len(X_val_pd) > 0 or len(X_val_fl) > 0)
        os.makedirs(path2saveResults, exist_ok=True)
        fig, axe = plt.subplots(figsize=[8, 6])
        plotLoss(axe, merged_history, has_validation=has_validation)
        plt.savefig(os.path.join(path2saveResults, f'Plot_loss_joint_{modelFit_mode}.pdf'))
        plt.show()

        save_weights_with_recruitment_and_poly(model_nn, Psi_model, path2saveResults, 'trained_weights.txt')

        # v4: also save Keras-format weights + curve vocabulary so this run can
        # be used as --init_from for a later fine-tune. The file contains every
        # layer weight needed to reconstruct the model; the vocabulary preserves
        # the curve_id -> protocol-name mapping for per-curve offsets.
        # Keras 3 / TF 2.16+ requires the .weights.h5 extension for save_weights;
        # earlier Keras accepted plain .h5. Try the new convention first, fall
        # back to legacy, and finally surface the real error if both fail.
        save_attempts = ['final_weights.weights.h5', 'final_weights.h5']
        save_err = None
        for fname in save_attempts:
            try:
                keras_weights_path = os.path.join(path2saveResults, fname)
                model_nn.save_weights(keras_weights_path)
                with open(os.path.join(path2saveResults, 'curve_vocabulary.json'), 'w') as vf:
                    json.dump(curve_vocabulary, vf, indent=2)
                print(f"\nSaved Keras checkpoint: {keras_weights_path}")
                print(f"Saved curve vocabulary: {os.path.join(path2saveResults, 'curve_vocabulary.json')}")
                print("  (use these as --init_from for warm-start fine-tuning)")
                save_err = None
                break
            except Exception as exc:
                save_err = exc
                continue
        if save_err is not None:
            print(f"[save_weights_h5] WARNING: could not save Keras checkpoint "
                  f"as any of {save_attempts}: {save_err}")

        # Final-equation report (which terms survived above the threshold)
        if args.report_label:
            report_label = args.report_label
        else:
            report_label = os.path.splitext(os.path.basename(args.sample_file))[0]
        save_final_equation_report(
            model_nn, Psi_model, path2saveResults,
            threshold=args.equation_thresh,
            sample_name=report_label,
            args=args,
            filename='final_equation.txt',
        )

        save_model_predictions_for_plotting(
            model_nn,
            Fl, Pd,
            save_folder=path2saveResults,
            prefix=f"CANNmodel_{filename}",
            curve_vocabulary=curve_vocabulary,
        )

        # ============================================================
        # v4 CSV / TXT outputs
        # ============================================================
        # learned_recruitment.csv -- per-family recruitment parameters
        rec_layer = next((l for l in model_nn.layers
                          if isinstance(l, FiberRecruitmentLayer)), None)
        if rec_layer is not None:
            lb = rec_layer.lambda_lb().numpy()
            ub = rec_layer.lambda_ub().numpy()
            alpha_t, beta_t = rec_layer.beta_ab()
            alpha = alpha_t.numpy(); beta = beta_t.numpy()
            tau = rec_layer.tau().numpy()
            rec_rows = []
            for i, family in enumerate(rec_layer.family_names):
                rec_rows.append(dict(
                    family=family, lambda_lb=float(lb[i]), lambda_ub=float(ub[i]),
                    alpha=float(alpha[i]), beta=float(beta[i]), tau=float(tau[i]),
                    dist_type=rec_layer.dist_type, shift_mode=rec_layer.shift_mode,
                    ref_lambda_theta=rec_layer.ref_lambda_theta,
                    ref_lambda_z=rec_layer.ref_lambda_z,
                ))
                if rec_layer.dist_type == 'lognormal':
                    rec_rows[-1]['logn_mu'] = float(rec_layer.logn_mu().numpy()[i])
                    rec_rows[-1]['logn_sigma'] = float(rec_layer.logn_sig().numpy()[i])
            pd.DataFrame(rec_rows).to_csv(
                os.path.join(path2saveResults, 'learned_recruitment.csv'), index=False)

            # learned_curve_offsets.csv -- per-curve, per-family delta_lb (only when
            # per_curve_recruit is on; otherwise the file is still written as an empty
            # table so downstream tooling has a stable path).
            offs = rec_layer.curve_offsets()
            offset_rows = []
            if offs is not None:
                offs_np = offs.numpy()
                for c_idx, c_name in enumerate(curve_vocabulary):
                    for f_idx, family in enumerate(rec_layer.family_names):
                        offset_rows.append(dict(
                            curve_id=str(c_name), family=family,
                            delta_lb=float(offs_np[c_idx, f_idx]),
                        ))
            pd.DataFrame(offset_rows).to_csv(
                os.path.join(path2saveResults, 'learned_curve_offsets.csv'),
                index=False)

        # term_importance.csv + active_terms.csv -- from prune-refit importance.
        if not importance:
            importance = _compute_term_importance(
                model_nn, Psi_model, X_train_pd, X_train_fl,
                curve_vocabulary=curve_vocabulary,
                cid_train_pd=cid_train_pd, cid_train_fl=cid_train_fl,
            )
        imp_df = pd.DataFrame(importance)
        imp_df.to_csv(os.path.join(path2saveResults, 'term_importance.csv'), index=False)
        active_mask = imp_df['importance'].abs() >= args.equation_thresh \
            if 'importance' in imp_df.columns else pd.Series([], dtype=bool)
        imp_df.loc[active_mask].to_csv(
            os.path.join(path2saveResults, 'active_terms.csv'), index=False)

        # per_curve_metrics.csv -- R2, RMSE per physical PD/FL protocol.
        per_curve_rows = []
        if X_train_pd.size:
            bf_pd = np.zeros((X_train_pd.shape[0], 1), dtype=np.float32)
            cid_pd_int, _ = encode_curve_ids(cid_train_pd, curve_vocabulary)
            preds_pd, _ = model_nn.predict(
                [X_train_pd[:,0:1], X_train_pd[:,1:2],
                 bf_pd, cid_pd_int.reshape(-1,1)], verbose=0)
            preds_pd = preds_pd.reshape(-1) * float(model_nn.theta_max)
            for c in np.unique(cid_train_pd):
                m = (cid_train_pd == c)
                if m.sum() == 0:
                    continue
                per_curve_rows.append(dict(
                    branch='PD', curve_id=str(c), n=int(m.sum()),
                    R2=float(r2_score(y_train_theta[m], preds_pd[m])) if m.sum() > 1 else float('nan'),
                    RMSE=float(np.sqrt(mean_squared_error(y_train_theta[m], preds_pd[m]))),
                ))
            shape_pd = shape_diagnostics_per_curve(
                X_train_pd, y_train_theta, preds_pd, cid_train_pd, stretch_index=0)
        else:
            shape_pd = []
        if X_train_fl.size:
            bf_fl = np.ones((X_train_fl.shape[0], 1), dtype=np.float32)
            cid_fl_int, _ = encode_curve_ids(cid_train_fl, curve_vocabulary)
            _, preds_fl = model_nn.predict(
                [X_train_fl[:,0:1], X_train_fl[:,1:2],
                 bf_fl, cid_fl_int.reshape(-1,1)], verbose=0)
            preds_fl = preds_fl.reshape(-1) * float(model_nn.z_max)
            for c in np.unique(cid_train_fl):
                m = (cid_train_fl == c)
                if m.sum() == 0:
                    continue
                per_curve_rows.append(dict(
                    branch='FL', curve_id=str(c), n=int(m.sum()),
                    R2=float(r2_score(y_train_z[m], preds_fl[m])) if m.sum() > 1 else float('nan'),
                    RMSE=float(np.sqrt(mean_squared_error(y_train_z[m], preds_fl[m]))),
                ))
            shape_fl = shape_diagnostics_per_curve(
                X_train_fl, y_train_z, preds_fl, cid_train_fl, stretch_index=1)
        else:
            shape_fl = []
        pd.DataFrame(per_curve_rows).to_csv(
            os.path.join(path2saveResults, 'per_curve_metrics.csv'), index=False)

        # shape_metrics.csv -- post-training shape diagnostics per curve.
        shape_rows = []
        for r in shape_pd:
            r2 = dict(r); r2['branch'] = 'PD'; shape_rows.append(r2)
        for r in shape_fl:
            r2 = dict(r); r2['branch'] = 'FL'; shape_rows.append(r2)
        pd.DataFrame(shape_rows).to_csv(
            os.path.join(path2saveResults, 'shape_metrics.csv'), index=False)

        # AIC/BIC -- joint, PD, FL. Saved both to its own CSV and folded into
        # the master results dict so it appears in metrics.txt + fit_metrics.csv.
        aic_bic = compute_aic_bic_v4(
            model_nn, Psi_model, importance, args, curve_vocabulary,
            X_train_pd, y_train_theta, cid_train_pd,
            X_train_fl, y_train_z, cid_train_fl,
        )
        pd.DataFrame([aic_bic]).to_csv(
            os.path.join(path2saveResults, 'aic_bic.csv'), index=False)
        print("\nAIC/BIC summary:")
        print(f"  k_total = {aic_bic['k_total']} "
              f"(terms={aic_bic['k_terms_active']}, recruit={aic_bic['k_recruit']}, "
              f"disp={aic_bic['k_disp']}, offsets={aic_bic['k_offsets']})")
        print(f"  pooled: AIC={aic_bic['AIC_pooled']:.2f}  BIC={aic_bic['BIC_pooled']:.2f}  "
              f"n={aic_bic['n_total']}")
        print(f"  PD:     AIC={aic_bic['AIC_PD']:.2f}  BIC={aic_bic['BIC_PD']:.2f}  "
              f"n={aic_bic['n_PD']}  RSS={aic_bic['RSS_PD']:.4g}")
        print(f"  FL:     AIC={aic_bic['AIC_FL']:.2f}  BIC={aic_bic['BIC_FL']:.2f}  "
              f"n={aic_bic['n_FL']}  RSS={aic_bic['RSS_FL']:.4g}")
        # Stash for the main metrics dict; collected below where `results` lives.
        _v4_aic_bic = aic_bic

    # Comprehensive evaluation and plotting
    print("\n" + "="*60)
    print("EVALUATING MODEL AND CREATING PLOTS")
    print("="*60)

    results = {}

    # Evaluate training set
    if len(X_train_pd) > 0 or len(X_train_fl) > 0:
        train_results = evaluate_on_dataset(
            model_nn, X_train_pd, X_train_fl, y_train_theta, y_train_z,
            dataset_name="Training", save_path=path2saveResults,
            cid_pd=cid_train_pd, cid_fl=cid_train_fl,
            curve_vocabulary=curve_vocabulary,
        )
        results.update({f"R2_train_{k.split('_')[1]}": v for k, v in train_results.items() if 'R2' in k})
        results.update({f"RMSE_train_{k.split('_')[1]}": v for k, v in train_results.items() if 'RMSE' in k})

    # Evaluate validation set
    if len(X_val_pd) > 0 or len(X_val_fl) > 0:
        val_results = evaluate_on_dataset(
            model_nn, X_val_pd, X_val_fl, y_val_theta, y_val_z,
            dataset_name="Validation", save_path=path2saveResults,
            cid_pd=cid_val_pd, cid_fl=cid_val_fl,
            curve_vocabulary=curve_vocabulary,
        )
        results.update({f"R2_val_{k.split('_')[1]}": v for k, v in val_results.items() if 'R2' in k})
        results.update({f"RMSE_val_{k.split('_')[1]}": v for k, v in val_results.items() if 'RMSE' in k})

        if 'RMS' in val_results:
            results['RMS_val'] = val_results['RMS']

    # Save metrics to file (always, not only when validation data exists)
    os.makedirs(path2saveResults, exist_ok=True)
    metrics_file = os.path.join(path2saveResults, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODEL PERFORMANCE METRICS\n")
        f.write("="*60 + "\n\n")
        f.write("OBJECTIVE:\n")
        if args.objective == "stress_hybrid_log":
            f.write("  hybrid normalized stress + log stress residuals\n")
            f.write(f"  log_stress_scale: {args.log_stress_scale} kPa\n")
            f.write(f"  log_loss_weight: {args.log_loss_weight}\n")
        elif args.objective == "stress":
            f.write("  direct normalized stress residuals: sigma_theta for PD, sigma_z for FL\n")
        else:
            f.write("  obj6 reconstructed loads: pressure residuals for PD, vessel-force residuals for FL\n")
            f.write("  Model outputs remain normalized stresses for plotting/report compatibility.\n")
        f.write("\n")

        f.write("HYPERPARAMETERS:\n")
        for k, v in sorted(vars(args).items()):
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write("TRAINING PROTOCOLS:\n")
        f.write(f"  PD: {pd_train_protocols}\n")
        f.write(f"  FL: {fl_train_protocols}\n\n")

        f.write("TRAINING SET:\n")
        f.write("-"*60 + "\n")
        if 'R2_train_theta' in results:
            f.write(f"Circumferential R²:   {results['R2_train_theta']:.6f}\n")
            f.write(f"Circumferential RMSE: {results['RMSE_train_theta']:.6f} kPa\n")
        if 'R2_train_z' in results:
            f.write(f"Axial R²:             {results['R2_train_z']:.6f}\n")
            f.write(f"Axial RMSE:           {results['RMSE_train_z']:.6f} kPa\n")
        if 'R2_train_overall' in results:
            f.write(f"Overall R² (pooled):  {results['R2_train_overall']:.6f}\n")
            f.write(f"Overall RMSE:         {results['RMSE_train_overall']:.6f} kPa\n")
        if 'RMS_train' in results:
            f.write(f"Combined RMS:         {results['RMS_train']:.6f}\n")

        if 'R2_val_theta' in results or 'R2_val_z' in results:
            f.write("\n" + "="*60 + "\n")
            f.write("VALIDATION SET:\n")
            f.write("-"*60 + "\n")
            if 'R2_val_theta' in results:
                f.write(f"Circumferential R²:   {results['R2_val_theta']:.6f}\n")
                f.write(f"Circumferential RMSE: {results['RMSE_val_theta']:.6f} kPa\n")
            if 'R2_val_z' in results:
                f.write(f"Axial R²:             {results['R2_val_z']:.6f}\n")
                f.write(f"Axial RMSE:           {results['RMSE_val_z']:.6f} kPa\n")
            if 'R2_val_overall' in results:
                f.write(f"Overall R² (pooled):  {results['R2_val_overall']:.6f}\n")
                f.write(f"Overall RMSE:         {results['RMSE_val_overall']:.6f} kPa\n")
            if 'RMS_val' in results:
                f.write(f"Combined RMS:         {results['RMS_val']:.6f}\n")

        # AIC / BIC -- model selection metrics. Effective parameter count comes
        # from count_effective_parameters() (active candidates + recruitment +
        # dispersion + active per-curve offsets).
        if _v4_aic_bic is not None:
            f.write("\n" + "="*60 + "\n")
            f.write("AIC / BIC (training set):\n")
            f.write("-"*60 + "\n")
            f.write(f"  Effective k:            {_v4_aic_bic['k_total']}\n")
            f.write(f"    active candidates:    {_v4_aic_bic['k_terms_active']} "
                    f"(thresh {args.equation_thresh:g})\n")
            f.write(f"    recruitment params:   {_v4_aic_bic['k_recruit']} "
                    f"(dist={_v4_aic_bic['aic_dist_type']})\n")
            f.write(f"    dispersion / angles:  {_v4_aic_bic['k_disp']}\n")
            f.write(f"    per-curve offsets:    {_v4_aic_bic['k_offsets']}\n")
            f.write(f"  Pooled (PD+FL):  n={_v4_aic_bic['n_total']}, "
                    f"AIC={_v4_aic_bic['AIC_pooled']:.4f}, "
                    f"BIC={_v4_aic_bic['BIC_pooled']:.4f}\n")
            f.write(f"  PD branch:       n={_v4_aic_bic['n_PD']}, "
                    f"RSS={_v4_aic_bic['RSS_PD']:.6g}, "
                    f"AIC={_v4_aic_bic['AIC_PD']:.4f}, "
                    f"BIC={_v4_aic_bic['BIC_PD']:.4f}\n")
            f.write(f"  FL branch:       n={_v4_aic_bic['n_FL']}, "
                    f"RSS={_v4_aic_bic['RSS_FL']:.6g}, "
                    f"AIC={_v4_aic_bic['AIC_FL']:.4f}, "
                    f"BIC={_v4_aic_bic['BIC_FL']:.4f}\n")
            # Fold scalar AIC/BIC values into `results` so they appear in
            # fit_metrics.csv as well.
            for key in ('AIC_pooled', 'BIC_pooled',
                       'AIC_PD', 'BIC_PD', 'AIC_FL', 'BIC_FL',
                       'k_total', 'k_terms_active'):
                results[key] = _v4_aic_bic[key]

    # Print summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY:")
    print("="*60)
    for key, value in results.items():
        try:
            print(f"{key}: {float(value):.4f}")
        except (TypeError, ValueError):
            print(f"{key}: {value}")
    print("="*60)

    # fit_metrics.csv -- compact CSV mirror of the dict written into metrics.txt
    fit_metrics_path = os.path.join(path2saveResults, 'fit_metrics.csv')
    pd.DataFrame([results]).to_csv(fit_metrics_path, index=False)
    print(f"fit_metrics.csv written to {fit_metrics_path}")
    print(f"\nMetrics saved to {metrics_file}")
    print(f"All figures saved to {path2saveResults}")

    # Plot recruitment curves
    plot_recruitment_curves(model_nn, save_path=path2saveResults, strain_range=(0, 0.5))

    # v4: publication-style two-panel figure (PD + FL) with black data dots,
    # red dashed model curve, mathtext labels, large fonts, and 3 ticks per axis.
    # Title shows the animal name + per-direction R^2 values.
    if args.report_label:
        _animal_for_fig = args.report_label
    else:
        _animal_for_fig = os.path.splitext(os.path.basename(args.sample_file))[0]
    try:
        save_publication_figure(
            model_nn,
            X_train_pd if len(X_train_pd) > 0 else None,
            y_train_theta if len(y_train_theta) > 0 else None,
            cid_train_pd if cid_train_pd is not None and len(cid_train_pd) > 0 else None,
            X_train_fl if len(X_train_fl) > 0 else None,
            y_train_z if len(y_train_z) > 0 else None,
            cid_train_fl if cid_train_fl is not None and len(cid_train_fl) > 0 else None,
            curve_vocabulary,
            animal_name=_animal_for_fig,
            r2_theta=results.get('R2_train_theta'),
            r2_z=results.get('R2_train_z'),
            r2_overall=results.get('R2_train_overall'),
            save_path=path2saveResults,
        )
        save_publication_figure(
            model_nn,
            X_train_pd if len(X_train_pd) > 0 else None,
            y_train_theta if len(y_train_theta) > 0 else None,
            cid_train_pd if cid_train_pd is not None and len(cid_train_pd) > 0 else None,
            X_train_fl if len(X_train_fl) > 0 else None,
            y_train_z if len(y_train_z) > 0 else None,
            cid_train_fl if cid_train_fl is not None and len(cid_train_fl) > 0 else None,
            curve_vocabulary,
            animal_name=_animal_for_fig,
            r2_theta=results.get('R2_train_theta'),
            r2_z=results.get('R2_train_z'),
            r2_overall=results.get('R2_train_overall'),
            save_path=path2saveResults,
            output_prefix='publication_figure_2',
            curve_color_mode='protocol',
            show_legend=True,
            legend_loc='upper left',
            model_linewidth=3.2,
        )
        plt.show()
    except Exception as _pub_exc:
        print(f"[publication_figure] WARNING: could not save figure: {_pub_exc}")

    print("\n" + "="*60)
    print("ALL TRAINING RUNS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

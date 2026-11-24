#!/usr/bin/env python3
"""Greedily remove highly correlated terms from a Sobol-ranked CANN library.

The input directory must be one cohort folder produced by
``sobol_sensitivity_analysis.py``. Correlations are computed independently on
the physiological and extended Psi matrices, combined across grids, and used
to prune candidates in Sobol-rank order.

This script performs pairwise numerical decorrelation only. Structural
within-family collapse belongs to the following model-selection stage.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_RANK_COLUMN = "mean_ST_rank_primary"
CATEGORIES = ("iso", "aniso", "coupling")


def load_inputs(sensitivity_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    """Load the Sobol ranking and both Psi grids, with strict validation."""
    ranking_path = sensitivity_dir / "ranked_terms_sobol.csv"
    if not ranking_path.is_file():
        raise FileNotFoundError(f"Missing sensitivity ranking: {ranking_path}")
    ranking = pd.read_csv(ranking_path)
    if "term_key" not in ranking.columns:
        raise ValueError(f"{ranking_path} does not contain a term_key column")
    if ranking["term_key"].duplicated().any():
        duplicates = ranking.loc[ranking["term_key"].duplicated(), "term_key"].tolist()
        raise ValueError(f"Duplicate term_key values in ranking: {duplicates}")

    grids: Dict[str, Dict[str, np.ndarray]] = {}
    for grid_name in ("physio", "extended"):
        path = sensitivity_dir / f"psi_matrix_{grid_name}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing sensitivity basis matrix: {path}")
        with np.load(path, allow_pickle=True) as data:
            required = {"psi", "term_keys"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
            psi = np.asarray(data["psi"], dtype=np.float64)
            term_keys = np.asarray(data["term_keys"]).astype(str)
        if psi.ndim != 2:
            raise ValueError(f"{path}: psi must be two-dimensional, got {psi.shape}")
        if psi.shape[1] != len(term_keys):
            raise ValueError(
                f"{path}: psi has {psi.shape[1]} columns but {len(term_keys)} term keys"
            )
        if not np.isfinite(psi).all():
            raise ValueError(f"{path}: psi contains NaN or infinite values")
        if len(set(term_keys.tolist())) != len(term_keys):
            raise ValueError(f"{path}: term_keys contains duplicates")
        grids[grid_name] = {"psi": psi, "term_keys": term_keys}

    phys_keys = grids["physio"]["term_keys"]
    ext_keys = grids["extended"]["term_keys"]
    if not np.array_equal(phys_keys, ext_keys):
        raise ValueError("term_keys differ between physiological and extended Psi files")

    ranked_keys = set(ranking["term_key"].astype(str))
    psi_keys = set(phys_keys.tolist())
    missing_from_ranking = sorted(psi_keys - ranked_keys)
    missing_from_psi = sorted(ranked_keys - psi_keys)
    if missing_from_ranking or missing_from_psi:
        raise ValueError(
            "Ranking/Psi term mismatch. "
            f"Missing from ranking: {missing_from_ranking}; "
            f"missing from Psi: {missing_from_psi}"
        )
    return ranking, grids


def absolute_pearson_matrix(psi: np.ndarray) -> np.ndarray:
    """Return absolute Pearson correlations between the columns of ``psi``.

    Constant columns are assigned zero correlation with other columns. Their
    diagonal is set to one for a conventional correlation-matrix display.
    """
    centered = psi - psi.mean(axis=0, keepdims=True)
    scale = psi.std(axis=0, ddof=0)
    usable = scale > 1e-12
    standardized = np.zeros_like(centered, dtype=np.float64)
    standardized[:, usable] = centered[:, usable] / scale[usable]
    corr = standardized.T @ standardized / max(psi.shape[0], 1)
    corr = np.clip(np.abs(corr), 0.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def combine_correlations(physio: np.ndarray, extended: np.ndarray, mode: str) -> np.ndarray:
    if mode == "max":
        return np.maximum(physio, extended)
    if mode == "mean":
        return 0.5 * (physio + extended)
    raise ValueError(f"Unknown correlation combination mode: {mode}")


def normalize_category(value: object) -> str:
    category = str(value).strip().lower()
    if category == "cross":
        return "coupling"
    return category if category in CATEGORIES else "iso"


def parse_term_list(values: Optional[Sequence[str]]) -> List[str]:
    """Parse repeatable options; comma-separated values remain supported."""
    terms: List[str] = []
    for value in values or []:
        terms.extend(part.strip() for part in value.split(",") if part.strip())
    return list(dict.fromkeys(terms))


def parse_category_ratios(spec: Optional[str]) -> Optional[Dict[str, float]]:
    if not spec:
        return None
    parsed = {category: 0.0 for category in CATEGORIES}
    for chunk in spec.split(","):
        key, separator, value = chunk.partition("=")
        if not separator:
            raise ValueError(
                f"Invalid --category_quota component {chunk!r}; expected category=value"
            )
        category = "coupling" if key.strip().lower() == "cross" else key.strip().lower()
        if category not in CATEGORIES:
            raise ValueError(f"Unknown category {category!r}; expected {CATEGORIES}")
        parsed[category] = float(value)
    if any(value < 0 for value in parsed.values()):
        raise ValueError("Category quota values cannot be negative")
    total = sum(parsed.values())
    if total <= 0:
        raise ValueError("Category quota values must have a positive sum")
    return {key: value / total for key, value in parsed.items()}


def category_targets(max_terms: int, ratios: Dict[str, float]) -> Dict[str, int]:
    raw = {category: ratios[category] * max_terms for category in CATEGORIES}
    targets = {category: int(np.floor(value)) for category, value in raw.items()}
    remaining = max_terms - sum(targets.values())
    order = sorted(CATEGORIES, key=lambda category: raw[category] - targets[category], reverse=True)
    for category in order[:remaining]:
        targets[category] += 1
    return targets


def prune_terms(
    ranked: pd.DataFrame,
    term_keys: np.ndarray,
    correlation: np.ndarray,
    threshold: float,
    max_terms: int,
    protected_terms: Sequence[str],
    blocked_terms: set[str],
    category_limits: Optional[Dict[str, int]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform deterministic rank-ordered greedy correlation pruning."""
    key_to_index = {key: index for index, key in enumerate(term_keys)}
    ranking_by_key = ranked.set_index("term_key", drop=False)
    kept: List[Dict[str, object]] = []
    dropped: List[Dict[str, object]] = []
    kept_keys: set[str] = set()
    category_counts = {category: 0 for category in CATEGORIES}

    if len(protected_terms) > max_terms:
        raise ValueError(
            f"{len(protected_terms)} protected terms exceed --max_terms={max_terms}"
        )

    for key in protected_terms:
        if key in blocked_terms:
            dropped.append(
                {
                    "term_key": key,
                    "reason": "blocked_overrides_protection",
                    "dropped_for": "",
                    "max_corr": np.nan,
                    "category": "",
                    "rank_sobol": np.nan,
                }
            )
            continue
        if key not in key_to_index or key not in ranking_by_key.index:
            dropped.append(
                {
                    "term_key": key,
                    "reason": "protected_term_not_found",
                    "dropped_for": "",
                    "max_corr": np.nan,
                    "category": "",
                    "rank_sobol": np.nan,
                }
            )
            continue
        row = ranking_by_key.loc[key]
        category = normalize_category(row.get("category", "iso"))
        candidate_index = key_to_index[key]
        worst_corr = 0.0
        worst_key = ""
        for kept_row in kept:
            kept_key = str(kept_row["term_key"])
            value = float(correlation[candidate_index, key_to_index[kept_key]])
            if value > worst_corr:
                worst_corr = value
                worst_key = kept_key
        kept.append(
            {
                "term_key": key,
                "category": category,
                "rank_sobol": int(row["rank_sobol"]),
                "max_corr_with_kept": worst_corr,
                "most_correlated_kept_term": worst_key,
                "protected": True,
            }
        )
        kept_keys.add(key)
        category_counts[category] += 1

    for _, row in ranked.iterrows():
        key = str(row["term_key"])
        category = normalize_category(row.get("category", "iso"))
        rank = int(row["rank_sobol"])
        if key in kept_keys or key in protected_terms:
            continue
        if key in blocked_terms:
            dropped.append(
                {
                    "term_key": key,
                    "reason": "user_blocked",
                    "dropped_for": "",
                    "max_corr": np.nan,
                    "category": category,
                    "rank_sobol": rank,
                }
            )
            continue
        if len(kept) >= max_terms:
            dropped.append(
                {
                    "term_key": key,
                    "reason": "max_terms_reached",
                    "dropped_for": "",
                    "max_corr": np.nan,
                    "category": category,
                    "rank_sobol": rank,
                }
            )
            continue
        if category_limits and category_counts[category] >= category_limits[category]:
            dropped.append(
                {
                    "term_key": key,
                    "reason": "category_quota_exhausted",
                    "dropped_for": category,
                    "max_corr": np.nan,
                    "category": category,
                    "rank_sobol": rank,
                }
            )
            continue

        candidate_index = key_to_index[key]
        worst_corr = 0.0
        worst_key = ""
        for kept_row in kept:
            kept_key = str(kept_row["term_key"])
            value = float(correlation[candidate_index, key_to_index[kept_key]])
            if value > worst_corr:
                worst_corr = value
                worst_key = kept_key
        if worst_corr >= threshold:
            dropped.append(
                {
                    "term_key": key,
                    "reason": "collinear",
                    "dropped_for": worst_key,
                    "max_corr": worst_corr,
                    "category": category,
                    "rank_sobol": rank,
                }
            )
            continue

        kept.append(
            {
                "term_key": key,
                "category": category,
                "rank_sobol": rank,
                "max_corr_with_kept": worst_corr,
                "most_correlated_kept_term": worst_key,
                "protected": False,
            }
        )
        kept_keys.add(key)
        category_counts[category] += 1

    kept_columns = [
        "term_key", "category", "rank_sobol", "max_corr_with_kept",
        "most_correlated_kept_term", "protected",
    ]
    dropped_columns = [
        "term_key", "reason", "dropped_for", "max_corr", "category", "rank_sobol",
    ]
    return (
        pd.DataFrame(kept, columns=kept_columns),
        pd.DataFrame(dropped, columns=dropped_columns),
    )


def save_matrix(path: Path, matrix: np.ndarray, term_keys: Iterable[str]) -> None:
    frame = pd.DataFrame(matrix, index=term_keys, columns=term_keys)
    frame.index.name = "term_key"
    frame.to_csv(path)


def plot_heatmap(
    matrix: np.ndarray,
    term_keys: np.ndarray,
    categories: Sequence[str],
    kept_terms: set[str],
    threshold: float,
    combine_mode: str,
    output_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("--make_plot requires matplotlib") from exc

    order = sorted(range(len(term_keys)), key=lambda i: (categories[i], term_keys[i]))
    ordered = matrix[np.ix_(order, order)]
    labels = [term_keys[i] for i in order]
    size = max(9.0, len(labels) * 0.19)
    fig, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(ordered, vmin=0.0, vmax=1.0, cmap="magma")
    colorbar = fig.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    colorbar.set_label("absolute Pearson correlation")
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=90, fontsize=6)
    axis.set_yticklabels(labels, fontsize=6)
    for index, label in enumerate(labels):
        if label in kept_terms:
            axis.add_patch(
                plt.Rectangle(
                    (-0.5, index - 0.5), len(labels), 1,
                    fill=False, edgecolor="white", linewidth=0.35, alpha=0.55,
                )
            )
    axis.set_title(
        f"Candidate-term correlation ({combine_mode} across grids; threshold={threshold:g})"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_summary(
    args: argparse.Namespace,
    kept: pd.DataFrame,
    dropped: pd.DataFrame,
    category_limits: Optional[Dict[str, int]],
) -> str:
    lines = [
        "Sobol-ranked pairwise decorrelation",
        f"source: {args.sensitivity_dir}",
        f"rank column: {args.rank_col}",
        f"correlation threshold: {args.corr_threshold}",
        f"grid combination: {args.corr_combine}",
        f"maximum kept terms: {args.max_terms}",
        f"category limits: {category_limits or '(none)'}",
        f"kept terms: {len(kept)}",
        f"logged drops: {len(dropped)}",
        "",
        "Kept terms:",
    ]
    for _, row in kept.sort_values("rank_sobol").iterrows():
        protected = " [PROTECTED]" if bool(row["protected"]) else ""
        lines.append(
            f"  rank={int(row['rank_sobol']):>3}  "
            f"category={row['category']:<8}  "
            f"max_corr={float(row['max_corr_with_kept']):.4f}  "
            f"{row['term_key']}{protected}"
        )
    if not dropped.empty:
        lines.extend(["", "Drop counts:"])
        for reason, count in dropped["reason"].value_counts().items():
            lines.append(f"  {reason}: {int(count)}")
        collinear = dropped[dropped["reason"] == "collinear"]
        if not collinear.empty:
            lines.extend(["", "Terms dropped for collinearity:"])
            for _, row in collinear.sort_values("rank_sobol").iterrows():
                lines.append(
                    f"  rank={int(row['rank_sobol']):>3}  "
                    f"rho={float(row['max_corr']):.4f}  "
                    f"{row['term_key']}  ~  {row['dropped_for']}"
                )
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sensitivity_dir", type=Path, required=True,
        help="One cohort folder produced by sobol_sensitivity_analysis.py.",
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--rank_col", default=DEFAULT_RANK_COLUMN,
        help="Ranking column; smaller values are considered first.",
    )
    parser.add_argument(
        "--corr_threshold", type=float, default=0.98,
        help="Drop a term when combined absolute correlation is >= this value.",
    )
    parser.add_argument(
        "--corr_combine", choices=("max", "mean"), default="max",
        help="Combine the two grid correlations conservatively by max or by mean.",
    )
    parser.add_argument("--max_terms", type=int, default=25)
    parser.add_argument(
        "--category_quota", default=None,
        help="Optional relative allocation, e.g. iso=4,aniso=14,coupling=2.",
    )
    parser.add_argument(
        "--protect_term", action="append", default=[],
        help="Term key to seed into the kept set; repeat as needed.",
    )
    parser.add_argument(
        "--protect_strain_linear", action="store_true",
        help="Protect the linear E_theta and E_z strain terms.",
    )
    parser.add_argument(
        "--block_term", action="append", default=[],
        help="Term key to exclude; repeat as needed.",
    )
    parser.add_argument(
        "--block_pattern", default=None,
        help="Regular expression for term keys to exclude.",
    )
    parser.add_argument(
        "--block_i8fs_nonlinear", action="store_true",
        help="Exclude every I8fs shape except Linear.",
    )
    parser.add_argument("--make_plot", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not 0.0 <= args.corr_threshold <= 1.0:
        raise SystemExit("--corr_threshold must be between 0 and 1")
    if args.max_terms <= 0:
        raise SystemExit("--max_terms must be positive")

    ranking, grids = load_inputs(args.sensitivity_dir)
    if args.rank_col not in ranking.columns:
        rank_columns = [column for column in ranking.columns if "rank" in column.lower()]
        raise SystemExit(
            f"--rank_col={args.rank_col!r} is unavailable; choose from {rank_columns}"
        )
    ranking = ranking.copy()
    ranking[args.rank_col] = pd.to_numeric(ranking[args.rank_col], errors="coerce")
    if ranking[args.rank_col].isna().any():
        raise ValueError(f"Ranking column {args.rank_col!r} contains nonnumeric values")
    if "category" in ranking.columns:
        ranking["category"] = ranking["category"].map(normalize_category)
    else:
        ranking["category"] = "iso"
    ranking = ranking.sort_values(
        [args.rank_col, "term_key"], ascending=[True, True], kind="mergesort"
    ).reset_index(drop=True)
    ranking["rank_sobol"] = np.arange(1, len(ranking) + 1)

    physio_corr = absolute_pearson_matrix(grids["physio"]["psi"])
    extended_corr = absolute_pearson_matrix(grids["extended"]["psi"])
    max_corr = combine_correlations(physio_corr, extended_corr, "max")
    mean_corr = combine_correlations(physio_corr, extended_corr, "mean")
    combined_corr = max_corr if args.corr_combine == "max" else mean_corr
    term_keys = grids["physio"]["term_keys"]

    protected = parse_term_list(args.protect_term)
    if args.protect_strain_linear:
        protected = list(
            dict.fromkeys(
                protected
                + ["strain | strain | E_θ", "strain | strain | E_z"]
            )
        )
    blocked = set(parse_term_list(args.block_term))
    if args.block_pattern:
        try:
            pattern = re.compile(args.block_pattern)
        except re.error as exc:
            raise SystemExit(f"Invalid --block_pattern: {exc}") from exc
        blocked.update(key for key in term_keys if pattern.search(str(key)))
    if args.block_i8fs_nonlinear:
        blocked.update(
            key for key in term_keys
            if " | I8fs | " in str(key) and not str(key).endswith(" | Linear")
        )

    ratios = parse_category_ratios(args.category_quota)
    category_limits = category_targets(args.max_terms, ratios) if ratios else None
    kept, dropped = prune_terms(
        ranking,
        term_keys,
        combined_corr,
        args.corr_threshold,
        args.max_terms,
        protected,
        blocked,
        category_limits,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_matrix(args.out_dir / "correlation_matrix_physio.csv", physio_corr, term_keys)
    save_matrix(args.out_dir / "correlation_matrix_extended.csv", extended_corr, term_keys)
    save_matrix(args.out_dir / "correlation_matrix_max.csv", max_corr, term_keys)
    save_matrix(args.out_dir / "correlation_matrix_mean.csv", mean_corr, term_keys)
    save_matrix(args.out_dir / "correlation_matrix_combined.csv", combined_corr, term_keys)

    kept_metadata = kept.drop(columns=["category"], errors="ignore")
    pruned = ranking.merge(kept_metadata, on=["term_key", "rank_sobol"], how="inner")
    pruned = pruned.sort_values("rank_sobol").reset_index(drop=True)
    pruned.to_csv(args.out_dir / "ranked_terms_sobol_pruned.csv", index=False)
    dropped.to_csv(args.out_dir / "pruning_log.csv", index=False)
    (args.out_dir / "decollinearized_summary.txt").write_text(
        build_summary(args, kept, dropped, category_limits), encoding="utf-8"
    )

    if args.make_plot:
        categories_by_key = ranking.set_index("term_key")["category"].to_dict()
        categories = [categories_by_key[str(key)] for key in term_keys]
        plot_heatmap(
            combined_corr,
            term_keys,
            categories,
            set(kept["term_key"].astype(str)),
            args.corr_threshold,
            args.corr_combine,
            args.out_dir / "collinearity_heatmap.png",
        )

    print(f"Wrote decorrelation results to {args.out_dir}")
    print(f"Kept {len(kept)} of {len(ranking)} ranked terms")
    print(f"Dropped {int((dropped['reason'] == 'collinear').sum())} for collinearity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

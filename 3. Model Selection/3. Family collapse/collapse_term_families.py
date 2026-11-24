#!/usr/bin/env python3
"""Collapse decorrelated CANN terms to a limited number of shapes per family.

Pairwise decorrelation removes numerically redundant basis columns. This stage
adds a structural interpretability constraint: terms that are alternative
shapes of the same scalar constitutive family compete for a limited number of
family slots.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd


INVARIANT_FAMILIES = ("I1", "I2", "I4f", "I4s", "I4el", "I4smc", "I8fs")
STRAIN_FAMILIES = (
    "strain_Etheta_pure",
    "strain_Ez_pure",
    "strain_mixed",
)

STRAIN_ETHETA_PURE = {"E_θ", "E_θ²", "E_θ³", "E_θ⁴"}
STRAIN_EZ_PURE = {"E_z", "E_z²", "E_z³", "E_z⁴"}
STRAIN_MIXED = {"E_θ·E_z", "E_θ²·E_z", "E_θ·E_z²", "E_θ²·E_z²"}

SHAPE_COMPLEXITY = {
    "linear": 1,
    "quadratic": 2,
    "cubic": 3,
    "xlog linear": 4,
    "xlog quadratic": 5,
    "exp linear": 6,
    "exp quadratic": 7,
}

STRAIN_COMPLEXITY = {
    "E_θ": 1,
    "E_z": 1,
    "E_θ²": 2,
    "E_z²": 2,
    "E_θ·E_z": 2,
    "E_θ³": 3,
    "E_z³": 3,
    "E_θ²·E_z": 3,
    "E_θ·E_z²": 3,
    "E_θ⁴": 4,
    "E_z⁴": 4,
    "E_θ²·E_z²": 4,
}

LOG_COLUMNS = (
    "collapse_family",
    "term_key",
    "rank_used",
    "shape_complexity",
    "protected",
    "kept_after_family_collapse",
    "selected_representatives",
    "collapse_reason",
)


def split_term_key(term_key: str) -> Tuple[str, str, str]:
    parts = tuple(part.strip() for part in str(term_key).split("|"))
    if len(parts) != 3:
        raise ValueError(f"Cannot parse term_key={term_key!r}; expected group | inv | term")
    return parts[0], parts[1], parts[2]


def ensure_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate term identity and fill group/inv/term from term_key if needed."""
    if "term_key" not in frame.columns:
        required = {"group", "inv", "term"}
        if not required.issubset(frame.columns):
            raise ValueError("Input needs term_key or group, inv, and term columns")
        frame = frame.copy()
        frame["term_key"] = (
            frame["group"].astype(str)
            + " | "
            + frame["inv"].astype(str)
            + " | "
            + frame["term"].astype(str)
        )
    if frame["term_key"].duplicated().any():
        duplicates = frame.loc[frame["term_key"].duplicated(), "term_key"].tolist()
        raise ValueError(f"Input contains duplicate term keys: {duplicates}")

    out = frame.copy()
    parsed = out["term_key"].map(split_term_key)
    for position, column in enumerate(("group", "inv", "term")):
        values = parsed.map(lambda parts: parts[position])
        if column not in out.columns:
            out[column] = values
        else:
            current = out[column].astype(str).str.strip()
            mismatch = current.ne(values)
            if mismatch.any():
                bad = out.loc[mismatch, ["term_key", column]].to_dict("records")
                raise ValueError(f"{column} disagrees with term_key: {bad}")
    return out


def classify_family(group: str, invariant: str, term: str) -> Optional[str]:
    if str(group).lower() == "invariant" and invariant in INVARIANT_FAMILIES:
        return invariant
    if str(group).lower() == "strain":
        if term in STRAIN_ETHETA_PURE:
            return "strain_Etheta_pure"
        if term in STRAIN_EZ_PURE:
            return "strain_Ez_pure"
        if term in STRAIN_MIXED:
            return "strain_mixed"
    return None


def shape_complexity(group: str, term: str) -> int:
    if str(group).lower() == "invariant":
        return SHAPE_COMPLEXITY.get(str(term).strip().lower(), 99)
    if str(group).lower() == "strain":
        return STRAIN_COMPLEXITY.get(str(term).strip(), 99)
    return 99


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    truthy = {"1", "true", "yes", "y", "on"}
    return series.fillna(False).map(lambda value: str(value).strip().lower() in truthy)


def collapse_families(
    frame: pd.DataFrame,
    rank_column: str,
    max_shapes_per_invariant: int,
    max_shapes_per_strain_family: int,
    prefer_simple_shapes: bool,
    tie_tolerance: float,
    merge_helical_i4: bool,
    merge_direction_strain_pairs: bool,
    allow_unclassified: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return the collapsed term table and a complete per-term audit log."""
    if rank_column not in frame.columns:
        available = [column for column in frame.columns if "rank" in column.lower()]
        raise ValueError(f"Rank column {rank_column!r} is unavailable; choose from {available}")
    if max_shapes_per_invariant < 1 or max_shapes_per_strain_family < 1:
        raise ValueError("Family shape limits must be at least one")
    if tie_tolerance < 0:
        raise ValueError("Tie tolerance cannot be negative")

    work = ensure_metadata(frame).reset_index(drop=True)
    work["_rank_used"] = pd.to_numeric(work[rank_column], errors="coerce")
    if work["_rank_used"].isna().any():
        keys = work.loc[work["_rank_used"].isna(), "term_key"].tolist()
        raise ValueError(f"Rank column contains missing or nonnumeric values for: {keys}")
    work["_base_family"] = work.apply(
        lambda row: classify_family(str(row["group"]), str(row["inv"]), str(row["term"])),
        axis=1,
    )
    unclassified = work[work["_base_family"].isna()]["term_key"].tolist()
    if unclassified and not allow_unclassified:
        raise ValueError(
            "Unclassified terms found; correct their labels or use --allow_unclassified: "
            f"{unclassified}"
        )
    work["_collapse_family"] = work["_base_family"]
    if merge_helical_i4:
        work["_collapse_family"] = work["_collapse_family"].replace(
            {"I4f": "I4h", "I4s": "I4h"}
        )
    if merge_direction_strain_pairs:
        work["_collapse_family"] = work["_collapse_family"].replace(
            {
                "strain_Etheta_pure": "I4smc",
                "strain_Ez_pure": "I4el",
            }
        )
    work["_complexity"] = work.apply(
        lambda row: shape_complexity(str(row["group"]), str(row["term"])), axis=1
    )
    if "protected" in work.columns:
        work["_protected"] = parse_bool_series(work["protected"])
    else:
        work["_protected"] = False

    kept_keys: set[str] = set()
    log_rows: List[Dict[str, object]] = []
    for family, family_rows in work.groupby("_collapse_family", sort=False, dropna=False):
        if pd.isna(family):
            for _, row in family_rows.iterrows():
                key = str(row["term_key"])
                kept_keys.add(key)
                log_rows.append(
                    {
                        "collapse_family": "(unclassified)",
                        "term_key": key,
                        "rank_used": float(row["_rank_used"]),
                        "shape_complexity": int(row["_complexity"]),
                        "protected": bool(row["_protected"]),
                        "kept_after_family_collapse": True,
                        "selected_representatives": key,
                        "collapse_reason": "unclassified term passed through explicitly",
                    }
                )
            continue

        family_name = str(family)
        cap = (
            max_shapes_per_strain_family
            if family_name in STRAIN_FAMILIES
            else max_shapes_per_invariant
        )
        ordered = family_rows.sort_values(
            ["_rank_used", "_complexity", "term_key"], kind="mergesort"
        )
        protected_keys = ordered.loc[ordered["_protected"], "term_key"].astype(str).tolist()
        unprotected_rows = ordered.loc[~ordered["_protected"]].copy()
        open_slots = max(cap - len(protected_keys), 0)
        selected_unprotected: List[str] = []

        if open_slots and not unprotected_rows.empty:
            if prefer_simple_shapes:
                best_rank = float(unprotected_rows["_rank_used"].min())
                close = unprotected_rows[
                    unprotected_rows["_rank_used"] <= best_rank + tie_tolerance
                ].sort_values(
                    ["_complexity", "_rank_used", "term_key"], kind="mergesort"
                )
                selected_unprotected.extend(
                    close["term_key"].astype(str).head(open_slots).tolist()
                )
                if len(selected_unprotected) < open_slots:
                    already = set(selected_unprotected)
                    for key in unprotected_rows["term_key"].astype(str):
                        if key not in already:
                            selected_unprotected.append(key)
                            already.add(key)
                        if len(selected_unprotected) == open_slots:
                            break
            else:
                selected_unprotected = (
                    unprotected_rows["term_key"].astype(str).head(open_slots).tolist()
                )

        representatives = protected_keys + selected_unprotected
        kept_keys.update(representatives)
        representative_text = "; ".join(representatives)
        for _, row in ordered.iterrows():
            key = str(row["term_key"])
            is_protected = bool(row["_protected"])
            is_kept = key in representatives
            if is_protected:
                reason = "protected term bypasses the family cap"
            elif is_kept and prefer_simple_shapes:
                reason = f"selected within rank tie tolerance {tie_tolerance:g} using simplicity"
            elif is_kept:
                reason = "best-ranked available term within the family cap"
            else:
                reason = f"family {family_name} collapsed to selected representative(s)"
            log_rows.append(
                {
                    "collapse_family": family_name,
                    "term_key": key,
                    "rank_used": float(row["_rank_used"]),
                    "shape_complexity": int(row["_complexity"]),
                    "protected": is_protected,
                    "kept_after_family_collapse": is_kept,
                    "selected_representatives": representative_text,
                    "collapse_reason": reason,
                }
            )

    audit = pd.DataFrame(log_rows, columns=LOG_COLUMNS)
    collapsed = work[work["term_key"].astype(str).isin(kept_keys)].copy()
    audit_by_key = audit.set_index("term_key")
    collapsed["collapse_family"] = collapsed["term_key"].map(
        audit_by_key["collapse_family"]
    )
    collapsed["family_rank_used"] = collapsed["term_key"].map(
        audit_by_key["rank_used"]
    )
    collapsed["shape_complexity"] = collapsed["term_key"].map(
        audit_by_key["shape_complexity"]
    )
    collapsed = collapsed.sort_values(
        ["_rank_used", "term_key"], kind="mergesort"
    ).drop(
        columns=[
            "_rank_used",
            "_base_family",
            "_collapse_family",
            "_complexity",
            "_protected",
        ]
    )
    return collapsed.reset_index(drop=True), audit


def build_summary(args: argparse.Namespace, source: pd.DataFrame, collapsed: pd.DataFrame, audit: pd.DataFrame) -> str:
    represented = collapsed["collapse_family"].value_counts().to_dict()
    lines = [
        "Within-family structural collapse",
        f"source: {args.input}",
        f"rank column: {args.rank_col}",
        f"maximum shapes per invariant family: {args.max_shapes_per_invariant}",
        f"maximum shapes per strain family: {args.max_shapes_per_strain_family}",
        f"prefer simple shapes: {args.prefer_simple_shapes}",
        f"rank tie tolerance: {args.tie_tolerance}",
        f"merge helical I4 families: {args.merge_helical_i4}",
        f"merge direction/strain families: {args.merge_direction_strain_pairs}",
        f"input terms: {len(source)}",
        f"retained terms: {len(collapsed)}",
        f"removed by family collapse: {int((~audit['kept_after_family_collapse']).sum())}",
        "",
        "Retained family representation:",
    ]
    for family, count in sorted(represented.items()):
        lines.append(f"  {family}: {count}")
    lines.extend(["", "Retained terms:"])
    for _, row in collapsed.iterrows():
        protected = " [PROTECTED]" if bool(row.get("protected", False)) else ""
        lines.append(
            f"  rank={float(row['family_rank_used']):>7.2f}  "
            f"family={row['collapse_family']:<20}  {row['term_key']}{protected}"
        )
    dropped = audit[~audit["kept_after_family_collapse"]]
    if not dropped.empty:
        lines.extend(["", "Removed terms:"])
        for _, row in dropped.sort_values(["collapse_family", "rank_used"]).iterrows():
            lines.append(
                f"  family={row['collapse_family']:<20}  {row['term_key']}  "
                f"-> {row['selected_representatives']}"
            )
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True,
        help="ranked_terms_sobol_pruned.csv from the decorrelation stage.",
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--rank_col", default="mean_ST_rank_primary",
        help="Numeric ranking column; smaller values are better.",
    )
    parser.add_argument("--max_shapes_per_invariant", type=int, default=1)
    parser.add_argument("--max_shapes_per_strain_family", type=int, default=1)
    parser.add_argument(
        "--prefer_simple_shapes", action=argparse.BooleanOptionalAction, default=True,
        help="Within the rank tie band, choose lower-complexity shapes first.",
    )
    parser.add_argument(
        "--tie_tolerance", type=float, default=5.0,
        help="Rank-distance band within which simplicity can override rank.",
    )
    parser.add_argument(
        "--merge_helical_i4", action=argparse.BooleanOptionalAction, default=False,
        help="Treat I4f and I4s as one logical I4h family.",
    )
    parser.add_argument(
        "--merge_direction_strain_pairs", action=argparse.BooleanOptionalAction, default=False,
        help="Fold pure E_theta into I4smc and pure E_z into I4el.",
    )
    parser.add_argument(
        "--allow_unclassified", action=argparse.BooleanOptionalAction, default=False,
        help="Pass unknown term families through instead of failing.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")
    source = pd.read_csv(args.input)
    collapsed, audit = collapse_families(
        source,
        rank_column=args.rank_col,
        max_shapes_per_invariant=args.max_shapes_per_invariant,
        max_shapes_per_strain_family=args.max_shapes_per_strain_family,
        prefer_simple_shapes=args.prefer_simple_shapes,
        tie_tolerance=args.tie_tolerance,
        merge_helical_i4=args.merge_helical_i4,
        merge_direction_strain_pairs=args.merge_direction_strain_pairs,
        allow_unclassified=args.allow_unclassified,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    collapsed.to_csv(args.out_dir / "ranked_terms_family_collapsed.csv", index=False)
    audit.to_csv(args.out_dir / "family_collapse_log.csv", index=False)
    (args.out_dir / "family_collapse_summary.txt").write_text(
        build_summary(args, source, collapsed, audit), encoding="utf-8"
    )
    print(f"Wrote family-collapse results to {args.out_dir}")
    print(f"Retained {len(collapsed)} of {len(source)} decorrelated terms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

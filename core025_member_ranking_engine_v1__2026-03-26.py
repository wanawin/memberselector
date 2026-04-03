#!/usr/bin/env python3
# core025_member_specific_top1_calibration_engine__2026-04-03_v1.py
#
# BUILD: core025_member_specific_top1_calibration_engine__2026-04-03_v1
#
# Full file. No placeholders.
#
# Purpose
# -------
# Mine member-specific Top1 calibration signals from real Core025 LAB outputs.
#
# This app is built for the exact problem now confirmed in testing:
# - traits and separators exist
# - winner member varies by row (0025 / 0225 / 0255)
# - current misses suggest the member-specific scoring / promotion balance is off
#
# So this engine does NOT ask for generic Top1 traits.
# It compares, for each member separately:
#
# A) TRUE_TOP1
#    winner = member and member was correctly played as Top1-only
#
# B) UNDER_PROMOTED
#    winner = member but member was not played as Top1-only
#    (it landed in Top2 / Top3 or required wider play)
#
# C) FALSE_TOP1
#    member was played as Top1-only but lost
#
# From those groups, the engine computes per-member calibration scores for:
# - numeric signals (gap, ratio, exclusivity, alignment, etc.)
# - fired rule IDs
# - top failed columns / near-miss evidence (when present)
#
# The outputs tell us which signals should:
# - promote a member to Top1
# - penalize a member from false Top1 promotion
# - rescue a member that is being under-promoted
#
# Inputs
# ------
# Required:
# - core025_lab_per_event__*.csv
#
# Optional:
# - core025_lab_summary__*.csv
# - promoted separator library CSV (for mapping rule ids to winner members / stacks)
#
# Outputs
# -------
# - core025_member_top1_groups__2026-04-03_v1.csv
# - core025_member_top1_numeric_profile__2026-04-03_v1.csv
# - core025_member_top1_rule_profile__2026-04-03_v1.csv
# - core025_member_top1_failed_column_profile__2026-04-03_v1.csv
# - core025_member_top1_calibration_recommendations__2026-04-03_v1.csv

from __future__ import annotations

import io
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

BUILD_MARKER = "BUILD: core025_member_specific_top1_calibration_engine__2026-04-03_v1"
CORE025 = ["0025", "0225", "0255"]


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def load_table(f) -> pd.DataFrame:
    name = f.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(f)
    if name.endswith(".txt") or name.endswith(".tsv"):
        data = f.getvalue()
        try:
            return pd.read_csv(io.BytesIO(data), sep="\t")
        except Exception:
            return pd.read_csv(io.BytesIO(data), sep=None, engine="python")
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(f)
    raise ValueError(f"Unsupported file type: {f.name}")


def normalize_member_code(raw: object) -> Optional[str]:
    if raw is None or pd.isna(raw):
        return None
    digits = "".join(re.findall(r"\d", str(raw)))
    if digits in {"25", "025", "0025"}:
        return "0025"
    if digits in {"225", "0225"}:
        return "0225"
    if digits in {"255", "0255"}:
        return "0255"
    return None


def safe_float(x: object) -> float:
    try:
        if pd.isna(x):
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def safe_int(x: object) -> int:
    try:
        if pd.isna(x):
            return 0
        return int(float(x))
    except Exception:
        return 0


def parse_rule_ids(rule_text: object) -> List[str]:
    if rule_text is None or pd.isna(rule_text):
        return []
    return re.findall(r"RID\d+", str(rule_text))


def parse_failed_columns(txt: object) -> List[str]:
    if txt is None or pd.isna(txt):
        return []
    parts = [p.strip() for p in str(txt).split("||") if p.strip()]
    cols = []
    for p in parts:
        col = p.split(":", 1)[0].strip()
        if col:
            cols.append(col)
    return cols


def parse_promoted_library(df: pd.DataFrame) -> pd.DataFrame:
    req = {"winner_member", "trait_stack"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Promoted library missing required columns: {sorted(missing)}")
    out = df.copy().reset_index(drop=True)
    out["rule_id"] = [f"RID{i+1}" for i in range(len(out))]
    out["winner_member"] = out["winner_member"].map(normalize_member_code)
    return out


# -----------------------------------------------------------------------------
# Validation / prep
# -----------------------------------------------------------------------------

def validate_per_event(df: pd.DataFrame) -> None:
    required = {
        "event_id",
        "stream",
        "seed",
        "winning_member",
        "Top1",
        "Top2",
        "Top3",
        "play_mode",
        "top1_hit",
        "top2_hit",
        "play_rule_hit",
        "gap",
        "ratio",
        "exclusivity_strength",
        "rule_gap_top12",
        "boost_gap_top12",
        "rule_alignment_ratio",
        "boost_alignment_ratio",
        "blended_alignment_ratio",
        "Top1_score",
        "Top2_score",
        "fired_rules",
        "top_failed_columns",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"LAB per-event file is missing required columns: {missing}")


def prep_per_event(df: pd.DataFrame) -> pd.DataFrame:
    validate_per_event(df)
    out = df.copy()
    out["winning_member"] = out["winning_member"].map(normalize_member_code)
    out["Top1"] = out["Top1"].map(normalize_member_code)
    out["Top2"] = out["Top2"].map(normalize_member_code)
    out["Top3"] = out["Top3"].map(normalize_member_code)
    out["fired_rule_ids"] = out["fired_rules"].apply(parse_rule_ids)
    out["failed_cols_list"] = out["top_failed_columns"].apply(parse_failed_columns)
    return out


# -----------------------------------------------------------------------------
# Group assignment using locked user definitions
# -----------------------------------------------------------------------------

def assign_member_group(row: pd.Series, member: str) -> Optional[str]:
    winner = row["winning_member"]
    top1 = row["Top1"]
    play_mode = str(row["play_mode"])

    if winner == member and play_mode == "PLAY_TOP1" and top1 == member:
        return "TRUE_TOP1"

    if winner == member and not (play_mode == "PLAY_TOP1" and top1 == member):
        return "UNDER_PROMOTED"

    if play_mode == "PLAY_TOP1" and top1 == member and winner != member:
        return "FALSE_TOP1"

    return None


def build_member_groups(per_event: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in per_event.iterrows():
        for member in CORE025:
            group = assign_member_group(row, member)
            if group is None:
                continue
            rows.append(
                {
                    "event_id": safe_int(row["event_id"]),
                    "member": member,
                    "group": group,
                    "stream": row["stream"],
                    "seed": row["seed"],
                    "winning_member": row["winning_member"],
                    "Top1": row["Top1"],
                    "Top2": row["Top2"],
                    "play_mode": row["play_mode"],
                    "gap": safe_float(row["gap"]),
                    "ratio": safe_float(row["ratio"]),
                    "Top1_score": safe_float(row["Top1_score"]),
                    "Top2_score": safe_float(row["Top2_score"]),
                    "exclusivity_strength": safe_float(row["exclusivity_strength"]),
                    "rule_gap_top12": safe_float(row["rule_gap_top12"]),
                    "boost_gap_top12": safe_float(row["boost_gap_top12"]),
                    "rule_alignment_ratio": safe_float(row["rule_alignment_ratio"]),
                    "boost_alignment_ratio": safe_float(row["boost_alignment_ratio"]),
                    "blended_alignment_ratio": safe_float(row["blended_alignment_ratio"]),
                    "fired_rule_ids": row["fired_rule_ids"],
                    "failed_cols_list": row["failed_cols_list"],
                }
            )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Numeric profiling
# -----------------------------------------------------------------------------

def build_numeric_profile(groups_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "gap",
        "ratio",
        "Top1_score",
        "Top2_score",
        "exclusivity_strength",
        "rule_gap_top12",
        "boost_gap_top12",
        "rule_alignment_ratio",
        "boost_alignment_ratio",
        "blended_alignment_ratio",
    ]
    rows = []
    for member in CORE025:
        sub = groups_df[groups_df["member"] == member].copy()
        if len(sub) == 0:
            continue
        bucket_means = {}
        for group in ["TRUE_TOP1", "UNDER_PROMOTED", "FALSE_TOP1"]:
            grp = sub[sub["group"] == group]
            for col in numeric_cols:
                bucket_means[(group, col)] = float(grp[col].mean()) if len(grp) else None

        for col in numeric_cols:
            true_top1 = bucket_means[("TRUE_TOP1", col)]
            under = bucket_means[("UNDER_PROMOTED", col)]
            false_top1 = bucket_means[("FALSE_TOP1", col)]
            promote_lift = None
            false_penalty_gap = None
            rescue_gap = None
            if true_top1 is not None and false_top1 is not None:
                promote_lift = true_top1 - false_top1
            if true_top1 is not None and under is not None:
                rescue_gap = true_top1 - under
            if false_top1 is not None and under is not None:
                false_penalty_gap = false_top1 - under
            rows.append(
                {
                    "member": member,
                    "metric": col,
                    "true_top1_mean": true_top1,
                    "under_promoted_mean": under,
                    "false_top1_mean": false_top1,
                    "promote_lift_true_minus_false": promote_lift,
                    "rescue_gap_true_minus_under": rescue_gap,
                    "false_penalty_gap_false_minus_under": false_penalty_gap,
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["member", "metric"]).reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# Rule profiling
# -----------------------------------------------------------------------------

def build_rule_profile(groups_df: pd.DataFrame, promoted_library: Optional[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for member in CORE025:
        sub = groups_df[groups_df["member"] == member].copy()
        if len(sub) == 0:
            continue

        counters = {g: Counter() for g in ["TRUE_TOP1", "UNDER_PROMOTED", "FALSE_TOP1"]}
        event_counts = {g: int((sub["group"] == g).sum()) for g in counters.keys()}

        for _, row in sub.iterrows():
            g = row["group"]
            for rid in row["fired_rule_ids"]:
                counters[g][rid] += 1

        all_rule_ids = sorted(set().union(*[set(c.keys()) for c in counters.values()]))
        for rid in all_rule_ids:
            true_hits = counters["TRUE_TOP1"][rid]
            under_hits = counters["UNDER_PROMOTED"][rid]
            false_hits = counters["FALSE_TOP1"][rid]
            total_hits = true_hits + under_hits + false_hits
            true_rate = true_hits / max(1, event_counts["TRUE_TOP1"])
            under_rate = under_hits / max(1, event_counts["UNDER_PROMOTED"])
            false_rate = false_hits / max(1, event_counts["FALSE_TOP1"])
            promote_score = true_rate - false_rate
            rescue_score = true_rate - under_rate
            harm_score = false_rate - true_rate
            rows.append(
                {
                    "member": member,
                    "rule_id": rid,
                    "true_top1_hits": int(true_hits),
                    "under_promoted_hits": int(under_hits),
                    "false_top1_hits": int(false_hits),
                    "total_hits": int(total_hits),
                    "true_top1_rate": float(true_rate),
                    "under_promoted_rate": float(under_rate),
                    "false_top1_rate": float(false_rate),
                    "promote_score": float(promote_score),
                    "rescue_score": float(rescue_score),
                    "harm_score": float(harm_score),
                }
            )
    out = pd.DataFrame(rows)
    if promoted_library is not None and len(out):
        join_cols = [c for c in ["rule_id", "winner_member", "pair", "trait_stack", "winner_rate", "pair_gap", "support", "stack_size"] if c in promoted_library.columns]
        lib = promoted_library[join_cols].copy()
        out = out.merge(lib, on="rule_id", how="left")
    if len(out):
        out = out.sort_values(["member", "promote_score", "harm_score", "total_hits"], ascending=[True, False, True, False]).reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# Failed-column profiling
# -----------------------------------------------------------------------------

def build_failed_column_profile(groups_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for member in CORE025:
        sub = groups_df[groups_df["member"] == member].copy()
        if len(sub) == 0:
            continue
        counters = {g: Counter() for g in ["TRUE_TOP1", "UNDER_PROMOTED", "FALSE_TOP1"]}
        event_counts = {g: int((sub["group"] == g).sum()) for g in counters.keys()}
        for _, row in sub.iterrows():
            g = row["group"]
            for col in row["failed_cols_list"]:
                counters[g][col] += 1
        all_cols = sorted(set().union(*[set(c.keys()) for c in counters.values()]))
        for col in all_cols:
            true_rate = counters["TRUE_TOP1"][col] / max(1, event_counts["TRUE_TOP1"])
            under_rate = counters["UNDER_PROMOTED"][col] / max(1, event_counts["UNDER_PROMOTED"])
            false_rate = counters["FALSE_TOP1"][col] / max(1, event_counts["FALSE_TOP1"])
            rows.append(
                {
                    "member": member,
                    "failed_column": col,
                    "true_top1_rate": float(true_rate),
                    "under_promoted_rate": float(under_rate),
                    "false_top1_rate": float(false_rate),
                    "false_minus_true": float(false_rate - true_rate),
                    "under_minus_true": float(under_rate - true_rate),
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["member", "false_minus_true", "under_minus_true"], ascending=[True, False, False]).reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# Recommendations
# -----------------------------------------------------------------------------

def build_recommendations(numeric_profile: pd.DataFrame, rule_profile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for member in CORE025:
        nsub = numeric_profile[numeric_profile["member"] == member].copy()
        rsub = rule_profile[rule_profile["member"] == member].copy()
        if len(nsub) == 0:
            continue

        promote_metrics = nsub.sort_values("promote_lift_true_minus_false", ascending=False)
        rescue_metrics = nsub.sort_values("rescue_gap_true_minus_under", ascending=False)
        harm_metrics = nsub.sort_values("false_penalty_gap_false_minus_under", ascending=False)

        top_promote_rules = rsub.sort_values(["promote_score", "total_hits"], ascending=[False, False]).head(10)
        top_harm_rules = rsub.sort_values(["harm_score", "false_top1_hits"], ascending=[False, False]).head(10)
        top_rescue_rules = rsub.sort_values(["rescue_score", "under_promoted_hits"], ascending=[False, False]).head(10)

        rows.append(
            {
                "member": member,
                "recommendation_type": "NUMERIC_PROMOTION_SIGNALS",
                "details": " | ".join(
                    [
                        f"{r.metric}: true_minus_false={r.promote_lift_true_minus_false:.4f}"
                        for _, r in promote_metrics.head(5).iterrows()
                        if pd.notna(r.promote_lift_true_minus_false)
                    ]
                ),
            }
        )
        rows.append(
            {
                "member": member,
                "recommendation_type": "NUMERIC_UNDERPROMOTION_SIGNALS",
                "details": " | ".join(
                    [
                        f"{r.metric}: true_minus_under={r.rescue_gap_true_minus_under:.4f}"
                        for _, r in rescue_metrics.head(5).iterrows()
                        if pd.notna(r.rescue_gap_true_minus_under)
                    ]
                ),
            }
        )
        rows.append(
            {
                "member": member,
                "recommendation_type": "RULE_PROMOTION_CANDIDATES",
                "details": " | ".join(
                    [
                        f"{r.rule_id}: promote={r.promote_score:.4f} hits={int(r.total_hits)}"
                        for _, r in top_promote_rules.iterrows()
                    ]
                ),
            }
        )
        rows.append(
            {
                "member": member,
                "recommendation_type": "RULE_HARM_CANDIDATES",
                "details": " | ".join(
                    [
                        f"{r.rule_id}: harm={r.harm_score:.4f} false_hits={int(r.false_top1_hits)}"
                        for _, r in top_harm_rules.iterrows()
                    ]
                ),
            }
        )
        rows.append(
            {
                "member": member,
                "recommendation_type": "RULE_RESCUE_CANDIDATES",
                "details": " | ".join(
                    [
                        f"{r.rule_id}: rescue={r.rescue_score:.4f} under_hits={int(r.under_promoted_hits)}"
                        for _, r in top_rescue_rules.iterrows()
                    ]
                ),
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["member", "recommendation_type"]).reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------

def run_engine(per_event_df: pd.DataFrame, promoted_library_df: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_event = prep_per_event(per_event_df)
    promoted = parse_promoted_library(promoted_library_df) if promoted_library_df is not None else None
    groups_df = build_member_groups(per_event)
    numeric_profile = build_numeric_profile(groups_df)
    rule_profile = build_rule_profile(groups_df, promoted)
    failed_profile = build_failed_column_profile(groups_df)
    recommendations = build_recommendations(numeric_profile, rule_profile)
    return groups_df, numeric_profile, rule_profile, failed_profile, recommendations


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Core025 Member-Specific Top1 Calibration Engine", layout="wide")
    st.title("Core025 Member-Specific Top1 Calibration Engine")
    st.caption("Mines member-specific Top1 promotion, false-promotion, and under-promotion signals from real LAB output.")
    st.code(BUILD_MARKER, language="text")

    with st.sidebar:
        st.markdown(f"**{BUILD_MARKER}**")
        rows_to_show = st.number_input("Rows to display", min_value=10, value=50, step=10)

    per_event_file = st.file_uploader("Upload core025_lab_per_event CSV", key="core025_member_top1_per_event")
    promoted_library_file = st.file_uploader("Optional: upload promoted separator library CSV", key="core025_member_top1_promoted_lib")
    summary_file = st.file_uploader("Optional: upload core025_lab_summary CSV", key="core025_member_top1_summary")

    if not per_event_file:
        st.info("Upload the LAB per-event CSV to run the member-specific Top1 calibration engine.")
        return

    try:
        per_event_df = load_table(per_event_file)
        promoted_library_df = load_table(promoted_library_file) if promoted_library_file is not None else None
        groups_df, numeric_profile, rule_profile, failed_profile, recommendations = run_engine(per_event_df, promoted_library_df)
    except Exception as e:
        st.exception(e)
        return

    st.subheader("Member-group counts")
    counts = groups_df.groupby(["member", "group"], dropna=False).size().reset_index(name="events")
    st.dataframe(counts, use_container_width=True)

    st.subheader("Member-specific event groups")
    st.dataframe(groups_df.head(int(rows_to_show)), use_container_width=True)

    st.subheader("Numeric profile")
    st.dataframe(numeric_profile.head(int(rows_to_show)), use_container_width=True)

    st.subheader("Rule profile")
    st.dataframe(rule_profile.head(int(rows_to_show)), use_container_width=True)

    st.subheader("Failed-column profile")
    st.dataframe(failed_profile.head(int(rows_to_show)), use_container_width=True)

    st.subheader("Calibration recommendations")
    st.dataframe(recommendations, use_container_width=True)

    if summary_file is not None:
        try:
            uploaded_summary = load_table(summary_file)
            st.subheader("Uploaded LAB summary")
            st.dataframe(uploaded_summary, use_container_width=True)
        except Exception:
            pass

    st.download_button(
        "Download core025_member_top1_groups__2026-04-03_v1.csv",
        data=groups_df.to_csv(index=False),
        file_name="core025_member_top1_groups__2026-04-03_v1.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download core025_member_top1_numeric_profile__2026-04-03_v1.csv",
        data=numeric_profile.to_csv(index=False),
        file_name="core025_member_top1_numeric_profile__2026-04-03_v1.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download core025_member_top1_rule_profile__2026-04-03_v1.csv",
        data=rule_profile.to_csv(index=False),
        file_name="core025_member_top1_rule_profile__2026-04-03_v1.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download core025_member_top1_failed_column_profile__2026-04-03_v1.csv",
        data=failed_profile.to_csv(index=False),
        file_name="core025_member_top1_failed_column_profile__2026-04-03_v1.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download core025_member_top1_calibration_recommendations__2026-04-03_v1.csv",
        data=recommendations.to_csv(index=False),
        file_name="core025_member_top1_calibration_recommendations__2026-04-03_v1.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()

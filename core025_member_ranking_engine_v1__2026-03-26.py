#!/usr/bin/env python3
# core025_lab_decision_audit__2026-03-31.py
#
# BUILD: core025_lab_decision_audit__2026-03-31
#
# Full file. No placeholders.
#
# Purpose
# -------
# Audits LAB walk-forward output from the Core025 separator system and breaks
# results into the decision buckets that matter for correction:
#
# 1) TOP1_ONLY_WIN
#    Recommendation was PLAY_TOP1 and Top1 won.
#
# 2) TOP12_TOP1_WIN
#    Recommendation was PLAY_TOP2 (Top1+Top2) and Top1 won.
#
# 3) TOP12_TOP2_WIN
#    Recommendation was PLAY_TOP2 (Top1+Top2) and Top2 won.
#
# 4) MISS
#    Recommendation did not capture the winner.
#
# This audit is intended to answer:
# - Are traits inaccurate?
# - Is Top1 vs Top2 separation inaccurate?
# - Is the recommendation logic misclassifying rows?
#
# Inputs
# ------
# Required:
# - core025_lab_per_event__*.csv
#
# Optional:
# - core025_lab_per_stream__*.csv
# - core025_lab_per_date__*.csv
# - core025_lab_summary__*.csv
#
# Outputs
# -------
# - core025_decision_audit_event_buckets__2026-03-31.csv
# - core025_decision_audit_bucket_summary__2026-03-31.csv
# - core025_decision_audit_stream_summary__2026-03-31.csv
# - core025_decision_audit_trait_pressure__2026-03-31.csv
# - core025_decision_audit_failure_patterns__2026-03-31.csv
#
# Notes
# -----
# - This file does not alter the separator engine.
# - It only audits the LAB output so the next correction can be evidence-based.

from __future__ import annotations

import io
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

BUILD_MARKER = "BUILD: core025_lab_decision_audit__2026-03-31"
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
    txt = str(rule_text)
    return re.findall(r"RID\d+", txt)


# -----------------------------------------------------------------------------
# Validation and prep
# -----------------------------------------------------------------------------

def validate_per_event(df: pd.DataFrame) -> None:
    required = {
        "event_id",
        "transition_date",
        "stream",
        "seed",
        "winning_member",
        "Top1",
        "Top2",
        "Top3",
        "Top1_score",
        "Top2_score",
        "Top3_score",
        "gap",
        "ratio",
        "play_mode",
        "dominance_state",
        "top1_hit",
        "top2_hit",
        "top3_hit",
        "play_rule_hit",
        "compression_factor",
        "exclusivity_strength",
        "rule_gap_top12",
        "boost_gap_top12",
        "rule_margin_top1_top2",
        "boost_margin_top1_top2",
        "rules_0025",
        "rules_0225",
        "rules_0255",
        "boost_0025",
        "boost_0225",
        "boost_0255",
        "fired_rule_count",
        "fired_rules",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"LAB per-event file is missing required columns: {missing}")



def prep_per_event(df: pd.DataFrame) -> pd.DataFrame:
    validate_per_event(df)
    out = df.copy()
    out["transition_date"] = pd.to_datetime(out["transition_date"], errors="coerce")
    out["winning_member"] = out["winning_member"].map(normalize_member_code)
    out["Top1"] = out["Top1"].map(normalize_member_code)
    out["Top2"] = out["Top2"].map(normalize_member_code)
    out["Top3"] = out["Top3"].map(normalize_member_code)
    return out


# -----------------------------------------------------------------------------
# Decision bucket assignment
# -----------------------------------------------------------------------------

def assign_decision_bucket(row: pd.Series) -> str:
    play_mode = str(row["play_mode"])
    winner = normalize_member_code(row["winning_member"])
    top1 = normalize_member_code(row["Top1"])
    top2 = normalize_member_code(row["Top2"])

    if play_mode == "PLAY_TOP1" and top1 == winner:
        return "TOP1_ONLY_WIN"
    if play_mode == "PLAY_TOP2" and top1 == winner:
        return "TOP12_TOP1_WIN"
    if play_mode == "PLAY_TOP2" and top2 == winner:
        return "TOP12_TOP2_WIN"
    if play_mode == "SKIP":
        return "SKIP_MISS"
    return "MISS"



def add_audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["decision_bucket"] = out.apply(assign_decision_bucket, axis=1)
    out["captured_any"] = out["play_rule_hit"].apply(safe_int)
    out["winner_is_top1"] = (out["winning_member"] == out["Top1"]).astype(int)
    out["winner_is_top2"] = (out["winning_member"] == out["Top2"]).astype(int)
    out["winner_is_top3"] = (out["winning_member"] == out["Top3"]).astype(int)
    out["top1_rules_for_predicted"] = out.apply(lambda r: safe_int(r.get(f"rules_{normalize_member_code(r['Top1'])}", 0)), axis=1)
    out["top2_rules_for_predicted"] = out.apply(lambda r: safe_int(r.get(f"rules_{normalize_member_code(r['Top2'])}", 0)), axis=1)
    out["top1_boost_for_predicted"] = out.apply(lambda r: safe_float(r.get(f"boost_{normalize_member_code(r['Top1'])}", 0.0)), axis=1)
    out["top2_boost_for_predicted"] = out.apply(lambda r: safe_float(r.get(f"boost_{normalize_member_code(r['Top2'])}", 0.0)), axis=1)
    out["winner_rule_count"] = out.apply(lambda r: safe_int(r.get(f"rules_{normalize_member_code(r['winning_member'])}", 0)), axis=1)
    out["winner_boost"] = out.apply(lambda r: safe_float(r.get(f"boost_{normalize_member_code(r['winning_member'])}", 0.0)), axis=1)
    out["winner_rank_position"] = out.apply(_winner_rank_position, axis=1)
    out["fired_rule_ids"] = out["fired_rules"].apply(parse_rule_ids)
    return out


def _winner_rank_position(row: pd.Series) -> int:
    winner = normalize_member_code(row["winning_member"])
    if winner == normalize_member_code(row["Top1"]):
        return 1
    if winner == normalize_member_code(row["Top2"]):
        return 2
    if winner == normalize_member_code(row["Top3"]):
        return 3
    return 0


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------

def build_bucket_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    total = len(df)
    for bucket, grp in df.groupby("decision_bucket", dropna=False):
        rows.append(
            {
                "decision_bucket": bucket,
                "events": int(len(grp)),
                "event_share": float(len(grp) / total) if total else 0.0,
                "avg_top1_score": float(pd.to_numeric(grp["Top1_score"], errors="coerce").fillna(0.0).mean()),
                "avg_top2_score": float(pd.to_numeric(grp["Top2_score"], errors="coerce").fillna(0.0).mean()),
                "avg_gap": float(pd.to_numeric(grp["gap"], errors="coerce").fillna(0.0).mean()),
                "avg_ratio": float(pd.to_numeric(grp["ratio"], errors="coerce").fillna(0.0).mean()),
                "avg_compression_factor": float(pd.to_numeric(grp["compression_factor"], errors="coerce").fillna(0.0).mean()),
                "avg_exclusivity_strength": float(pd.to_numeric(grp["exclusivity_strength"], errors="coerce").fillna(0.0).mean()),
                "avg_rule_gap_top12": float(pd.to_numeric(grp["rule_gap_top12"], errors="coerce").fillna(0.0).mean()),
                "avg_boost_gap_top12": float(pd.to_numeric(grp["boost_gap_top12"], errors="coerce").fillna(0.0).mean()),
                "avg_rule_margin_top1_top2": float(pd.to_numeric(grp["rule_margin_top1_top2"], errors="coerce").fillna(0.0).mean()),
                "avg_boost_margin_top1_top2": float(pd.to_numeric(grp["boost_margin_top1_top2"], errors="coerce").fillna(0.0).mean()),
                "avg_top1_rules_predicted": float(pd.to_numeric(grp["top1_rules_for_predicted"], errors="coerce").fillna(0.0).mean()),
                "avg_top2_rules_predicted": float(pd.to_numeric(grp["top2_rules_for_predicted"], errors="coerce").fillna(0.0).mean()),
                "avg_winner_rule_count": float(pd.to_numeric(grp["winner_rule_count"], errors="coerce").fillna(0.0).mean()),
                "avg_winner_boost": float(pd.to_numeric(grp["winner_boost"], errors="coerce").fillna(0.0).mean()),
                "winner_rank_1_pct": float((grp["winner_rank_position"] == 1).mean()),
                "winner_rank_2_pct": float((grp["winner_rank_position"] == 2).mean()),
                "winner_rank_3_pct": float((grp["winner_rank_position"] == 3).mean()),
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        order = {"TOP1_ONLY_WIN": 1, "TOP12_TOP1_WIN": 2, "TOP12_TOP2_WIN": 3, "MISS": 4, "SKIP_MISS": 5}
        out["_order"] = out["decision_bucket"].map(order).fillna(999)
        out = out.sort_values(["_order", "decision_bucket"]).drop(columns=["_order"]).reset_index(drop=True)
    return out



def build_stream_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for stream, grp in df.groupby("stream", dropna=False):
        bucket_counts = grp["decision_bucket"].value_counts(dropna=False).to_dict()
        rows.append(
            {
                "stream": stream,
                "events": int(len(grp)),
                "top1_only_wins": int(bucket_counts.get("TOP1_ONLY_WIN", 0)),
                "top12_top1_wins": int(bucket_counts.get("TOP12_TOP1_WIN", 0)),
                "top12_top2_wins": int(bucket_counts.get("TOP12_TOP2_WIN", 0)),
                "misses": int(bucket_counts.get("MISS", 0)),
                "skip_misses": int(bucket_counts.get("SKIP_MISS", 0)),
                "capture_pct": float(grp["captured_any"].mean()),
                "avg_gap": float(pd.to_numeric(grp["gap"], errors="coerce").fillna(0.0).mean()),
                "avg_ratio": float(pd.to_numeric(grp["ratio"], errors="coerce").fillna(0.0).mean()),
                "avg_exclusivity_strength": float(pd.to_numeric(grp["exclusivity_strength"], errors="coerce").fillna(0.0).mean()),
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["capture_pct", "events", "stream"], ascending=[False, False, True]).reset_index(drop=True)
    return out



def build_trait_pressure(df: pd.DataFrame) -> pd.DataFrame:
    bucket_rule_counter: Dict[str, Counter] = defaultdict(Counter)
    bucket_event_counter: Counter = Counter()

    for _, row in df.iterrows():
        bucket = str(row["decision_bucket"])
        bucket_event_counter[bucket] += 1
        for rid in row["fired_rule_ids"]:
            bucket_rule_counter[bucket][rid] += 1

    rows: List[Dict[str, object]] = []
    buckets = ["TOP1_ONLY_WIN", "TOP12_TOP1_WIN", "TOP12_TOP2_WIN", "MISS", "SKIP_MISS"]
    all_rule_ids = sorted({rid for c in bucket_rule_counter.values() for rid in c.keys()})
    for rid in all_rule_ids:
        row = {"rule_id": rid}
        top1_only = bucket_rule_counter["TOP1_ONLY_WIN"][rid]
        top12_top2 = bucket_rule_counter["TOP12_TOP2_WIN"][rid]
        misses = bucket_rule_counter["MISS"][rid] + bucket_rule_counter["SKIP_MISS"][rid]
        total_hits = sum(bucket_rule_counter[b][rid] for b in buckets)
        row["total_rule_hits"] = int(total_hits)
        row["top1_only_hits"] = int(top1_only)
        row["top12_top1_hits"] = int(bucket_rule_counter["TOP12_TOP1_WIN"][rid])
        row["top12_top2_hits"] = int(top12_top2)
        row["miss_hits"] = int(bucket_rule_counter["MISS"][rid])
        row["skip_miss_hits"] = int(bucket_rule_counter["SKIP_MISS"][rid])
        row["top1_only_share_of_rule_hits"] = float(top1_only / total_hits) if total_hits else 0.0
        row["top12_top2_share_of_rule_hits"] = float(top12_top2 / total_hits) if total_hits else 0.0
        row["miss_share_of_rule_hits"] = float(misses / total_hits) if total_hits else 0.0
        row["quality_signal"] = float((top1_only + top12_top2) / total_hits) if total_hits else 0.0
        row["harm_signal"] = float(misses / total_hits) if total_hits else 0.0
        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["harm_signal", "quality_signal", "total_rule_hits", "rule_id"], ascending=[False, False, False, True]).reset_index(drop=True)
    return out



def build_failure_patterns(df: pd.DataFrame) -> pd.DataFrame:
    miss_df = df[df["decision_bucket"].isin(["MISS", "SKIP_MISS"])].copy()
    rows: List[Dict[str, object]] = []
    if len(miss_df) == 0:
        return pd.DataFrame(columns=["pattern", "events"])

    def gap_band(x: float) -> str:
        if x < 0.05:
            return "gap_lt_0.05"
        if x < 0.10:
            return "gap_0.05_0.09"
        if x < 0.20:
            return "gap_0.10_0.19"
        return "gap_ge_0.20"

    def ratio_band(x: float) -> str:
        if x >= 0.95:
            return "ratio_ge_0.95"
        if x >= 0.90:
            return "ratio_0.90_0.949"
        if x >= 0.80:
            return "ratio_0.80_0.899"
        return "ratio_lt_0.80"

    def excl_band(x: float) -> str:
        if x < 0.10:
            return "excl_lt_0.10"
        if x < 0.20:
            return "excl_0.10_0.19"
        if x < 0.30:
            return "excl_0.20_0.29"
        return "excl_ge_0.30"

    miss_df["gap_band"] = miss_df["gap"].apply(safe_float).apply(gap_band)
    miss_df["ratio_band"] = miss_df["ratio"].apply(safe_float).apply(ratio_band)
    miss_df["excl_band"] = miss_df["exclusivity_strength"].apply(safe_float).apply(excl_band)

    grouped = (
        miss_df.groupby(["play_mode", "dominance_state", "gap_band", "ratio_band", "excl_band"], dropna=False)
        .size()
        .reset_index(name="events")
        .sort_values(["events", "play_mode", "dominance_state"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    grouped["pattern"] = grouped.apply(
        lambda r: f"{r['play_mode']} | {r['dominance_state']} | {r['gap_band']} | {r['ratio_band']} | {r['excl_band']}",
        axis=1,
    )
    cols = ["pattern", "events", "play_mode", "dominance_state", "gap_band", "ratio_band", "excl_band"]
    return grouped[cols]


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------

def run_audit(per_event_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_event = prep_per_event(per_event_df)
    per_event = add_audit_columns(per_event)
    bucket_summary = build_bucket_summary(per_event)
    stream_summary = build_stream_summary(per_event)
    trait_pressure = build_trait_pressure(per_event)
    failure_patterns = build_failure_patterns(per_event)
    return per_event, bucket_summary, stream_summary, trait_pressure, failure_patterns


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Core025 LAB Decision Audit", layout="wide")
    st.title("Core025 LAB Decision Audit")
    st.caption("Audits LAB walk-forward output to show where Top1, Top2, and misses are really coming from.")
    st.code(BUILD_MARKER, language="text")

    with st.sidebar:
        st.markdown(f"**{BUILD_MARKER}**")
        rows_to_show = st.number_input("Rows to display", min_value=10, value=50, step=10)

    per_event_file = st.file_uploader("Upload core025_lab_per_event CSV", key="core025_lab_per_event_audit")
    per_stream_file = st.file_uploader("Optional: upload core025_lab_per_stream CSV", key="core025_lab_per_stream_audit")
    per_date_file = st.file_uploader("Optional: upload core025_lab_per_date CSV", key="core025_lab_per_date_audit")
    summary_file = st.file_uploader("Optional: upload core025_lab_summary CSV", key="core025_lab_summary_audit")

    if not per_event_file:
        st.info("Upload the LAB per-event CSV to run the decision audit.")
        return

    try:
        per_event_raw = load_table(per_event_file)
        per_event, bucket_summary, stream_summary, trait_pressure, failure_patterns = run_audit(per_event_raw)
    except Exception as e:
        st.exception(e)
        return

    st.subheader("Decision bucket summary")
    st.dataframe(bucket_summary, use_container_width=True)

    st.subheader("Event bucket preview")
    preview_cols = [
        "event_id",
        "transition_date",
        "stream",
        "seed",
        "winning_member",
        "Top1",
        "Top2",
        "Top3",
        "play_mode",
        "decision_bucket",
        "dominance_state",
        "gap",
        "ratio",
        "exclusivity_strength",
        "rule_margin_top1_top2",
        "boost_margin_top1_top2",
        "winner_rank_position",
    ]
    show_cols = [c for c in preview_cols if c in per_event.columns]
    st.dataframe(per_event[show_cols].head(int(rows_to_show)), use_container_width=True)

    st.subheader("Stream summary")
    st.dataframe(stream_summary.head(int(rows_to_show)), use_container_width=True)

    st.subheader("Trait pressure / rule quality audit")
    st.dataframe(trait_pressure.head(int(rows_to_show)), use_container_width=True)

    st.subheader("Failure patterns")
    st.dataframe(failure_patterns.head(int(rows_to_show)), use_container_width=True)

    if summary_file is not None:
        try:
            uploaded_summary = load_table(summary_file)
            st.subheader("Uploaded LAB summary")
            st.dataframe(uploaded_summary, use_container_width=True)
        except Exception:
            pass

    if per_date_file is not None:
        try:
            uploaded_per_date = load_table(per_date_file)
            st.subheader("Uploaded LAB per-date preview")
            st.dataframe(uploaded_per_date.head(int(rows_to_show)), use_container_width=True)
        except Exception:
            pass

    if per_stream_file is not None:
        try:
            uploaded_per_stream = load_table(per_stream_file)
            st.subheader("Uploaded LAB per-stream preview")
            st.dataframe(uploaded_per_stream.head(int(rows_to_show)), use_container_width=True)
        except Exception:
            pass

    st.download_button(
        "Download core025_decision_audit_event_buckets__2026-03-31.csv",
        data=per_event.to_csv(index=False),
        file_name="core025_decision_audit_event_buckets__2026-03-31.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download core025_decision_audit_bucket_summary__2026-03-31.csv",
        data=bucket_summary.to_csv(index=False),
        file_name="core025_decision_audit_bucket_summary__2026-03-31.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download core025_decision_audit_stream_summary__2026-03-31.csv",
        data=stream_summary.to_csv(index=False),
        file_name="core025_decision_audit_stream_summary__2026-03-31.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download core025_decision_audit_trait_pressure__2026-03-31.csv",
        data=trait_pressure.to_csv(index=False),
        file_name="core025_decision_audit_trait_pressure__2026-03-31.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download core025_decision_audit_failure_patterns__2026-03-31.csv",
        data=failure_patterns.to_csv(index=False),
        file_name="core025_decision_audit_failure_patterns__2026-03-31.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()

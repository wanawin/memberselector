#!/usr/bin/env python3
# core025_dual_lab_review_and_daily_splitter__2026-04-04_v12.py
#
# BUILD: core025_dual_lab_review_and_daily_splitter__2026-04-04_v12
#
# FULL REPLACEMENT
#
# PURPOSE
# -------
# Lightweight, trustworthy review app that keeps the two systems SEPARATE:
#   1) Winner engine (Frozen V4 / no-skip)
#   2) Skip engine
#
# This app does NOT try to merge both walk-forwards into one risky engine path.
# Instead it:
#   - reviews both walk-forward results separately
#   - shows accuracy summaries side by side
#   - compares stream overlap safely
#   - builds daily split lists from two separate uploads
#   - highlights / marks streams to STRIP for the day
#
# WHY THIS VERSION
# ----------------
# You asked for a working, trustworthy application today without continuing to
# risk progress or burn Streamlit resources. This version is intentionally light:
# it analyzes uploaded outputs instead of running heavy combined computation.
#
# WHAT TO UPLOAD
# --------------
# LAB REVIEW TAB:
#   - Winner Engine LAB per-event CSV (from the known-good no-skip engine)
#   - Skip LAB per-event CSV (from the standalone skip walk-forward)
#
# DAILY SPLITTER TAB:
#   - Winner Engine daily playlist CSV
#   - Skip daily scored CSV OR skip daily playable CSV
#
# OUTPUTS
# -------
# LAB REVIEW:
#   - separate summaries for both systems
#   - side-by-side comparison table
#   - stream overlap tables
#
# DAILY SPLITTER:
#   - KEEP list (playable rows)
#   - STRIP list (rows to remove)
#   - merged daily board with explicit action column
#
# NO PLACEHOLDERS
# ---------------
# This file is fully self-contained.

from __future__ import annotations

import io
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

BUILD_MARKER = "BUILD: core025_dual_lab_review_and_daily_splitter__2026-04-04_v12"


# =============================================================================
# BASIC HELPERS
# =============================================================================

def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    seen: Dict[str, int] = {}
    cols: List[str] = []
    for col in df.columns:
        name = str(col)
        if name not in seen:
            seen[name] = 0
            cols.append(name)
        else:
            seen[name] += 1
            cols.append(f"{name}__dup{seen[name]}")
    out = df.copy()
    out.columns = cols
    return out


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return dedupe_columns(df).to_csv(index=False).encode("utf-8")


def load_table(upload) -> pd.DataFrame:
    name = upload.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(upload)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(upload)
    if name.endswith(".txt") or name.endswith(".tsv"):
        data = upload.getvalue()
        try:
            return pd.read_csv(io.BytesIO(data), sep="\t")
        except Exception:
            return pd.read_csv(io.BytesIO(data), sep=None, engine="python")
    raise ValueError(f"Unsupported file type: {upload.name}")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    mapping = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in mapping:
            return mapping[key]
    for cand in candidates:
        key = _norm(cand)
        for norm_name, real_name in mapping.items():
            if key in norm_name:
                return real_name
    if required:
        raise ValueError(f"Could not find any of these columns: {candidates}. Available: {list(df.columns)}")
    return None


def normalize_member(raw: object) -> str:
    if raw is None or pd.isna(raw):
        return ""
    digits = "".join(re.findall(r"\d", str(raw)))
    if digits in {"25", "025", "0025"}:
        return "0025"
    if digits in {"225", "0225"}:
        return "0225"
    if digits in {"255", "0255"}:
        return "0255"
    return digits


def normalize_stream(raw: object) -> str:
    if raw is None or pd.isna(raw):
        return ""
    return str(raw).strip()


def normalize_seed(raw: object) -> str:
    if raw is None or pd.isna(raw):
        return ""
    return str(raw).strip()


def ensure_datetime_if_present(df: pd.DataFrame, col: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if col and col in out.columns:
        out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


# =============================================================================
# WINNER ENGINE LAB REVIEW
# =============================================================================

def prep_winner_lab_per_event(df: pd.DataFrame) -> pd.DataFrame:
    df = dedupe_columns(df.copy())
    stream_col = find_col(df, ["stream", "stream_id"])
    seed_col = find_col(df, ["seed"])
    date_col = find_col(df, ["transition_date", "event_date", "date"], required=False)
    winning_col = find_col(df, ["winning_member", "winner_member", "next_member"], required=False)
    top1_hit_col = find_col(df, ["top1_hit", "top1_win"])
    top2_hit_col = find_col(df, ["top2_hit", "top2_win"])
    top3_hit_col = find_col(df, ["top3_hit", "top3_loss"], required=False)
    play_rule_hit_col = find_col(df, ["play_rule_hit", "combined_hit", "capture_hit"], required=False)
    play_mode_col = find_col(df, ["play_mode"], required=False)
    top1_col = find_col(df, ["top1"])
    top2_col = find_col(df, ["top2"], required=False)
    out = pd.DataFrame({
        "stream": df[stream_col].map(normalize_stream),
        "seed": df[seed_col].map(normalize_seed),
        "event_date": df[date_col] if date_col else pd.NaT,
        "winning_member": df[winning_col].map(normalize_member) if winning_col else "",
        "top1_hit": pd.to_numeric(df[top1_hit_col], errors="coerce").fillna(0).astype(int),
        "top2_hit": pd.to_numeric(df[top2_hit_col], errors="coerce").fillna(0).astype(int),
        "top3_hit_or_loss": pd.to_numeric(df[top3_hit_col], errors="coerce").fillna(0).astype(int) if top3_hit_col else 0,
        "play_rule_hit": pd.to_numeric(df[play_rule_hit_col], errors="coerce").fillna(0).astype(int) if play_rule_hit_col else 0,
        "play_mode": df[play_mode_col].astype(str) if play_mode_col else "",
        "Top1": df[top1_col].map(normalize_member),
        "Top2": df[top2_col].map(normalize_member) if top2_col else "",
    })
    out = ensure_datetime_if_present(out, "event_date")
    return out


def summarize_winner_lab(df: pd.DataFrame) -> pd.DataFrame:
    events = len(df)
    top1 = int(df["top1_hit"].sum())
    top2 = int(df["top2_hit"].sum())
    play_rule = int(df["play_rule_hit"].sum()) if "play_rule_hit" in df.columns else top2
    summary = pd.DataFrame([
        {"metric": "events", "value": events},
        {"metric": "top1_hits", "value": top1},
        {"metric": "top2_hits", "value": top2},
        {"metric": "play_rule_hits", "value": play_rule},
        {"metric": "top1_capture_pct", "value": float(top1 / max(1, events))},
        {"metric": "top2_capture_pct", "value": float(top2 / max(1, events))},
        {"metric": "play_rule_capture_pct", "value": float(play_rule / max(1, events))},
        {"metric": "rows_play_top1", "value": int((df["play_mode"] == "PLAY_TOP1").sum()) if "play_mode" in df.columns else 0},
        {"metric": "rows_play_top2", "value": int((df["play_mode"] == "PLAY_TOP2").sum()) if "play_mode" in df.columns else 0},
        {"metric": "rows_skip", "value": int((df["play_mode"] == "SKIP").sum()) if "play_mode" in df.columns else 0},
    ])
    return summary


# =============================================================================
# SKIP LAB REVIEW
# =============================================================================

def prep_skip_lab_per_event(df: pd.DataFrame) -> pd.DataFrame:
    df = dedupe_columns(df.copy())
    stream_col = find_col(df, ["stream", "stream_id"])
    seed_col = find_col(df, ["seed"])
    date_col = find_col(df, ["transition_date", "event_date", "date"], required=False)
    winner_stream_flag_col = find_col(df, ["winner_stream_flag", "winning_stream_flag"], required=False)
    survived_col = find_col(df, ["winner_survived_skip", "winning_streams_survived", "winner_kept"], required=False)
    skip_class_col = find_col(df, ["skip_class", "class"])
    skip_score_col = find_col(df, ["skip_score"])
    out = pd.DataFrame({
        "stream": df[stream_col].map(normalize_stream),
        "seed": df[seed_col].map(normalize_seed),
        "event_date": df[date_col] if date_col else pd.NaT,
        "winner_stream_flag": pd.to_numeric(df[winner_stream_flag_col], errors="coerce").fillna(0).astype(int) if winner_stream_flag_col else 0,
        "winner_survived_skip": pd.to_numeric(df[survived_col], errors="coerce").fillna(0).astype(int) if survived_col else 0,
        "skip_class": df[skip_class_col].astype(str),
        "skip_score": pd.to_numeric(df[skip_score_col], errors="coerce").fillna(0.0),
    })
    out = ensure_datetime_if_present(out, "event_date")
    return out


def summarize_skip_lab(df: pd.DataFrame) -> pd.DataFrame:
    events = len(df)
    skips = int((df["skip_class"] == "SKIP").sum())
    playable = int((df["skip_class"] == "PLAY").sum())
    winning_stream_rows = int(df["winner_stream_flag"].sum()) if "winner_stream_flag" in df.columns else 0
    survived = int(df["winner_survived_skip"].sum()) if "winner_survived_skip" in df.columns else 0
    summary = pd.DataFrame([
        {"metric": "events", "value": events},
        {"metric": "skip_rows", "value": skips},
        {"metric": "play_rows", "value": playable},
        {"metric": "avg_skip_score", "value": float(df["skip_score"].mean()) if events else 0.0},
        {"metric": "winning_stream_rows", "value": winning_stream_rows},
        {"metric": "winning_streams_survived", "value": survived},
        {"metric": "winning_stream_survival_pct", "value": float(survived / max(1, winning_stream_rows))},
    ])
    return summary


# =============================================================================
# COMPARISON
# =============================================================================

def build_lab_comparison(winner_summary: pd.DataFrame, skip_summary: pd.DataFrame) -> pd.DataFrame:
    ws = dict(zip(winner_summary["metric"], winner_summary["value"]))
    ss = dict(zip(skip_summary["metric"], skip_summary["value"]))
    rows = [
        {"comparison": "winner_engine_events", "value": ws.get("events", 0)},
        {"comparison": "winner_engine_top1_capture_pct", "value": ws.get("top1_capture_pct", 0)},
        {"comparison": "winner_engine_top2_capture_pct", "value": ws.get("top2_capture_pct", 0)},
        {"comparison": "winner_engine_play_rule_capture_pct", "value": ws.get("play_rule_capture_pct", 0)},
        {"comparison": "skip_events", "value": ss.get("events", 0)},
        {"comparison": "skip_play_rows", "value": ss.get("play_rows", 0)},
        {"comparison": "skip_rows", "value": ss.get("skip_rows", 0)},
        {"comparison": "skip_winning_stream_survival_pct", "value": ss.get("winning_stream_survival_pct", 0)},
    ]
    return pd.DataFrame(rows)


def build_lab_overlap(winner_df: pd.DataFrame, skip_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    w = winner_df[["stream", "seed"]].copy()
    w["winner_engine_row"] = 1
    s = skip_df[["stream", "seed", "skip_class"]].copy()
    merged = w.merge(s, on=["stream", "seed"], how="outer")
    merged["winner_engine_row"] = merged["winner_engine_row"].fillna(0).astype(int)
    merged["skip_class"] = merged["skip_class"].fillna("MISSING_IN_SKIP_FILE")
    merged["relation"] = merged.apply(
        lambda r: "Winner row kept by skip" if r["winner_engine_row"] == 1 and r["skip_class"] == "PLAY"
        else ("Winner row stripped by skip" if r["winner_engine_row"] == 1 and r["skip_class"] == "SKIP"
        else ("Missing from skip file" if r["winner_engine_row"] == 1 else "Skip-only row")),
        axis=1,
    )
    relation_summary = merged.groupby("relation", dropna=False).size().reset_index(name="rows").sort_values(["rows", "relation"], ascending=[False, True]).reset_index(drop=True)
    return merged, relation_summary


# =============================================================================
# DAILY SPLITTER
# =============================================================================

def prep_winner_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = dedupe_columns(df.copy())
    stream_col = find_col(df, ["stream", "stream_id"])
    seed_col = find_col(df, ["seed"])
    top1_col = find_col(df, ["top1"])
    top2_col = find_col(df, ["top2"], required=False)
    top3_col = find_col(df, ["top3"], required=False)
    play_mode_col = find_col(df, ["play_mode"], required=False)
    top1_score_col = find_col(df, ["top1_score"], required=False)
    gap_col = find_col(df, ["gap"], required=False)
    ratio_col = find_col(df, ["ratio"], required=False)
    out = pd.DataFrame({
        "stream": df[stream_col].map(normalize_stream),
        "seed": df[seed_col].map(normalize_seed),
        "Top1": df[top1_col].map(normalize_member),
        "Top2": df[top2_col].map(normalize_member) if top2_col else "",
        "Top3": df[top3_col].map(normalize_member) if top3_col else "",
        "play_mode": df[play_mode_col].astype(str) if play_mode_col else "",
        "Top1_score": pd.to_numeric(df[top1_score_col], errors="coerce").fillna(0.0) if top1_score_col else 0.0,
        "gap": pd.to_numeric(df[gap_col], errors="coerce").fillna(0.0) if gap_col else 0.0,
        "ratio": pd.to_numeric(df[ratio_col], errors="coerce").fillna(0.0) if ratio_col else 0.0,
    })
    return out


def prep_skip_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = dedupe_columns(df.copy())
    stream_col = find_col(df, ["stream", "stream_id"])
    seed_col = find_col(df, ["seed"])
    skip_class_col = find_col(df, ["skip_class", "class"], required=False)
    playable_marker_col = find_col(df, ["playable_marker", "play", "PLAY"], required=False)
    skip_score_col = find_col(df, ["skip_score"], required=False)
    out = pd.DataFrame({
        "stream": df[stream_col].map(normalize_stream),
        "seed": df[seed_col].map(normalize_seed),
        "skip_class": df[skip_class_col].astype(str) if skip_class_col else "",
        "playable_marker": df[playable_marker_col].astype(str) if playable_marker_col else "",
        "skip_score": pd.to_numeric(df[skip_score_col], errors="coerce").fillna(0.0) if skip_score_col else 0.0,
    })
    if (out["skip_class"] == "").all() and (out["playable_marker"] != "").any():
        out["skip_class"] = out["playable_marker"].map(lambda x: "PLAY" if "PLAY" in str(x).upper() else "SKIP")
    return out


def build_daily_split(winner_daily: pd.DataFrame, skip_daily: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = winner_daily.merge(skip_daily[["stream", "seed", "skip_class", "skip_score", "playable_marker"]], on=["stream", "seed"], how="left")
    merged["skip_class"] = merged["skip_class"].fillna("MISSING_IN_SKIP_FILE")
    merged["playable_marker"] = merged["playable_marker"].fillna("")
    merged["action"] = merged["skip_class"].map(
        lambda x: "KEEP" if x == "PLAY" else ("STRIP" if x == "SKIP" else "REVIEW")
    )
    merged["display_stream"] = merged.apply(
        lambda r: f"~~{r['stream']}~~" if r["action"] == "STRIP" else r["stream"],
        axis=1,
    )
    keep_df = merged[merged["action"] == "KEEP"].copy().reset_index(drop=True)
    strip_df = merged[merged["action"] == "STRIP"].copy().reset_index(drop=True)
    merged = merged.sort_values(["action", "Top1_score", "gap"], ascending=[True, False, False]).reset_index(drop=True)
    return keep_df, strip_df, merged


# =============================================================================
# UI
# =============================================================================

def main() -> None:
    st.set_page_config(page_title="Core025 Dual Lab Review + Daily Splitter", layout="wide")
    st.title("Core025 Dual Lab Review + Daily Splitter")
    st.code(BUILD_MARKER, language="text")

    tab1, tab2 = st.tabs(["LAB Review", "Daily Splitter"])

    with tab1:
        st.subheader("Separate walk-forward review")
        st.caption("Upload the winner-engine LAB per-event CSV and the skip LAB per-event CSV. This app reviews them separately and compares them without combining logic.")

        winner_lab_file = st.file_uploader("Upload Winner Engine LAB per-event CSV", key="winner_lab_file")
        skip_lab_file = st.file_uploader("Upload Skip LAB per-event CSV", key="skip_lab_file")

        if winner_lab_file and skip_lab_file:
            try:
                winner_lab_raw = load_table(winner_lab_file)
                skip_lab_raw = load_table(skip_lab_file)
                winner_lab = prep_winner_lab_per_event(winner_lab_raw)
                skip_lab = prep_skip_lab_per_event(skip_lab_raw)
                winner_summary = summarize_winner_lab(winner_lab)
                skip_summary = summarize_skip_lab(skip_lab)
                comparison = build_lab_comparison(winner_summary, skip_summary)
                overlap_detail, overlap_summary = build_lab_overlap(winner_lab, skip_lab)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Winner Engine Summary**")
                    st.dataframe(winner_summary, use_container_width=True)
                with c2:
                    st.markdown("**Skip Summary**")
                    st.dataframe(skip_summary, use_container_width=True)

                st.markdown("**Side-by-side comparison**")
                st.dataframe(comparison, use_container_width=True)

                st.markdown("**Overlap summary**")
                st.dataframe(overlap_summary, use_container_width=True)

                st.markdown("**Overlap detail**")
                st.dataframe(overlap_detail.head(500), use_container_width=True)

                st.download_button(
                    "Download lab_review_comparison__2026-04-04_v12.csv",
                    df_to_csv_bytes(comparison),
                    file_name="lab_review_comparison__2026-04-04_v12.csv",
                    mime="text/csv",
                )
                st.download_button(
                    "Download lab_overlap_detail__2026-04-04_v12.csv",
                    df_to_csv_bytes(overlap_detail),
                    file_name="lab_overlap_detail__2026-04-04_v12.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.exception(e)

    with tab2:
        st.subheader("Daily keep / strip splitter")
        st.caption("Upload the winner-engine daily playlist and the separate skip daily file. This app creates KEEP and STRIP lists without altering either engine.")

        winner_daily_file = st.file_uploader("Upload Winner Engine daily playlist CSV", key="winner_daily_file")
        skip_daily_file = st.file_uploader("Upload Skip daily scored/playable CSV", key="skip_daily_file")

        if winner_daily_file and skip_daily_file:
            try:
                winner_daily_raw = load_table(winner_daily_file)
                skip_daily_raw = load_table(skip_daily_file)
                winner_daily = prep_winner_daily(winner_daily_raw)
                skip_daily = prep_skip_daily(skip_daily_raw)
                keep_df, strip_df, merged_df = build_daily_split(winner_daily, skip_daily)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**KEEP list**")
                    st.dataframe(keep_df.head(500), use_container_width=True)
                with c2:
                    st.markdown("**STRIP list**")
                    st.dataframe(strip_df.head(500), use_container_width=True)

                st.markdown("**Merged daily board**")
                st.dataframe(merged_df.head(1000), use_container_width=True)

                st.download_button(
                    "Download daily_keep_list__2026-04-04_v12.csv",
                    df_to_csv_bytes(keep_df),
                    file_name="daily_keep_list__2026-04-04_v12.csv",
                    mime="text/csv",
                )
                st.download_button(
                    "Download daily_strip_list__2026-04-04_v12.csv",
                    df_to_csv_bytes(strip_df),
                    file_name="daily_strip_list__2026-04-04_v12.csv",
                    mime="text/csv",
                )
                st.download_button(
                    "Download daily_merged_board__2026-04-04_v12.csv",
                    df_to_csv_bytes(merged_df),
                    file_name="daily_merged_board__2026-04-04_v12.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.exception(e)


if __name__ == "__main__":
    main()

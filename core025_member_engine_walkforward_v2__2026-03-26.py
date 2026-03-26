#!/usr/bin/env python3
# core025_member_engine_walkforward_v2__2026-03-26.py
#
# Fast walk-forward validator for Core025 Member Engine
# - Full file
# - Uses only uploaded history
# - True walk-forward on historical Core025 HIT events
# - Measures Top1 / Top2 / Top3 / play-rule capture
# - Much faster than scoring every event on the full board
#
# Why this version exists:
# The prior validator scored every transition against all prior transitions.
# On a full multi-year history that becomes too slow for Streamlit Cloud.
# This version validates the member engine on the events that matter most:
# the historical Core025 hit events, using only earlier data each time.

from __future__ import annotations

import io
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

CORE025 = ["0025", "0225", "0255"]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def norm_result(r: object) -> Optional[str]:
    d = re.findall(r"\d", str(r))
    return "".join(d[:4]) if len(d) >= 4 else None


def to_member(r4: Optional[str]) -> Optional[str]:
    if r4 is None:
        return None
    s = "".join(sorted(r4))
    return s if s in CORE025 else None


def load_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".txt") or name.endswith(".tsv"):
        data = uploaded_file.getvalue()
        try:
            return pd.read_csv(io.BytesIO(data), sep="\t", header=None)
        except Exception:
            return pd.read_csv(io.BytesIO(data), sep=None, engine="python", header=None)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def safe_display_df(df: pd.DataFrame, rows: int) -> pd.DataFrame:
    return df.head(int(rows)).copy()


# ---------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------

def features(seed: object) -> Optional[Dict[str, int]]:
    if seed is None:
        return None
    d = re.findall(r"\d", str(seed))
    if len(d) < 4:
        return None
    d = [int(x) for x in d[:4]]
    cnt = Counter(d)
    return {
        "sum": sum(d),
        "spread": max(d) - min(d),
        "even": sum(x % 2 == 0 for x in d),
        "high": sum(x >= 5 for x in d),
        "unique": len(set(d)),
        "pair": int(len(set(d)) < 4),
        "pos1": d[0],
        "pos2": d[1],
        "pos3": d[2],
        "pos4": d[3],
        "max_rep": max(cnt.values()),
    }


def similarity(a: Dict[str, int], b: pd.Series) -> float:
    score = 0.0
    if a["sum"] == b["sum"]:
        score += 3
    elif abs(a["sum"] - b["sum"]) <= 2:
        score += 1

    if a["spread"] == b["spread"]:
        score += 3
    elif abs(a["spread"] - b["spread"]) <= 1:
        score += 1

    if a["even"] == b["even"]:
        score += 2
    if a["high"] == b["high"]:
        score += 2
    if a["unique"] == b["unique"]:
        score += 2
    if a["pair"] == b["pair"]:
        score += 2
    if a["max_rep"] == b["max_rep"]:
        score += 2

    if a["pos1"] == b["pos1"]:
        score += 2
    if a["pos2"] == b["pos2"]:
        score += 2
    if a["pos3"] == b["pos3"]:
        score += 1
    if a["pos4"] == b["pos4"]:
        score += 1

    return score


# ---------------------------------------------------------
# History prep
# ---------------------------------------------------------

def prepare_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if len(df.columns) == 4:
        df.columns = ["date", "jurisdiction", "game", "result"]
    else:
        cols = [str(c).lower() for c in df.columns]
        df.columns = cols
        rename_map = {}
        if "date" not in df.columns:
            for c in df.columns:
                if "date" in c:
                    rename_map[c] = "date"
                    break
        if "jurisdiction" not in df.columns:
            for c in df.columns:
                if "jurisdiction" in c or "state" in c:
                    rename_map[c] = "jurisdiction"
                    break
        if "game" not in df.columns:
            for c in df.columns:
                if "game" in c or "stream" in c:
                    rename_map[c] = "game"
                    break
        if "result" not in df.columns:
            for c in df.columns:
                if "result" in c:
                    rename_map[c] = "result"
                    break
        df = df.rename(columns=rename_map)
        needed = {"date", "jurisdiction", "game", "result"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["r4"] = df["result"].apply(norm_result)
    df["member"] = df["r4"].apply(to_member)
    df["stream"] = df["jurisdiction"].astype(str) + "|" + df["game"].astype(str)
    df = df.dropna(subset=["r4", "date"]).reset_index(drop=True)

    feat_series = df["r4"].apply(features)
    valid_mask = feat_series.notna()
    df = df.loc[valid_mask].reset_index(drop=True)
    feat_df = feat_series.loc[valid_mask].apply(pd.Series).reset_index(drop=True)
    return pd.concat([df.reset_index(drop=True), feat_df], axis=1)


def build_transition_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stream, g in df.groupby("stream"):
        g = g.sort_values("date").reset_index(drop=True)
        for i in range(1, len(g)):
            seed = g.loc[i - 1, "r4"]
            nxt_member = g.loc[i, "member"]
            nxt_r4 = g.loc[i, "r4"]
            feat = features(seed)
            if feat is None:
                continue
            rows.append({
                "stream": stream,
                "jurisdiction": g.loc[i, "jurisdiction"],
                "game": g.loc[i, "game"],
                "seed_date": g.loc[i - 1, "date"],
                "event_date": g.loc[i, "date"],
                "seed": seed,
                "next_r4": nxt_r4,
                "next_member": nxt_member,
                "is_core025_hit": int(nxt_member is not None),
                **feat,
            })
    out = pd.DataFrame(rows)
    out = out.sort_values(["event_date", "stream", "seed"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------
# Member scoring
# ---------------------------------------------------------

def score_seed_against_pool(seed: str, pool: pd.DataFrame, stream_filter: Optional[str], min_stream_history: int) -> List[Tuple[str, float]]:
    seed_feat = features(seed)
    if seed_feat is None:
        return [(m, 1 / 3) for m in CORE025]

    # stream-specific pool only if enough prior stream examples
    if stream_filter is not None:
        stream_subset = pool[pool["stream"] == stream_filter].copy()
        if len(stream_subset) >= int(min_stream_history):
            pool = stream_subset

    if len(pool) == 0:
        return [(m, 1 / 3) for m in CORE025]

    pool = pool.sort_values("event_date").reset_index(drop=True)
    recency_weights = np.linspace(0.50, 1.50, len(pool)) if len(pool) else np.array([])

    scores = {m: 0.0 for m in CORE025}
    total_weight = 0.0

    for idx, r in pool.iterrows():
        if pd.isna(r["next_member"]) or r["next_member"] is None:
            continue
        sim = similarity(seed_feat, r)
        if sim <= 0:
            continue
        weight = float(sim) * float(recency_weights[idx])
        scores[r["next_member"]] += weight
        total_weight += weight

    if total_weight <= 0:
        return [(m, 1 / 3) for m in CORE025]

    probs = {m: scores[m] / total_weight for m in CORE025}
    ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    return ranked


def classify_play(top1_score: float, gap12: float, top1_only_threshold: float, play_two_threshold: float, min_gap_for_top1_only: float) -> str:
    if top1_score >= top1_only_threshold and gap12 >= min_gap_for_top1_only:
        return "Top1 only"
    if top1_score >= play_two_threshold:
        return "Top1 + Top2"
    return "Skip member play"


# ---------------------------------------------------------
# Fast walk-forward: score only Core025 HIT events
# ---------------------------------------------------------

def run_walkforward_hit_events(
    transitions: pd.DataFrame,
    top1_only_threshold: float,
    play_two_threshold: float,
    min_gap_for_top1_only: float,
    min_global_history: int,
    min_stream_history: int,
) -> pd.DataFrame:
    rows = []

    hit_indices = transitions.index[transitions["is_core025_hit"] == 1].tolist()
    progress_bar = st.progress(0.0)
    status = st.empty()

    total = len(hit_indices)

    for n, idx in enumerate(hit_indices, start=1):
        current = transitions.iloc[idx]
        prior = transitions.iloc[:idx].copy()

        if len(prior) < int(min_global_history):
            continue

        ranked = score_seed_against_pool(
            seed=current["seed"],
            pool=prior,
            stream_filter=current["stream"],
            min_stream_history=int(min_stream_history),
        )

        top1, top1_score = ranked[0]
        top2, top2_score = ranked[1]
        top3, top3_score = ranked[2]
        gap12 = top1_score - top2_score
        recommendation = classify_play(
            top1_score=top1_score,
            gap12=gap12,
            top1_only_threshold=float(top1_only_threshold),
            play_two_threshold=float(play_two_threshold),
            min_gap_for_top1_only=float(min_gap_for_top1_only),
        )

        actual_member = current["next_member"] if pd.notna(current["next_member"]) else ""

        rows.append({
            "event_date": current["event_date"],
            "stream": current["stream"],
            "jurisdiction": current["jurisdiction"],
            "game": current["game"],
            "seed": current["seed"],
            "next_r4": current["next_r4"],
            "actual_member": actual_member,
            "Top1": top1,
            "Top1_score": top1_score,
            "Top2": top2,
            "Top2_score": top2_score,
            "Top3": top3,
            "Top3_score": top3_score,
            "Top1_minus_Top2": gap12,
            "recommendation": recommendation,
            "top1_hit": int(actual_member == top1),
            "top2_hit": int(actual_member in [top1, top2]),
            "top3_hit": int(actual_member in [top1, top2, top3]),
            "member_play_hit": int(
                (recommendation == "Top1 only" and actual_member == top1) or
                (recommendation == "Top1 + Top2" and actual_member in [top1, top2])
            ),
        })

        if total:
            progress_bar.progress(n / total)
            if n % 10 == 0 or n == total:
                status.write(f"Scored {n:,} of {total:,} Core025 hit events...")

    progress_bar.empty()
    status.empty()
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Summaries
# ---------------------------------------------------------

def summarize_hit_capture(wf_hits: pd.DataFrame) -> pd.DataFrame:
    total_hits = len(wf_hits)
    rows = [
        {
            "metric": "Top1 capture on Core025 hit events",
            "numerator": int(wf_hits["top1_hit"].sum()),
            "denominator": total_hits,
            "rate": float(wf_hits["top1_hit"].mean()) if total_hits else np.nan,
        },
        {
            "metric": "Top2 capture on Core025 hit events",
            "numerator": int(wf_hits["top2_hit"].sum()),
            "denominator": total_hits,
            "rate": float(wf_hits["top2_hit"].mean()) if total_hits else np.nan,
        },
        {
            "metric": "Top3 capture on Core025 hit events",
            "numerator": int(wf_hits["top3_hit"].sum()),
            "denominator": total_hits,
            "rate": float(wf_hits["top3_hit"].mean()) if total_hits else np.nan,
        },
        {
            "metric": "Play-rule capture on Core025 hit events",
            "numerator": int(wf_hits["member_play_hit"].sum()),
            "denominator": total_hits,
            "rate": float(wf_hits["member_play_hit"].mean()) if total_hits else np.nan,
        },
    ]
    return pd.DataFrame(rows)


def summarize_by_recommendation(wf_hits: pd.DataFrame) -> pd.DataFrame:
    out = wf_hits.groupby("recommendation", dropna=False).agg(
        events=("recommendation", "size"),
        avg_top1_score=("Top1_score", "mean"),
        avg_gap12=("Top1_minus_Top2", "mean"),
        top1_capture=("top1_hit", "mean"),
        top2_capture=("top2_hit", "mean"),
        play_rule_capture=("member_play_hit", "mean"),
    ).reset_index()
    return out.sort_values("events", ascending=False).reset_index(drop=True)


def summarize_top2_needed(wf_hits: pd.DataFrame) -> pd.DataFrame:
    hits = wf_hits.sort_values(["Top1_minus_Top2", "Top1_score"], ascending=[True, True]).reset_index(drop=True)
    if len(hits) == 0:
        return pd.DataFrame()
    q = min(10, len(hits))
    hits["bucket"] = pd.qcut(hits.index + 1, q=q, duplicates="drop")
    out = hits.groupby("bucket").agg(
        events=("bucket", "size"),
        avg_gap12=("Top1_minus_Top2", "mean"),
        avg_top1_score=("Top1_score", "mean"),
        top1_capture=("top1_hit", "mean"),
        top2_capture=("top2_hit", "mean"),
        play_rule_capture=("member_play_hit", "mean"),
    ).reset_index()
    return out


# ---------------------------------------------------------
# App
# ---------------------------------------------------------

def app():
    st.set_page_config(page_title="Core025 Member Engine Walk-Forward v2", layout="wide")
    st.title("Core025 Member Engine Walk-Forward Validator v2")
    st.caption("Fast true walk-forward on historical Core025 hit events. This is the right validator for confirming whether the member engine reaches 75%+ historical capture.")

    with st.sidebar:
        st.header("Walk-forward controls")
        min_global_history = st.number_input("Minimum prior transitions before scoring", min_value=10, value=100, step=10)
        min_stream_history = st.number_input("Minimum stream-specific history to use stream-only pool", min_value=0, value=20, step=5)

        st.header("Play-rule controls")
        top1_only_threshold = st.slider("Top1-only threshold", min_value=0.33, max_value=0.80, value=0.375, step=0.005)
        play_two_threshold = st.slider("Top1+Top2 threshold", min_value=0.33, max_value=0.80, value=0.350, step=0.005)
        min_gap_for_top1_only = st.slider("Minimum Top1-Top2 gap for Top1-only", min_value=0.00, max_value=0.20, value=0.015, step=0.001)

        st.header("Target")
        target_capture = st.slider("Required capture target", min_value=0.50, max_value=0.95, value=0.75, step=0.01)
        rows_to_show = st.number_input("Rows to display", min_value=5, value=25, step=5)

    hist_file = st.file_uploader("Upload FULL history file", key="wf_hist_file_v2")
    if not hist_file:
        st.info("Upload the full history file to begin.")
        return

    try:
        hist = prepare_history(load_table(hist_file))
    except Exception as e:
        st.exception(e)
        return

    transitions = build_transition_rows(hist)
    core025_hit_events = int(transitions["is_core025_hit"].sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Transitions", f"{len(transitions):,}")
    c2.metric("Core025 hit events", f"{core025_hit_events:,}")
    c3.metric("Base Core025 rate", f"{transitions['is_core025_hit'].mean():.4f}")

    if st.button("Run Member Engine Walk-Forward", type="primary"):
        try:
            with st.spinner("Running fast true walk-forward on Core025 hit events..."):
                wf_hits = run_walkforward_hit_events(
                    transitions=transitions,
                    top1_only_threshold=float(top1_only_threshold),
                    play_two_threshold=float(play_two_threshold),
                    min_gap_for_top1_only=float(min_gap_for_top1_only),
                    min_global_history=int(min_global_history),
                    min_stream_history=int(min_stream_history),
                )
            st.session_state["member_wf_v2_results"] = {
                "wf_hits": wf_hits,
                "summary_capture": summarize_hit_capture(wf_hits),
                "summary_recommendation": summarize_by_recommendation(wf_hits),
                "summary_top2": summarize_top2_needed(wf_hits),
            }
            st.rerun()
        except Exception as e:
            st.exception(e)
            return

    results = st.session_state.get("member_wf_v2_results")
    if results is None:
        return

    wf_hits = results["wf_hits"]
    summary_capture = results["summary_capture"]
    summary_recommendation = results["summary_recommendation"]
    summary_top2 = results["summary_top2"]

    st.subheader("Core025 hit-event capture summary")
    st.dataframe(summary_capture, use_container_width=True)

    play_rule_row = summary_capture[summary_capture["metric"] == "Play-rule capture on Core025 hit events"]
    if len(play_rule_row):
        rate = float(play_rule_row.iloc[0]["rate"])
        if rate >= float(target_capture):
            st.success(f"Target met: play-rule capture = {rate:.2%}, at or above the {target_capture:.2%} target.")
        else:
            st.error(f"Target not met: play-rule capture = {rate:.2%}, below the {target_capture:.2%} target.")

    st.subheader("By recommendation")
    st.dataframe(summary_recommendation, use_container_width=True)

    st.subheader("Top2-needed pockets by low Top1-Top2 gap")
    st.dataframe(summary_top2, use_container_width=True)

    st.subheader("Walk-forward Core025 hit-event table")
    st.dataframe(safe_display_df(wf_hits, int(rows_to_show)), use_container_width=True)

    st.download_button("Download walk-forward hit-event table CSV", data=df_to_csv_bytes(wf_hits), file_name="core025_member_engine_walkforward_v2_hit_events__2026-03-26.csv", mime="text/csv")
    st.download_button("Download capture summary CSV", data=df_to_csv_bytes(summary_capture), file_name="core025_member_engine_walkforward_v2_capture_summary__2026-03-26.csv", mime="text/csv")
    st.download_button("Download recommendation summary CSV", data=df_to_csv_bytes(summary_recommendation), file_name="core025_member_engine_walkforward_v2_recommendation_summary__2026-03-26.csv", mime="text/csv")
    st.download_button("Download Top2-needed summary CSV", data=df_to_csv_bytes(summary_top2), file_name="core025_member_engine_walkforward_v2_top2_needed_summary__2026-03-26.csv", mime="text/csv")


if __name__ == "__main__":
    if "member_wf_v2_results" not in st.session_state:
        st.session_state["member_wf_v2_results"] = None
    app()

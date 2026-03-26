#!/usr/bin/env python3
# core025_member_engine_v2_1__2026-03-26.py
#
# Member Ranking Engine v2.1
# - Trait-based member ranking for 0025 / 0225 / 0255
# - Safe handling of malformed seeds
# - Sidebar controls for Top1-only / Top1+Top2 / Skip thresholds
# - Plays shown in bold in the on-screen recommendation table
# - Full export with explicit play flags
#
# Full file. No placeholders.

from __future__ import annotations

import io
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

CORE025 = ["0025", "0225", "0255"]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def norm(r: object) -> Optional[str]:
    d = re.findall(r"\d", str(r))
    return "".join(d[:4]) if len(d) >= 4 else None


def to_member(r: Optional[str]) -> Optional[str]:
    if r is None:
        return None
    s = "".join(sorted(r))
    return s if s in CORE025 else None


def load_table(f) -> pd.DataFrame:
    name = f.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(f)
    if name.endswith(".txt") or name.endswith(".tsv"):
        data = f.getvalue()
        try:
            return pd.read_csv(io.BytesIO(data), sep="\t", header=None)
        except Exception:
            return pd.read_csv(io.BytesIO(data), sep=None, engine="python", header=None)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(f)
    raise ValueError(f"Unsupported file type: {f.name}")


# ---------------------------------------------------------
# Feature builder (safe)
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


# ---------------------------------------------------------
# Prepare history
# ---------------------------------------------------------

def prep_history(df: pd.DataFrame) -> pd.DataFrame:
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
    df["r4"] = df["result"].apply(norm)
    df["member"] = df["r4"].apply(to_member)
    df["stream"] = df["jurisdiction"].astype(str) + "|" + df["game"].astype(str)

    df = df.dropna(subset=["r4"]).reset_index(drop=True)

    feat_series = df["r4"].apply(features)
    valid_mask = feat_series.notna()
    df = df.loc[valid_mask].reset_index(drop=True)
    feat_df = feat_series.loc[valid_mask].apply(pd.Series).reset_index(drop=True)

    out = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
    return out


def prep_survivors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "stream_id" in df.columns and "stream" not in df.columns:
        df["stream"] = df["stream_id"]

    if "seed" not in df.columns:
        raise ValueError("Survivor file must contain a 'seed' column.")

    if "stream" not in df.columns:
        raise ValueError("Survivor file must contain 'stream' or 'stream_id'.")

    df = df[df["seed"].notna()].copy()
    df["seed"] = df["seed"].astype(str)
    df = df[df["seed"].str.len() >= 4].copy().reset_index(drop=True)

    feat_series = df["seed"].apply(features)
    valid_mask = feat_series.notna()
    df = df.loc[valid_mask].reset_index(drop=True)
    feat_df = feat_series.loc[valid_mask].apply(pd.Series).reset_index(drop=True)

    out = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
    return out


# ---------------------------------------------------------
# Build transitions
# ---------------------------------------------------------

def build_transitions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for stream, g in df.groupby("stream"):
        g = g.sort_values("date").reset_index(drop=True)
        for i in range(1, len(g)):
            seed = g.loc[i - 1, "r4"]
            nxt_member = g.loc[i, "member"]

            feat = features(seed)
            if feat is None:
                continue

            rows.append({
                "stream": stream,
                "seed": seed,
                "next_member": nxt_member,
                "transition_date": g.loc[i, "date"],
                **feat,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Similarity scoring
# ---------------------------------------------------------

def similarity(a: Dict[str, int], b: pd.Series) -> int:
    score = 0

    # stronger global shape matching
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

    # position weighting
    if a["pos1"] == b["pos1"]:
        score += 2
    if a["pos2"] == b["pos2"]:
        score += 2
    if a["pos3"] == b["pos3"]:
        score += 1
    if a["pos4"] == b["pos4"]:
        score += 1

    return score


def score_seed(seed: str, transitions: pd.DataFrame, stream_filter: Optional[str] = None) -> List[tuple[str, float]]:
    seed_feat = features(seed)
    if seed_feat is None:
        return [(m, 1 / 3) for m in CORE025]

    pool = transitions.copy()
    if stream_filter is not None and "stream" in pool.columns:
        stream_subset = pool[pool["stream"] == stream_filter].copy()
        # Use stream-specific history only if there is enough support
        if len(stream_subset) >= 20:
            pool = stream_subset

    scores = {m: 0.0 for m in CORE025}
    total_weight = 0.0

    if "transition_date" in pool.columns:
        pool = pool.sort_values("transition_date").reset_index(drop=True)
        # simple recency weighting
        recency_weights = np.linspace(0.5, 1.5, len(pool)) if len(pool) else np.array([])
    else:
        recency_weights = np.ones(len(pool))

    for idx, r in pool.iterrows():
        if r["next_member"] is None or pd.isna(r["next_member"]):
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


# ---------------------------------------------------------
# Apply to survivors
# ---------------------------------------------------------

def classify_play(top1_score: float, top2_score: float, top1_minus_top2: float, top1_only_threshold: float, play_two_threshold: float, min_gap_for_top1_only: float) -> tuple[str, bool, bool, bool]:
    if top1_score >= top1_only_threshold and top1_minus_top2 >= min_gap_for_top1_only:
        return "Top1 only", True, False, False

    if top1_score >= play_two_threshold:
        return "Top1 + Top2", True, True, False

    return "Skip member play", False, False, False


def apply_survivors(
    surv: pd.DataFrame,
    transitions: pd.DataFrame,
    top1_only_threshold: float,
    play_two_threshold: float,
    min_gap_for_top1_only: float,
) -> pd.DataFrame:
    rows = []

    for _, r in surv.iterrows():
        seed = str(r["seed"])
        stream = r.get("stream")

        ranked = score_seed(seed, transitions, stream_filter=stream)

        top1, top1_score = ranked[0]
        top2, top2_score = ranked[1]
        top3, top3_score = ranked[2]
        gap12 = top1_score - top2_score
        recommendation, play_top1, play_top2, play_top3 = classify_play(
            top1_score=top1_score,
            top2_score=top2_score,
            top1_minus_top2=gap12,
            top1_only_threshold=top1_only_threshold,
            play_two_threshold=play_two_threshold,
            min_gap_for_top1_only=min_gap_for_top1_only,
        )

        rows.append({
            "stream": stream,
            "seed": seed,
            "Top1": top1,
            "Top1_score": top1_score,
            "Top2": top2,
            "Top2_score": top2_score,
            "Top3": top3,
            "Top3_score": top3_score,
            "Top1_minus_Top2": gap12,
            "recommendation": recommendation,
            "play_top1": int(play_top1),
            "play_top2": int(play_top2),
            "play_top3": int(play_top3),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(["Top1_score", "Top1_minus_Top2"], ascending=[False, False]).reset_index(drop=True)
    return out


# ---------------------------------------------------------
# Display helpers
# ---------------------------------------------------------

def format_bold_member_table(df: pd.DataFrame) -> str:
    display_df = df.copy()

    def bold_if(flag: int, val: object) -> str:
        txt = str(val)
        return f"<b>{txt}</b>" if int(flag) == 1 else txt

    display_df["Top1"] = [bold_if(f, v) for f, v in zip(display_df["play_top1"], display_df["Top1"])]
    display_df["Top2"] = [bold_if(f, v) for f, v in zip(display_df["play_top2"], display_df["Top2"])]
    display_df["Top3"] = [bold_if(f, v) for f, v in zip(display_df["play_top3"], display_df["Top3"])]

    show_cols = [
        "stream", "seed",
        "Top1", "Top1_score",
        "Top2", "Top2_score",
        "Top3", "Top3_score",
        "Top1_minus_Top2",
        "recommendation",
    ]

    fmt = display_df[show_cols].copy()
    fmt["Top1_score"] = fmt["Top1_score"].map(lambda x: f"{x:.4f}")
    fmt["Top2_score"] = fmt["Top2_score"].map(lambda x: f"{x:.4f}")
    fmt["Top3_score"] = fmt["Top3_score"].map(lambda x: f"{x:.4f}")
    fmt["Top1_minus_Top2"] = fmt["Top1_minus_Top2"].map(lambda x: f"{x:.4f}")

    return fmt.to_html(index=False, escape=False)


# ---------------------------------------------------------
# App
# ---------------------------------------------------------

def app():
    st.set_page_config(page_title="Member Ranking Engine v2.1", layout="wide")
    st.title("Member Ranking Engine v2.1 (Trait-Based + Bold Plays)")

    with st.sidebar:
        st.header("Play controls")
        top1_only_threshold = st.slider("Top1-only threshold", min_value=0.33, max_value=0.80, value=0.375, step=0.005)
        play_two_threshold = st.slider("Top1+Top2 threshold", min_value=0.33, max_value=0.80, value=0.350, step=0.005)
        min_gap_for_top1_only = st.slider("Minimum Top1-Top2 gap for Top1-only", min_value=0.00, max_value=0.20, value=0.015, step=0.001)
        rows_to_show = st.number_input("Rows to display", min_value=5, value=50, step=5)

    hist_file = st.file_uploader("Upload FULL history file", key="hist_file")
    surv_file = st.file_uploader("Upload PLAY survivors file", key="surv_file")

    if not hist_file or not surv_file:
        st.info("Upload both the full history file and the PLAY survivors file to begin.")
        return

    try:
        hist = prep_history(load_table(hist_file))
        surv = prep_survivors(load_table(surv_file))
    except Exception as e:
        st.exception(e)
        return

    transitions = build_transitions(hist)
    results = apply_survivors(
        surv=surv,
        transitions=transitions,
        top1_only_threshold=float(top1_only_threshold),
        play_two_threshold=float(play_two_threshold),
        min_gap_for_top1_only=float(min_gap_for_top1_only),
    )

    st.subheader("Recommendation summary")
    summary = results["recommendation"].value_counts(dropna=False).rename_axis("recommendation").reset_index(name="count")
    st.dataframe(summary, use_container_width=True)

    st.subheader("Member rankings")
    html_table = format_bold_member_table(results.head(int(rows_to_show)))
    st.markdown(html_table, unsafe_allow_html=True)

    st.caption("Only the members actually to be played are shown in bold. If a row is Top1 only, only Top1 is bold.")

    st.download_button(
        "Download member_rankings_v2_1.csv",
        results.to_csv(index=False),
        "member_rankings_v2_1.csv",
        "text/csv",
    )


if __name__ == "__main__":
    app()

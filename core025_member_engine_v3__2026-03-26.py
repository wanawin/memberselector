#!/usr/bin/env python3
# core025_member_engine_v3__2026-03-26.py
#
# Core025 Member Ranking Engine v3
# Full file. No placeholders.
#
# What is new in v3:
# - Trait-based similarity scoring
# - Stream-specific member bias
# - Direct seed->member transition weighting
# - Seed family / sorted-seed transition weighting
# - Recency weighting
# - Gap-aware recommendations
# - Bold on-screen play display
#
# Purpose:
# Rank 0025 / 0225 / 0255 for current PLAY survivors from the skip ladder.

from __future__ import annotations

import io
import re
from collections import Counter, defaultdict
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
# Feature builder
# ---------------------------------------------------------

def features(seed: object) -> Optional[Dict[str, object]]:
    if seed is None:
        return None

    d = re.findall(r"\d", str(seed))
    if len(d) < 4:
        return None

    d = [int(x) for x in d[:4]]
    cnt = Counter(d)

    pair_tokens = []
    for i in range(4):
        for j in range(i + 1, 4):
            pair_tokens.append("".join(sorted([str(d[i]), str(d[j])])))

    feat = {
        "sum": sum(d),
        "spread": max(d) - min(d),
        "even": sum(x % 2 == 0 for x in d),
        "high": sum(x >= 5 for x in d),
        "unique": len(set(d)),
        "pair": int(len(set(d)) < 4),
        "max_rep": max(cnt.values()),
        "pos1": d[0],
        "pos2": d[1],
        "pos3": d[2],
        "pos4": d[3],
        "sorted_seed": "".join(map(str, sorted(d))),
        "first2": f"{d[0]}{d[1]}",
        "last2": f"{d[2]}{d[3]}",
        "pair_tokens": tuple(sorted(pair_tokens)),
    }

    for k in range(10):
        feat[f"has{k}"] = int(k in cnt)
        feat[f"cnt{k}"] = int(cnt.get(k, 0))

    return feat


def similarity(a: Dict[str, object], b: pd.Series) -> float:
    score = 0.0

    # stronger whole-shape matching
    if a["sum"] == b["sum"]:
        score += 4
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

    # positions
    if a["pos1"] == b["pos1"]:
        score += 2
    if a["pos2"] == b["pos2"]:
        score += 2
    if a["pos3"] == b["pos3"]:
        score += 1
    if a["pos4"] == b["pos4"]:
        score += 1

    # exact structural signatures
    if a["sorted_seed"] == b["sorted_seed"]:
        score += 6
    if a["first2"] == b["first2"]:
        score += 1
    if a["last2"] == b["last2"]:
        score += 1

    # digit presence / counts
    for k in range(10):
        if a[f"has{k}"] == b[f"has{k}"]:
            score += 0.35
        if a[f"cnt{k}"] == b[f"cnt{k}"]:
            score += 0.20

    # pair token overlap
    pair_overlap = len(set(a["pair_tokens"]).intersection(set(b["pair_tokens"])))
    score += pair_overlap * 0.50

    return float(score)


# ---------------------------------------------------------
# Data preparation
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
    df["r4"] = df["result"].apply(norm_result)
    df["member"] = df["r4"].apply(to_member)
    df["stream"] = df["jurisdiction"].astype(str) + "|" + df["game"].astype(str)

    df = df.dropna(subset=["date", "r4"]).reset_index(drop=True)

    feat_series = df["r4"].apply(features)
    valid_mask = feat_series.notna()
    df = df.loc[valid_mask].reset_index(drop=True)
    feat_df = feat_series.loc[valid_mask].apply(pd.Series).reset_index(drop=True)

    return pd.concat([df.reset_index(drop=True), feat_df], axis=1)


def prep_survivors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "stream_id" in df.columns and "stream" not in df.columns:
        df["stream"] = df["stream_id"]

    if "stream" not in df.columns:
        raise ValueError("Survivor file must contain 'stream' or 'stream_id'.")
    if "seed" not in df.columns:
        raise ValueError("Survivor file must contain 'seed'.")

    df = df[df["seed"].notna()].copy()
    df["seed"] = df["seed"].astype(str)
    df = df[df["seed"].str.len() >= 4].reset_index(drop=True)

    feat_series = df["seed"].apply(features)
    valid_mask = feat_series.notna()
    df = df.loc[valid_mask].reset_index(drop=True)
    feat_df = feat_series.loc[valid_mask].apply(pd.Series).reset_index(drop=True)

    return pd.concat([df.reset_index(drop=True), feat_df], axis=1)


def build_transitions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for stream, g in df.groupby("stream"):
        g = g.sort_values("date").reset_index(drop=True)
        for i in range(1, len(g)):
            seed = g.loc[i - 1, "r4"]
            next_member = g.loc[i, "member"]

            feat = features(seed)
            if feat is None:
                continue

            rows.append({
                "stream": stream,
                "seed": seed,
                "seed_date": g.loc[i - 1, "date"],
                "transition_date": g.loc[i, "date"],
                "next_member": next_member,
                **feat,
            })

    out = pd.DataFrame(rows)
    out = out.sort_values(["transition_date", "stream", "seed"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------
# Transition weighting model
# ---------------------------------------------------------

def build_transition_maps(transitions: pd.DataFrame) -> Dict[str, object]:
    exact_seed_map = defaultdict(Counter)
    sorted_seed_map = defaultdict(Counter)
    stream_member_map = defaultdict(Counter)
    global_member_map = Counter()

    for _, r in transitions.iterrows():
        member = r["next_member"]
        if member is None or pd.isna(member):
            continue

        exact_seed_map[str(r["seed"])][member] += 1
        sorted_seed_map[str(r["sorted_seed"])][member] += 1
        stream_member_map[str(r["stream"])][member] += 1
        global_member_map[member] += 1

    return {
        "exact_seed_map": exact_seed_map,
        "sorted_seed_map": sorted_seed_map,
        "stream_member_map": stream_member_map,
        "global_member_map": global_member_map,
    }


def counter_to_probs(c: Counter) -> Dict[str, float]:
    total = sum(c.values())
    if total <= 0:
        return {m: 1 / 3 for m in CORE025}
    return {m: c.get(m, 0) / total for m in CORE025}


def score_seed_v3(
    seed: str,
    stream: Optional[str],
    transitions: pd.DataFrame,
    maps: Dict[str, object],
    min_stream_history: int,
    use_stream_bias_weight: float,
    use_exact_seed_weight: float,
    use_sorted_seed_weight: float,
    use_similarity_weight: float,
) -> List[Tuple[str, float]]:
    seed_feat = features(seed)
    if seed_feat is None:
        return [(m, 1 / 3) for m in CORE025]

    exact_seed_map = maps["exact_seed_map"]
    sorted_seed_map = maps["sorted_seed_map"]
    stream_member_map = maps["stream_member_map"]
    global_member_map = maps["global_member_map"]

    score_accum = {m: 0.0 for m in CORE025}

    # 1) global member baseline
    global_probs = counter_to_probs(global_member_map)
    for m in CORE025:
        score_accum[m] += global_probs[m] * 0.25

    # 2) stream bias
    if stream is not None and sum(stream_member_map[str(stream)].values()) >= int(min_stream_history):
        stream_probs = counter_to_probs(stream_member_map[str(stream)])
        for m in CORE025:
            score_accum[m] += stream_probs[m] * float(use_stream_bias_weight)

    # 3) exact seed -> member
    if str(seed) in exact_seed_map and sum(exact_seed_map[str(seed)].values()) > 0:
        exact_probs = counter_to_probs(exact_seed_map[str(seed)])
        for m in CORE025:
            score_accum[m] += exact_probs[m] * float(use_exact_seed_weight)

    # 4) sorted seed -> member
    sorted_key = str(seed_feat["sorted_seed"])
    if sorted_key in sorted_seed_map and sum(sorted_seed_map[sorted_key].values()) > 0:
        sorted_probs = counter_to_probs(sorted_seed_map[sorted_key])
        for m in CORE025:
            score_accum[m] += sorted_probs[m] * float(use_sorted_seed_weight)

    # 5) similarity pool with optional stream-only subset if enough support
    pool = transitions
    if stream is not None:
        stream_subset = transitions[transitions["stream"] == str(stream)].copy()
        if len(stream_subset) >= int(min_stream_history):
            pool = stream_subset

    if len(pool):
        pool = pool.sort_values("transition_date").reset_index(drop=True)
        recency_weights = np.linspace(0.50, 1.50, len(pool))
        similarity_scores = {m: 0.0 for m in CORE025}
        total_sim_weight = 0.0

        for idx, r in pool.iterrows():
            member = r["next_member"]
            if member is None or pd.isna(member):
                continue
            sim = similarity(seed_feat, r)
            if sim <= 0:
                continue
            weight = float(sim) * float(recency_weights[idx])
            similarity_scores[member] += weight
            total_sim_weight += weight

        if total_sim_weight > 0:
            for m in CORE025:
                score_accum[m] += (similarity_scores[m] / total_sim_weight) * float(use_similarity_weight)

    total = sum(score_accum.values())
    if total <= 0:
        return [(m, 1 / 3) for m in CORE025]

    probs = {m: score_accum[m] / total for m in CORE025}
    ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    return ranked


# ---------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------

def classify_play(
    top1_score: float,
    top2_score: float,
    top1_minus_top2: float,
    top1_only_threshold: float,
    play_two_threshold: float,
    min_gap_for_top1_only: float,
    weak_skip_threshold: float,
) -> Tuple[str, int, int, int]:
    if top1_score < weak_skip_threshold:
        return "Skip member play", 0, 0, 0

    if top1_score >= top1_only_threshold and top1_minus_top2 >= min_gap_for_top1_only:
        return "Top1 only", 1, 0, 0

    if top1_score >= play_two_threshold:
        return "Top1 + Top2", 1, 1, 0

    return "Skip member play", 0, 0, 0


def apply_survivors_v3(
    surv: pd.DataFrame,
    transitions: pd.DataFrame,
    maps: Dict[str, object],
    min_stream_history: int,
    use_stream_bias_weight: float,
    use_exact_seed_weight: float,
    use_sorted_seed_weight: float,
    use_similarity_weight: float,
    top1_only_threshold: float,
    play_two_threshold: float,
    min_gap_for_top1_only: float,
    weak_skip_threshold: float,
) -> pd.DataFrame:
    rows = []

    for _, r in surv.iterrows():
        seed = str(r["seed"])
        stream = str(r["stream"])

        ranked = score_seed_v3(
            seed=seed,
            stream=stream,
            transitions=transitions,
            maps=maps,
            min_stream_history=int(min_stream_history),
            use_stream_bias_weight=float(use_stream_bias_weight),
            use_exact_seed_weight=float(use_exact_seed_weight),
            use_sorted_seed_weight=float(use_sorted_seed_weight),
            use_similarity_weight=float(use_similarity_weight),
        )

        top1, top1_score = ranked[0]
        top2, top2_score = ranked[1]
        top3, top3_score = ranked[2]
        gap12 = top1_score - top2_score

        recommendation, play_top1, play_top2, play_top3 = classify_play(
            top1_score=top1_score,
            top2_score=top2_score,
            top1_minus_top2=gap12,
            top1_only_threshold=float(top1_only_threshold),
            play_two_threshold=float(play_two_threshold),
            min_gap_for_top1_only=float(min_gap_for_top1_only),
            weak_skip_threshold=float(weak_skip_threshold),
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

def format_bold_table(df: pd.DataFrame) -> str:
    work = df.copy()

    def bold(flag: int, val: object) -> str:
        txt = str(val)
        return f"<b>{txt}</b>" if int(flag) == 1 else txt

    work["Top1"] = [bold(f, v) for f, v in zip(work["play_top1"], work["Top1"])]
    work["Top2"] = [bold(f, v) for f, v in zip(work["play_top2"], work["Top2"])]
    work["Top3"] = [bold(f, v) for f, v in zip(work["play_top3"], work["Top3"])]

    cols = [
        "stream", "seed",
        "Top1", "Top1_score",
        "Top2", "Top2_score",
        "Top3", "Top3_score",
        "Top1_minus_Top2",
        "recommendation",
    ]

    show = work[cols].copy()
    for c in ["Top1_score", "Top2_score", "Top3_score", "Top1_minus_Top2"]:
        show[c] = show[c].map(lambda x: f"{x:.4f}")

    return show.to_html(index=False, escape=False)


# ---------------------------------------------------------
# App
# ---------------------------------------------------------

def app():
    st.set_page_config(page_title="Core025 Member Engine v3", layout="wide")
    st.title("Core025 Member Ranking Engine v3")
    st.caption("Weighted member selector using stream bias, direct seed transitions, sorted-seed transitions, recency, and trait similarity.")

    with st.sidebar:
        st.header("Model weights")
        use_stream_bias_weight = st.slider("Stream-bias weight", min_value=0.0, max_value=3.0, value=1.20, step=0.05)
        use_exact_seed_weight = st.slider("Exact-seed weight", min_value=0.0, max_value=3.0, value=1.50, step=0.05)
        use_sorted_seed_weight = st.slider("Sorted-seed weight", min_value=0.0, max_value=3.0, value=1.10, step=0.05)
        use_similarity_weight = st.slider("Similarity weight", min_value=0.0, max_value=3.0, value=1.80, step=0.05)
        min_stream_history = st.number_input("Minimum stream history for stream-only bias/pool", min_value=0, value=20, step=5)

        st.header("Play controls")
        top1_only_threshold = st.slider("Top1-only threshold", min_value=0.33, max_value=0.80, value=0.390, step=0.005)
        play_two_threshold = st.slider("Top1+Top2 threshold", min_value=0.33, max_value=0.80, value=0.360, step=0.005)
        min_gap_for_top1_only = st.slider("Minimum Top1-Top2 gap for Top1-only", min_value=0.00, max_value=0.20, value=0.040, step=0.001)
        weak_skip_threshold = st.slider("Weak-score skip threshold", min_value=0.00, max_value=0.50, value=0.345, step=0.001)

        rows_to_show = st.number_input("Rows to display", min_value=5, value=50, step=5)

    hist_file = st.file_uploader("Upload FULL history file", key="hist_file_v3")
    surv_file = st.file_uploader("Upload PLAY survivors file", key="surv_file_v3")

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
    maps = build_transition_maps(transitions)

    results = apply_survivors_v3(
        surv=surv,
        transitions=transitions,
        maps=maps,
        min_stream_history=int(min_stream_history),
        use_stream_bias_weight=float(use_stream_bias_weight),
        use_exact_seed_weight=float(use_exact_seed_weight),
        use_sorted_seed_weight=float(use_sorted_seed_weight),
        use_similarity_weight=float(use_similarity_weight),
        top1_only_threshold=float(top1_only_threshold),
        play_two_threshold=float(play_two_threshold),
        min_gap_for_top1_only=float(min_gap_for_top1_only),
        weak_skip_threshold=float(weak_skip_threshold),
    )

    st.subheader("Recommendation summary")
    reco_summary = results["recommendation"].value_counts(dropna=False).rename_axis("recommendation").reset_index(name="count")
    st.dataframe(reco_summary, use_container_width=True)

    st.subheader("Member rankings")
    st.markdown(format_bold_table(results.head(int(rows_to_show))), unsafe_allow_html=True)
    st.caption("Only the members actually to be played are bold. If a row is Top1 only, only Top1 is bold.")

    st.download_button(
        "Download member_rankings_v3.csv",
        results.to_csv(index=False),
        "member_rankings_v3.csv",
        "text/csv",
    )


if __name__ == "__main__":
    app()

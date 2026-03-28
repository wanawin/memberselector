#!/usr/bin/env python3
# core025_top2_optimization_walkforward_validator_v1__2026-03-28.py
#
# BUILD: core025_top2_optimization_walkforward_validator_v1__2026-03-28

from __future__ import annotations

import io
import re
from collections import Counter, defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

CORE025 = ["0025", "0225", "0255"]
BUILD_MARKER = "BUILD: core025_top2_optimization_walkforward_validator_v1__2026-03-28"


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


def load_trait_df(f) -> pd.DataFrame:
    df = load_table(f)
    if df.empty:
        return pd.DataFrame(columns=["trait"])
    if "trait" not in df.columns:
        raise ValueError(f"Trait CSV must contain a 'trait' column. File: {f.name}")
    return df.copy()


def norm_result(r: object) -> Optional[str]:
    d = re.findall(r"\d", str(r))
    return "".join(d[:4]) if len(d) >= 4 else None


def to_member(r4: Optional[str]) -> Optional[str]:
    if r4 is None:
        return None
    s = "".join(sorted(r4))
    return s if s in CORE025 else None


def features(seed: object) -> Optional[Dict[str, object]]:
    if seed is None:
        return None
    d = re.findall(r"\d", str(seed))
    if len(d) < 4:
        return None
    digs = [int(x) for x in d[:4]]
    cnt = Counter(digs)
    pair_tokens = []
    for i in range(4):
        for j in range(i + 1, 4):
            pair_tokens.append("".join(sorted([str(digs[i]), str(digs[j])])))
    unique_sorted = sorted(set(digs))
    consec_links = 0
    for a, b in zip(unique_sorted[:-1], unique_sorted[1:]):
        if b - a == 1:
            consec_links += 1
    feat = {
        "sum": sum(digs),
        "spread": max(digs) - min(digs),
        "even": sum(x % 2 == 0 for x in digs),
        "odd": sum(x % 2 != 0 for x in digs),
        "high": sum(x >= 5 for x in digs),
        "low": sum(x <= 4 for x in digs),
        "unique": len(set(digs)),
        "pair": int(len(set(digs)) < 4),
        "max_rep": max(cnt.values()),
        "sorted_seed": "".join(map(str, sorted(digs))),
        "first2": f"{digs[0]}{digs[1]}",
        "last2": f"{digs[2]}{digs[3]}",
        "consec_links": consec_links,
        "parity_pattern": "".join("E" if x % 2 == 0 else "O" for x in digs),
        "highlow_pattern": "".join("H" if x >= 5 else "L" for x in digs),
        "pair_token_pattern": "|".join(sorted(pair_tokens)),
    }
    for k in range(10):
        feat[f"has{k}"] = int(k in cnt)
        feat[f"cnt{k}"] = int(cnt.get(k, 0))
    return feat


def parse_trait_string(trait: str) -> Tuple[str, str]:
    if "=" not in str(trait):
        raise ValueError(f"Invalid trait format: {trait}")
    col, val = str(trait).split("=", 1)
    return col, val


def normalize_scalar_for_compare(x: object) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def row_matches_trait(row: pd.Series, trait: str) -> bool:
    col, val = parse_trait_string(trait)
    if col not in row.index:
        return False
    return normalize_scalar_for_compare(row[col]) == val


def matched_traits_for_row(row: pd.Series, trait_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if trait_df is None or len(trait_df) == 0:
        return pd.DataFrame()
    mask = trait_df["trait"].apply(lambda t: row_matches_trait(row, t))
    return trait_df.loc[mask].copy().reset_index(drop=True)


def similarity(a: Dict[str, object], b: Dict[str, object]) -> float:
    score = 0.0
    if a["sorted_seed"] == b["sorted_seed"]:
        score += 6
    if a["pair_token_pattern"] == b["pair_token_pattern"]:
        score += 4
    if a["parity_pattern"] == b["parity_pattern"]:
        score += 2
    if a["highlow_pattern"] == b["highlow_pattern"]:
        score += 2
    if a["unique"] == b["unique"]:
        score += 1
    if a["max_rep"] == b["max_rep"]:
        score += 1
    return float(score)


def counter_to_probs(c: Counter) -> Dict[str, float]:
    total = sum(c.values())
    if total <= 0:
        return {m: 1/3 for m in CORE025}
    return {m: c.get(m, 0) / total for m in CORE025}


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
                    rename_map[c] = "date"; break
        if "jurisdiction" not in df.columns:
            for c in df.columns:
                if "jurisdiction" in c or "state" in c:
                    rename_map[c] = "jurisdiction"; break
        if "game" not in df.columns:
            for c in df.columns:
                if "game" in c or "stream" in c:
                    rename_map[c] = "game"; break
        if "result" not in df.columns:
            for c in df.columns:
                if "result" in c:
                    rename_map[c] = "result"; break
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


def build_transitions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stream, g in df.groupby("stream"):
        g = g.sort_values("date").reset_index(drop=True)
        for i in range(1, len(g)):
            seed = g.loc[i-1, "r4"]
            next_member = g.loc[i, "member"]
            next_r4 = g.loc[i, "r4"]
            feat = features(seed)
            if feat is None:
                continue
            rows.append({
                "stream": stream,
                "jurisdiction": g.loc[i, "jurisdiction"],
                "game": g.loc[i, "game"],
                "seed_date": g.loc[i-1, "date"],
                "event_date": g.loc[i, "date"],
                "year_month": g.loc[i, "date"].to_period("M").strftime("%Y-%m"),
                "seed": seed,
                "next_r4": next_r4,
                "next_member": next_member,
                "is_core025_hit": int(next_member is not None),
                **feat,
            })
    return pd.DataFrame(rows).sort_values(["event_date", "stream", "seed"]).reset_index(drop=True)


def initialize_state(max_global_similarity_pool: int, max_stream_similarity_pool: int) -> Dict[str, object]:
    return {
        "exact_seed_map": defaultdict(Counter),
        "sorted_seed_map": defaultdict(Counter),
        "stream_member_map": defaultdict(Counter),
        "global_member_map": Counter(),
        "global_recent_pool": deque(maxlen=int(max_global_similarity_pool)),
        "stream_recent_pool": defaultdict(lambda: deque(maxlen=int(max_stream_similarity_pool))),
        "transitions_seen": 0,
    }


def update_state_with_event(state: Dict[str, object], current: pd.Series) -> None:
    member = current["next_member"] if pd.notna(current["next_member"]) else None
    if member is not None:
        state["exact_seed_map"][str(current["seed"])][member] += 1
        state["sorted_seed_map"][str(current["sorted_seed"])][member] += 1
        state["stream_member_map"][str(current["stream"])][member] += 1
        state["global_member_map"][member] += 1
    pool_row = {
        "next_member": member,
        "sorted_seed": current["sorted_seed"],
        "pair_token_pattern": current["pair_token_pattern"],
        "parity_pattern": current["parity_pattern"],
        "highlow_pattern": current["highlow_pattern"],
        "unique": current["unique"],
        "max_rep": current["max_rep"],
    }
    state["global_recent_pool"].append(pool_row)
    state["stream_recent_pool"][str(current["stream"])].append(pool_row)
    state["transitions_seen"] += 1


def score_seed_incremental(
    seed_feat: Dict[str, object], seed: str, stream: str, state: Dict[str, object],
    min_stream_history: int, stream_bias_weight: float, exact_seed_weight: float,
    sorted_seed_weight: float, similarity_weight: float
) -> List[Tuple[str, float]]:
    score_accum = {m: 0.0 for m in CORE025}
    global_probs = counter_to_probs(state["global_member_map"])
    for m in CORE025:
        score_accum[m] += global_probs[m] * 0.25
    if sum(state["stream_member_map"][stream].values()) >= int(min_stream_history):
        stream_probs = counter_to_probs(state["stream_member_map"][stream])
        for m in CORE025:
            score_accum[m] += stream_probs[m] * float(stream_bias_weight)
    if seed in state["exact_seed_map"] and sum(state["exact_seed_map"][seed].values()) > 0:
        exact_probs = counter_to_probs(state["exact_seed_map"][seed])
        for m in CORE025:
            score_accum[m] += exact_probs[m] * float(exact_seed_weight)
    sorted_key = str(seed_feat["sorted_seed"])
    if sorted_key in state["sorted_seed_map"] and sum(state["sorted_seed_map"][sorted_key].values()) > 0:
        sorted_probs = counter_to_probs(state["sorted_seed_map"][sorted_key])
        for m in CORE025:
            score_accum[m] += sorted_probs[m] * float(sorted_seed_weight)
    pool = list(state["global_recent_pool"])
    if len(state["stream_recent_pool"][stream]) >= int(min_stream_history):
        pool = list(state["stream_recent_pool"][stream])
    if len(pool) > 0:
        sim_scores = {m: 0.0 for m in CORE025}
        total_sim = 0.0
        recency_weights = np.linspace(0.50, 1.50, len(pool))
        for idx, r in enumerate(pool):
            member = r["next_member"]
            if member is None:
                continue
            sim = similarity(seed_feat, r)
            if sim <= 0:
                continue
            weight = float(sim) * float(recency_weights[idx])
            sim_scores[member] += weight
            total_sim += weight
        if total_sim > 0:
            for m in CORE025:
                score_accum[m] += (sim_scores[m] / total_sim) * float(similarity_weight)
    total = sum(score_accum.values())
    if total <= 0:
        return [(m, 1/3) for m in CORE025]
    probs = {m: score_accum[m] / total for m in CORE025}
    return sorted(probs.items(), key=lambda x: x[1], reverse=True)


def score_current_event(
    current: pd.Series, state: Dict[str, object], top2_needed_traits: pd.DataFrame, skip_danger_traits: pd.DataFrame,
    min_global_history: int, min_stream_history: int, stream_bias_weight: float, exact_seed_weight: float,
    sorted_seed_weight: float, similarity_weight: float, top1_only_threshold: float, top2_score_floor: float,
    weak_skip_threshold: float, top2_needed_min_rate: float, top2_gap_max: float, skip_danger_min_rate: float,
    skip_score_max: float, min_top2_trait_support: int, min_skip_trait_support: int, top2_hard_cap_fraction: float
) -> Optional[Dict[str, object]]:
    if int(current["is_core025_hit"]) != 1:
        return None
    if state["transitions_seen"] < int(min_global_history):
        return None

    seed = str(current["seed"])
    stream = str(current["stream"])
    seed_feat = features(seed)
    ranked = score_seed_incremental(
        seed_feat=seed_feat, seed=seed, stream=stream, state=state,
        min_stream_history=int(min_stream_history), stream_bias_weight=float(stream_bias_weight),
        exact_seed_weight=float(exact_seed_weight), sorted_seed_weight=float(sorted_seed_weight),
        similarity_weight=float(similarity_weight)
    )
    top1, top1_score = ranked[0]
    top2, top2_score = ranked[1]
    top3, top3_score = ranked[2]
    gap12 = top1_score - top2_score

    matched_top2 = matched_traits_for_row(current, top2_needed_traits)
    matched_skip = matched_traits_for_row(current, skip_danger_traits)

    if top1_score < float(weak_skip_threshold):
        recommendation = "Skip member play"
        decision_source = "score_default"
        decision_reason = "Weak score default"
    elif top1_score >= float(top1_only_threshold):
        recommendation = "Top1 only"
        decision_source = "score_default"
        decision_reason = "Strong Top1 score"
    elif top1_score >= float(top2_score_floor):
        recommendation = "Top1 only"
        decision_source = "score_default"
        decision_reason = "Borderline score kept Top1 only"
    else:
        recommendation = "Skip member play"
        decision_source = "score_default"
        decision_reason = "Below Top2 score floor"

    best_skip_trait = ""
    best_skip_rate = np.nan
    best_skip_support = 0
    skip_score_gate_passed = 0
    skip_blocked = 0
    if len(matched_skip):
        sort_cols = [c for c in ["skip_danger_rate", "support_skipped_hits", "hit_event_support"] if c in matched_skip.columns]
        best_skip = matched_skip.sort_values(sort_cols, ascending=[False] * len(sort_cols)).iloc[0]
        best_skip_trait = str(best_skip["trait"])
        best_skip_rate = float(best_skip.get("skip_danger_rate", np.nan))
        best_skip_support = int(best_skip.get("support_skipped_hits", 0))
        skip_score_gate_passed = int(top1_score <= float(skip_score_max))
        strong_skip_trait = (best_skip_rate >= float(skip_danger_min_rate)) and (best_skip_support >= int(min_skip_trait_support))
        if strong_skip_trait and recommendation == "Skip member play" and skip_score_gate_passed == 0:
            recommendation = "Top1 only"
            decision_source = "skip_danger_block"
            decision_reason = f"Skip blocked by dangerous skip trait: {best_skip_trait}"
            skip_blocked = 1

    best_top2_trait = ""
    best_top2_rate = np.nan
    best_top2_support = 0
    top2_gap_gate_passed = 0
    top2_allowed = 0
    if len(matched_top2):
        sort_cols = [c for c in ["top2_needed_rate", "support_top2_needed", "hit_event_support"] if c in matched_top2.columns]
        best_top2 = matched_top2.sort_values(sort_cols, ascending=[False] * len(sort_cols)).iloc[0]
        best_top2_trait = str(best_top2["trait"])
        best_top2_rate = float(best_top2.get("top2_needed_rate", np.nan))
        best_top2_support = int(best_top2.get("support_top2_needed", 0))
        top2_gap_gate_passed = int(gap12 <= float(top2_gap_max))
        strong_top2_trait = (best_top2_rate >= float(top2_needed_min_rate)) and (best_top2_support >= int(min_top2_trait_support))
        score_justifies_top2 = (gap12 <= float(top2_gap_max)) or (top1_score < float(top1_only_threshold) * float(top2_hard_cap_fraction))
        if strong_top2_trait and score_justifies_top2 and recommendation != "Skip member play":
            recommendation = "Top1 + Top2"
            decision_source = "optimized_top2"
            decision_reason = f"Strong Top2-needed trait + justified score gap: {best_top2_trait}"
            top2_allowed = 1

    actual_member = current["next_member"] if pd.notna(current["next_member"]) else ""
    plays_used = 0 if recommendation == "Skip member play" else (2 if recommendation == "Top1 + Top2" else 1)

    return {
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
        "decision_source": decision_source,
        "decision_reason": decision_reason,
        "matched_top2_needed_traits": len(matched_top2),
        "best_top2_trait": best_top2_trait,
        "best_top2_rate": best_top2_rate,
        "best_top2_support": best_top2_support,
        "top2_gap_gate_passed": top2_gap_gate_passed,
        "top2_allowed": top2_allowed,
        "matched_skip_danger_traits": len(matched_skip),
        "best_skip_trait": best_skip_trait,
        "best_skip_rate": best_skip_rate,
        "best_skip_support": best_skip_support,
        "skip_score_gate_passed": skip_score_gate_passed,
        "skip_blocked": skip_blocked,
        "top1_hit": int(actual_member == top1),
        "top2_hit": int(actual_member in [top1, top2]),
        "top3_hit": int(actual_member in [top1, top2, top3]),
        "play_rule_hit": int((recommendation == "Top1 only" and actual_member == top1) or (recommendation == "Top1 + Top2" and actual_member in [top1, top2])),
        "plays_used": plays_used,
    }


def run_walkforward(
    transitions: pd.DataFrame, top2_needed_traits: pd.DataFrame, skip_danger_traits: pd.DataFrame,
    min_global_history: int, min_stream_history: int, stream_bias_weight: float, exact_seed_weight: float,
    sorted_seed_weight: float, similarity_weight: float, top1_only_threshold: float, top2_score_floor: float,
    weak_skip_threshold: float, top2_needed_min_rate: float, top2_gap_max: float, skip_danger_min_rate: float,
    skip_score_max: float, min_top2_trait_support: int, min_skip_trait_support: int, top2_hard_cap_fraction: float,
    max_global_similarity_pool: int, max_stream_similarity_pool: int, chunk_size: int
) -> pd.DataFrame:
    state = initialize_state(max_global_similarity_pool, max_stream_similarity_pool)
    rows = []
    total_hit_events = int(transitions["is_core025_hit"].sum())
    processed_hit_events = 0
    progress = st.progress(0.0)
    status = st.empty()
    n = len(transitions)
    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        chunk = transitions.iloc[start:end]
        for _, current in chunk.iterrows():
            scored = score_current_event(
                current=current, state=state, top2_needed_traits=top2_needed_traits, skip_danger_traits=skip_danger_traits,
                min_global_history=int(min_global_history), min_stream_history=int(min_stream_history),
                stream_bias_weight=float(stream_bias_weight), exact_seed_weight=float(exact_seed_weight),
                sorted_seed_weight=float(sorted_seed_weight), similarity_weight=float(similarity_weight),
                top1_only_threshold=float(top1_only_threshold), top2_score_floor=float(top2_score_floor),
                weak_skip_threshold=float(weak_skip_threshold), top2_needed_min_rate=float(top2_needed_min_rate),
                top2_gap_max=float(top2_gap_max), skip_danger_min_rate=float(skip_danger_min_rate),
                skip_score_max=float(skip_score_max), min_top2_trait_support=int(min_top2_trait_support),
                min_skip_trait_support=int(min_skip_trait_support), top2_hard_cap_fraction=float(top2_hard_cap_fraction)
            )
            if scored is not None:
                rows.append(scored)
                processed_hit_events += 1
            update_state_with_event(state, current)
        progress.progress(processed_hit_events / total_hit_events if total_hit_events else 1.0)
        status.write(f"Processed transitions {start+1:,}–{end:,} of {n:,} | Scored Core025 hit events: {processed_hit_events:,} / {total_hit_events:,}")
    progress.empty()
    status.empty()
    return pd.DataFrame(rows)


def summarize_capture(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    return pd.DataFrame([
        {"metric": "Top1 capture", "numerator": int(df["top1_hit"].sum()), "denominator": total, "rate": float(df["top1_hit"].mean()) if total else np.nan},
        {"metric": "Top2 capture", "numerator": int(df["top2_hit"].sum()), "denominator": total, "rate": float(df["top2_hit"].mean()) if total else np.nan},
        {"metric": "Top3 capture", "numerator": int(df["top3_hit"].sum()), "denominator": total, "rate": float(df["top3_hit"].mean()) if total else np.nan},
        {"metric": "Play-rule capture", "numerator": int(df["play_rule_hit"].sum()), "denominator": total, "rate": float(df["play_rule_hit"].mean()) if total else np.nan},
        {"metric": "Average plays per hit-event row", "numerator": float(df["plays_used"].sum()), "denominator": total, "rate": float(df["plays_used"].mean()) if total else np.nan},
        {"metric": "Top2 usage rate", "numerator": int((df["recommendation"] == "Top1 + Top2").sum()), "denominator": total, "rate": float((df["recommendation"] == "Top1 + Top2").mean()) if total else np.nan},
    ])


def summarize_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("recommendation", dropna=False).agg(
        rows=("recommendation", "size"),
        avg_top1_score=("Top1_score", "mean"),
        avg_gap12=("Top1_minus_Top2", "mean"),
        capture=("play_rule_hit", "mean"),
        avg_plays=("plays_used", "mean"),
    ).reset_index().sort_values("rows", ascending=False).reset_index(drop=True)


def summarize_sources(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("decision_source", dropna=False).agg(
        rows=("decision_source", "size"),
        capture=("play_rule_hit", "mean"),
        avg_top1_score=("Top1_score", "mean"),
        avg_gap12=("Top1_minus_Top2", "mean"),
    ).reset_index().sort_values("rows", ascending=False).reset_index(drop=True)


def summarize_daily(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["date"] = pd.to_datetime(work["event_date"]).dt.date
    return work.groupby("date", dropna=False).agg(
        rows=("date", "size"),
        plays=("plays_used", "sum"),
        top1_hits=("top1_hit", "sum"),
        top2_hits=("top2_hit", "sum"),
        play_rule_hits=("play_rule_hit", "sum"),
        skips=("recommendation", lambda s: int((s == "Skip member play").sum())),
        top1_only=("recommendation", lambda s: int((s == "Top1 only").sum())),
        top1_top2=("recommendation", lambda s: int((s == "Top1 + Top2").sum())),
    ).reset_index()


def main():
    st.set_page_config(page_title="Core025 Top2 Optimization Walk-Forward Validator v1", layout="wide")
    st.title("Core025 Top2 Optimization Walk-Forward Validator v1")
    st.caption("Full historical walk-forward for the Top2 optimization engine.")
    st.code(BUILD_MARKER, language="text")

    with st.sidebar:
        st.markdown(f"**{BUILD_MARKER}**")
        st.header("History controls")
        min_global_history = st.number_input("Minimum prior transitions before scoring", min_value=10, value=100, step=10)
        min_stream_history = st.number_input("Minimum stream history", min_value=0, value=20, step=5)
        st.header("Scoring weights")
        stream_bias_weight = st.slider("Stream-bias weight", 0.0, 3.0, 1.20, 0.05)
        exact_seed_weight = st.slider("Exact-seed weight", 0.0, 3.0, 1.50, 0.05)
        sorted_seed_weight = st.slider("Sorted-seed weight", 0.0, 3.0, 1.10, 0.05)
        similarity_weight = st.slider("Similarity weight", 0.0, 3.0, 1.80, 0.05)
        st.header("Top2 optimization")
        top1_only_threshold = st.slider("Top1-only threshold", 0.33, 0.95, 0.48, 0.005)
        top2_score_floor = st.slider("Top2 score floor", 0.33, 0.95, 0.36, 0.005)
        top2_needed_min_rate = st.slider("Minimum Top2-needed rate", 0.05, 0.95, 0.25, 0.01)
        min_top2_trait_support = st.number_input("Minimum Top2-needed support", min_value=1, value=25, step=1)
        top2_gap_max = st.slider("Maximum Top1-Top2 gap to allow Top2", 0.00, 0.30, 0.08, 0.005)
        top2_hard_cap_fraction = st.slider("Extra Top2 allowance fraction of Top1 threshold", 0.50, 1.00, 0.90, 0.01)
        st.header("Skip controls")
        weak_skip_threshold = st.slider("Weak-score skip threshold", 0.00, 0.50, 0.36, 0.001)
        skip_danger_min_rate = st.slider("Minimum skip-danger rate", 0.05, 0.95, 0.25, 0.01)
        min_skip_trait_support = st.number_input("Minimum skip-danger support", min_value=1, value=25, step=1)
        skip_score_max = st.slider("Top1 score max for skip trait to force true skip", 0.00, 0.60, 0.45, 0.005)
        st.header("Performance controls")
        max_global_similarity_pool = st.number_input("Max global similarity pool rows", min_value=200, value=4000, step=100)
        max_stream_similarity_pool = st.number_input("Max stream similarity pool rows", min_value=50, value=600, step=50)
        chunk_size = st.number_input("Transition chunk size", min_value=100, value=1000, step=100)
        rows_to_show = st.number_input("Rows to display", min_value=5, value=50, step=5)

    hist_file = st.file_uploader("Upload FULL history file", key="wfopt_hist")
    top2_needed_file = st.file_uploader("Upload strict Top2-needed traits CSV", key="wfopt_top2")
    skip_danger_file = st.file_uploader("Upload strict skip-danger traits CSV", key="wfopt_skip")

    if not all([hist_file, top2_needed_file, skip_danger_file]):
        st.info("Upload the full history file, strict Top2-needed traits CSV, and strict skip-danger traits CSV.")
        return

    if st.button("Run Top2 Optimization Walk-Forward", type="primary"):
        try:
            hist = prepare_history(load_table(hist_file))
            top2_needed_traits = load_trait_df(top2_needed_file)
            skip_danger_traits = load_trait_df(skip_danger_file)
            transitions = build_transitions(hist)
            wf = run_walkforward(
                transitions=transitions, top2_needed_traits=top2_needed_traits, skip_danger_traits=skip_danger_traits,
                min_global_history=int(min_global_history), min_stream_history=int(min_stream_history),
                stream_bias_weight=float(stream_bias_weight), exact_seed_weight=float(exact_seed_weight),
                sorted_seed_weight=float(sorted_seed_weight), similarity_weight=float(similarity_weight),
                top1_only_threshold=float(top1_only_threshold), top2_score_floor=float(top2_score_floor),
                weak_skip_threshold=float(weak_skip_threshold), top2_needed_min_rate=float(top2_needed_min_rate),
                top2_gap_max=float(top2_gap_max), skip_danger_min_rate=float(skip_danger_min_rate),
                skip_score_max=float(skip_score_max), min_top2_trait_support=int(min_top2_trait_support),
                min_skip_trait_support=int(min_skip_trait_support), top2_hard_cap_fraction=float(top2_hard_cap_fraction),
                max_global_similarity_pool=int(max_global_similarity_pool), max_stream_similarity_pool=int(max_stream_similarity_pool),
                chunk_size=int(chunk_size)
            )
            st.session_state["wfopt_rows"] = wf
            st.session_state["wfopt_capture"] = summarize_capture(wf)
            st.session_state["wfopt_reco"] = summarize_recommendations(wf)
            st.session_state["wfopt_source"] = summarize_sources(wf)
            st.session_state["wfopt_daily"] = summarize_daily(wf)
        except Exception as e:
            st.exception(e)
            return

    if "wfopt_rows" not in st.session_state or st.session_state["wfopt_rows"] is None:
        return

    wf = st.session_state["wfopt_rows"]
    capture = st.session_state["wfopt_capture"]
    reco = st.session_state["wfopt_reco"]
    source = st.session_state["wfopt_source"]
    daily = st.session_state["wfopt_daily"]

    st.subheader("Capture summary")
    st.dataframe(capture, use_container_width=True)
    st.subheader("Recommendation summary")
    st.dataframe(reco, use_container_width=True)
    st.subheader("Decision source summary")
    st.dataframe(source, use_container_width=True)
    st.subheader("Daily summary")
    st.dataframe(daily.head(int(rows_to_show)), use_container_width=True)
    st.subheader("Hit-event table")
    st.dataframe(wf.head(int(rows_to_show)), use_container_width=True)

    st.download_button("Download Top2 optimization walk-forward hit-events CSV", wf.to_csv(index=False), "core025_top2_optimization_walkforward_hit_events_v1__2026-03-28.csv", "text/csv")
    st.download_button("Download capture summary CSV", capture.to_csv(index=False), "core025_top2_optimization_walkforward_capture_summary_v1__2026-03-28.csv", "text/csv")
    st.download_button("Download recommendation summary CSV", reco.to_csv(index=False), "core025_top2_optimization_walkforward_recommendation_summary_v1__2026-03-28.csv", "text/csv")
    st.download_button("Download decision source summary CSV", source.to_csv(index=False), "core025_top2_optimization_walkforward_decision_source_summary_v1__2026-03-28.csv", "text/csv")
    st.download_button("Download daily summary CSV", daily.to_csv(index=False), "core025_top2_optimization_walkforward_daily_summary_v1__2026-03-28.csv", "text/csv")


if __name__ == "__main__":
    if "wfopt_rows" not in st.session_state:
        st.session_state["wfopt_rows"] = None
    main()

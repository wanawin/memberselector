#!/usr/bin/env python3
# core025_hybrid_walkforward_validator_v1__2026-03-28.py
#
# BUILD: core025_hybrid_walkforward_validator_v1__2026-03-28
#
# Full file. No placeholders.
# True historical walk-forward validator for the current hybrid Step 5 logic.

from __future__ import annotations

import io
import re
from collections import Counter, defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

BUILD_MARKER = "BUILD: core025_hybrid_walkforward_validator_v1__2026-03-28"
CORE025 = ["0025", "0225", "0255"]


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
    d = [int(x) for x in d[:4]]
    cnt = Counter(d)
    pair_tokens = []
    for i in range(4):
        for j in range(i + 1, 4):
            pair_tokens.append("".join(sorted([str(d[i]), str(d[j])])))
    unique_sorted = sorted(set(d))
    consec_links = 0
    for a, b in zip(unique_sorted[:-1], unique_sorted[1:]):
        if b - a == 1:
            consec_links += 1
    feat = {
        "sum": sum(d),
        "spread": max(d) - min(d),
        "even": sum(x % 2 == 0 for x in d),
        "odd": sum(x % 2 != 0 for x in d),
        "high": sum(x >= 5 for x in d),
        "low": sum(x <= 4 for x in d),
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
        "sum_mod3": sum(d) % 3,
        "sum_mod4": sum(d) % 4,
        "sum_mod5": sum(d) % 5,
        "consec_links": consec_links,
        "pair_tokens": tuple(sorted(pair_tokens)),
        "parity_pattern": "".join("E" if x % 2 == 0 else "O" for x in d),
        "highlow_pattern": "".join("H" if x >= 5 else "L" for x in d),
    }
    for k in range(10):
        feat[f"has{k}"] = int(k in cnt)
        feat[f"cnt{k}"] = int(cnt.get(k, 0))
    return feat


def candidate_columns() -> List[str]:
    cols = [
        "sum", "spread", "even", "odd", "high", "low", "unique", "pair", "max_rep",
        "pos1", "pos2", "pos3", "pos4", "sorted_seed", "first2", "last2",
        "sum_mod3", "sum_mod4", "sum_mod5", "consec_links",
        "parity_pattern", "highlow_pattern",
    ]
    cols.extend([f"has{k}" for k in range(10)])
    cols.extend([f"cnt{k}" for k in range(10)])
    return cols


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
    if a["pos1"] == b["pos1"]:
        score += 2
    if a["pos2"] == b["pos2"]:
        score += 2
    if a["pos3"] == b["pos3"]:
        score += 1
    if a["pos4"] == b["pos4"]:
        score += 1
    if a["sorted_seed"] == b["sorted_seed"]:
        score += 6
    if a["first2"] == b["first2"]:
        score += 1
    if a["last2"] == b["last2"]:
        score += 1
    for k in range(10):
        if a[f"has{k}"] == b[f"has{k}"]:
            score += 0.35
        if a[f"cnt{k}"] == b[f"cnt{k}"]:
            score += 0.20
    pair_overlap = len(set(a["pair_tokens"]).intersection(set(b["pair_tokens"])))
    score += pair_overlap * 0.50
    return float(score)


def counter_to_probs(c: Counter) -> Dict[str, float]:
    total = sum(c.values())
    if total <= 0:
        return {m: 1 / 3 for m in CORE025}
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


def build_transitions(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for stream, g in df.groupby("stream"):
        g = g.sort_values("date").reset_index(drop=True)
        for i in range(1, len(g)):
            seed = g.loc[i - 1, "r4"]
            next_member = g.loc[i, "member"]
            next_r4 = g.loc[i, "r4"]
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
                "next_r4": next_r4,
                "next_member": next_member,
                "is_core025_hit": int(next_member is not None),
                **feat,
            })
    out = pd.DataFrame(rows)
    out = out.sort_values(["event_date", "stream", "seed"]).reset_index(drop=True)
    return out


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
        "sum": current["sum"],
        "spread": current["spread"],
        "even": current["even"],
        "odd": current["odd"],
        "high": current["high"],
        "low": current["low"],
        "unique": current["unique"],
        "pair": current["pair"],
        "max_rep": current["max_rep"],
        "pos1": current["pos1"],
        "pos2": current["pos2"],
        "pos3": current["pos3"],
        "pos4": current["pos4"],
        "sorted_seed": current["sorted_seed"],
        "first2": current["first2"],
        "last2": current["last2"],
        "sum_mod3": current["sum_mod3"],
        "sum_mod4": current["sum_mod4"],
        "sum_mod5": current["sum_mod5"],
        "consec_links": current["consec_links"],
        "pair_tokens": current["pair_tokens"],
        "parity_pattern": current["parity_pattern"],
        "highlow_pattern": current["highlow_pattern"],
        **{f"has{k}": current[f"has{k}"] for k in range(10)},
        **{f"cnt{k}": current[f"cnt{k}"] for k in range(10)},
    }
    state["global_recent_pool"].append(pool_row)
    state["stream_recent_pool"][str(current["stream"])].append(pool_row)
    state["transitions_seen"] += 1


def score_seed_incremental(
    seed_feat: Dict[str, object], seed: str, stream: str, state: Dict[str, object],
    min_stream_history: int, stream_bias_weight: float, exact_seed_weight: float,
    sorted_seed_weight: float, similarity_weight: float,
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
        similarity_scores = {m: 0.0 for m in CORE025}
        total_sim_weight = 0.0
        recency_weights = np.linspace(0.50, 1.50, len(pool))
        for idx, r in enumerate(pool):
            member = r["next_member"]
            if member is None:
                continue
            sim = similarity(seed_feat, r)
            if sim <= 0:
                continue
            weight = float(sim) * float(recency_weights[idx])
            similarity_scores[member] += weight
            total_sim_weight += weight
        if total_sim_weight > 0:
            for m in CORE025:
                score_accum[m] += (similarity_scores[m] / total_sim_weight) * float(similarity_weight)

    total = sum(score_accum.values())
    if total <= 0:
        return [(m, 1 / 3) for m in CORE025]
    probs = {m: score_accum[m] / total for m in CORE025}
    return sorted(probs.items(), key=lambda x: x[1], reverse=True)


def choose_member_from_traits(
    matched_sep: pd.DataFrame, matched_0025: pd.DataFrame, matched_0225: pd.DataFrame, matched_0255: pd.DataFrame,
    sep_min_rate: float, sep_min_gap: float, member_min_rate: float
) -> Tuple[Optional[str], Optional[str], float]:
    candidates: List[Dict[str, object]] = []
    if matched_sep is not None and len(matched_sep):
        sub = matched_sep[
            (matched_sep["winning_member_rate"] >= float(sep_min_rate)) &
            (matched_sep["separation_gap"] >= float(sep_min_gap))
        ].copy()
        for _, r in sub.iterrows():
            candidates.append({
                "member": r["winning_member"],
                "trait": r["trait"],
                "strength": float(r["winning_member_rate"]),
                "tie_break": float(r["separation_gap"]),
                "support": int(r["support"]),
                "source": "separation",
            })
    for label, mdf in [("0025", matched_0025), ("0225", matched_0225), ("0255", matched_0255)]:
        if mdf is None or len(mdf) == 0:
            continue
        sub = mdf[mdf["target_member_rate"] >= float(member_min_rate)].copy()
        for _, r in sub.iterrows():
            candidates.append({
                "member": label,
                "trait": r["trait"],
                "strength": float(r["target_member_rate"]),
                "tie_break": float(r["target_member_rate"] - (1.0 - r["target_member_rate"])),
                "support": int(r["support"]),
                "source": f"{label}_specific",
            })
    if not candidates:
        return None, None, 0.0
    best = sorted(candidates, key=lambda x: (x["strength"], x["tie_break"], x["support"]), reverse=True)[0]
    return str(best["member"]), f"{best['source']} trait: {best['trait']}", float(best["strength"])


def score_current_hit_event(
    current: pd.Series, state: Dict[str, object], sep_traits: pd.DataFrame,
    top2_needed_traits: pd.DataFrame, skip_danger_traits: pd.DataFrame,
    traits_0025: Optional[pd.DataFrame], traits_0225: Optional[pd.DataFrame], traits_0255: Optional[pd.DataFrame],
    min_global_history: int, min_stream_history: int,
    stream_bias_weight: float, exact_seed_weight: float, sorted_seed_weight: float, similarity_weight: float,
    top1_only_threshold: float, top2_threshold: float, weak_skip_threshold: float,
    skip_danger_min_rate: float, top2_needed_min_rate: float, sep_min_rate: float, sep_min_gap: float,
    member_min_rate: float, top2_gap_max: float, skip_score_max: float
) -> Optional[Dict[str, object]]:
    if int(current["is_core025_hit"]) != 1:
        return None
    if state["transitions_seen"] < int(min_global_history):
        return None

    seed = str(current["seed"])
    stream = str(current["stream"])
    feat_keys = set(candidate_columns() + ["pair_tokens"])
    seed_feat = {k: current[k] for k in current.index if k in feat_keys}

    ranked = score_seed_incremental(
        seed_feat=seed_feat, seed=seed, stream=stream, state=state,
        min_stream_history=int(min_stream_history),
        stream_bias_weight=float(stream_bias_weight),
        exact_seed_weight=float(exact_seed_weight),
        sorted_seed_weight=float(sorted_seed_weight),
        similarity_weight=float(similarity_weight),
    )

    top1, top1_score = ranked[0]
    top2, top2_score = ranked[1]
    top3, top3_score = ranked[2]
    gap12 = top1_score - top2_score

    matched_sep = matched_traits_for_row(current, sep_traits)
    matched_top2 = matched_traits_for_row(current, top2_needed_traits)
    matched_skip = matched_traits_for_row(current, skip_danger_traits)
    matched_0025 = matched_traits_for_row(current, traits_0025) if traits_0025 is not None else pd.DataFrame()
    matched_0225 = matched_traits_for_row(current, traits_0225) if traits_0225 is not None else pd.DataFrame()
    matched_0255 = matched_traits_for_row(current, traits_0255) if traits_0255 is not None else pd.DataFrame()

    forced_member, member_reason, trait_strength = choose_member_from_traits(
        matched_sep, matched_0025, matched_0225, matched_0255,
        float(sep_min_rate), float(sep_min_gap), float(member_min_rate)
    )
    forced_member_source = member_reason if forced_member is not None else ""

    if forced_member is not None:
        remaining = [(m, s) for m, s in ranked if m != forced_member]
        forced_score = max(dict(ranked).get(forced_member, 0.34), trait_strength)
        reordered = [(forced_member, forced_score)] + remaining
        top1, top1_score = reordered[0]
        top2, top2_score = reordered[1]
        top3, top3_score = reordered[2]
        gap12 = top1_score - top2_score

    if top1_score < float(weak_skip_threshold):
        recommendation = "Skip member play"
        decision_source = "score_default"
        decision_reason = "Weak score default"
    elif top1_score >= float(top1_only_threshold):
        recommendation = "Top1 only"
        decision_source = "score_default"
        decision_reason = "Strong Top1 score"
    elif top1_score >= float(top2_threshold):
        recommendation = "Top1 + Top2"
        decision_source = "score_default"
        decision_reason = "Mid score default"
    else:
        recommendation = "Skip member play"
        decision_source = "score_default"
        decision_reason = "Below Top2 threshold"

    best_skip_trait = ""
    best_skip_rate = np.nan
    skip_score_gate_passed = 0
    if len(matched_skip):
        best_skip = matched_skip.sort_values(
            ["skip_danger_rate", "support_skipped_hits", "hit_event_support"],
            ascending=[False, False, False]
        ).iloc[0]
        best_skip_trait = str(best_skip["trait"])
        best_skip_rate = float(best_skip["skip_danger_rate"])
        skip_score_gate_passed = int(top1_score <= float(skip_score_max))
        if best_skip_rate >= float(skip_danger_min_rate) and skip_score_gate_passed == 1:
            recommendation = "Skip member play"
            decision_source = "hybrid_skip"
            decision_reason = f"Skip trait + weak score: {best_skip_trait}"

    best_top2_trait = ""
    best_top2_rate = np.nan
    top2_gap_gate_passed = 0
    if len(matched_top2):
        best_top2 = matched_top2.sort_values(
            ["top2_needed_rate", "support_top2_needed", "hit_event_support"],
            ascending=[False, False, False]
        ).iloc[0]
        best_top2_trait = str(best_top2["trait"])
        best_top2_rate = float(best_top2["top2_needed_rate"])
        top2_gap_gate_passed = int(gap12 <= float(top2_gap_max))
        if best_top2_rate >= float(top2_needed_min_rate) and top2_gap_gate_passed == 1:
            recommendation = "Top1 + Top2"
            decision_source = "hybrid_top2"
            if forced_member is not None:
                decision_reason = f"{member_reason} | Top2 trait + tight gap: {best_top2_trait}"
            else:
                decision_reason = f"Top2 trait + tight gap: {best_top2_trait}"

    if forced_member is not None and decision_source == "score_default" and recommendation != "Skip member play":
        recommendation = "Top1 only"
        decision_source = "hybrid_member_force"
        decision_reason = member_reason

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
        "trait_strength": trait_strength,
        "matched_sep_traits": len(matched_sep),
        "forced_member": forced_member if forced_member is not None else "",
        "forced_member_reason": forced_member_source,
        "matched_top2_needed_traits": len(matched_top2),
        "best_top2_trait": best_top2_trait,
        "best_top2_rate": best_top2_rate,
        "top2_gap_gate_passed": top2_gap_gate_passed,
        "matched_skip_danger_traits": len(matched_skip),
        "best_skip_trait": best_skip_trait,
        "best_skip_rate": best_skip_rate,
        "skip_score_gate_passed": skip_score_gate_passed,
        "top1_hit": int(actual_member == top1),
        "top2_hit": int(actual_member in [top1, top2]),
        "top3_hit": int(actual_member in [top1, top2, top3]),
        "play_rule_hit": int(
            (recommendation == "Top1 only" and actual_member == top1) or
            (recommendation == "Top1 + Top2" and actual_member in [top1, top2])
        ),
        "plays_used": plays_used,
    }


def run_hybrid_walkforward(
    transitions: pd.DataFrame, sep_traits: pd.DataFrame, top2_needed_traits: pd.DataFrame, skip_danger_traits: pd.DataFrame,
    traits_0025: Optional[pd.DataFrame], traits_0225: Optional[pd.DataFrame], traits_0255: Optional[pd.DataFrame],
    min_global_history: int, min_stream_history: int,
    stream_bias_weight: float, exact_seed_weight: float, sorted_seed_weight: float, similarity_weight: float,
    top1_only_threshold: float, top2_threshold: float, weak_skip_threshold: float,
    skip_danger_min_rate: float, top2_needed_min_rate: float, sep_min_rate: float, sep_min_gap: float,
    member_min_rate: float, top2_gap_max: float, skip_score_max: float,
    max_global_similarity_pool: int, max_stream_similarity_pool: int, chunk_size: int
) -> pd.DataFrame:
    state = initialize_state(max_global_similarity_pool, max_stream_similarity_pool)
    rows: List[Dict[str, object]] = []
    total_hit_events = int(transitions["is_core025_hit"].sum())
    processed_hit_events = 0

    progress = st.progress(0.0)
    status = st.empty()

    n = len(transitions)
    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        chunk = transitions.iloc[start:end]
        for _, current in chunk.iterrows():
            scored = score_current_hit_event(
                current=current, state=state, sep_traits=sep_traits, top2_needed_traits=top2_needed_traits,
                skip_danger_traits=skip_danger_traits, traits_0025=traits_0025, traits_0225=traits_0225, traits_0255=traits_0255,
                min_global_history=int(min_global_history), min_stream_history=int(min_stream_history),
                stream_bias_weight=float(stream_bias_weight), exact_seed_weight=float(exact_seed_weight),
                sorted_seed_weight=float(sorted_seed_weight), similarity_weight=float(similarity_weight),
                top1_only_threshold=float(top1_only_threshold), top2_threshold=float(top2_threshold),
                weak_skip_threshold=float(weak_skip_threshold), skip_danger_min_rate=float(skip_danger_min_rate),
                top2_needed_min_rate=float(top2_needed_min_rate), sep_min_rate=float(sep_min_rate),
                sep_min_gap=float(sep_min_gap), member_min_rate=float(member_min_rate),
                top2_gap_max=float(top2_gap_max), skip_score_max=float(skip_score_max)
            )
            if scored is not None:
                rows.append(scored)
                processed_hit_events += 1
            update_state_with_event(state, current)

        progress.progress(processed_hit_events / total_hit_events if total_hit_events else 1.0)
        status.write(
            f"Processed transitions {start+1:,}–{end:,} of {n:,} | "
            f"Scored Core025 hit events: {processed_hit_events:,} / {total_hit_events:,}"
        )

    progress.empty()
    status.empty()
    return pd.DataFrame(rows)


def summarize_capture(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    rows = [
        {"metric": "Top1 capture", "numerator": int(df["top1_hit"].sum()), "denominator": total, "rate": float(df["top1_hit"].mean()) if total else np.nan},
        {"metric": "Top2 capture", "numerator": int(df["top2_hit"].sum()), "denominator": total, "rate": float(df["top2_hit"].mean()) if total else np.nan},
        {"metric": "Top3 capture", "numerator": int(df["top3_hit"].sum()), "denominator": total, "rate": float(df["top3_hit"].mean()) if total else np.nan},
        {"metric": "Play-rule capture", "numerator": int(df["play_rule_hit"].sum()), "denominator": total, "rate": float(df["play_rule_hit"].mean()) if total else np.nan},
        {"metric": "Average plays per hit-event row", "numerator": float(df["plays_used"].sum()), "denominator": total, "rate": float(df["plays_used"].mean()) if total else np.nan},
    ]
    return pd.DataFrame(rows)


def summarize_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("recommendation", dropna=False).agg(
        rows=("recommendation", "size"),
        avg_top1_score=("Top1_score", "mean"),
        avg_gap12=("Top1_minus_Top2", "mean"),
        capture=("play_rule_hit", "mean"),
        avg_plays=("plays_used", "mean"),
    ).reset_index()
    return out.sort_values("rows", ascending=False).reset_index(drop=True)


def summarize_decision_sources(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("decision_source", dropna=False).agg(
        rows=("decision_source", "size"),
        capture=("play_rule_hit", "mean"),
        avg_top1_score=("Top1_score", "mean"),
        avg_gap12=("Top1_minus_Top2", "mean"),
    ).reset_index()
    return out.sort_values("rows", ascending=False).reset_index(drop=True)


def summarize_daily(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["date"] = pd.to_datetime(work["event_date"]).dt.date
    out = work.groupby("date", dropna=False).agg(
        rows=("date", "size"),
        plays=("plays_used", "sum"),
        top1_hits=("top1_hit", "sum"),
        top2_hits=("top2_hit", "sum"),
        play_rule_hits=("play_rule_hit", "sum"),
        skips=("recommendation", lambda s: int((s == "Skip member play").sum())),
        top1_only=("recommendation", lambda s: int((s == "Top1 only").sum())),
        top1_top2=("recommendation", lambda s: int((s == "Top1 + Top2").sum())),
    ).reset_index()
    return out


def summarize_skip_diag(df: pd.DataFrame) -> pd.DataFrame:
    skip_rows = df[df["recommendation"] == "Skip member play"]
    return pd.DataFrame([{
        "rows_with_skip_traits": int((df["matched_skip_danger_traits"] > 0).sum()),
        "rows_skip_score_gate_passed": int((df["skip_score_gate_passed"] == 1).sum()),
        "rows_final_skip": int(len(skip_rows)),
        "play_rule_capture_on_skipped_rows": float(skip_rows["play_rule_hit"].mean()) if len(skip_rows) else np.nan,
    }])


def summarize_top2_diag(df: pd.DataFrame) -> pd.DataFrame:
    top2_rows = df[df["recommendation"] == "Top1 + Top2"]
    return pd.DataFrame([{
        "rows_with_top2_traits": int((df["matched_top2_needed_traits"] > 0).sum()),
        "rows_top2_gap_gate_passed": int((df["top2_gap_gate_passed"] == 1).sum()),
        "rows_final_top1_plus_top2": int(len(top2_rows)),
        "play_rule_capture_on_top1_plus_top2_rows": float(top2_rows["play_rule_hit"].mean()) if len(top2_rows) else np.nan,
    }])


def main():
    st.set_page_config(page_title="Core025 Hybrid Walk-Forward Validator v1", layout="wide")
    st.title("Core025 Hybrid Walk-Forward Validator v1")
    st.caption("Exact walk-forward validator for the working hybrid Step 5 logic.")
    st.code(BUILD_MARKER, language="text")

    with st.sidebar:
        st.markdown(f"**{BUILD_MARKER}**")
        st.header("History controls")
        min_global_history = st.number_input("Minimum prior transitions before scoring", min_value=10, value=100, step=10)
        min_stream_history = st.number_input("Minimum stream history", min_value=0, value=20, step=5)

        st.header("Scoring weights")
        stream_bias_weight = st.slider("Stream-bias weight", min_value=0.0, max_value=3.0, value=1.20, step=0.05)
        exact_seed_weight = st.slider("Exact-seed weight", min_value=0.0, max_value=3.0, value=1.50, step=0.05)
        sorted_seed_weight = st.slider("Sorted-seed weight", min_value=0.0, max_value=3.0, value=1.10, step=0.05)
        similarity_weight = st.slider("Similarity weight", min_value=0.0, max_value=3.0, value=1.80, step=0.05)

        st.header("Trait thresholds")
        skip_danger_min_rate = st.slider("Skip-danger minimum rate", min_value=0.05, max_value=0.95, value=0.30, step=0.01)
        top2_needed_min_rate = st.slider("Top2-needed minimum rate", min_value=0.05, max_value=0.95, value=0.20, step=0.01)
        sep_min_rate = st.slider("Separation minimum rate", min_value=0.34, max_value=0.95, value=0.55, step=0.01)
        sep_min_gap = st.slider("Separation minimum gap", min_value=0.00, max_value=0.50, value=0.05, step=0.01)
        member_min_rate = st.slider("Member-specific minimum rate", min_value=0.34, max_value=0.95, value=0.55, step=0.01)

        st.header("Score confirmation rules")
        top1_only_threshold = st.slider("Top1-only threshold", min_value=0.33, max_value=0.95, value=0.48, step=0.005)
        top2_threshold = st.slider("Top1+Top2 threshold", min_value=0.33, max_value=0.95, value=0.36, step=0.005)
        weak_skip_threshold = st.slider("Weak-score skip threshold", min_value=0.00, max_value=0.50, value=0.36, step=0.001)
        top2_gap_max = st.slider("Top2 max gap for Top2-needed trait", min_value=0.00, max_value=0.30, value=0.08, step=0.005)
        skip_score_max = st.slider("Top1 score max for skip trait to force skip", min_value=0.00, max_value=0.60, value=0.45, step=0.005)

        st.header("Performance controls")
        max_global_similarity_pool = st.number_input("Max global similarity pool rows", min_value=200, value=4000, step=100)
        max_stream_similarity_pool = st.number_input("Max stream similarity pool rows", min_value=50, value=600, step=50)
        chunk_size = st.number_input("Transition chunk size", min_value=100, value=1000, step=100)
        rows_to_show = st.number_input("Rows to display", min_value=5, value=50, step=5)

    hist_file = st.file_uploader("Upload FULL history file", key="hwf_hist")
    sep_file = st.file_uploader("Upload separation traits CSV", key="hwf_sep")
    top2_needed_file = st.file_uploader("Upload Top2-needed traits CSV", key="hwf_top2")
    skip_danger_file = st.file_uploader("Upload skip-danger traits CSV", key="hwf_skip")
    t0025_file = st.file_uploader("Optional: upload 0025 traits CSV", key="hwf_0025")
    t0225_file = st.file_uploader("Optional: upload 0225 traits CSV", key="hwf_0225")
    t0255_file = st.file_uploader("Optional: upload 0255 traits CSV", key="hwf_0255")

    if not all([hist_file, sep_file, top2_needed_file, skip_danger_file]):
        st.info("Upload the full history file, separation traits, Top2-needed traits, and skip-danger traits to begin.")
        return

    if st.button("Run Hybrid Walk-Forward", type="primary"):
        try:
            hist = prepare_history(load_table(hist_file))
            sep_traits = load_trait_df(sep_file)
            top2_needed_traits = load_trait_df(top2_needed_file)
            skip_danger_traits = load_trait_df(skip_danger_file)
            traits_0025 = load_trait_df(t0025_file) if t0025_file is not None else None
            traits_0225 = load_trait_df(t0225_file) if t0225_file is not None else None
            traits_0255 = load_trait_df(t0255_file) if t0255_file is not None else None

            transitions = build_transitions(hist)

            wf_hits = run_hybrid_walkforward(
                transitions=transitions,
                sep_traits=sep_traits,
                top2_needed_traits=top2_needed_traits,
                skip_danger_traits=skip_danger_traits,
                traits_0025=traits_0025,
                traits_0225=traits_0225,
                traits_0255=traits_0255,
                min_global_history=int(min_global_history),
                min_stream_history=int(min_stream_history),
                stream_bias_weight=float(stream_bias_weight),
                exact_seed_weight=float(exact_seed_weight),
                sorted_seed_weight=float(sorted_seed_weight),
                similarity_weight=float(similarity_weight),
                top1_only_threshold=float(top1_only_threshold),
                top2_threshold=float(top2_threshold),
                weak_skip_threshold=float(weak_skip_threshold),
                skip_danger_min_rate=float(skip_danger_min_rate),
                top2_needed_min_rate=float(top2_needed_min_rate),
                sep_min_rate=float(sep_min_rate),
                sep_min_gap=float(sep_min_gap),
                member_min_rate=float(member_min_rate),
                top2_gap_max=float(top2_gap_max),
                skip_score_max=float(skip_score_max),
                max_global_similarity_pool=int(max_global_similarity_pool),
                max_stream_similarity_pool=int(max_stream_similarity_pool),
                chunk_size=int(chunk_size),
            )

            st.session_state["hybrid_wf_hits"] = wf_hits
            st.session_state["hybrid_wf_capture"] = summarize_capture(wf_hits)
            st.session_state["hybrid_wf_reco"] = summarize_recommendations(wf_hits)
            st.session_state["hybrid_wf_source"] = summarize_decision_sources(wf_hits)
            st.session_state["hybrid_wf_daily"] = summarize_daily(wf_hits)
            st.session_state["hybrid_wf_skipdiag"] = summarize_skip_diag(wf_hits)
            st.session_state["hybrid_wf_top2diag"] = summarize_top2_diag(wf_hits)
        except Exception as e:
            st.exception(e)
            return

    if "hybrid_wf_hits" not in st.session_state or st.session_state["hybrid_wf_hits"] is None:
        return

    wf_hits = st.session_state["hybrid_wf_hits"]
    capture = st.session_state["hybrid_wf_capture"]
    reco = st.session_state["hybrid_wf_reco"]
    source = st.session_state["hybrid_wf_source"]
    daily = st.session_state["hybrid_wf_daily"]
    skipdiag = st.session_state["hybrid_wf_skipdiag"]
    top2diag = st.session_state["hybrid_wf_top2diag"]

    st.subheader("Capture summary")
    st.dataframe(capture, use_container_width=True)

    st.subheader("Recommendation summary")
    st.dataframe(reco, use_container_width=True)

    st.subheader("Decision source summary")
    st.dataframe(source, use_container_width=True)

    st.subheader("Skip diagnostics")
    st.dataframe(skipdiag, use_container_width=True)

    st.subheader("Top2 diagnostics")
    st.dataframe(top2diag, use_container_width=True)

    st.subheader("Daily play summary")
    st.dataframe(daily.head(int(rows_to_show)), use_container_width=True)

    st.subheader("Historical hit-event table")
    st.dataframe(wf_hits.head(int(rows_to_show)), use_container_width=True)

    st.download_button("Download hit-event table CSV", wf_hits.to_csv(index=False), "hybrid_walkforward_hit_events_v1__2026-03-28.csv", "text/csv")
    st.download_button("Download capture summary CSV", capture.to_csv(index=False), "hybrid_walkforward_capture_summary_v1__2026-03-28.csv", "text/csv")
    st.download_button("Download recommendation summary CSV", reco.to_csv(index=False), "hybrid_walkforward_recommendation_summary_v1__2026-03-28.csv", "text/csv")
    st.download_button("Download decision source summary CSV", source.to_csv(index=False), "hybrid_walkforward_decision_source_summary_v1__2026-03-28.csv", "text/csv")
    st.download_button("Download daily play summary CSV", daily.to_csv(index=False), "hybrid_walkforward_daily_play_summary_v1__2026-03-28.csv", "text/csv")
    st.download_button("Download skip diagnostics CSV", skipdiag.to_csv(index=False), "hybrid_walkforward_skip_diagnostics_v1__2026-03-28.csv", "text/csv")
    st.download_button("Download Top2 diagnostics CSV", top2diag.to_csv(index=False), "hybrid_walkforward_top2_diagnostics_v1__2026-03-28.csv", "text/csv")


if __name__ == "__main__":
    if "hybrid_wf_hits" not in st.session_state:
        st.session_state["hybrid_wf_hits"] = None
    main()

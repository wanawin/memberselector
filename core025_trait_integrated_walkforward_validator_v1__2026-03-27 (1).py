#!/usr/bin/env python3
# core025_trait_integrated_walkforward_validator_v1__2026-03-27.py
#
# Full file. No placeholders.
# True historical walk-forward validator for the trait-integrated Core025 member engine.

from __future__ import annotations

import io
import re
from collections import Counter, defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

CORE025 = ["0025", "0225", "0255"]


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def safe_display_df(df: pd.DataFrame, rows: int) -> pd.DataFrame:
    return df.head(int(rows)).copy()


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


def choose_best_separation_override(
    matched_sep: pd.DataFrame,
    matched_0025: Optional[pd.DataFrame],
    matched_0225: Optional[pd.DataFrame],
    matched_0255: Optional[pd.DataFrame],
    sep_min_rate: float,
    sep_min_gap: float,
) -> Tuple[Optional[str], Optional[str], Optional[float], Optional[float]]:
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
                "rate": float(r["winning_member_rate"]),
                "gap": float(r["separation_gap"]),
                "support": int(r["support"]),
            })

    for label, mdf in [("0025", matched_0025), ("0225", matched_0225), ("0255", matched_0255)]:
        if mdf is None or len(mdf) == 0:
            continue
        sub = mdf[mdf["target_member_rate"] >= float(sep_min_rate)].copy()
        for _, r in sub.iterrows():
            candidates.append({
                "member": label,
                "trait": r["trait"],
                "rate": float(r["target_member_rate"]),
                "gap": float(r["target_member_rate"] - (1.0 - r["target_member_rate"])),
                "support": int(r["support"]),
            })

    if not candidates:
        return None, None, None, None

    best = sorted(candidates, key=lambda x: (x["rate"], x["gap"], x["support"]), reverse=True)[0]
    return best["member"], best["trait"], best["rate"], best["gap"]


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
    seed_feat: Dict[str, object],
    seed: str,
    stream: str,
    state: Dict[str, object],
    min_stream_history: int,
    stream_bias_weight: float,
    exact_seed_weight: float,
    sorted_seed_weight: float,
    similarity_weight: float,
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


def score_current_hit_event(
    current: pd.Series,
    state: Dict[str, object],
    sep_traits: pd.DataFrame,
    top2_needed_traits: pd.DataFrame,
    skip_danger_traits: pd.DataFrame,
    traits_0025: Optional[pd.DataFrame],
    traits_0225: Optional[pd.DataFrame],
    traits_0255: Optional[pd.DataFrame],
    min_global_history: int,
    min_stream_history: int,
    stream_bias_weight: float,
    exact_seed_weight: float,
    sorted_seed_weight: float,
    similarity_weight: float,
    top1_only_threshold: float,
    play_two_threshold: float,
    weak_skip_threshold: float,
    sep_min_rate: float,
    sep_min_gap: float,
    top2_needed_min_rate: float,
    skip_danger_min_rate: float,
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
        seed_feat=seed_feat,
        seed=seed,
        stream=stream,
        state=state,
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
    matched_skip_danger = matched_traits_for_row(current, skip_danger_traits)
    matched_0025 = matched_traits_for_row(current, traits_0025) if traits_0025 is not None else pd.DataFrame()
    matched_0225 = matched_traits_for_row(current, traits_0225) if traits_0225 is not None else pd.DataFrame()
    matched_0255 = matched_traits_for_row(current, traits_0255) if traits_0255 is not None else pd.DataFrame()

    override_reason = ""
    override_type = ""
    forced_member = None

    sep_member, sep_trait, _, _ = choose_best_separation_override(
        matched_sep=matched_sep,
        matched_0025=matched_0025,
        matched_0225=matched_0225,
        matched_0255=matched_0255,
        sep_min_rate=float(sep_min_rate),
        sep_min_gap=float(sep_min_gap),
    )
    if sep_member is not None:
        forced_member = sep_member
        override_type = "member_force"
        override_reason = f"Separation override: {sep_trait}"

    if forced_member is not None:
        remaining = [(m, s) for m, s in ranked if m != forced_member]
        forced_score = dict(ranked).get(forced_member, max(top1_score, 0.34))
        new_ranked = [(forced_member, forced_score)] + remaining
        top1, top1_score = new_ranked[0]
        top2, top2_score = new_ranked[1]
        top3, top3_score = new_ranked[2]
        gap12 = top1_score - top2_score

    if top1_score < float(weak_skip_threshold):
        recommendation = "Skip member play"
    elif top1_score >= float(top1_only_threshold):
        recommendation = "Top1 only"
    elif top1_score >= float(play_two_threshold):
        recommendation = "Top1 + Top2"
    else:
        recommendation = "Skip member play"

    if len(matched_top2):
        best_top2 = matched_top2.sort_values(
            ["top2_needed_rate", "support_top2_needed", "hit_event_support"],
            ascending=[False, False, False]
        ).head(1)
        rate = float(best_top2.iloc[0]["top2_needed_rate"])
        if rate >= float(top2_needed_min_rate):
            recommendation = "Top1 + Top2"
            override_type = "top2_needed" if override_type == "" else override_type + "+top2_needed"
            add_reason = f"Top2-needed override: {best_top2.iloc[0]['trait']}"
            override_reason = add_reason if override_reason == "" else override_reason + " | " + add_reason

    if len(matched_skip_danger):
        best_sd = matched_skip_danger.sort_values(
            ["skip_danger_rate", "support_skipped_hits", "hit_event_support"],
            ascending=[False, False, False]
        ).head(1)
        rate = float(best_sd.iloc[0]["skip_danger_rate"])
        if rate >= float(skip_danger_min_rate) and recommendation == "Skip member play":
            recommendation = "Top1 + Top2"
            override_type = "skip_danger" if override_type == "" else override_type + "+skip_danger"
            add_reason = f"Skip-danger override: {best_sd.iloc[0]['trait']}"
            override_reason = add_reason if override_reason == "" else override_reason + " | " + add_reason

    actual_member = current["next_member"] if pd.notna(current["next_member"]) else ""

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
        "override_type": override_type,
        "override_reason": override_reason,
        "matched_sep_traits": len(matched_sep),
        "matched_top2_needed_traits": len(matched_top2),
        "matched_skip_danger_traits": len(matched_skip_danger),
        "top1_hit": int(actual_member == top1),
        "top2_hit": int(actual_member in [top1, top2]),
        "top3_hit": int(actual_member in [top1, top2, top3]),
        "member_play_hit": int(
            (recommendation == "Top1 only" and actual_member == top1) or
            (recommendation == "Top1 + Top2" and actual_member in [top1, top2])
        ),
    }


def run_trait_integrated_walkforward(
    transitions: pd.DataFrame,
    sep_traits: pd.DataFrame,
    top2_needed_traits: pd.DataFrame,
    skip_danger_traits: pd.DataFrame,
    traits_0025: Optional[pd.DataFrame],
    traits_0225: Optional[pd.DataFrame],
    traits_0255: Optional[pd.DataFrame],
    min_global_history: int,
    min_stream_history: int,
    stream_bias_weight: float,
    exact_seed_weight: float,
    sorted_seed_weight: float,
    similarity_weight: float,
    top1_only_threshold: float,
    play_two_threshold: float,
    weak_skip_threshold: float,
    sep_min_rate: float,
    sep_min_gap: float,
    top2_needed_min_rate: float,
    skip_danger_min_rate: float,
    max_global_similarity_pool: int,
    max_stream_similarity_pool: int,
    chunk_size: int,
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
                current=current,
                state=state,
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
                play_two_threshold=float(play_two_threshold),
                weak_skip_threshold=float(weak_skip_threshold),
                sep_min_rate=float(sep_min_rate),
                sep_min_gap=float(sep_min_gap),
                top2_needed_min_rate=float(top2_needed_min_rate),
                skip_danger_min_rate=float(skip_danger_min_rate),
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


def summarize_capture(wf_hits: pd.DataFrame) -> pd.DataFrame:
    total_hits = len(wf_hits)
    rows = [
        {"metric": "Top1 capture on Core025 hit events", "numerator": int(wf_hits["top1_hit"].sum()), "denominator": total_hits, "rate": float(wf_hits["top1_hit"].mean()) if total_hits else np.nan},
        {"metric": "Top2 capture on Core025 hit events", "numerator": int(wf_hits["top2_hit"].sum()), "denominator": total_hits, "rate": float(wf_hits["top2_hit"].mean()) if total_hits else np.nan},
        {"metric": "Top3 capture on Core025 hit events", "numerator": int(wf_hits["top3_hit"].sum()), "denominator": total_hits, "rate": float(wf_hits["top3_hit"].mean()) if total_hits else np.nan},
        {"metric": "Play-rule capture on Core025 hit events", "numerator": int(wf_hits["member_play_hit"].sum()), "denominator": total_hits, "rate": float(wf_hits["member_play_hit"].mean()) if total_hits else np.nan},
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


def summarize_by_override(wf_hits: pd.DataFrame) -> pd.DataFrame:
    tmp = wf_hits.copy()
    tmp["override_type"] = tmp["override_type"].replace("", "none")
    out = tmp.groupby("override_type", dropna=False).agg(
        events=("override_type", "size"),
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


def app():
    st.set_page_config(page_title="Core025 Trait-Integrated Walk-Forward Validator", layout="wide")
    st.title("Core025 Trait-Integrated Walk-Forward Validator")
    st.caption("True historical walk-forward for the trait-integrated member engine. Uses trait overrides plus weighted scoring, and only prior data for each scored event.")

    with st.sidebar:
        st.header("Walk-forward controls")
        min_global_history = st.number_input("Minimum prior transitions before scoring", min_value=10, value=100, step=10)
        min_stream_history = st.number_input("Minimum stream-specific history to use stream-only bias/pool", min_value=0, value=20, step=5)

        st.header("Model weights")
        stream_bias_weight = st.slider("Stream-bias weight", min_value=0.0, max_value=3.0, value=1.20, step=0.05)
        exact_seed_weight = st.slider("Exact-seed weight", min_value=0.0, max_value=3.0, value=1.50, step=0.05)
        sorted_seed_weight = st.slider("Sorted-seed weight", min_value=0.0, max_value=3.0, value=1.10, step=0.05)
        similarity_weight = st.slider("Similarity weight", min_value=0.0, max_value=3.0, value=1.80, step=0.05)

        st.header("Decision thresholds")
        top1_only_threshold = st.slider("Top1-only threshold", min_value=0.33, max_value=0.95, value=0.48, step=0.005)
        play_two_threshold = st.slider("Top1+Top2 threshold", min_value=0.33, max_value=0.95, value=0.36, step=0.005)
        weak_skip_threshold = st.slider("Weak-score skip threshold", min_value=0.00, max_value=0.50, value=0.33, step=0.001)

        st.header("Trait override thresholds")
        sep_min_rate = st.slider("Minimum separation rate to force member", min_value=0.34, max_value=0.95, value=0.60, step=0.01)
        sep_min_gap = st.slider("Minimum separation gap to force member", min_value=0.00, max_value=0.50, value=0.10, step=0.01)
        top2_needed_min_rate = st.slider("Minimum Top2-needed rate to force Top1+Top2", min_value=0.05, max_value=0.95, value=0.30, step=0.01)
        skip_danger_min_rate = st.slider("Minimum skip-danger rate to block skip", min_value=0.05, max_value=0.95, value=0.30, step=0.01)

        st.header("Performance controls")
        max_global_similarity_pool = st.number_input("Max global similarity pool rows", min_value=200, value=4000, step=100)
        max_stream_similarity_pool = st.number_input("Max stream similarity pool rows", min_value=50, value=600, step=50)
        chunk_size = st.number_input("Transition chunk size", min_value=100, value=1000, step=100)

        st.header("Target")
        target_capture = st.slider("Required capture target", min_value=0.50, max_value=0.95, value=0.75, step=0.01)
        rows_to_show = st.number_input("Rows to display", min_value=5, value=25, step=5)

    hist_file = st.file_uploader("Upload FULL history file", key="tiwf_hist")
    sep_file = st.file_uploader("Upload separation traits CSV", key="tiwf_sep")
    top2_needed_file = st.file_uploader("Upload Top2-needed traits CSV", key="tiwf_top2")
    skip_danger_file = st.file_uploader("Upload skip-danger traits CSV", key="tiwf_skip")
    t0025_file = st.file_uploader("Optional: upload 0025 traits CSV", key="tiwf_0025")
    t0225_file = st.file_uploader("Optional: upload 0225 traits CSV", key="tiwf_0225")
    t0255_file = st.file_uploader("Optional: upload 0255 traits CSV", key="tiwf_0255")

    if not all([hist_file, sep_file, top2_needed_file, skip_danger_file]):
        st.info("Upload the full history file, separation traits, Top2-needed traits, and skip-danger traits to begin.")
        return

    try:
        hist = prepare_history(load_table(hist_file))
        sep_traits = load_trait_df(sep_file)
        top2_needed_traits = load_trait_df(top2_needed_file)
        skip_danger_traits = load_trait_df(skip_danger_file)
        traits_0025 = load_trait_df(t0025_file) if t0025_file is not None else None
        traits_0225 = load_trait_df(t0225_file) if t0225_file is not None else None
        traits_0255 = load_trait_df(t0255_file) if t0255_file is not None else None
    except Exception as e:
        st.exception(e)
        return

    transitions = build_transitions(hist)
    core025_hit_events = int(transitions["is_core025_hit"].sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Transitions", f"{len(transitions):,}")
    c2.metric("Core025 hit events", f"{core025_hit_events:,}")
    c3.metric("Base Core025 rate", f"{transitions['is_core025_hit'].mean():.4f}")

    if st.button("Run Trait-Integrated Walk-Forward", type="primary"):
        try:
            with st.spinner("Running trait-integrated walk-forward..."):
                wf_hits = run_trait_integrated_walkforward(
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
                    play_two_threshold=float(play_two_threshold),
                    weak_skip_threshold=float(weak_skip_threshold),
                    sep_min_rate=float(sep_min_rate),
                    sep_min_gap=float(sep_min_gap),
                    top2_needed_min_rate=float(top2_needed_min_rate),
                    skip_danger_min_rate=float(skip_danger_min_rate),
                    max_global_similarity_pool=int(max_global_similarity_pool),
                    max_stream_similarity_pool=int(max_stream_similarity_pool),
                    chunk_size=int(chunk_size),
                )

            st.session_state["tiwf_results"] = {
                "wf_hits": wf_hits,
                "summary_capture": summarize_capture(wf_hits),
                "summary_recommendation": summarize_by_recommendation(wf_hits),
                "summary_override": summarize_by_override(wf_hits),
                "summary_top2": summarize_top2_needed(wf_hits),
            }
            st.rerun()
        except Exception as e:
            st.exception(e)
            return

    results = st.session_state.get("tiwf_results")
    if results is None:
        return

    wf_hits = results["wf_hits"]
    summary_capture = results["summary_capture"]
    summary_recommendation = results["summary_recommendation"]
    summary_override = results["summary_override"]
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

    st.subheader("By override type")
    st.dataframe(summary_override, use_container_width=True)

    st.subheader("Top2-needed pockets by low Top1-Top2 gap")
    st.dataframe(summary_top2, use_container_width=True)

    st.subheader("Historical hit-event table")
    st.dataframe(safe_display_df(wf_hits, int(rows_to_show)), use_container_width=True)

    st.download_button("Download hit-event table CSV", data=df_to_csv_bytes(wf_hits), file_name="core025_trait_integrated_walkforward_hit_events__2026-03-27.csv", mime="text/csv")
    st.download_button("Download capture summary CSV", data=df_to_csv_bytes(summary_capture), file_name="core025_trait_integrated_walkforward_capture_summary__2026-03-27.csv", mime="text/csv")
    st.download_button("Download recommendation summary CSV", data=df_to_csv_bytes(summary_recommendation), file_name="core025_trait_integrated_walkforward_recommendation_summary__2026-03-27.csv", mime="text/csv")
    st.download_button("Download override summary CSV", data=df_to_csv_bytes(summary_override), file_name="core025_trait_integrated_walkforward_override_summary__2026-03-27.csv", mime="text/csv")
    st.download_button("Download Top2-needed gap summary CSV", data=df_to_csv_bytes(summary_top2), file_name="core025_trait_integrated_walkforward_top2_needed_summary__2026-03-27.csv", mime="text/csv")


if __name__ == "__main__":
    if "tiwf_results" not in st.session_state:
        st.session_state["tiwf_results"] = None
    app()

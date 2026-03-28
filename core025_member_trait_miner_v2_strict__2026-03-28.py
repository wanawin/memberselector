#!/usr/bin/env python3
# core025_member_trait_miner_v2_strict__2026-03-28.py
#
# BUILD: core025_member_trait_miner_v2_strict__2026-03-28
#
# Full file. No placeholders.
#
# Purpose
# -------
# Rebuilt strict trait miner for Core025.
#
# Key changes vs prior miner
# --------------------------
# - Raises support thresholds
# - Removes noisy single-position digit traits and modulo traits
# - Focuses on more stable features
# - Adds stream/time stability checks
# - Produces stricter separation, Top2-needed, and skip-danger traits
# - Emits member-specific trait files for review, but downstream member forcing
#   should be disabled in the engine until those files prove useful historically

from __future__ import annotations

import io
import re
from collections import Counter
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

CORE025 = ["0025", "0225", "0255"]
BUILD_MARKER = "BUILD: core025_member_trait_miner_v2_strict__2026-03-28"


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def safe_display_df(df: pd.DataFrame, rows: int) -> pd.DataFrame:
    return df.head(int(rows)).copy()


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


def norm_result(r: object) -> Optional[str]:
    d = re.findall(r"\d", str(r))
    return "".join(d[:4]) if len(d) >= 4 else None


def to_member(r4: Optional[str]) -> Optional[str]:
    if r4 is None:
        return None
    s = "".join(sorted(r4))
    return s if s in CORE025 else None


def sum_bucket(x: int) -> str:
    if x <= 9:
        return "sum_00_09"
    if x <= 13:
        return "sum_10_13"
    if x <= 17:
        return "sum_14_17"
    if x <= 21:
        return "sum_18_21"
    return "sum_22_plus"


def spread_bucket(x: int) -> str:
    if x <= 2:
        return "spread_0_2"
    if x <= 4:
        return "spread_3_4"
    if x <= 6:
        return "spread_5_6"
    return "spread_7_plus"


def pair_token_pattern(digs: List[int]) -> str:
    tokens = []
    for i in range(4):
        for j in range(i + 1, 4):
            tokens.append("".join(sorted([str(digs[i]), str(digs[j])])))
    return "|".join(sorted(tokens))


def structure_label(digs: List[int]) -> str:
    counts = sorted(Counter(digs).values(), reverse=True)
    if counts == [1, 1, 1, 1]:
        return "ABCD"
    if counts == [2, 1, 1]:
        return "AABC"
    if counts == [2, 2]:
        return "AABB"
    if counts == [3, 1]:
        return "AAAB"
    if counts == [4]:
        return "AAAA"
    return "OTHER"


def features(seed: object) -> Optional[Dict[str, object]]:
    if seed is None:
        return None
    d = re.findall(r"\d", str(seed))
    if len(d) < 4:
        return None
    digs = [int(x) for x in d[:4]]
    cnt = Counter(digs)
    unique_sorted = sorted(set(digs))
    consec_links = 0
    for a, b in zip(unique_sorted[:-1], unique_sorted[1:]):
        if b - a == 1:
            consec_links += 1
    s = sum(digs)
    spread = max(digs) - min(digs)
    feat = {
        "sum": s,
        "sum_bucket": sum_bucket(s),
        "spread": spread,
        "spread_bucket": spread_bucket(spread),
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
        "pair_token_pattern": pair_token_pattern(digs),
        "structure": structure_label(digs),
    }
    for k in range(10):
        feat[f"has{k}"] = int(k in cnt)
        feat[f"cnt{k}"] = int(cnt.get(k, 0))
    return feat


def miner_feature_columns() -> List[str]:
    # intentionally excludes noisy single-position digits and modulo features
    return [
        "sum_bucket",
        "spread_bucket",
        "even",
        "odd",
        "high",
        "low",
        "unique",
        "pair",
        "max_rep",
        "sorted_seed",
        "first2",
        "last2",
        "consec_links",
        "parity_pattern",
        "highlow_pattern",
        "pair_token_pattern",
        "structure",
    ] + [f"has{k}" for k in range(10)] + [f"cnt{k}" for k in range(10)]


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
    df = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    return df


def build_transitions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
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
                "year_month": g.loc[i, "date"].to_period("M").strftime("%Y-%m"),
                "seed": seed,
                "next_r4": next_r4,
                "next_member": next_member,
                "is_core025_hit": int(next_member is not None),
                **feat,
            })
    out = pd.DataFrame(rows)
    out = out.sort_values(["event_date", "stream", "seed"]).reset_index(drop=True)
    return out


def transition_maps(prior_transitions: pd.DataFrame):
    exact_seed_map = {}
    sorted_seed_map = {}
    stream_member_map = {}
    global_member_map = Counter()

    for _, r in prior_transitions.iterrows():
        member = r["next_member"]
        if member is None or pd.isna(member):
            continue
        exact_seed_map.setdefault(str(r["seed"]), Counter())[member] += 1
        sorted_seed_map.setdefault(str(r["sorted_seed"]), Counter())[member] += 1
        stream_member_map.setdefault(str(r["stream"]), Counter())[member] += 1
        global_member_map[member] += 1
    return exact_seed_map, sorted_seed_map, stream_member_map, global_member_map


def counter_to_probs(c: Counter) -> Dict[str, float]:
    total = sum(c.values())
    if total <= 0:
        return {m: 1 / 3 for m in CORE025}
    return {m: c.get(m, 0) / total for m in CORE025}


def similarity(a: Dict[str, object], b: pd.Series) -> float:
    score = 0.0
    if a["sum_bucket"] == b["sum_bucket"]:
        score += 3
    if a["spread_bucket"] == b["spread_bucket"]:
        score += 2
    if a["parity_pattern"] == b["parity_pattern"]:
        score += 2
    if a["highlow_pattern"] == b["highlow_pattern"]:
        score += 2
    if a["structure"] == b["structure"]:
        score += 3
    if a["sorted_seed"] == b["sorted_seed"]:
        score += 6
    if a["pair_token_pattern"] == b["pair_token_pattern"]:
        score += 4
    if a["unique"] == b["unique"]:
        score += 1
    if a["max_rep"] == b["max_rep"]:
        score += 1
    for k in range(10):
        if a[f"has{k}"] == b[f"has{k}"]:
            score += 0.2
        if a[f"cnt{k}"] == b[f"cnt{k}"]:
            score += 0.1
    return float(score)


def score_seed_v3(seed: str, stream: str, prior_transitions: pd.DataFrame, min_stream_history: int = 20) -> List[Tuple[str, float]]:
    seed_feat = features(seed)
    if seed_feat is None:
        return [(m, 1 / 3) for m in CORE025]

    exact_seed_map, sorted_seed_map, stream_member_map, global_member_map = transition_maps(prior_transitions)

    score_accum = {m: 0.0 for m in CORE025}
    global_probs = counter_to_probs(global_member_map)
    for m in CORE025:
        score_accum[m] += global_probs[m] * 0.25

    if stream is not None and str(stream) in stream_member_map and sum(stream_member_map[str(stream)].values()) >= int(min_stream_history):
        stream_probs = counter_to_probs(stream_member_map[str(stream)])
        for m in CORE025:
            score_accum[m] += stream_probs[m] * 1.20

    if str(seed) in exact_seed_map:
        exact_probs = counter_to_probs(exact_seed_map[str(seed)])
        for m in CORE025:
            score_accum[m] += exact_probs[m] * 1.50

    sorted_key = str(seed_feat["sorted_seed"])
    if sorted_key in sorted_seed_map:
        sorted_probs = counter_to_probs(sorted_seed_map[sorted_key])
        for m in CORE025:
            score_accum[m] += sorted_probs[m] * 1.10

    pool = prior_transitions.copy()
    if stream is not None:
        stream_subset = prior_transitions[prior_transitions["stream"] == str(stream)].copy()
        if len(stream_subset) >= int(min_stream_history):
            pool = stream_subset

    if len(pool):
        pool = pool.sort_values("event_date").reset_index(drop=True)
        sim_scores = {m: 0.0 for m in CORE025}
        total_sim = 0.0
        for _, r in pool.iterrows():
            member = r["next_member"]
            if member is None or pd.isna(member):
                continue
            sim = similarity(seed_feat, r)
            if sim <= 0:
                continue
            sim_scores[member] += sim
            total_sim += sim
        if total_sim > 0:
            for m in CORE025:
                score_accum[m] += (sim_scores[m] / total_sim) * 1.80

    total = sum(score_accum.values())
    if total <= 0:
        return [(m, 1 / 3) for m in CORE025]
    probs = {m: score_accum[m] / total for m in CORE025}
    return sorted(probs.items(), key=lambda x: x[1], reverse=True)


def classify_score_tier(top1_score: float, top1_only_score_threshold: float, top1_top2_score_threshold: float) -> str:
    if top1_score >= float(top1_only_score_threshold):
        return "Top1 only"
    if top1_score >= float(top1_top2_score_threshold):
        return "Top1 + Top2"
    return "Skip member play"


def build_hit_event_predictions(
    transitions: pd.DataFrame,
    min_global_history: int,
    min_stream_history: int,
    top1_only_score_threshold: float,
    top1_top2_score_threshold: float,
) -> pd.DataFrame:
    rows = []
    for i in range(len(transitions)):
        current = transitions.iloc[i]
        if int(current["is_core025_hit"]) != 1:
            continue
        prior = transitions.iloc[:i].copy()
        if len(prior) < int(min_global_history):
            continue

        ranked = score_seed_v3(
            seed=str(current["seed"]),
            stream=str(current["stream"]),
            prior_transitions=prior,
            min_stream_history=int(min_stream_history),
        )

        top1, top1_score = ranked[0]
        top2, top2_score = ranked[1]
        top3, top3_score = ranked[2]
        gap12 = top1_score - top2_score
        recommendation = classify_score_tier(
            top1_score=top1_score,
            top1_only_score_threshold=float(top1_only_score_threshold),
            top1_top2_score_threshold=float(top1_top2_score_threshold),
        )
        actual_member = current["next_member"] if pd.notna(current["next_member"]) else ""

        rows.append({
            "event_date": current["event_date"],
            "year_month": current["year_month"],
            "stream": current["stream"],
            "seed": current["seed"],
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
            "member_play_hit": int(
                (recommendation == "Top1 only" and actual_member == top1) or
                (recommendation == "Top1 + Top2" and actual_member in [top1, top2])
            ),
        })
    return pd.DataFrame(rows)


def stability_stats(sub: pd.DataFrame) -> Dict[str, int]:
    return {
        "stream_count": int(sub["stream"].nunique()),
        "month_count": int(sub["year_month"].nunique()),
    }


def build_member_separation_traits(
    core_hits: pd.DataFrame, min_support: int, preferred_support: int, min_dom_rate: float,
    min_gap: float, min_streams: int, min_months: int
) -> pd.DataFrame:
    rows = []
    for col in miner_feature_columns():
        if col not in core_hits.columns:
            continue
        vals = core_hits[col].dropna().unique().tolist()
        try:
            vals = sorted(vals)
        except Exception:
            pass
        for val in vals:
            sub = core_hits[core_hits[col] == val].copy()
            support = len(sub)
            if support < int(min_support):
                continue
            counts = sub["next_member"].value_counts()
            probs = {m: counts.get(m, 0) / support for m in CORE025}
            winner = max(probs, key=probs.get)
            winner_rate = probs[winner]
            second_rate = sorted(probs.values(), reverse=True)[1]
            separation_gap = winner_rate - second_rate
            stable = stability_stats(sub)
            if winner_rate < float(min_dom_rate):
                continue
            if separation_gap < float(min_gap):
                continue
            if stable["stream_count"] < int(min_streams):
                continue
            if stable["month_count"] < int(min_months):
                continue
            rows.append({
                "trait": f"{col}={val}",
                "support": support,
                "preferred_support_met": int(support >= int(preferred_support)),
                "winning_member": winner,
                "winning_member_rate": winner_rate,
                "second_best_rate": second_rate,
                "separation_gap": separation_gap,
                "stream_count": stable["stream_count"],
                "month_count": stable["month_count"],
                "rate_0025": probs["0025"],
                "rate_0225": probs["0225"],
                "rate_0255": probs["0255"],
            })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["preferred_support_met", "winning_member_rate", "separation_gap", "support", "stream_count", "month_count"],
            ascending=[False, False, False, False, False, False]
        ).reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=[
            "trait", "support", "preferred_support_met", "winning_member", "winning_member_rate",
            "second_best_rate", "separation_gap", "stream_count", "month_count",
            "rate_0025", "rate_0225", "rate_0255"
        ])
    return out


def build_member_specific_traits(
    core_hits: pd.DataFrame, min_support: int, preferred_support: int, target_member: str,
    min_member_rate: float, min_streams: int, min_months: int
) -> pd.DataFrame:
    rows = []
    for col in miner_feature_columns():
        if col not in core_hits.columns:
            continue
        vals = core_hits[col].dropna().unique().tolist()
        try:
            vals = sorted(vals)
        except Exception:
            pass
        for val in vals:
            sub = core_hits[core_hits[col] == val].copy()
            support = len(sub)
            if support < int(min_support):
                continue
            rate = float((sub["next_member"] == target_member).mean())
            stable = stability_stats(sub)
            if rate < float(min_member_rate):
                continue
            if stable["stream_count"] < int(min_streams):
                continue
            if stable["month_count"] < int(min_months):
                continue
            rows.append({
                "trait": f"{col}={val}",
                "support": support,
                "preferred_support_met": int(support >= int(preferred_support)),
                "target_member": target_member,
                "target_member_rate": rate,
                "non_target_rate": 1.0 - rate,
                "stream_count": stable["stream_count"],
                "month_count": stable["month_count"],
            })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["preferred_support_met", "target_member_rate", "support", "stream_count", "month_count"],
            ascending=[False, False, False, False, False]
        ).reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=[
            "trait", "support", "preferred_support_met", "target_member", "target_member_rate", "non_target_rate",
            "stream_count", "month_count"
        ])
    return out


def build_top2_needed_traits(
    pred_hits: pd.DataFrame, transitions: pd.DataFrame, min_support: int, preferred_support: int,
    min_top2_rate: float, min_streams: int, min_months: int
) -> pd.DataFrame:
    pockets = pred_hits[(pred_hits["top1_hit"] == 0) & (pred_hits["top2_hit"] == 1)].copy()
    if len(pockets) == 0:
        return pd.DataFrame(columns=["trait", "support_top2_needed", "hit_event_support", "top2_needed_rate"])
    trait_source = transitions[["event_date", "stream", "year_month"] + [c for c in miner_feature_columns() if c in transitions.columns]].copy()
    data = pockets.merge(trait_source, on=["event_date", "stream"], how="left")
    rows = []
    cols = [c for c in miner_feature_columns() if c in data.columns]
    for col in cols:
        vals = data[col].dropna().unique().tolist()
        try:
            vals = sorted(vals)
        except Exception:
            pass
        for val in vals:
            sub = data[data[col] == val].copy()
            support = len(sub)
            if support < int(min_support):
                continue
            denom_df = transitions[(transitions["is_core025_hit"] == 1) & (transitions[col] == val)].copy()
            denom = len(denom_df)
            if denom < int(min_support):
                continue
            top2_needed_rate = support / denom
            stable = stability_stats(sub)
            if top2_needed_rate < float(min_top2_rate):
                continue
            if stable["stream_count"] < int(min_streams):
                continue
            if stable["month_count"] < int(min_months):
                continue
            rows.append({
                "trait": f"{col}={val}",
                "support_top2_needed": support,
                "preferred_support_met": int(support >= int(preferred_support)),
                "hit_event_support": denom,
                "top2_needed_rate": top2_needed_rate,
                "stream_count": stable["stream_count"],
                "month_count": stable["month_count"],
            })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["preferred_support_met", "top2_needed_rate", "support_top2_needed", "stream_count", "month_count"],
            ascending=[False, False, False, False, False]
        ).reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=[
            "trait", "support_top2_needed", "preferred_support_met", "hit_event_support", "top2_needed_rate",
            "stream_count", "month_count"
        ])
    return out


def build_skip_danger_traits(
    pred_hits: pd.DataFrame, transitions: pd.DataFrame, min_support: int, preferred_support: int,
    min_skip_danger_rate: float, min_streams: int, min_months: int
) -> pd.DataFrame:
    danger = pred_hits[pred_hits["recommendation"] == "Skip member play"].copy()
    if len(danger) == 0:
        return pd.DataFrame(columns=["trait", "support_skipped_hits", "hit_event_support", "skip_danger_rate"])
    trait_source = transitions[["event_date", "stream", "year_month"] + [c for c in miner_feature_columns() if c in transitions.columns]].copy()
    data = danger.merge(trait_source, on=["event_date", "stream"], how="left")
    rows = []
    cols = [c for c in miner_feature_columns() if c in data.columns]
    for col in cols:
        vals = data[col].dropna().unique().tolist()
        try:
            vals = sorted(vals)
        except Exception:
            pass
        for val in vals:
            sub = data[data[col] == val].copy()
            support = len(sub)
            if support < int(min_support):
                continue
            denom_df = transitions[(transitions["is_core025_hit"] == 1) & (transitions[col] == val)].copy()
            denom = len(denom_df)
            if denom < int(min_support):
                continue
            skip_danger_rate = support / denom
            stable = stability_stats(sub)
            if skip_danger_rate < float(min_skip_danger_rate):
                continue
            if stable["stream_count"] < int(min_streams):
                continue
            if stable["month_count"] < int(min_months):
                continue
            rows.append({
                "trait": f"{col}={val}",
                "support_skipped_hits": support,
                "preferred_support_met": int(support >= int(preferred_support)),
                "hit_event_support": denom,
                "skip_danger_rate": skip_danger_rate,
                "stream_count": stable["stream_count"],
                "month_count": stable["month_count"],
            })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["preferred_support_met", "skip_danger_rate", "support_skipped_hits", "stream_count", "month_count"],
            ascending=[False, False, False, False, False]
        ).reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=[
            "trait", "support_skipped_hits", "preferred_support_met", "hit_event_support", "skip_danger_rate",
            "stream_count", "month_count"
        ])
    return out


def main():
    st.set_page_config(page_title="Core025 Member Trait Miner v2 Strict", layout="wide")
    st.title("Core025 Member Trait Miner v2 Strict")
    st.caption("Strict miner with stronger support/stability filters and reduced noisy features.")
    st.code(BUILD_MARKER, language="text")

    with st.sidebar:
        st.markdown(f"**{BUILD_MARKER}**")

        st.header("Strict trait filters")
        min_support = st.number_input("Minimum support", min_value=10, value=25, step=1)
        preferred_support = st.number_input("Preferred support", min_value=10, value=40, step=1)
        min_dom_rate = st.slider("Minimum dominance rate for separation", min_value=0.34, max_value=0.95, value=0.60, step=0.01)
        min_gap = st.slider("Minimum separation gap", min_value=0.00, max_value=0.50, value=0.10, step=0.01)
        min_member_rate = st.slider("Minimum member-specific rate", min_value=0.34, max_value=0.95, value=0.60, step=0.01)
        min_top2_rate = st.slider("Minimum Top2-needed rate", min_value=0.05, max_value=0.95, value=0.25, step=0.01)
        min_skip_danger_rate = st.slider("Minimum skip-danger rate", min_value=0.05, max_value=0.95, value=0.25, step=0.01)
        min_streams = st.number_input("Minimum distinct streams", min_value=1, value=3, step=1)
        min_months = st.number_input("Minimum distinct months", min_value=1, value=2, step=1)

        st.header("Prediction-context controls")
        min_global_history = st.number_input("Minimum prior transitions before prediction mining", min_value=10, value=100, step=10)
        min_stream_history = st.number_input("Minimum stream-specific history", min_value=0, value=20, step=5)
        top1_only_score_threshold = st.slider("Top1-only score threshold", min_value=0.33, max_value=0.95, value=0.48, step=0.005)
        top1_top2_score_threshold = st.slider("Top1+Top2 score threshold", min_value=0.33, max_value=0.95, value=0.36, step=0.005)

        rows_to_show = st.number_input("Rows to display", min_value=5, value=25, step=5)

    hist_file = st.file_uploader("Upload FULL history file", key="strict_trait_hist")
    if not hist_file:
        st.info("Upload the full history file to begin.")
        return

    try:
        hist = prepare_history(load_table(hist_file))
    except Exception as e:
        st.exception(e)
        return

    transitions = build_transitions(hist)
    core_hits = transitions[transitions["is_core025_hit"] == 1].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Transitions", f"{len(transitions):,}")
    c2.metric("Core025 hit events", f"{len(core_hits):,}")
    c3.metric("Core025 base rate", f"{transitions['is_core025_hit'].mean():.4f}")

    if st.button("Run Strict Trait Miner", type="primary"):
        try:
            with st.spinner("Mining strict traits..."):
                sep_traits = build_member_separation_traits(
                    core_hits=core_hits,
                    min_support=int(min_support),
                    preferred_support=int(preferred_support),
                    min_dom_rate=float(min_dom_rate),
                    min_gap=float(min_gap),
                    min_streams=int(min_streams),
                    min_months=int(min_months),
                )
                traits_0025 = build_member_specific_traits(
                    core_hits=core_hits,
                    min_support=int(min_support),
                    preferred_support=int(preferred_support),
                    target_member="0025",
                    min_member_rate=float(min_member_rate),
                    min_streams=int(min_streams),
                    min_months=int(min_months),
                )
                traits_0225 = build_member_specific_traits(
                    core_hits=core_hits,
                    min_support=int(min_support),
                    preferred_support=int(preferred_support),
                    target_member="0225",
                    min_member_rate=float(min_member_rate),
                    min_streams=int(min_streams),
                    min_months=int(min_months),
                )
                traits_0255 = build_member_specific_traits(
                    core_hits=core_hits,
                    min_support=int(min_support),
                    preferred_support=int(preferred_support),
                    target_member="0255",
                    min_member_rate=float(min_member_rate),
                    min_streams=int(min_streams),
                    min_months=int(min_months),
                )
                pred_hits = build_hit_event_predictions(
                    transitions=transitions,
                    min_global_history=int(min_global_history),
                    min_stream_history=int(min_stream_history),
                    top1_only_score_threshold=float(top1_only_score_threshold),
                    top1_top2_score_threshold=float(top1_top2_score_threshold),
                )
                top2_needed = build_top2_needed_traits(
                    pred_hits=pred_hits,
                    transitions=transitions,
                    min_support=int(min_support),
                    preferred_support=int(preferred_support),
                    min_top2_rate=float(min_top2_rate),
                    min_streams=int(min_streams),
                    min_months=int(min_months),
                )
                skip_danger = build_skip_danger_traits(
                    pred_hits=pred_hits,
                    transitions=transitions,
                    min_support=int(min_support),
                    preferred_support=int(preferred_support),
                    min_skip_danger_rate=float(min_skip_danger_rate),
                    min_streams=int(min_streams),
                    min_months=int(min_months),
                )

            st.session_state["strict_miner_results"] = {
                "sep_traits": sep_traits,
                "traits_0025": traits_0025,
                "traits_0225": traits_0225,
                "traits_0255": traits_0255,
                "pred_hits": pred_hits,
                "top2_needed": top2_needed,
                "skip_danger": skip_danger,
            }
        except Exception as e:
            st.exception(e)
            return

    if "strict_miner_results" not in st.session_state or st.session_state["strict_miner_results"] is None:
        return

    results = st.session_state["strict_miner_results"]

    st.subheader("Strict separation traits")
    st.dataframe(safe_display_df(results["sep_traits"], int(rows_to_show)), use_container_width=True)
    st.download_button("Download strict separation traits CSV", df_to_csv_bytes(results["sep_traits"]), "core025_member_trait_miner_v2_strict_separation_traits__2026-03-28.csv", "text/csv")

    st.subheader("0025 member-specific traits")
    st.dataframe(safe_display_df(results["traits_0025"], int(rows_to_show)), use_container_width=True)
    st.download_button("Download strict 0025 traits CSV", df_to_csv_bytes(results["traits_0025"]), "core025_member_trait_miner_v2_strict_0025_traits__2026-03-28.csv", "text/csv")

    st.subheader("0225 member-specific traits")
    st.dataframe(safe_display_df(results["traits_0225"], int(rows_to_show)), use_container_width=True)
    st.download_button("Download strict 0225 traits CSV", df_to_csv_bytes(results["traits_0225"]), "core025_member_trait_miner_v2_strict_0225_traits__2026-03-28.csv", "text/csv")

    st.subheader("0255 member-specific traits")
    st.dataframe(safe_display_df(results["traits_0255"], int(rows_to_show)), use_container_width=True)
    st.download_button("Download strict 0255 traits CSV", df_to_csv_bytes(results["traits_0255"]), "core025_member_trait_miner_v2_strict_0255_traits__2026-03-28.csv", "text/csv")

    st.subheader("Strict Top2-needed traits")
    st.dataframe(safe_display_df(results["top2_needed"], int(rows_to_show)), use_container_width=True)
    st.download_button("Download strict Top2-needed traits CSV", df_to_csv_bytes(results["top2_needed"]), "core025_member_trait_miner_v2_strict_top2_needed_traits__2026-03-28.csv", "text/csv")

    st.subheader("Strict skip-danger traits")
    st.dataframe(safe_display_df(results["skip_danger"], int(rows_to_show)), use_container_width=True)
    st.download_button("Download strict skip-danger traits CSV", df_to_csv_bytes(results["skip_danger"]), "core025_member_trait_miner_v2_strict_skip_danger_traits__2026-03-28.csv", "text/csv")

    st.subheader("Historical hit-event prediction table")
    st.dataframe(safe_display_df(results["pred_hits"], int(rows_to_show)), use_container_width=True)
    st.download_button("Download strict hit-event prediction table CSV", df_to_csv_bytes(results["pred_hits"]), "core025_member_trait_miner_v2_strict_hit_event_predictions__2026-03-28.csv", "text/csv")


if __name__ == "__main__":
    if "strict_miner_results" not in st.session_state:
        st.session_state["strict_miner_results"] = None
    main()

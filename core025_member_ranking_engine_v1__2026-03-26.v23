#!/usr/bin/env python3
# core025_separator_engine_plus_lab_walkforward__2026-04-11_v23_safe_mined_pockets_locked.py
#
# BUILD: core025_separator_engine_plus_lab_walkforward__2026-04-11_v23_safe_mined_pockets_locked
#
# Full file. No placeholders.
#
# Purpose
# -------
# Unified Core025 separator app with:
# 1) Regular Run mode for current survivor playlist ranking
# 2) Optional LAB mode for full no-lookahead walk-forward validation
#
# This version adds score-level member calibration based on the mined member-
# specific Top1 calibration outputs.
#
# Locked optimization goal
# ----------------------
# - increase Top1 accuracy
# - avoid regressions that merely increase Top2 usage
# - use Top2 only when truly necessary
#
# V8 focus
# --------
# - 0025 was over-promoted -> apply score suppression in weak-dominance pockets
# - 0225 stays mostly stable with a light promotion assist in cleaner pockets
# - 0255 was under-promoted -> apply score boosts before ranking when its
#   calibration signals are present
# - member-specific gates remain in place after scoring
#
# Outputs
# -------
# Regular Run:
# - core025_separator_ranked_playlist__2026-04-11_v23_safe_mined_pockets_locked.csv
# - core025_separator_summary__2026-04-11_v23_safe_mined_pockets_locked.csv
#
# LAB Walk-Forward:
# - core025_lab_per_event__2026-04-11_v23_safe_mined_pockets_locked.csv
# - core025_lab_per_date__2026-04-11_v23_safe_mined_pockets_locked.csv
# - core025_lab_per_stream__2026-04-11_v23_safe_mined_pockets_locked.csv
# - core025_lab_summary__2026-04-11_v23_safe_mined_pockets_locked.csv

from __future__ import annotations

import io
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

CORE025 = ["0025", "0225", "0255"]
BUILD_MARKER = "BUILD: core025_separator_engine_plus_lab_walkforward__2026-04-11_v23_safe_mined_pockets_locked"
BUILD_SLUG = BUILD_MARKER.replace("BUILD: ", "")
DEFAULT_SKIP_SCORE_CUTOFF = 0.515465


def build_operational_views(per_event: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(per_event) == 0:
        empty = pd.DataFrame()
        return empty, empty, empty

    out = per_event.copy()
    out["is_play_top3"] = (out.get("play_mode", "") == "PLAY_TOP3").astype(int)
    out["recommended_play_count"] = 0
    out.loc[out["is_play_top1"] == 1, "recommended_play_count"] = 1
    out.loc[out["is_play_top2"] == 1, "recommended_play_count"] = 2
    out.loc[out["is_play_top3"] == 1, "recommended_play_count"] = 3

    out["bucket_top1_win"] = ((out["is_play_top1"] == 1) & (out["Top1"] == out["winning_member"])).astype(int)
    out["bucket_top2_win"] = ((out["is_play_top2"] == 1) & (out["top2_hit"] == 1)).astype(int)
    out["bucket_top3_win"] = ((out["is_play_top3"] == 1) & (out["top3_hit"] == 1)).astype(int)
    out["bucket_miss"] = (out["play_rule_hit"] == 0).astype(int)

    out["top2_top1_would_have_won"] = ((out["is_play_top2"] == 1) & (out["Top1"] == out["winning_member"])).astype(int)
    out["top2_top2_needed"] = ((out["is_play_top2"] == 1) & (out["Top1"] != out["winning_member"]) & (out["Top2"] == out["winning_member"])).astype(int)
    out["top3_top1_would_have_won"] = ((out["is_play_top3"] == 1) & (out["Top1"] == out["winning_member"])).astype(int)
    out["top3_top2_would_have_won"] = ((out["is_play_top3"] == 1) & (out["Top1"] != out["winning_member"]) & (out["Top2"] == out["winning_member"])).astype(int)
    out["top3_top3_needed"] = ((out["is_play_top3"] == 1) & (out["Top1"] != out["winning_member"]) & (out["Top2"] != out["winning_member"]) & (out["Top3"] == out["winning_member"])).astype(int)

    out["winning_member_plays"] = out["play_rule_hit"].astype(int)
    out["losing_member_plays_on_winner_events"] = out["recommended_play_count"] - out["winning_member_plays"]

    bucket_summary = pd.DataFrame([
        {"metric": "winner_event_rows", "value": int(len(out))},
        {"metric": "top1_wins__play_top1_and_top1_won", "value": int(out["bucket_top1_win"].sum())},
        {"metric": "top2_wins__play_top2_and_capture", "value": int(out["bucket_top2_win"].sum())},
        {"metric": "top3_wins__play_top3_and_capture", "value": int(out["bucket_top3_win"].sum())},
        {"metric": "misses__recommended_set_did_not_capture", "value": int(out["bucket_miss"].sum())},
        {"metric": "play_top1_rows", "value": int(out["is_play_top1"].sum())},
        {"metric": "play_top2_rows", "value": int(out["is_play_top2"].sum())},
        {"metric": "play_top3_rows", "value": int(out["is_play_top3"].sum())},
        {"metric": "plays_spent_on_winner_event_rows", "value": int(out["recommended_play_count"].sum())},
        {"metric": "losing_member_plays_on_winner_event_rows", "value": int(out["losing_member_plays_on_winner_events"].sum())},
        {"metric": "top2_rows_where_top1_would_have_won_anyway", "value": int(out["top2_top1_would_have_won"].sum())},
        {"metric": "top2_rows_where_top2_was_actually_needed", "value": int(out["top2_top2_needed"].sum())},
        {"metric": "avg_plays_spent_per_winner_event_row", "value": float(out["recommended_play_count"].mean())},
        {"metric": "avg_losing_member_plays_per_winner_event_row", "value": float(out["losing_member_plays_on_winner_events"].mean())},
    ])

    per_date_oper = (
        out.groupby("transition_date", dropna=False)
        .agg(
            events=("event_id", "count"),
            top1_wins=("bucket_top1_win", "sum"),
            top2_wins=("bucket_top2_win", "sum"),
            top3_wins=("bucket_top3_win", "sum"),
            misses=("bucket_miss", "sum"),
            play_top1_rows=("is_play_top1", "sum"),
            play_top2_rows=("is_play_top2", "sum"),
            play_top3_rows=("is_play_top3", "sum"),
            plays_spent=("recommended_play_count", "sum"),
            losing_member_plays=("losing_member_plays_on_winner_events", "sum"),
            top2_top1_would_have_won=("top2_top1_would_have_won", "sum"),
            top2_top2_needed=("top2_top2_needed", "sum"),
        )
        .reset_index()
        .sort_values("transition_date")
        .reset_index(drop=True)
    )

    per_stream_oper = (
        out.groupby("stream", dropna=False)
        .agg(
            events=("event_id", "count"),
            top1_wins=("bucket_top1_win", "sum"),
            top2_wins=("bucket_top2_win", "sum"),
            top3_wins=("bucket_top3_win", "sum"),
            misses=("bucket_miss", "sum"),
            play_top1_rows=("is_play_top1", "sum"),
            play_top2_rows=("is_play_top2", "sum"),
            play_top3_rows=("is_play_top3", "sum"),
            plays_spent=("recommended_play_count", "sum"),
            losing_member_plays=("losing_member_plays_on_winner_events", "sum"),
            top2_top1_would_have_won=("top2_top1_would_have_won", "sum"),
            top2_top2_needed=("top2_top2_needed", "sum"),
        )
        .reset_index()
        .sort_values(["misses", "plays_spent", "stream"], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    return bucket_summary, per_date_oper, per_stream_oper


# -----------------------------------------------------------------------------
# Basic IO / normalization helpers
# -----------------------------------------------------------------------------

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


def norm_result(r: object) -> Optional[str]:
    d = re.findall(r"\d", str(r))
    return "".join(d[:4]) if len(d) >= 4 else None


def to_member(r4: Optional[str]) -> Optional[str]:
    if r4 is None:
        return None
    s = "".join(sorted(r4))
    return s if s in CORE025 else None


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


def normalize_scalar(x: object) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if float(x).is_integer():
            return str(int(x))
        return str(x).strip()
    return str(x).strip()


# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Data prep
# -----------------------------------------------------------------------------

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
            rows.append(
                {
                    "stream": stream,
                    "seed": seed,
                    "seed_date": g.loc[i - 1, "date"],
                    "transition_date": g.loc[i, "date"],
                    "next_member": next_member,
                    **feat,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["stream", "seed", "seed_date", "transition_date", "next_member"])
    out = pd.DataFrame(rows).sort_values(["transition_date", "stream", "seed"]).reset_index(drop=True)
    out["event_id"] = range(1, len(out) + 1)
    return out


# -----------------------------------------------------------------------------
# Baseline maps and incremental scorer
# -----------------------------------------------------------------------------

def counter_to_probs(c: Counter) -> Dict[str, float]:
    total = sum(c.values())
    if total <= 0:
        return {m: 1 / 3 for m in CORE025}
    return {m: c.get(m, 0) / total for m in CORE025}


@dataclass
class BaselineMaps:
    exact_seed_map: Dict[str, Counter]
    sorted_seed_map: Dict[str, Counter]
    stream_member_map: Dict[str, Counter]
    global_member_map: Counter


def init_baseline_maps() -> BaselineMaps:
    return BaselineMaps(
        exact_seed_map=defaultdict(Counter),
        sorted_seed_map=defaultdict(Counter),
        stream_member_map=defaultdict(Counter),
        global_member_map=Counter(),
    )


def add_transition_to_maps(maps: BaselineMaps, row: pd.Series) -> None:
    member = normalize_member_code(row.get("next_member"))
    if member is None:
        return
    seed = str(row["seed"])
    stream = str(row["stream"])
    sorted_seed = str(row["sorted_seed"])
    maps.exact_seed_map[seed][member] += 1
    maps.sorted_seed_map[sorted_seed][member] += 1
    maps.stream_member_map[stream][member] += 1
    maps.global_member_map[member] += 1


def baseline_scores_from_maps(seed_row: pd.Series, maps: BaselineMaps, min_stream_history: int = 20) -> Dict[str, float]:
    seed = str(seed_row["seed"])
    stream = str(seed_row["stream"])
    score_accum = {m: 0.0 for m in CORE025}

    global_probs = counter_to_probs(maps.global_member_map)
    for m in CORE025:
        score_accum[m] += global_probs[m] * 0.25

    if stream is not None and sum(maps.stream_member_map[stream].values()) >= int(min_stream_history):
        stream_probs = counter_to_probs(maps.stream_member_map[stream])
        for m in CORE025:
            score_accum[m] += stream_probs[m] * 1.20

    if seed in maps.exact_seed_map and sum(maps.exact_seed_map[seed].values()) > 0:
        exact_probs = counter_to_probs(maps.exact_seed_map[seed])
        for m in CORE025:
            score_accum[m] += exact_probs[m] * 1.50

    sorted_key = str(seed_row["sorted_seed"])
    if sorted_key in maps.sorted_seed_map and sum(maps.sorted_seed_map[sorted_key].values()) > 0:
        sorted_probs = counter_to_probs(maps.sorted_seed_map[sorted_key])
        for m in CORE025:
            score_accum[m] += sorted_probs[m] * 1.10

    total = sum(score_accum.values())
    if total <= 0:
        return {m: 1 / 3 for m in CORE025}
    return {m: score_accum[m] / total for m in CORE025}


def build_maps_from_transitions(transitions: pd.DataFrame) -> BaselineMaps:
    maps = init_baseline_maps()
    for _, row in transitions.iterrows():
        add_transition_to_maps(maps, row)
    return maps


# -----------------------------------------------------------------------------
# Separator library
# -----------------------------------------------------------------------------

def parse_trait_stack(stack_text: str) -> List[Tuple[str, str]]:
    parts = [p.strip() for p in str(stack_text).split("&&") if p.strip()]
    out: List[Tuple[str, str]] = []
    for p in parts:
        if "=" not in p:
            continue
        col, val = p.split("=", 1)
        out.append((col.strip(), val.strip()))
    return out


def load_separator_library(df: pd.DataFrame) -> List[Dict[str, object]]:
    req = {"pair", "trait_stack", "winner_member", "winner_rate", "pair_gap", "support", "stack_size"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Separator library missing required columns: {sorted(missing)}")

    rules: List[Dict[str, object]] = []
    for idx, r in df.iterrows():
        stack = parse_trait_stack(str(r["trait_stack"]))
        if not stack:
            continue
        winner_norm = normalize_member_code(r["winner_member"])
        if winner_norm is None:
            continue
        rules.append(
            {
                "rule_id": idx + 1,
                "pair": str(r["pair"]),
                "trait_stack": str(r["trait_stack"]),
                "conditions": stack,
                "winner_member": winner_norm,
                "winner_rate": float(r["winner_rate"]),
                "pair_gap": float(r["pair_gap"]),
                "support": int(r["support"]),
                "stack_size": int(r["stack_size"]),
            }
        )
    return rules


def match_rule(row: pd.Series, rule: Dict[str, object]) -> Tuple[bool, int, int, List[str]]:
    matched = 0
    total = len(rule["conditions"])
    failed_cols: List[str] = []
    for col, val in rule["conditions"]:
        if col not in row.index:
            failed_cols.append(f"{col}:missing")
            continue
        cur = normalize_scalar(row[col])
        if cur == val:
            matched += 1
        else:
            failed_cols.append(f"{col}:{cur}!={val}")
    return matched == total, matched, total, failed_cols


# -----------------------------------------------------------------------------
# Scoring helpers
# -----------------------------------------------------------------------------

def apply_separator_rules(
    row: pd.Series,
    rules: List[Dict[str, object]],
    per_rule_cap: float,
    total_boost_cap: float,
    diminishing_return_factor: float,
    rule_count_norm_factor: float,
    max_rules_per_member: int,
) -> Tuple[Dict[str, float], Dict[str, int], List[str], List[Dict[str, object]], Counter, Dict[str, float]]:
    boosts = {m: 0.0 for m in CORE025}
    fired_counts = {m: 0 for m in CORE025}
    raw_boosts = {m: 0.0 for m in CORE025}
    fired_rules: List[str] = []
    near_misses: List[Dict[str, object]] = []
    fail_counter: Counter = Counter()

    for rule in rules:
        is_full, matched, total, failed_cols = match_rule(row, rule)
        if is_full:
            winner = rule["winner_member"]
            if winner not in boosts:
                fail_counter[f"invalid_winner_{winner}"] += 1
                continue
            if fired_counts[winner] >= int(max_rules_per_member):
                continue

            raw_score = (rule["winner_rate"] * 0.60) + (rule["pair_gap"] * 0.90)
            raw_score += min(rule["support"], 50) / 100.0
            raw_score += 0.03 * max(rule["stack_size"] - 1, 0)
            raw_score = min(raw_score, float(per_rule_cap))

            diminishing_scale = 1.0 / (1.0 + fired_counts[winner] * float(diminishing_return_factor))
            count_norm_scale = 1.0 / (1.0 + fired_counts[winner] * float(rule_count_norm_factor))
            scaled_score = raw_score * diminishing_scale * count_norm_scale

            raw_boosts[winner] += raw_score
            boosts[winner] += scaled_score
            boosts[winner] = min(boosts[winner], float(total_boost_cap))
            fired_counts[winner] += 1

            fired_rules.append(
                f"RID{rule['rule_id']} | {rule['pair']} | {rule['trait_stack']} | winner={winner} | raw={raw_score:.3f} | scaled={scaled_score:.3f} | wr={rule['winner_rate']:.3f} | gap={rule['pair_gap']:.3f} | sup={rule['support']}"
            )
        else:
            for fc in failed_cols:
                fail_counter[fc.split(":")[0]] += 1
            if matched > 0:
                near_misses.append(
                    {
                        "rule_id": rule["rule_id"],
                        "pair": rule["pair"],
                        "trait_stack": rule["trait_stack"],
                        "winner_member": rule["winner_member"],
                        "matched_conditions": matched,
                        "total_conditions": total,
                        "failed_cols": " | ".join(failed_cols[:10]),
                        "winner_rate": rule["winner_rate"],
                        "pair_gap": rule["pair_gap"],
                        "support": rule["support"],
                    }
                )

    near_misses = sorted(
        near_misses,
        key=lambda x: (
            x["matched_conditions"] / x["total_conditions"],
            x["winner_rate"],
            x["pair_gap"],
            x["support"],
        ),
        reverse=True,
    )[:20]

    return boosts, fired_counts, fired_rules, near_misses, fail_counter, raw_boosts


def compress_member_scores(
    base_scores: Dict[str, float],
    boosts: Dict[str, float],
    fired_counts: Dict[str, int],
    compression_alpha: float,
    exclusivity_rule_bonus: float,
    exclusivity_boost_bonus: float,
    exclusivity_cap: float,
    min_compression_factor: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    pre_scores = {m: float(base_scores.get(m, 0.0)) + float(boosts.get(m, 0.0)) for m in CORE025}
    mean_score = sum(pre_scores.values()) / len(CORE025)

    ranked_rule_counts = sorted(fired_counts.values(), reverse=True)
    top_rule_count = ranked_rule_counts[0] if ranked_rule_counts else 0
    second_rule_count = ranked_rule_counts[1] if len(ranked_rule_counts) > 1 else 0
    rule_gap = max(0, top_rule_count - second_rule_count)

    ranked_boosts = sorted(boosts.values(), reverse=True)
    top_boost = ranked_boosts[0] if ranked_boosts else 0.0
    second_boost = ranked_boosts[1] if len(ranked_boosts) > 1 else 0.0
    boost_gap = max(0.0, top_boost - second_boost)

    exclusivity_strength = min(
        float(exclusivity_cap),
        (rule_gap * float(exclusivity_rule_bonus)) + (boost_gap * float(exclusivity_boost_bonus)),
    )

    compression_factor = max(
        float(min_compression_factor),
        min(1.0, float(compression_alpha) + exclusivity_strength),
    )

    compressed = {}
    for m in CORE025:
        delta = pre_scores[m] - mean_score
        compressed[m] = mean_score + (delta * compression_factor)

    diagnostics = {
        "pre_mean_score": mean_score,
        "compression_factor": compression_factor,
        "exclusivity_strength": exclusivity_strength,
        "rule_gap_top12": float(rule_gap),
        "boost_gap_top12": float(boost_gap),
    }
    return compressed, diagnostics


def compute_alignment(top1_rule_count: int, top2_rule_count: int, top1_boost: float, top2_boost: float) -> Dict[str, float]:
    total_rules = max(1, int(top1_rule_count) + int(top2_rule_count))
    total_boost = max(1e-9, float(top1_boost) + float(top2_boost))
    rule_alignment_ratio = float(top1_rule_count) / float(total_rules)
    boost_alignment_ratio = float(top1_boost) / float(total_boost)
    blended_alignment_ratio = (rule_alignment_ratio * 0.50) + (boost_alignment_ratio * 0.50)
    return {
        "rule_alignment_ratio": rule_alignment_ratio,
        "boost_alignment_ratio": boost_alignment_ratio,
        "blended_alignment_ratio": blended_alignment_ratio,
    }


def apply_member_score_calibration(
    scores: Dict[str, float],
    top2_score_proxy: float,
    member_alignment: Dict[str, float],
    member_boost_gap: Dict[str, float],
    m0025_penalty_top2_score_min: float,
    m0025_penalty_alignment_max: float,
    m0025_penalty_multiplier_top2: float,
    m0025_penalty_multiplier_align: float,
    m0225_boost_alignment_min: float,
    m0225_boost_multiplier: float,
    m0255_boost_gap_min: float,
    m0255_boost_alignment_min: float,
    m0255_boost_multiplier_gap: float,
    m0255_boost_multiplier_align: float,
) -> Dict[str, float]:
    adjusted = {m: float(v) for m, v in scores.items()}

    # 0025 suppression in weak-cleanliness pockets
    if adjusted.get("0025", 0.0) > 0:
        if top2_score_proxy > float(m0025_penalty_top2_score_min):
            adjusted["0025"] *= float(m0025_penalty_multiplier_top2)
        if member_alignment.get("0025", 0.0) < float(m0025_penalty_alignment_max):
            adjusted["0025"] *= float(m0025_penalty_multiplier_align)

    # 0225 light assist in clean pockets
    if adjusted.get("0225", 0.0) > 0:
        if member_alignment.get("0225", 0.0) >= float(m0225_boost_alignment_min):
            adjusted["0225"] *= float(m0225_boost_multiplier)

    # 0255 under-promotion rescue
    if adjusted.get("0255", 0.0) > 0:
        if member_boost_gap.get("0255", 0.0) >= float(m0255_boost_gap_min):
            adjusted["0255"] *= float(m0255_boost_multiplier_gap)
        if member_alignment.get("0255", 0.0) >= float(m0255_boost_alignment_min):
            adjusted["0255"] *= float(m0255_boost_multiplier_align)

    return adjusted


def member_specific_top1_gate(
    top1_member: str,
    gap: float,
    ratio: float,
    top2_score: float,
    boost_gap_top12: float,
    blended_alignment_ratio: float,
    m0025_boost_gap_min: float,
    m0025_alignment_min: float,
    m0025_top2_score_max: float,
    m0225_boost_gap_min: float,
    m0225_alignment_min: float,
    m0225_ratio_max: float,
    m0255_boost_gap_min: float,
    m0255_alignment_min: float,
    m0255_gap_min: float,
) -> Tuple[bool, str]:
    if top1_member == "0025":
        if (
            boost_gap_top12 >= float(m0025_boost_gap_min)
            and blended_alignment_ratio >= float(m0025_alignment_min)
            and top2_score <= float(m0025_top2_score_max)
        ):
            return True, "0025-specific Top1 gate passed"
        return False, "0025-specific Top1 gate failed"

    if top1_member == "0225":
        if (
            boost_gap_top12 >= float(m0225_boost_gap_min)
            and blended_alignment_ratio >= float(m0225_alignment_min)
            and ratio <= float(m0225_ratio_max)
        ):
            return True, "0225-specific Top1 gate passed"
        return False, "0225-specific Top1 gate failed"

    if top1_member == "0255":
        if (
            boost_gap_top12 >= float(m0255_boost_gap_min)
            and blended_alignment_ratio >= float(m0255_alignment_min)
            and gap >= float(m0255_gap_min)
        ):
            return True, "0255-specific Top1 gate passed"
        return False, "0255-specific Top1 gate failed"

    return False, "Unknown member-specific Top1 gate"


def classify_dominance_state(
    gap: float,
    ratio: float,
    exclusivity_strength: float,
    rule_gap_top12: float,
    blended_alignment_ratio: float,
    dominant_gap_strict: float,
    dominant_ratio_max_strict: float,
    dominant_exclusivity_min: float,
    dominant_rule_gap_min: float,
    dominant_alignment_min: float,
    contested_gap_max: float,
    contested_ratio_min: float,
) -> str:
    if (
        gap >= float(dominant_gap_strict)
        and ratio <= float(dominant_ratio_max_strict)
        and exclusivity_strength >= float(dominant_exclusivity_min)
        and rule_gap_top12 >= float(dominant_rule_gap_min)
        and blended_alignment_ratio >= float(dominant_alignment_min)
    ):
        return "DOMINANT"

    # V10 soft-dominant expansion: promote strong middle-tier rows out of WEAK
    if gap >= 0.55 and blended_alignment_ratio >= 0.58:
        return "DOMINANT"

    if gap <= float(contested_gap_max) or ratio >= float(contested_ratio_min):
        return "CONTESTED"

    return "WEAK"


def decide_play_mode(
    top1_member: str,
    top1_score: float,
    top2_score: float,
    gap: float,
    ratio: float,
    exclusivity_strength: float,
    boost_gap_top12: float,
    blended_alignment_ratio: float,
    dominance_state: str,
    weak_top1_score_floor: float,
    top2_ratio_trigger: float,
    top2_gap_trigger: float,
    top2_alignment_ceiling: float,
    top2_exclusivity_ceiling: float,
    m0025_boost_gap_min: float,
    m0025_alignment_min: float,
    m0025_top2_score_max: float,
    m0225_boost_gap_min: float,
    m0225_alignment_min: float,
    m0225_ratio_max: float,
    m0255_boost_gap_min: float,
    m0255_alignment_min: float,
    m0255_gap_min: float,
    cnt0: int,
    cnt6: int,
    has8: int,
    structure: str,
) -> Tuple[str, str]:
    if top1_score < float(weak_top1_score_floor):
        return "SKIP", "Top1 score too weak"

    member_gate_passed, member_gate_reason = member_specific_top1_gate(
        top1_member=top1_member,
        gap=gap,
        ratio=ratio,
        top2_score=top2_score,
        boost_gap_top12=boost_gap_top12,
        blended_alignment_ratio=blended_alignment_ratio,
        m0025_boost_gap_min=m0025_boost_gap_min,
        m0025_alignment_min=m0025_alignment_min,
        m0025_top2_score_max=m0025_top2_score_max,
        m0225_boost_gap_min=m0225_boost_gap_min,
        m0225_alignment_min=m0225_alignment_min,
        m0225_ratio_max=m0225_ratio_max,
        m0255_boost_gap_min=m0255_boost_gap_min,
        m0255_alignment_min=m0255_alignment_min,
        m0255_gap_min=m0255_gap_min,
    )

    if dominance_state == "DOMINANT" and member_gate_passed:
        return "PLAY_TOP1", f"Validated dominant Top1 | {member_gate_reason}"

    if member_gate_passed and ratio <= 0.93 and blended_alignment_ratio >= 0.50:
        return "PLAY_TOP1", "Member-specific Top1 promotion"
        return "PLAY_TOP1", f"Member-specific Top1 promotion | {member_gate_reason}"


    if ratio >= float(top2_ratio_trigger) and blended_alignment_ratio < float(top2_alignment_ceiling):
        return "PLAY_TOP2", "Tight ratio with weak alignment widened to Top1+Top2"

    if gap <= float(top2_gap_trigger) and exclusivity_strength <= float(top2_exclusivity_ceiling):
        return "PLAY_TOP2", "Small gap with weak exclusivity widened to Top1+Top2"

    if dominance_state == "CONTESTED":
        return "PLAY_TOP2", "Contested row"

    # v23: safe mined residual pockets only
    # These pockets were mined from the residual PLAY_TOP2 waste pool across v19/v20/v21
    # and selected only where support was clean relative to needed Top2 and misses.
    if (
        top1_member in {"0225", "0255"}
        and (
            (ratio <= 0.75 and exclusivity_strength >= 0.14)
            or (cnt6 >= 2 and ratio <= 0.75)
            or (cnt6 >= 2 and gap >= 0.40)
            or (cnt0 >= 2 and has8 == 1)
            or (cnt0 >= 2 and blended_alignment_ratio <= 0.50)
            or (ratio <= 0.75 and structure == "AABB")
        )
    ):
        return "PLAY_TOP1", "Safe mined residual Top1 conversion"

    if (
        not member_gate_passed
        and top1_member in {"0225", "0255"}
        and blended_alignment_ratio >= 0.54
        and ratio <= 0.93
        and exclusivity_strength >= 0.16
        and boost_gap_top12 >= 0.22
        and gap >= 0.28
    ):
        return "PLAY_TOP1", f"Trait-based Top1 conversion | {member_gate_reason}"

    if not member_gate_passed:
        return "PLAY_TOP2", f"Top1 blocked by member-specific gate | {member_gate_reason}"

    return "PLAY_TOP1", f"Top1-first default | {member_gate_reason}"


def rank_members_from_maps(
    row: pd.Series,
    maps: BaselineMaps,
    separator_rules: List[Dict[str, object]],
    min_stream_history: int,
    per_rule_cap: float,
    total_boost_cap: float,
    diminishing_return_factor: float,
    rule_count_norm_factor: float,
    max_rules_per_member: int,
    compression_alpha: float,
    exclusivity_rule_bonus: float,
    exclusivity_boost_bonus: float,
    exclusivity_cap: float,
    min_compression_factor: float,
    weak_top1_score_floor: float,
    dominant_gap_strict: float,
    dominant_ratio_max_strict: float,
    dominant_exclusivity_min: float,
    dominant_rule_gap_min: float,
    dominant_alignment_min: float,
    contested_gap_max: float,
    contested_ratio_min: float,
    top2_ratio_trigger: float,
    top2_gap_trigger: float,
    top2_alignment_ceiling: float,
    top2_exclusivity_ceiling: float,
    m0025_boost_gap_min: float,
    m0025_alignment_min: float,
    m0025_top2_score_max: float,
    m0225_boost_gap_min: float,
    m0225_alignment_min: float,
    m0225_ratio_max: float,
    m0255_boost_gap_min: float,
    m0255_alignment_min: float,
    m0255_gap_min: float,
    m0025_penalty_top2_score_min: float,
    m0025_penalty_alignment_max: float,
    m0025_penalty_multiplier_top2: float,
    m0025_penalty_multiplier_align: float,
    m0225_boost_alignment_min: float,
    m0225_boost_multiplier: float,
    m0255_boost_multiplier_gap: float,
    m0255_boost_multiplier_align: float,
) -> Dict[str, object]:
    base = baseline_scores_from_maps(row, maps, min_stream_history=int(min_stream_history))
    boosts, fired_counts, fired_rules, near_misses, fail_counter, raw_boosts = apply_separator_rules(
        row=row,
        rules=separator_rules,
        per_rule_cap=float(per_rule_cap),
        total_boost_cap=float(total_boost_cap),
        diminishing_return_factor=float(diminishing_return_factor),
        rule_count_norm_factor=float(rule_count_norm_factor),
        max_rules_per_member=int(max_rules_per_member),
    )

    compressed_scores, compression_diag = compress_member_scores(
        base_scores=base,
        boosts=boosts,
        fired_counts=fired_counts,
        compression_alpha=float(compression_alpha),
        exclusivity_rule_bonus=float(exclusivity_rule_bonus),
        exclusivity_boost_bonus=float(exclusivity_boost_bonus),
        exclusivity_cap=float(exclusivity_cap),
        min_compression_factor=float(min_compression_factor),
    )

    # Pre-calibration diagnostics by member
    member_alignment = {}
    member_boost_gap = {}
    for m in CORE025:
        other_boosts = sorted([boosts[o] for o in CORE025 if o != m], reverse=True)
        next_boost = other_boosts[0] if other_boosts else 0.0
        member_boost_gap[m] = float(boosts[m] - next_boost)
        member_alignment[m] = float(
            compute_alignment(
                top1_rule_count=fired_counts[m],
                top2_rule_count=max([fired_counts[o] for o in CORE025 if o != m], default=0),
                top1_boost=boosts[m],
                top2_boost=max([boosts[o] for o in CORE025 if o != m], default=0.0),
            )["blended_alignment_ratio"]
        )

    scores_after_calibration = apply_member_score_calibration(
        scores=compressed_scores,
        top2_score_proxy=max(compressed_scores.values()),
        member_alignment=member_alignment,
        member_boost_gap=member_boost_gap,
        m0025_penalty_top2_score_min=float(m0025_penalty_top2_score_min),
        m0025_penalty_alignment_max=float(m0025_penalty_alignment_max),
        m0025_penalty_multiplier_top2=float(m0025_penalty_multiplier_top2),
        m0025_penalty_multiplier_align=float(m0025_penalty_multiplier_align),
        m0225_boost_alignment_min=float(m0225_boost_alignment_min),
        m0225_boost_multiplier=float(m0225_boost_multiplier),
        m0255_boost_gap_min=float(m0255_boost_gap_min),
        m0255_boost_alignment_min=float(m0255_alignment_min),
        m0255_boost_multiplier_gap=float(m0255_boost_multiplier_gap),
        m0255_boost_multiplier_align=float(m0255_boost_multiplier_align),
    )

    # V14 production-balanced nudge: do not force Top1, lightly bias score instead
    if (
        max(scores_after_calibration.values()) > 0
        and compression_diag["boost_gap_top12"] >= 0.60
    ):
        # determine current leader after calibration and apply only a light nudge
        current_leader = max(scores_after_calibration.items(), key=lambda kv: kv[1])[0]
        current_alignment = member_alignment.get(current_leader, 0.0)
        if current_alignment >= 0.60:
            scores_after_calibration[current_leader] *= 1.08

    normalized_scores: Dict[str, float] = {}
    for k, v in scores_after_calibration.items():
        k_norm = normalize_member_code(k)
        if k_norm is None:
            continue
        normalized_scores[k_norm] = normalized_scores.get(k_norm, 0.0) + float(v)
    for m in CORE025:
        normalized_scores.setdefault(m, 0.0)

    ranked = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
    top1, top1_score = ranked[0]
    top2, top2_score = ranked[1]
    top3, top3_score = ranked[2]
    top1 = normalize_member_code(top1) or "0025"
    top2 = normalize_member_code(top2) or "0225"
    top3 = normalize_member_code(top3) or "0255"

    gap = top1_score - top2_score
    ratio = (top2_score / top1_score) if top1_score > 0 else 1.0
    top1_rule_count = fired_counts.get(top1, 0)
    top2_rule_count = fired_counts.get(top2, 0)
    top1_boost = boosts.get(top1, 0.0)
    top2_boost = boosts.get(top2, 0.0)
    rule_margin = top1_rule_count - top2_rule_count
    boost_margin = top1_boost - top2_boost

    alignment_diag = compute_alignment(
        top1_rule_count=top1_rule_count,
        top2_rule_count=top2_rule_count,
        top1_boost=top1_boost,
        top2_boost=top2_boost,
    )

    dominance_state = classify_dominance_state(
        gap=gap,
        ratio=ratio,
        exclusivity_strength=float(compression_diag["exclusivity_strength"]),
        rule_gap_top12=float(compression_diag["rule_gap_top12"]),
        blended_alignment_ratio=float(alignment_diag["blended_alignment_ratio"]),
        dominant_gap_strict=float(dominant_gap_strict),
        dominant_ratio_max_strict=float(dominant_ratio_max_strict),
        dominant_exclusivity_min=float(dominant_exclusivity_min),
        dominant_rule_gap_min=float(dominant_rule_gap_min),
        dominant_alignment_min=float(dominant_alignment_min),
        contested_gap_max=float(contested_gap_max),
        contested_ratio_min=float(contested_ratio_min),
    )

    play_mode, play_reason = decide_play_mode(
        top1_member=top1,
        top1_score=top1_score,
        top2_score=top2_score,
        gap=gap,
        ratio=ratio,
        exclusivity_strength=float(compression_diag["exclusivity_strength"]),
        boost_gap_top12=float(compression_diag["boost_gap_top12"]),
        blended_alignment_ratio=float(alignment_diag["blended_alignment_ratio"]),
        dominance_state=dominance_state,
        weak_top1_score_floor=float(weak_top1_score_floor),
        top2_ratio_trigger=float(top2_ratio_trigger),
        top2_gap_trigger=float(top2_gap_trigger),
        top2_alignment_ceiling=float(top2_alignment_ceiling),
        top2_exclusivity_ceiling=float(top2_exclusivity_ceiling),
        m0025_boost_gap_min=float(m0025_boost_gap_min),
        m0025_alignment_min=float(m0025_alignment_min),
        m0025_top2_score_max=float(m0025_top2_score_max),
        m0225_boost_gap_min=float(m0225_boost_gap_min),
        m0225_alignment_min=float(m0225_alignment_min),
        m0225_ratio_max=float(m0225_ratio_max),
        m0255_boost_gap_min=float(m0255_boost_gap_min),
        m0255_alignment_min=float(m0255_alignment_min),
        m0255_gap_min=float(m0255_gap_min),
        cnt0=int(row.get("cnt0", 0)),
        cnt6=int(row.get("cnt6", 0)),
        has8=int(row.get("has8", 0)),
        structure=str(row.get("structure", "")),
    )

    near_text = " || ".join(
        [
            f"RID{x['rule_id']} {x['pair']} {x['matched_conditions']}/{x['total_conditions']} winner={x['winner_member']} failed={x['failed_cols']}"
            for x in near_misses[:10]
        ]
    )
    fail_top = " || ".join([f"{k}:{v}" for k, v in fail_counter.most_common(10)])

    return {
        "base_0025": base["0025"],
        "base_0225": base["0225"],
        "base_0255": base["0255"],
        "raw_boost_0025": raw_boosts["0025"],
        "raw_boost_0225": raw_boosts["0225"],
        "raw_boost_0255": raw_boosts["0255"],
        "boost_0025": boosts["0025"],
        "boost_0225": boosts["0225"],
        "boost_0255": boosts["0255"],
        "rules_0025": fired_counts["0025"],
        "rules_0225": fired_counts["0225"],
        "rules_0255": fired_counts["0255"],
        "precompress_0025": base["0025"] + boosts["0025"],
        "precompress_0225": base["0225"] + boosts["0225"],
        "precompress_0255": base["0255"] + boosts["0255"],
        "final_0025": normalized_scores.get("0025", 0.0),
        "final_0225": normalized_scores.get("0225", 0.0),
        "final_0255": normalized_scores.get("0255", 0.0),
        "Top1": top1,
        "Top1_score": top1_score,
        "Top2": top2,
        "Top2_score": top2_score,
        "Top3": top3,
        "Top3_score": top3_score,
        "gap": gap,
        "ratio": ratio,
        "dominance_state": dominance_state,
        "play_mode": play_mode,
        "play_reason": play_reason,
        "compression_factor": compression_diag["compression_factor"],
        "exclusivity_strength": compression_diag["exclusivity_strength"],
        "rule_gap_top12": compression_diag["rule_gap_top12"],
        "boost_gap_top12": compression_diag["boost_gap_top12"],
        "rule_margin_top1_top2": rule_margin,
        "boost_margin_top1_top2": boost_margin,
        "rule_alignment_ratio": alignment_diag["rule_alignment_ratio"],
        "boost_alignment_ratio": alignment_diag["boost_alignment_ratio"],
        "blended_alignment_ratio": alignment_diag["blended_alignment_ratio"],
        "member_alignment_0025": member_alignment.get("0025", 0.0),
        "member_alignment_0225": member_alignment.get("0225", 0.0),
        "member_alignment_0255": member_alignment.get("0255", 0.0),
        "member_boost_gap_0025": member_boost_gap.get("0025", 0.0),
        "member_boost_gap_0225": member_boost_gap.get("0225", 0.0),
        "member_boost_gap_0255": member_boost_gap.get("0255", 0.0),
        "fired_rule_count": len(fired_rules),
        "fired_rules": " || ".join(fired_rules[:25]),
        "near_miss_rule_count": len(near_misses),
        "near_miss_rules": near_text,
        "top_failed_columns": fail_top,
    }


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------

def summarize_playlist(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if len(df) == 0:
        return pd.DataFrame(columns=["metric", "value"])
    rows.append({"metric": "playlist_rows", "value": len(df)})
    rows.append({"metric": "rows_play_top1", "value": int((df["play_mode"] == "PLAY_TOP1").sum())})
    rows.append({"metric": "rows_play_top2", "value": int((df["play_mode"] == "PLAY_TOP2").sum())})
    rows.append({"metric": "rows_skip", "value": int((df["play_mode"] == "SKIP").sum())})
    rows.append({"metric": "rows_dominant", "value": int((df["dominance_state"] == "DOMINANT").sum())})
    rows.append({"metric": "rows_contested", "value": int((df["dominance_state"] == "CONTESTED").sum())})
    rows.append({"metric": "rows_weak", "value": int((df["dominance_state"] == "WEAK").sum())})
    rows.append({"metric": "avg_gap", "value": float(df["gap"].mean())})
    rows.append({"metric": "avg_ratio", "value": float(df["ratio"].mean())})
    rows.append({"metric": "avg_fired_rule_count", "value": float(df["fired_rule_count"].mean())})
    rows.append({"metric": "avg_compression_factor", "value": float(df["compression_factor"].mean())})
    rows.append({"metric": "avg_exclusivity_strength", "value": float(df["exclusivity_strength"].mean())})
    rows.append({"metric": "avg_rule_gap_top12", "value": float(df["rule_gap_top12"].mean())})
    rows.append({"metric": "avg_boost_gap_top12", "value": float(df["boost_gap_top12"].mean())})
    rows.append({"metric": "avg_rule_alignment_ratio", "value": float(df["rule_alignment_ratio"].mean())})
    rows.append({"metric": "avg_boost_alignment_ratio", "value": float(df["boost_alignment_ratio"].mean())})
    rows.append({"metric": "avg_blended_alignment_ratio", "value": float(df["blended_alignment_ratio"].mean())})
    return pd.DataFrame(rows)


def summarize_lab(per_event: pd.DataFrame, total_transitions_seen: int, non_core025_transitions_skipped: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(per_event) == 0:
        summary = pd.DataFrame(
            [
                {"metric": "events", "value": 0},
                {"metric": "total_transitions_seen", "value": int(total_transitions_seen)},
                {"metric": "non_core025_transitions_skipped", "value": int(non_core025_transitions_skipped)},
                {"metric": "skip_score_cutoff_reference", "value": float(DEFAULT_SKIP_SCORE_CUTOFF)},
            ]
        )
        empty = pd.DataFrame()
        return empty, empty, empty, summary

    per_date = (
        per_event.groupby("transition_date", dropna=False)
        .agg(
            events=("event_id", "count"),
            top1_hits=("top1_hit", "sum"),
            top2_hits=("top2_hit", "sum"),
            top3_hits=("top3_hit", "sum"),
            play_rule_hits=("play_rule_hit", "sum"),
            play_top1_rows=("is_play_top1", "sum"),
            play_top2_rows=("is_play_top2", "sum"),
            skip_rows=("is_skip", "sum"),
        )
        .reset_index()
        .sort_values("transition_date")
        .reset_index(drop=True)
    )
    per_date["top1_capture_pct"] = per_date["top1_hits"] / per_date["events"]
    per_date["top2_capture_pct"] = per_date["top2_hits"] / per_date["events"]
    per_date["top3_capture_pct"] = per_date["top3_hits"] / per_date["events"]
    per_date["play_rule_capture_pct"] = per_date["play_rule_hits"] / per_date["events"]

    per_stream = (
        per_event.groupby("stream", dropna=False)
        .agg(
            events=("event_id", "count"),
            top1_hits=("top1_hit", "sum"),
            top2_hits=("top2_hit", "sum"),
            top3_hits=("top3_hit", "sum"),
            play_rule_hits=("play_rule_hit", "sum"),
            play_top1_rows=("is_play_top1", "sum"),
            play_top2_rows=("is_play_top2", "sum"),
            skip_rows=("is_skip", "sum"),
        )
        .reset_index()
        .sort_values(["play_rule_hits", "events", "stream"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    per_stream["top1_capture_pct"] = per_stream["top1_hits"] / per_stream["events"]
    per_stream["top2_capture_pct"] = per_stream["top2_hits"] / per_stream["events"]
    per_stream["top3_capture_pct"] = per_stream["top3_hits"] / per_stream["events"]
    per_stream["play_rule_capture_pct"] = per_stream["play_rule_hits"] / per_stream["events"]

    by_mode = (
        per_event.groupby("play_mode", dropna=False)
        .agg(
            events=("event_id", "count"),
            avg_top1_score=("Top1_score", "mean"),
            avg_gap=("gap", "mean"),
            avg_ratio=("ratio", "mean"),
            avg_alignment=("blended_alignment_ratio", "mean"),
            avg_boost_gap=("boost_gap_top12", "mean"),
            top1_capture=("top1_hit", "mean"),
            top2_capture=("top2_hit", "mean"),
            top3_capture=("top3_hit", "mean"),
            play_rule_capture=("play_rule_hit", "mean"),
        )
        .reset_index()
        .rename(columns={"play_mode": "bucket"})
        .sort_values("bucket")
        .reset_index(drop=True)
    )

    summary_rows = []
    summary_rows.append({"metric": "events", "value": int(len(per_event))})
    summary_rows.append({"metric": "total_transitions_seen", "value": int(total_transitions_seen)})
    summary_rows.append({"metric": "non_core025_transitions_skipped", "value": int(non_core025_transitions_skipped)})
    summary_rows.append({"metric": "top1_capture", "value": int(per_event["top1_hit"].sum())})
    summary_rows.append({"metric": "top2_capture", "value": int(per_event["top2_hit"].sum())})
    summary_rows.append({"metric": "top3_capture", "value": int(per_event["top3_hit"].sum())})
    summary_rows.append({"metric": "play_rule_capture", "value": int(per_event["play_rule_hit"].sum())})
    summary_rows.append({"metric": "top1_capture_pct", "value": float(per_event["top1_hit"].mean())})
    summary_rows.append({"metric": "top2_capture_pct", "value": float(per_event["top2_hit"].mean())})
    summary_rows.append({"metric": "top3_capture_pct", "value": float(per_event["top3_hit"].mean())})
    summary_rows.append({"metric": "play_rule_capture_pct", "value": float(per_event["play_rule_hit"].mean())})
    summary_rows.append({"metric": "play_top1_rows", "value": int(per_event["is_play_top1"].sum())})
    summary_rows.append({"metric": "play_top2_rows", "value": int(per_event["is_play_top2"].sum())})
    summary_rows.append({"metric": "skip_rows", "value": int(per_event["is_skip"].sum())})
    summary_rows.append({"metric": "avg_gap", "value": float(per_event["gap"].mean())})
    summary_rows.append({"metric": "avg_ratio", "value": float(per_event["ratio"].mean())})
    summary_rows.append({"metric": "avg_compression_factor", "value": float(per_event["compression_factor"].mean())})
    summary_rows.append({"metric": "avg_exclusivity_strength", "value": float(per_event["exclusivity_strength"].mean())})
    summary_rows.append({"metric": "avg_rule_alignment_ratio", "value": float(per_event["rule_alignment_ratio"].mean())})
    summary_rows.append({"metric": "avg_boost_alignment_ratio", "value": float(per_event["boost_alignment_ratio"].mean())})
    summary_rows.append({"metric": "avg_blended_alignment_ratio", "value": float(per_event["blended_alignment_ratio"].mean())})
    summary_rows.append({"metric": "avg_boost_gap_top12", "value": float(per_event["boost_gap_top12"].mean())})
    summary_rows.append({"metric": "rows_dominant", "value": int((per_event["dominance_state"] == "DOMINANT").sum())})
    summary_rows.append({"metric": "rows_contested", "value": int((per_event["dominance_state"] == "CONTESTED").sum())})
    summary_rows.append({"metric": "rows_weak", "value": int((per_event["dominance_state"] == "WEAK").sum())})
    summary_rows.append({"metric": "skip_score_cutoff_reference", "value": float(DEFAULT_SKIP_SCORE_CUTOFF)})
    summary = pd.DataFrame(summary_rows)
    return per_date, per_stream, by_mode, summary


# -----------------------------------------------------------------------------
# Regular run and LAB walk-forward runners
# -----------------------------------------------------------------------------

def run_regular_playlist(
    hist: pd.DataFrame,
    surv: pd.DataFrame,
    separator_rules: List[Dict[str, object]],
    params: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    transitions = build_transitions(hist)
    maps = build_maps_from_transitions(transitions)

    rows = []
    for _, row in surv.iterrows():
        ranked = rank_members_from_maps(
            row=row,
            maps=maps,
            separator_rules=separator_rules,
            min_stream_history=int(params["min_stream_history"]),
            per_rule_cap=float(params["per_rule_cap"]),
            total_boost_cap=float(params["total_boost_cap"]),
            diminishing_return_factor=float(params["diminishing_return_factor"]),
            rule_count_norm_factor=float(params["rule_count_norm_factor"]),
            max_rules_per_member=int(params["max_rules_per_member"]),
            compression_alpha=float(params["compression_alpha"]),
            exclusivity_rule_bonus=float(params["exclusivity_rule_bonus"]),
            exclusivity_boost_bonus=float(params["exclusivity_boost_bonus"]),
            exclusivity_cap=float(params["exclusivity_cap"]),
            min_compression_factor=float(params["min_compression_factor"]),
            weak_top1_score_floor=float(params["weak_top1_score_floor"]),
            dominant_gap_strict=float(params["dominant_gap_strict"]),
            dominant_ratio_max_strict=float(params["dominant_ratio_max_strict"]),
            dominant_exclusivity_min=float(params["dominant_exclusivity_min"]),
            dominant_rule_gap_min=float(params["dominant_rule_gap_min"]),
            dominant_alignment_min=float(params["dominant_alignment_min"]),
            contested_gap_max=float(params["contested_gap_max"]),
            contested_ratio_min=float(params["contested_ratio_min"]),
            top2_ratio_trigger=float(params["top2_ratio_trigger"]),
            top2_gap_trigger=float(params["top2_gap_trigger"]),
            top2_alignment_ceiling=float(params["top2_alignment_ceiling"]),
            top2_exclusivity_ceiling=float(params["top2_exclusivity_ceiling"]),
            m0025_boost_gap_min=float(params["m0025_boost_gap_min"]),
            m0025_alignment_min=float(params["m0025_alignment_min"]),
            m0025_top2_score_max=float(params["m0025_top2_score_max"]),
            m0225_boost_gap_min=float(params["m0225_boost_gap_min"]),
            m0225_alignment_min=float(params["m0225_alignment_min"]),
            m0225_ratio_max=float(params["m0225_ratio_max"]),
            m0255_boost_gap_min=float(params["m0255_boost_gap_min"]),
            m0255_alignment_min=float(params["m0255_alignment_min"]),
            m0255_gap_min=float(params["m0255_gap_min"]),
            m0025_penalty_top2_score_min=float(params["m0025_penalty_top2_score_min"]),
            m0025_penalty_alignment_max=float(params["m0025_penalty_alignment_max"]),
            m0025_penalty_multiplier_top2=float(params["m0025_penalty_multiplier_top2"]),
            m0025_penalty_multiplier_align=float(params["m0025_penalty_multiplier_align"]),
            m0225_boost_alignment_min=float(params["m0225_boost_alignment_min"]),
            m0225_boost_multiplier=float(params["m0225_boost_multiplier"]),
            m0255_boost_multiplier_gap=float(params["m0255_boost_multiplier_gap"]),
            m0255_boost_multiplier_align=float(params["m0255_boost_multiplier_align"]),
        )
        rows.append({"stream": row["stream"], "seed": row["seed"], **ranked})

    out = pd.DataFrame(rows).sort_values(
        ["Top1_score", "gap", "fired_rule_count", "ratio"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    summary = summarize_playlist(out)
    return out, summary


def run_lab_walkforward(
    hist: pd.DataFrame,
    separator_rules: List[Dict[str, object]],
    params: Dict[str, float],
    progress_bar=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transitions = build_transitions(hist)
    if len(transitions) == 0:
        empty = pd.DataFrame()
        summary = pd.DataFrame(
            [
                {"metric": "events", "value": 0},
                {"metric": "total_transitions_seen", "value": 0},
                {"metric": "non_core025_transitions_skipped", "value": 0},
                {"metric": "skip_score_cutoff_reference", "value": float(DEFAULT_SKIP_SCORE_CUTOFF)},
            ]
        )
        return empty, empty, empty, empty, summary

    maps = init_baseline_maps()
    rows = []
    total = len(transitions)
    total_transitions_seen = 0
    non_core025_transitions_skipped = 0

    for idx, (_, row) in enumerate(transitions.iterrows(), start=1):
        total_transitions_seen += 1
        winner_member = normalize_member_code(row["next_member"])

        ranked = rank_members_from_maps(
            row=row,
            maps=maps,
            separator_rules=separator_rules,
            min_stream_history=int(params["min_stream_history"]),
            per_rule_cap=float(params["per_rule_cap"]),
            total_boost_cap=float(params["total_boost_cap"]),
            diminishing_return_factor=float(params["diminishing_return_factor"]),
            rule_count_norm_factor=float(params["rule_count_norm_factor"]),
            max_rules_per_member=int(params["max_rules_per_member"]),
            compression_alpha=float(params["compression_alpha"]),
            exclusivity_rule_bonus=float(params["exclusivity_rule_bonus"]),
            exclusivity_boost_bonus=float(params["exclusivity_boost_bonus"]),
            exclusivity_cap=float(params["exclusivity_cap"]),
            min_compression_factor=float(params["min_compression_factor"]),
            weak_top1_score_floor=float(params["weak_top1_score_floor"]),
            dominant_gap_strict=float(params["dominant_gap_strict"]),
            dominant_ratio_max_strict=float(params["dominant_ratio_max_strict"]),
            dominant_exclusivity_min=float(params["dominant_exclusivity_min"]),
            dominant_rule_gap_min=float(params["dominant_rule_gap_min"]),
            dominant_alignment_min=float(params["dominant_alignment_min"]),
            contested_gap_max=float(params["contested_gap_max"]),
            contested_ratio_min=float(params["contested_ratio_min"]),
            top2_ratio_trigger=float(params["top2_ratio_trigger"]),
            top2_gap_trigger=float(params["top2_gap_trigger"]),
            top2_alignment_ceiling=float(params["top2_alignment_ceiling"]),
            top2_exclusivity_ceiling=float(params["top2_exclusivity_ceiling"]),
            m0025_boost_gap_min=float(params["m0025_boost_gap_min"]),
            m0025_alignment_min=float(params["m0025_alignment_min"]),
            m0025_top2_score_max=float(params["m0025_top2_score_max"]),
            m0225_boost_gap_min=float(params["m0225_boost_gap_min"]),
            m0225_alignment_min=float(params["m0225_alignment_min"]),
            m0225_ratio_max=float(params["m0225_ratio_max"]),
            m0255_boost_gap_min=float(params["m0255_boost_gap_min"]),
            m0255_alignment_min=float(params["m0255_alignment_min"]),
            m0255_gap_min=float(params["m0255_gap_min"]),
            m0025_penalty_top2_score_min=float(params["m0025_penalty_top2_score_min"]),
            m0025_penalty_alignment_max=float(params["m0025_penalty_alignment_max"]),
            m0025_penalty_multiplier_top2=float(params["m0025_penalty_multiplier_top2"]),
            m0025_penalty_multiplier_align=float(params["m0025_penalty_multiplier_align"]),
            m0225_boost_alignment_min=float(params["m0225_boost_alignment_min"]),
            m0225_boost_multiplier=float(params["m0225_boost_multiplier"]),
            m0255_boost_multiplier_gap=float(params["m0255_boost_multiplier_gap"]),
            m0255_boost_multiplier_align=float(params["m0255_boost_multiplier_align"]),
        )

        if winner_member is None:
            non_core025_transitions_skipped += 1
            add_transition_to_maps(maps, row)
            if progress_bar is not None:
                progress_bar.progress(idx / total)
            continue

        top1_hit = int(ranked["Top1"] == winner_member)
        top2_hit = int((ranked["Top1"] == winner_member) or (ranked["Top2"] == winner_member))
        top3_hit = int((ranked["Top1"] == winner_member) or (ranked["Top2"] == winner_member) or (ranked["Top3"] == winner_member))
        if ranked["play_mode"] == "PLAY_TOP1":
            play_rule_hit = int(ranked["Top1"] == winner_member)
        elif ranked["play_mode"] == "PLAY_TOP2":
            play_rule_hit = int((ranked["Top1"] == winner_member) or (ranked["Top2"] == winner_member))
        else:
            play_rule_hit = 0

        rows.append(
            {
                "event_id": int(row["event_id"]),
                "transition_date": row["transition_date"],
                "seed_date": row["seed_date"],
                "stream": row["stream"],
                "seed": row["seed"],
                "winning_member": winner_member,
                "top1_hit": top1_hit,
                "top2_hit": top2_hit,
                "top3_hit": top3_hit,
                "play_rule_hit": play_rule_hit,
                "is_play_top1": int(ranked["play_mode"] == "PLAY_TOP1"),
                "is_play_top2": int(ranked["play_mode"] == "PLAY_TOP2"),
                "is_skip": int(ranked["play_mode"] == "SKIP"),
                **ranked,
            }
        )

        add_transition_to_maps(maps, row)
        if progress_bar is not None:
            progress_bar.progress(idx / total)

    per_event = pd.DataFrame(rows).sort_values(["event_id"]).reset_index(drop=True)
    per_date, per_stream, by_mode, summary = summarize_lab(
        per_event=per_event,
        total_transitions_seen=total_transitions_seen,
        non_core025_transitions_skipped=non_core025_transitions_skipped,
    )
    return per_event, per_date, per_stream, by_mode, summary


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------

def collect_params_from_sidebar() -> Dict[str, float]:
    with st.sidebar:
        st.markdown(f"**{BUILD_MARKER}**")
        run_mode = st.radio("Mode", ["Regular Run", "LAB Walk-Forward"], index=0)
        min_stream_history = st.number_input("Minimum stream history for baseline fallback", min_value=0, value=20, step=5)

        st.header("Scaling controls")
        per_rule_cap = st.slider("Per-rule cap", min_value=0.10, max_value=5.00, value=2.50, step=0.05)
        total_boost_cap = st.slider("Total boost cap per member", min_value=0.50, max_value=20.00, value=10.00, step=0.10)
        diminishing_return_factor = st.slider("Diminishing return factor", min_value=0.00, max_value=3.00, value=0.35, step=0.01)
        rule_count_norm_factor = st.slider("Rule-count normalization factor", min_value=0.00, max_value=3.00, value=1.50, step=0.01)
        max_rules_per_member = st.number_input("Max fired rules per member", min_value=1, max_value=100, value=5, step=1)

        st.header("Cross-member compression")
        compression_alpha = st.slider("Base compression alpha", min_value=0.05, max_value=1.00, value=0.45, step=0.01)
        exclusivity_rule_bonus = st.slider("Exclusivity bonus per rule-gap", min_value=0.00, max_value=0.50, value=0.08, step=0.01)
        exclusivity_boost_bonus = st.slider("Exclusivity bonus per boost-gap", min_value=0.00, max_value=1.00, value=0.20, step=0.01)
        exclusivity_cap = st.slider("Exclusivity cap", min_value=0.00, max_value=1.00, value=0.35, step=0.01)
        min_compression_factor = st.slider("Minimum compression factor", min_value=0.05, max_value=1.00, value=0.30, step=0.01)

        st.header("Dominance thresholds")
        dominant_gap_strict = st.slider("Strict dominant gap threshold", min_value=0.00, max_value=2.00, value=0.65, step=0.01)
        dominant_ratio_max_strict = st.slider("Strict dominant max ratio", min_value=0.50, max_value=1.00, value=0.65, step=0.01)
        dominant_exclusivity_min = st.slider("Strict dominant min exclusivity", min_value=0.00, max_value=1.00, value=0.24, step=0.01)
        dominant_rule_gap_min = st.slider("Strict dominant min rule-gap", min_value=0.00, max_value=10.00, value=3.00, step=0.10)
        dominant_alignment_min = st.slider("Strict dominant min alignment", min_value=0.00, max_value=1.00, value=0.58, step=0.01)

        st.header("Top2 widening controls")
        contested_gap_max = st.slider("Contested gap max", min_value=0.00, max_value=2.00, value=0.12, step=0.01)
        contested_ratio_min = st.slider("Contested ratio min", min_value=0.50, max_value=1.00, value=0.97, step=0.01)
        top2_ratio_trigger = st.slider("Top2 widen ratio trigger", min_value=0.50, max_value=1.00, value=0.97, step=0.01)
        top2_gap_trigger = st.slider("Top2 widen gap trigger", min_value=0.00, max_value=2.00, value=0.08, step=0.01)
        top2_alignment_ceiling = st.slider("Top2 widen max alignment", min_value=0.00, max_value=1.00, value=0.62, step=0.01)
        top2_exclusivity_ceiling = st.slider("Top2 widen max exclusivity", min_value=0.00, max_value=1.00, value=0.22, step=0.01)

        st.header("Member-specific Top1 gates")
        m0025_boost_gap_min = st.slider("0025 min boost gap", min_value=0.00, max_value=2.00, value=0.60, step=0.01)
        m0025_alignment_min = st.slider("0025 min alignment", min_value=0.00, max_value=1.00, value=0.60, step=0.01)
        m0025_top2_score_max = st.slider("0025 max Top2 score", min_value=0.00, max_value=5.00, value=1.75, step=0.01)

        m0225_boost_gap_min = st.slider("0225 min boost gap", min_value=0.00, max_value=2.00, value=0.45, step=0.01)
        m0225_alignment_min = st.slider("0225 min alignment", min_value=0.00, max_value=1.00, value=0.58, step=0.01)
        m0225_ratio_max = st.slider("0225 max ratio", min_value=0.50, max_value=1.00, value=0.88, step=0.01)

        m0255_boost_gap_min = st.slider("0255 min boost gap", min_value=0.00, max_value=2.00, value=0.40, step=0.01)
        m0255_alignment_min = st.slider("0255 min alignment", min_value=0.00, max_value=1.00, value=0.55, step=0.01)
        m0255_gap_min = st.slider("0255 min gap", min_value=0.00, max_value=2.00, value=0.18, step=0.01)

        st.header("Score-level member calibration")
        m0025_penalty_top2_score_min = st.slider("0025 penalty if score above", min_value=0.00, max_value=5.00, value=1.70, step=0.01)
        m0025_penalty_alignment_max = st.slider("0025 penalty if alignment below", min_value=0.00, max_value=1.00, value=0.58, step=0.01)
        m0025_penalty_multiplier_top2 = st.slider("0025 score multiplier on high competition", min_value=0.50, max_value=1.20, value=0.88, step=0.01)
        m0025_penalty_multiplier_align = st.slider("0025 score multiplier on weak alignment", min_value=0.50, max_value=1.20, value=0.90, step=0.01)

        m0225_boost_alignment_min = st.slider("0225 boost if alignment at least", min_value=0.00, max_value=1.00, value=0.60, step=0.01)
        m0225_boost_multiplier = st.slider("0225 score multiplier on clean pocket", min_value=0.80, max_value=1.30, value=1.05, step=0.01)

        m0255_boost_multiplier_gap = st.slider("0255 score multiplier on boost-gap signal", min_value=0.80, max_value=1.50, value=1.30, step=0.01)
        m0255_boost_multiplier_align = st.slider("0255 score multiplier on alignment signal", min_value=0.80, max_value=1.50, value=1.24, step=0.01)

        st.header("Weak-row control")
        weak_top1_score_floor = st.slider("Weak Top1 score floor", min_value=0.00, max_value=5.00, value=0.20, step=0.01)

        rows_to_show = st.number_input("Rows to display", min_value=5, value=50, step=5)
        lab_max_events = st.number_input("LAB max events (0 = all)", min_value=0, value=0, step=50)

    return {
        "run_mode": run_mode,
        "min_stream_history": float(min_stream_history),
        "per_rule_cap": float(per_rule_cap),
        "total_boost_cap": float(total_boost_cap),
        "diminishing_return_factor": float(diminishing_return_factor),
        "rule_count_norm_factor": float(rule_count_norm_factor),
        "max_rules_per_member": float(max_rules_per_member),
        "compression_alpha": float(compression_alpha),
        "exclusivity_rule_bonus": float(exclusivity_rule_bonus),
        "exclusivity_boost_bonus": float(exclusivity_boost_bonus),
        "exclusivity_cap": float(exclusivity_cap),
        "min_compression_factor": float(min_compression_factor),
        "dominant_gap_strict": float(dominant_gap_strict),
        "dominant_ratio_max_strict": float(dominant_ratio_max_strict),
        "dominant_exclusivity_min": float(dominant_exclusivity_min),
        "dominant_rule_gap_min": float(dominant_rule_gap_min),
        "dominant_alignment_min": float(dominant_alignment_min),
        "contested_gap_max": float(contested_gap_max),
        "contested_ratio_min": float(contested_ratio_min),
        "top2_ratio_trigger": float(top2_ratio_trigger),
        "top2_gap_trigger": float(top2_gap_trigger),
        "top2_alignment_ceiling": float(top2_alignment_ceiling),
        "top2_exclusivity_ceiling": float(top2_exclusivity_ceiling),
        "m0025_boost_gap_min": float(m0025_boost_gap_min),
        "m0025_alignment_min": float(m0025_alignment_min),
        "m0025_top2_score_max": float(m0025_top2_score_max),
        "m0225_boost_gap_min": float(m0225_boost_gap_min),
        "m0225_alignment_min": float(m0225_alignment_min),
        "m0225_ratio_max": float(m0225_ratio_max),
        "m0255_boost_gap_min": float(m0255_boost_gap_min),
        "m0255_alignment_min": float(m0255_alignment_min),
        "m0255_gap_min": float(m0255_gap_min),
        "m0025_penalty_top2_score_min": float(m0025_penalty_top2_score_min),
        "m0025_penalty_alignment_max": float(m0025_penalty_alignment_max),
        "m0025_penalty_multiplier_top2": float(m0025_penalty_multiplier_top2),
        "m0025_penalty_multiplier_align": float(m0025_penalty_multiplier_align),
        "m0225_boost_alignment_min": float(m0225_boost_alignment_min),
        "m0225_boost_multiplier": float(m0225_boost_multiplier),
        "m0255_boost_multiplier_gap": float(m0255_boost_multiplier_gap),
        "m0255_boost_multiplier_align": float(m0255_boost_multiplier_align),
        "weak_top1_score_floor": float(weak_top1_score_floor),
        "rows_to_show": int(rows_to_show),
        "lab_max_events": int(lab_max_events),
    }


def render_regular_results(out: pd.DataFrame, summary: pd.DataFrame, rows_to_show: int) -> None:
    st.subheader("Regular Run summary")
    st.dataframe(summary, use_container_width=True)

    st.subheader("Playlist preview")
    st.dataframe(out.head(int(rows_to_show)), use_container_width=True)

    st.subheader("Play mode distribution")
    st.dataframe(
        out["play_mode"].value_counts(dropna=False).rename_axis("play_mode").reset_index(name="count"),
        use_container_width=True,
    )

    st.subheader("Dominance state distribution")
    st.dataframe(
        out["dominance_state"].value_counts(dropna=False).rename_axis("dominance_state").reset_index(name="count"),
        use_container_width=True,
    )

    st.subheader("Calibration diagnostics")
    preview_cols = [
        "stream",
        "seed",
        "Top1",
        "Top2",
        "Top1_score",
        "Top2_score",
        "gap",
        "ratio",
        "boost_gap_top12",
        "blended_alignment_ratio",
        "exclusivity_strength",
        "dominance_state",
        "play_mode",
        "play_reason",
    ]
    present_cols = [c for c in preview_cols if c in out.columns]
    st.dataframe(out[present_cols].head(int(rows_to_show)), use_container_width=True)

    st.download_button(
        "Download core025_separator_ranked_playlist__2026-04-11_v23_safe_mined_pockets_locked.csv",
        data=out.to_csv(index=False),
        file_name="core025_separator_ranked_playlist__2026-04-11_v23_safe_mined_pockets_locked.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download core025_separator_summary__2026-04-11_v23_safe_mined_pockets_locked.csv",
        data=summary.to_csv(index=False),
        file_name="core025_separator_summary__2026-04-11_v23_safe_mined_pockets_locked.csv",
        mime="text/csv",
    )


def render_lab_results(per_event: pd.DataFrame, per_date: pd.DataFrame, per_stream: pd.DataFrame, by_mode: pd.DataFrame, summary: pd.DataFrame, rows_to_show: int) -> None:
    bucket_summary, per_date_oper, per_stream_oper = build_operational_views(per_event)

    st.subheader("LAB summary")
    st.dataframe(summary, use_container_width=True)

    st.subheader("Operational winner-event summary")
    st.caption("This section separates the diagnostic cumulative capture stats from the bucketed play outcomes on winner-event rows.")
    st.dataframe(bucket_summary, use_container_width=True)

    st.subheader("Recommendation breakdown by bucket")
    st.dataframe(by_mode, use_container_width=True)

    st.subheader("Per-event preview")
    st.dataframe(per_event.head(int(rows_to_show)), use_container_width=True)

    st.subheader("Per-date summary")
    st.dataframe(per_date.head(int(rows_to_show)), use_container_width=True)

    st.subheader("Per-date operational summary")
    st.dataframe(per_date_oper.head(int(rows_to_show)), use_container_width=True)

    st.subheader("Per-stream summary")
    st.dataframe(per_stream.head(int(rows_to_show)), use_container_width=True)

    st.subheader("Per-stream operational summary")
    st.dataframe(per_stream_oper.head(int(rows_to_show)), use_container_width=True)

    per_event = per_event.copy(); per_event["build_marker"] = BUILD_SLUG
    per_date = per_date.copy(); per_date["build_marker"] = BUILD_SLUG
    per_date_oper = per_date_oper.copy(); per_date_oper["build_marker"] = BUILD_SLUG
    per_stream = per_stream.copy(); per_stream["build_marker"] = BUILD_SLUG
    per_stream_oper = per_stream_oper.copy(); per_stream_oper["build_marker"] = BUILD_SLUG
    bucket_summary = bucket_summary.copy(); bucket_summary["build_marker"] = BUILD_SLUG
    summary = summary.copy(); summary["build_marker"] = BUILD_SLUG

    st.download_button(
        f"Download core025_lab_per_event__{BUILD_SLUG}.csv",
        data=per_event.to_csv(index=False),
        file_name=f"core025_lab_per_event__{BUILD_SLUG}.csv",
        mime="text/csv",
    )
    st.download_button(
        f"Download core025_lab_per_date__{BUILD_SLUG}.csv",
        data=per_date.to_csv(index=False),
        file_name=f"core025_lab_per_date__{BUILD_SLUG}.csv",
        mime="text/csv",
    )
    st.download_button(
        f"Download core025_lab_per_date_operational__{BUILD_SLUG}.csv",
        data=per_date_oper.to_csv(index=False),
        file_name=f"core025_lab_per_date_operational__{BUILD_SLUG}.csv",
        mime="text/csv",
    )
    st.download_button(
        f"Download core025_lab_per_stream__{BUILD_SLUG}.csv",
        data=per_stream.to_csv(index=False),
        file_name=f"core025_lab_per_stream__{BUILD_SLUG}.csv",
        mime="text/csv",
    )
    st.download_button(
        f"Download core025_lab_per_stream_operational__{BUILD_SLUG}.csv",
        data=per_stream_oper.to_csv(index=False),
        file_name=f"core025_lab_per_stream_operational__{BUILD_SLUG}.csv",
        mime="text/csv",
    )
    st.download_button(
        f"Download core025_lab_operational_summary__{BUILD_SLUG}.csv",
        data=bucket_summary.to_csv(index=False),
        file_name=f"core025_lab_operational_summary__{BUILD_SLUG}.csv",
        mime="text/csv",
    )
    st.download_button(
        f"Download core025_lab_summary__{BUILD_SLUG}.csv",
        data=summary.to_csv(index=False),
        file_name=f"core025_lab_summary__{BUILD_SLUG}.csv",
        mime="text/csv",
    )


def main():
    st.set_page_config(page_title="Core025 Separator Engine + LAB Walk-Forward", layout="wide")
    st.title("Core025 Separator Engine + LAB Walk-Forward")
    st.caption("Regular current-slate separator run plus optional full no-lookahead LAB walk-forward in one file.")
    st.code(BUILD_MARKER, language="text")

    params = collect_params_from_sidebar()
    run_mode = params["run_mode"]
    rows_to_show = params["rows_to_show"]
    lab_max_events = params["lab_max_events"]

    hist_file = st.file_uploader("Upload FULL history file", key="core025_full_history")
    sep_library_file = st.file_uploader("Upload promoted separator library CSV", key="core025_separator_library")

    if run_mode == "Regular Run":
        surv_file = st.file_uploader("Upload PLAY survivors file", key="core025_survivors")
    else:
        surv_file = None

    if run_mode == "Regular Run" and not all([hist_file, sep_library_file, surv_file]):
        st.info("Upload the full history file, promoted separator library CSV, and play survivors file.")
        return
    if run_mode == "LAB Walk-Forward" and not all([hist_file, sep_library_file]):
        st.info("Upload the full history file and promoted separator library CSV.")
        return

    try:
        hist = prep_history(load_table(hist_file))
        sep_lib_df = load_table(sep_library_file)
        separator_rules = load_separator_library(sep_lib_df)
        if surv_file is not None:
            surv = prep_survivors(load_table(surv_file))
        else:
            surv = None
    except Exception as e:
        st.exception(e)
        return

    if run_mode == "Regular Run":
        if st.button("Run Regular Playlist", type="primary"):
            with st.spinner("Running regular playlist..."):
                out, summary = run_regular_playlist(hist, surv, separator_rules, params)
                st.session_state["core025_regular_out_v14"] = out
                st.session_state["core025_regular_summary_v14"] = summary

        if "core025_regular_out_v14" in st.session_state and "core025_regular_summary_v14" in st.session_state:
            render_regular_results(
                st.session_state["core025_regular_out_v14"],
                st.session_state["core025_regular_summary_v14"],
                rows_to_show,
            )

    else:
        if st.button("Run LAB Walk-Forward", type="primary"):
            with st.spinner("Running full no-lookahead LAB walk-forward..."):
                progress = st.progress(0.0)
                per_event, per_date, per_stream, by_mode, summary = run_lab_walkforward(hist, separator_rules, params, progress_bar=progress)
                progress.empty()
                if lab_max_events > 0:
                    per_event = per_event.head(lab_max_events).copy()
                    non_core = int(summary.loc[summary["metric"] == "non_core025_transitions_skipped", "value"].iloc[0]) if not summary.empty and (summary["metric"] == "non_core025_transitions_skipped").any() else 0
                    total_seen = int(summary.loc[summary["metric"] == "total_transitions_seen", "value"].iloc[0]) if not summary.empty and (summary["metric"] == "total_transitions_seen").any() else 0
                    per_date, per_stream, by_mode, summary = summarize_lab(per_event, total_seen, non_core)
                st.session_state["core025_lab_per_event_v14"] = per_event
                st.session_state["core025_lab_per_date_v14"] = per_date
                st.session_state["core025_lab_per_stream_v14"] = per_stream
                st.session_state["core025_lab_by_mode_v14"] = by_mode
                st.session_state["core025_lab_summary_v14"] = summary

        if all(
            key in st.session_state
            for key in [
                "core025_lab_per_event_v14",
                "core025_lab_per_date_v14",
                "core025_lab_per_stream_v14",
                "core025_lab_by_mode_v14",
                "core025_lab_summary_v14",
            ]
        ):
            render_lab_results(
                st.session_state["core025_lab_per_event_v14"],
                st.session_state["core025_lab_per_date_v14"],
                st.session_state["core025_lab_per_stream_v14"],
                st.session_state["core025_lab_by_mode_v14"],
                st.session_state["core025_lab_summary_v14"],
                rows_to_show,
            )


if __name__ == "__main__":
    main()

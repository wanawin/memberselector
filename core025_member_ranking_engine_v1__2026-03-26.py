#!/usr/bin/env python3
# core025_separator_engine_plus_lab_walkforward__2026-03-31.py
#
# BUILD: core025_separator_engine_plus_lab_walkforward__2026-03-31
#
# Full file. No placeholders.
#
# Purpose
# -------
# Unified Core025 separator app with:
# 1) Regular Run mode for current survivor playlist ranking
# 2) Optional LAB mode for full no-lookahead walk-forward validation
#
# This file preserves the calibrated separator behavior and adds a full LAB path
# without removing the regular run workflow.
#
# Outputs
# -------
# Regular Run:
# - core025_separator_ranked_playlist__2026-03-31.csv
# - core025_separator_summary__2026-03-31.csv
#
# LAB Walk-Forward:
# - core025_lab_per_event__2026-03-31.csv
# - core025_lab_per_date__2026-03-31.csv
# - core025_lab_per_stream__2026-03-31.csv
# - core025_lab_summary__2026-03-31.csv

from __future__ import annotations

import io
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

CORE025 = ["0025", "0225", "0255"]
BUILD_MARKER = "BUILD: core025_separator_engine_plus_lab_walkforward__2026-03-31"
DEFAULT_SKIP_SCORE_CUTOFF = 0.515465


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
# Separator scoring and calibrated dominance logic
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


def decide_play_mode(
    top1_score: float,
    top2_score: float,
    gap: float,
    ratio: float,
    top1_rule_count: int,
    top2_rule_count: int,
    top1_boost: float,
    top2_boost: float,
    dominant_gap: float,
    contested_gap: float,
    dominant_ratio_max: float,
    contested_ratio_min: float,
    weak_top1_score_floor: float,
    min_rule_margin_for_dominance: int,
    min_boost_margin_for_dominance: float,
) -> Tuple[str, str]:
    if top1_score < float(weak_top1_score_floor):
        return "SKIP", "Top1 score too weak"

    rule_margin = int(top1_rule_count) - int(top2_rule_count)
    boost_margin = float(top1_boost) - float(top2_boost)

    if (
        gap >= float(dominant_gap)
        and ratio <= float(dominant_ratio_max)
        and rule_margin >= int(min_rule_margin_for_dominance)
        and boost_margin >= float(min_boost_margin_for_dominance)
    ):
        return "PLAY_TOP1", "Dominant Top1"

    if (
        gap < float(contested_gap)
        or ratio >= float(contested_ratio_min)
        or top2_rule_count >= top1_rule_count
        or boost_margin < float(min_boost_margin_for_dominance)
    ):
        return "PLAY_TOP2", "Contested row"

    return "PLAY_TOP1", "Default Top1"


def rank_members_from_maps(
    row: pd.Series,
    maps: BaselineMaps,
    separator_rules: List[Dict[str, object]],
    min_stream_history: int,
    dominant_gap: float,
    contested_gap: float,
    dominant_ratio_max: float,
    contested_ratio_min: float,
    weak_top1_score_floor: float,
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
    min_rule_margin_for_dominance: int,
    min_boost_margin_for_dominance: float,
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

    normalized_scores: Dict[str, float] = {}
    for k, v in compressed_scores.items():
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

    if (
        gap >= float(dominant_gap)
        and ratio <= float(dominant_ratio_max)
        and rule_margin >= int(min_rule_margin_for_dominance)
        and boost_margin >= float(min_boost_margin_for_dominance)
    ):
        dominance_state = "DOMINANT"
    elif (
        gap < float(contested_gap)
        or ratio >= float(contested_ratio_min)
        or top2_rule_count >= top1_rule_count
        or boost_margin < float(min_boost_margin_for_dominance)
    ):
        dominance_state = "CONTESTED"
    else:
        dominance_state = "WEAK"

    play_mode, play_reason = decide_play_mode(
        top1_score=top1_score,
        top2_score=top2_score,
        gap=gap,
        ratio=ratio,
        top1_rule_count=top1_rule_count,
        top2_rule_count=top2_rule_count,
        top1_boost=top1_boost,
        top2_boost=top2_boost,
        dominant_gap=float(dominant_gap),
        contested_gap=float(contested_gap),
        dominant_ratio_max=float(dominant_ratio_max),
        contested_ratio_min=float(contested_ratio_min),
        weak_top1_score_floor=float(weak_top1_score_floor),
        min_rule_margin_for_dominance=int(min_rule_margin_for_dominance),
        min_boost_margin_for_dominance=float(min_boost_margin_for_dominance),
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
    return pd.DataFrame(rows)


def summarize_lab(per_event: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(per_event) == 0:
        empty = pd.DataFrame()
        return empty, empty, empty, pd.DataFrame(columns=["metric", "value"])

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
    total = len(per_event)
    summary_rows.append({"metric": "events", "value": total})
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
            dominant_gap=float(params["dominant_gap"]),
            contested_gap=float(params["contested_gap"]),
            dominant_ratio_max=float(params["dominant_ratio_max"]),
            contested_ratio_min=float(params["contested_ratio_min"]),
            weak_top1_score_floor=float(params["weak_top1_score_floor"]),
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
            min_rule_margin_for_dominance=int(params["min_rule_margin_for_dominance"]),
            min_boost_margin_for_dominance=float(params["min_boost_margin_for_dominance"]),
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
        return empty, empty, empty, empty, pd.DataFrame(columns=["metric", "value"])

    maps = init_baseline_maps()
    rows = []
    total = len(transitions)

    for idx, (_, row) in enumerate(transitions.iterrows(), start=1):
        ranked = rank_members_from_maps(
            row=row,
            maps=maps,
            separator_rules=separator_rules,
            min_stream_history=int(params["min_stream_history"]),
            dominant_gap=float(params["dominant_gap"]),
            contested_gap=float(params["contested_gap"]),
            dominant_ratio_max=float(params["dominant_ratio_max"]),
            contested_ratio_min=float(params["contested_ratio_min"]),
            weak_top1_score_floor=float(params["weak_top1_score_floor"]),
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
            min_rule_margin_for_dominance=int(params["min_rule_margin_for_dominance"]),
            min_boost_margin_for_dominance=float(params["min_boost_margin_for_dominance"]),
        )

        winner_member = normalize_member_code(row["next_member"])
        top1_hit = int(ranked["Top1"] == winner_member)
        top2_hit = int((ranked["Top1"] == winner_member) or (ranked["Top2"] == winner_member))
        top3_hit = int((ranked["Top1"] == winner_member) or (ranked["Top2"] == winner_member) or (ranked["Top3"] == winner_member))
        play_rule_hit = 0
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
    per_date, per_stream, by_mode, summary = summarize_lab(per_event)
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
        dominant_gap = st.slider("Dominant gap threshold", min_value=0.00, max_value=2.00, value=0.18, step=0.01)
        contested_gap = st.slider("Contested gap threshold", min_value=0.00, max_value=2.00, value=0.08, step=0.01)
        dominant_ratio_max = st.slider("Dominant max ratio", min_value=0.50, max_value=1.00, value=0.88, step=0.01)
        contested_ratio_min = st.slider("Contested min ratio", min_value=0.50, max_value=1.00, value=0.94, step=0.01)
        weak_top1_score_floor = st.slider("Weak Top1 score floor", min_value=0.00, max_value=5.00, value=0.20, step=0.01)
        min_rule_margin_for_dominance = st.number_input("Min Top1 rule margin for dominance", min_value=0, max_value=20, value=1, step=1)
        min_boost_margin_for_dominance = st.slider("Min Top1 boost margin for dominance", min_value=0.00, max_value=2.00, value=0.08, step=0.01)

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
        "dominant_gap": float(dominant_gap),
        "contested_gap": float(contested_gap),
        "dominant_ratio_max": float(dominant_ratio_max),
        "contested_ratio_min": float(contested_ratio_min),
        "weak_top1_score_floor": float(weak_top1_score_floor),
        "min_rule_margin_for_dominance": float(min_rule_margin_for_dominance),
        "min_boost_margin_for_dominance": float(min_boost_margin_for_dominance),
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

    st.subheader("Compression diagnostics")
    compression_cols = [
        "stream",
        "seed",
        "Top1",
        "Top2",
        "Top1_score",
        "Top2_score",
        "gap",
        "ratio",
        "compression_factor",
        "exclusivity_strength",
        "rule_gap_top12",
        "boost_gap_top12",
        "rule_margin_top1_top2",
        "boost_margin_top1_top2",
        "play_mode",
        "dominance_state",
    ]
    present_cols = [c for c in compression_cols if c in out.columns]
    st.dataframe(out[present_cols].head(int(rows_to_show)), use_container_width=True)

    st.download_button(
        "Download core025_separator_ranked_playlist__2026-03-31.csv",
        data=out.to_csv(index=False),
        file_name="core025_separator_ranked_playlist__2026-03-31.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download core025_separator_summary__2026-03-31.csv",
        data=summary.to_csv(index=False),
        file_name="core025_separator_summary__2026-03-31.csv",
        mime="text/csv",
    )


def render_lab_results(per_event: pd.DataFrame, per_date: pd.DataFrame, per_stream: pd.DataFrame, by_mode: pd.DataFrame, summary: pd.DataFrame, rows_to_show: int) -> None:
    st.subheader("LAB summary")
    st.dataframe(summary, use_container_width=True)

    st.subheader("Recommendation breakdown by bucket")
    st.dataframe(by_mode, use_container_width=True)

    st.subheader("Per-event preview")
    st.dataframe(per_event.head(int(rows_to_show)), use_container_width=True)

    st.subheader("Per-date summary")
    st.dataframe(per_date.head(int(rows_to_show)), use_container_width=True)

    st.subheader("Per-stream summary")
    st.dataframe(per_stream.head(int(rows_to_show)), use_container_width=True)

    st.download_button(
        "Download core025_lab_per_event__2026-03-31.csv",
        data=per_event.to_csv(index=False),
        file_name="core025_lab_per_event__2026-03-31.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download core025_lab_per_date__2026-03-31.csv",
        data=per_date.to_csv(index=False),
        file_name="core025_lab_per_date__2026-03-31.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download core025_lab_per_stream__2026-03-31.csv",
        data=per_stream.to_csv(index=False),
        file_name="core025_lab_per_stream__2026-03-31.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download core025_lab_summary__2026-03-31.csv",
        data=summary.to_csv(index=False),
        file_name="core025_lab_summary__2026-03-31.csv",
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
                st.session_state["core025_regular_out"] = out
                st.session_state["core025_regular_summary"] = summary

        if "core025_regular_out" in st.session_state and "core025_regular_summary" in st.session_state:
            render_regular_results(
                st.session_state["core025_regular_out"],
                st.session_state["core025_regular_summary"],
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
                    per_date, per_stream, by_mode, summary = summarize_lab(per_event)
                st.session_state["core025_lab_per_event"] = per_event
                st.session_state["core025_lab_per_date"] = per_date
                st.session_state["core025_lab_per_stream"] = per_stream
                st.session_state["core025_lab_by_mode"] = by_mode
                st.session_state["core025_lab_summary"] = summary

        if all(
            key in st.session_state
            for key in [
                "core025_lab_per_event",
                "core025_lab_per_date",
                "core025_lab_per_stream",
                "core025_lab_by_mode",
                "core025_lab_summary",
            ]
        ):
            render_lab_results(
                st.session_state["core025_lab_per_event"],
                st.session_state["core025_lab_per_date"],
                st.session_state["core025_lab_per_stream"],
                st.session_state["core025_lab_by_mode"],
                st.session_state["core025_lab_summary"],
                rows_to_show,
            )


if __name__ == "__main__":
    main()

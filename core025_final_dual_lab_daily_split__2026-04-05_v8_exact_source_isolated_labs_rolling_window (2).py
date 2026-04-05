#!/usr/bin/env python3
from __future__ import annotations

import io
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# FINAL BUILD MARKER
# -----------------------------------------------------------------------------
FINAL_BUILD_MARKER = 'BUILD: core025_final_dual_lab_daily_split__2026-04-05_v8_exact_source_isolated_labs_rolling_window | 2026-04-05 UTC'

# -----------------------------------------------------------------------------
# EXACT WINNER SOURCE BODY
# -----------------------------------------------------------------------------
CORE025 = ["0025", "0225", "0255"]
BUILD_MARKER = "BUILD: core025_separator_engine_plus_lab_walkforward__2026-04-03_v14"
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
        "Download core025_separator_ranked_playlist__2026-04-03_v14.csv",
        data=out.to_csv(index=False),
        file_name="core025_separator_ranked_playlist__2026-04-03_v14.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download core025_separator_summary__2026-04-03_v14.csv",
        data=summary.to_csv(index=False),
        file_name="core025_separator_summary__2026-04-03_v14.csv",
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
        "Download core025_lab_per_event__2026-04-03_v14.csv",
        data=per_event.to_csv(index=False),
        file_name="core025_lab_per_event__2026-04-03_v14.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download core025_lab_per_date__2026-04-03_v14.csv",
        data=per_date.to_csv(index=False),
        file_name="core025_lab_per_date__2026-04-03_v14.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download core025_lab_per_stream__2026-04-03_v14.csv",
        data=per_stream.to_csv(index=False),
        file_name="core025_lab_per_stream__2026-04-03_v14.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download core025_lab_summary__2026-04-03_v14.csv",
        data=summary.to_csv(index=False),
        file_name="core025_lab_summary__2026-04-03_v14.csv",
        mime="text/csv",
    )



# -----------------------------------------------------------------------------
# EXACT SKIP SOURCE BODY
# -----------------------------------------------------------------------------
CORE025_SET = {"0025", "0225", "0255"}
DIGITS = list(range(10))
MIRROR_PAIRS = {(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)}


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


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_col(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    cols = list(df.columns)
    nmap = {_norm(c): c for c in cols}
    for cand in candidates:
        key = _norm(cand)
        if key in nmap:
            return nmap[key]
    for cand in candidates:
        key = _norm(cand)
        for k, c in nmap.items():
            if key and key in k:
                return c
    if required:
        raise KeyError(f"Required column not found. Tried {list(candidates)}. Available columns: {cols}")
    return None


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return dedupe_columns(df).to_csv(index=False).encode("utf-8")


def safe_display_df(df: pd.DataFrame, rows: int) -> pd.DataFrame:
    return dedupe_columns(df).head(int(rows)).copy()


def percentile_rank_series(s: pd.Series) -> pd.Series:
    if len(s) == 0:
        return s
    return s.rank(method="average", pct=True)


def has_streamlit_context() -> bool:
    if st is None:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def read_uploaded_table(uploaded_file) -> pd.DataFrame:
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
    raise ValueError(f"Unsupported uploaded input type: {uploaded_file.name}")


def normalize_result_to_4digits(result_text: str) -> Optional[str]:
    if pd.isna(result_text):
        return None
    digits = re.findall(r"\d", str(result_text))
    if len(digits) < 4:
        return None
    return "".join(digits[:4])


def core025_member(result4: str) -> Optional[str]:
    if result4 is None:
        return None
    sorted4 = "".join(sorted(result4))
    return sorted4 if sorted4 in CORE025_SET else None


def prepare_history(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = dedupe_columns(df_raw.copy())

    if len(df.columns) == 4:
        c0, c1, c2, c3 = list(df.columns)
        df = df.rename(columns={c0: "date", c1: "jurisdiction", c2: "game", c3: "result_raw"})
    else:
        date_col = find_col(df, ["date"], required=True)
        juris_col = find_col(df, ["jurisdiction", "state", "province"], required=True)
        game_col = find_col(df, ["game", "stream"], required=True)
        result_col = find_col(df, ["result", "winning result", "draw result"], required=True)
        df = df.rename(columns={
            date_col: "date",
            juris_col: "jurisdiction",
            game_col: "game",
            result_col: "result_raw",
        })

    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df["result4"] = df["result_raw"].apply(normalize_result_to_4digits)
    df["member"] = df["result4"].apply(core025_member)
    df["is_core025_hit"] = df["member"].notna().astype(int)
    df["stream_id"] = df["jurisdiction"].astype(str).str.strip() + " | " + df["game"].astype(str).str.strip()
    df = df.dropna(subset=["result4"]).copy().reset_index(drop=True)
    df["file_order"] = np.arange(len(df))
    return dedupe_columns(df)


def build_transition_events(history_df: pd.DataFrame) -> pd.DataFrame:
    sort_df = history_df.sort_values(["stream_id", "date_dt", "file_order"], ascending=[True, True, False]).copy()
    rows: List[Dict[str, object]] = []

    for stream_id, g in sort_df.groupby("stream_id", sort=False):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue

        past_hit_positions: List[int] = []
        for i in range(1, len(g)):
            prev_row = g.iloc[i - 1]
            cur_row = g.iloc[i]

            last_hit_before_prev = past_hit_positions[-1] if len(past_hit_positions) > 0 else None
            current_gap_before_event = (i - 1 - last_hit_before_prev) if last_hit_before_prev is not None else i
            last50 = g.iloc[max(0, i - 50):i]
            recent_50_hit_rate = float(last50["is_core025_hit"].mean()) if len(last50) else 0.0

            rows.append({
                "stream_id": stream_id,
                "jurisdiction": cur_row["jurisdiction"],
                "game": cur_row["game"],
                "event_date": cur_row["date_dt"],
                "seed": prev_row["result4"],
                "next_result4": cur_row["result4"],
                "next_member": cur_row["member"] if pd.notna(cur_row["member"]) else "",
                "next_is_core025_hit": int(cur_row["is_core025_hit"]),
                "stream_event_index": int(i),
                "current_gap_before_event": int(current_gap_before_event),
                "recent_50_hit_rate_before_event": recent_50_hit_rate,
            })

            if int(cur_row["is_core025_hit"]) == 1:
                past_hit_positions.append(i)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise ValueError("No usable transitions could be created from the uploaded history.")
    return dedupe_columns(out)


def digit_list(seed: str) -> List[int]:
    return [int(ch) for ch in str(seed)]


def feature_dict(seed: str) -> Dict[str, object]:
    d = digit_list(seed)
    cnt = Counter(d)
    s = sum(d)
    spread = max(d) - min(d)
    even = sum(x % 2 == 0 for x in d)
    high = sum(x >= 5 for x in d)
    unique = len(cnt)

    consec_links = 0
    unique_sorted = sorted(set(d))
    for a, b in zip(unique_sorted[:-1], unique_sorted[1:]):
        if b - a == 1:
            consec_links += 1

    mirrorpair_cnt = sum(1 for a, b in MIRROR_PAIRS if a in cnt and b in cnt)

    out: Dict[str, object] = {
        "sum": s,
        "spread": spread,
        "even": even,
        "high": high,
        "unique": unique,
        "pair": int(unique < 4),
        "max_rep": max(cnt.values()),
        "pos1": d[0],
        "pos2": d[1],
        "pos3": d[2],
        "pos4": d[3],
        "consec_links": consec_links,
        "mirrorpair_cnt": mirrorpair_cnt,
    }
    for k in DIGITS:
        out[f"has{k}"] = int(k in cnt)
        out[f"cnt{k}"] = int(cnt.get(k, 0))
    return out


def build_feature_table(transitions_df: pd.DataFrame) -> pd.DataFrame:
    feats = [feature_dict(seed) for seed in transitions_df["seed"].astype(str)]
    feat_df = pd.DataFrame(feats)
    return dedupe_columns(pd.concat([transitions_df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1))


def mine_negative_traits(df: pd.DataFrame, min_support: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    base_rate = float(df["next_is_core025_hit"].mean())

    candidate_cols = ["sum", "spread", "even", "high", "unique", "pair", "max_rep", "pos1", "pos2", "pos3", "pos4", "consec_links", "mirrorpair_cnt"] + [f"has{k}" for k in DIGITS] + [f"cnt{k}" for k in DIGITS]

    for col in candidate_cols:
        vals = sorted(df[col].dropna().unique().tolist())
        for val in vals:
            mask = df[col] == val
            support = int(mask.sum())
            if support < int(min_support):
                continue
            hit_rate = float(df.loc[mask, "next_is_core025_hit"].mean())
            rows.append({
                "trait": f"{col}={val}",
                "support": support,
                "support_pct": support / len(df),
                "hit_rate": hit_rate,
                "gain_vs_base": base_rate - hit_rate,
                "zero_hit_trait": int(hit_rate == 0.0),
            })

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["zero_hit_trait", "hit_rate", "support"], ascending=[False, True, False]).reset_index(drop=True)
    return dedupe_columns(out)


def eval_single_trait(df: pd.DataFrame, trait: str) -> pd.Series:
    col, raw_val = trait.split("=", 1)
    series = df[col]
    if pd.api.types.is_numeric_dtype(series):
        try:
            val = int(raw_val)
        except Exception:
            try:
                val = float(raw_val)
            except Exception:
                val = raw_val
    else:
        val = raw_val
    return series == val


def build_skip_score_table(
    feat_df: pd.DataFrame,
    negative_traits_df: pd.DataFrame,
    top_negative_traits_to_use: int,
) -> pd.DataFrame:
    work = feat_df.copy()
    selected = negative_traits_df.head(int(top_negative_traits_to_use)).copy()

    fire_counts: List[int] = []
    fired_traits: List[str] = []

    trait_list = selected["trait"].tolist()

    for idx in work.index:
        row_df = work.loc[[idx]]
        fired: List[str] = []
        for t in trait_list:
            if bool(eval_single_trait(row_df, t).iloc[0]):
                fired.append(t)
        fire_counts.append(len(fired))
        fired_traits.append(" | ".join(fired))

    work["skip_fire_count"] = fire_counts
    work["fired_skip_traits"] = fired_traits

    # stronger skip score = more suppressive
    work["trait_fire_pct"] = percentile_rank_series(work["skip_fire_count"].fillna(0))
    work["stream_negative_pct"] = percentile_rank_series(1 - work.groupby("stream_id")["next_is_core025_hit"].transform("mean"))
    work["recent50_negative_pct"] = percentile_rank_series(1 - work["recent_50_hit_rate_before_event"].fillna(0))

    work["skip_score"] = (
        0.50 * work["trait_fire_pct"].fillna(0) +
        0.30 * work["stream_negative_pct"].fillna(0) +
        0.20 * work["recent50_negative_pct"].fillna(0)
    ).clip(lower=0, upper=1)

    return dedupe_columns(work)


def build_retention_ladder(
    scored_df: pd.DataFrame,
    rung_count: int,
) -> pd.DataFrame:
    df = scored_df.sort_values(["skip_score", "skip_fire_count"], ascending=[False, False]).reset_index(drop=True)
    total_events = len(df)
    total_hits = int(df["next_is_core025_hit"].sum())

    if total_events == 0:
        return pd.DataFrame()

    cutoffs = np.linspace(0, total_events, int(rung_count) + 1, dtype=int)[1:]
    rows: List[Dict[str, object]] = []

    for rank_cut in cutoffs:
        skip_mask = pd.Series([False] * total_events)
        skip_mask.iloc[:rank_cut] = True

        skipped = df[skip_mask]
        played = df[~skip_mask]

        plays_saved = int(len(skipped))
        hits_skipped = int(skipped["next_is_core025_hit"].sum()) if len(skipped) else 0
        hits_kept = int(played["next_is_core025_hit"].sum()) if len(played) else 0

        max_skip_score_included = float(skipped["skip_score"].min()) if len(skipped) else np.nan
        min_skip_score_not_included = float(played["skip_score"].max()) if len(played) else np.nan

        rows.append({
            "ladder_rank": len(rows) + 1,
            "events_marked_skip": plays_saved,
            "plays_saved_pct": plays_saved / total_events if total_events else 0.0,
            "hits_skipped": hits_skipped,
            "hits_kept": hits_kept,
            "hit_retention_pct": hits_kept / total_hits if total_hits else 0.0,
            "hit_rate_on_played_events": hits_kept / len(played) if len(played) else 0.0,
            "max_skip_score_included": max_skip_score_included,
            "next_score_after_cutoff": min_skip_score_not_included,
        })

    out = pd.DataFrame(rows)
    return dedupe_columns(out)


def recommend_cutoff(ladder_df: pd.DataFrame, target_retention_pct: float) -> pd.DataFrame:
    if len(ladder_df) == 0:
        return pd.DataFrame()
    ok = ladder_df[ladder_df["hit_retention_pct"] >= float(target_retention_pct)].copy()
    if len(ok) == 0:
        return ladder_df.head(1).copy()
    # choose most aggressive skip that still meets target = max plays_saved_pct
    best = ok.sort_values(["plays_saved_pct", "hit_rate_on_played_events"], ascending=[False, False]).head(1).copy()
    return dedupe_columns(best)


def current_seed_rows(history_df: pd.DataFrame, last24_history: Optional[pd.DataFrame]) -> pd.DataFrame:
    source = last24_history if last24_history is not None and len(last24_history) else history_df
    latest = source.sort_values(["stream_id", "date_dt", "file_order"], ascending=[True, True, False]).groupby("stream_id", as_index=False).tail(1).copy()
    feat_df = pd.DataFrame([feature_dict(x) for x in latest["result4"]])
    latest = latest.reset_index(drop=True)
    out = pd.concat([
        latest[["stream_id", "jurisdiction", "game", "date_dt", "result4"]].rename(columns={"date_dt": "seed_date", "result4": "seed"}),
        feat_df
    ], axis=1)
    return dedupe_columns(out)


def score_current_streams(
    current_df: pd.DataFrame,
    history_scored_df: pd.DataFrame,
    negative_traits_df: pd.DataFrame,
    top_negative_traits_to_use: int,
    chosen_skip_score_cutoff: float,
) -> pd.DataFrame:
    work = current_df.copy()
    selected = negative_traits_df.head(int(top_negative_traits_to_use)).copy()
    trait_list = selected["trait"].tolist()

    fire_counts: List[int] = []
    fired_traits: List[str] = []

    for idx in work.index:
        row_df = work.loc[[idx]]
        fired: List[str] = []
        for t in trait_list:
            if bool(eval_single_trait(row_df, t).iloc[0]):
                fired.append(t)
        fire_counts.append(len(fired))
        fired_traits.append(" | ".join(fired))

    work["skip_fire_count"] = fire_counts
    work["fired_skip_traits"] = fired_traits

    stream_hist = history_scored_df.groupby("stream_id")["next_is_core025_hit"].mean().rename("stream_hit_rate")
    stream_hist_recent = history_scored_df.groupby("stream_id")["recent_50_hit_rate_before_event"].mean().rename("stream_recent50")
    work = work.merge(stream_hist, on="stream_id", how="left")
    work = work.merge(stream_hist_recent, on="stream_id", how="left")

    work["trait_fire_pct"] = percentile_rank_series(work["skip_fire_count"].fillna(0))
    work["stream_negative_pct"] = percentile_rank_series(1 - work["stream_hit_rate"].fillna(history_scored_df["next_is_core025_hit"].mean()))
    work["recent50_negative_pct"] = percentile_rank_series(1 - work["stream_recent50"].fillna(history_scored_df["recent_50_hit_rate_before_event"].mean()))

    work["skip_score"] = (
        0.50 * work["trait_fire_pct"].fillna(0) +
        0.30 * work["stream_negative_pct"].fillna(0) +
        0.20 * work["recent50_negative_pct"].fillna(0)
    ).clip(lower=0, upper=1)

    work["skip_class"] = np.where(work["skip_score"] >= float(chosen_skip_score_cutoff), "SKIP", "PLAY")
    out = work[["stream_id", "jurisdiction", "game", "seed_date", "seed", "skip_fire_count", "fired_skip_traits", "skip_score", "skip_class"]].copy()
    out = out.sort_values(["skip_score", "skip_fire_count"], ascending=[False, False]).reset_index(drop=True)
    return dedupe_columns(out)


def build_summary_text(
    transitions_df: pd.DataFrame,
    negative_traits_df: pd.DataFrame,
    ladder_df: pd.DataFrame,
    recommended_df: pd.DataFrame,
) -> str:
    lines: List[str] = []
    lines.append("CORE 025 SKIP LADDER SUMMARY")
    lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    lines.append("")
    lines.append(f"Transition events: {len(transitions_df):,}")
    lines.append(f"Core025 hits: {int(transitions_df['next_is_core025_hit'].sum()):,}")
    lines.append(f"Core025 base rate: {float(transitions_df['next_is_core025_hit'].mean()):.4f}")
    lines.append("")
    lines.append("Top negative traits:")
    for _, r in negative_traits_df.head(10).iterrows():
        lines.append(f"  - {r['trait']} | support={int(r['support'])} | hit_rate={r['hit_rate']:.4f} | gain={r['gain_vs_base']:.4f}")
    lines.append("")
    if len(recommended_df):
        r = recommended_df.iloc[0]
        lines.append("Recommended cutoff at/above retention target:")
        lines.append(f"  - plays_saved_pct={r['plays_saved_pct']:.4f}")
        lines.append(f"  - hit_retention_pct={r['hit_retention_pct']:.4f}")
        lines.append(f"  - hit_rate_on_played_events={r['hit_rate_on_played_events']:.4f}")
        lines.append(f"  - skip_score_cutoff={r['max_skip_score_included']:.6f}")
    return "\n".join(lines)


def run_pipeline(
    main_raw_df: pd.DataFrame,
    last24_raw_df: Optional[pd.DataFrame],
    min_trait_support: int,
    top_negative_traits_to_use: int,
    rung_count: int,
    target_retention_pct: float,
) -> Dict[str, object]:
    main_history = prepare_history(main_raw_df)
    last24_history = prepare_history(last24_raw_df) if last24_raw_df is not None else None

    transitions_df = build_transition_events(main_history)
    feat_df = build_feature_table(transitions_df)
    negative_traits_df = mine_negative_traits(feat_df, min_support=int(min_trait_support))
    scored_df = build_skip_score_table(
        feat_df=feat_df,
        negative_traits_df=negative_traits_df,
        top_negative_traits_to_use=int(top_negative_traits_to_use),
    )
    ladder_df = build_retention_ladder(scored_df, rung_count=int(rung_count))
    recommended_df = recommend_cutoff(ladder_df, target_retention_pct=float(target_retention_pct))

    chosen_cutoff = float(recommended_df.iloc[0]["max_skip_score_included"]) if len(recommended_df) else 1.0
    current_df = current_seed_rows(main_history, last24_history)
    current_scored_df = score_current_streams(
        current_df=current_df,
        history_scored_df=scored_df,
        negative_traits_df=negative_traits_df,
        top_negative_traits_to_use=int(top_negative_traits_to_use),
        chosen_skip_score_cutoff=chosen_cutoff,
    )

    summary_text = build_summary_text(
        transitions_df=transitions_df,
        negative_traits_df=negative_traits_df,
        ladder_df=ladder_df,
        recommended_df=recommended_df,
    )

    return {
        "main_history": main_history,
        "last24_history": last24_history,
        "transitions": transitions_df,
        "features": feat_df,
        "negative_traits": negative_traits_df,
        "scored_events": scored_df,
        "retention_ladder": ladder_df,
        "recommended_cutoff": recommended_df,
        "current_scored_streams": current_scored_df,
        "summary_text": summary_text,
        "completed_at_utc": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
    }



# -----------------------------------------------------------------------------
# FINAL APP OVERRIDES / WRAPPER HELPERS
# -----------------------------------------------------------------------------
BUILD_MARKER = FINAL_BUILD_MARKER
APP_DOWNLOAD_STEM = 'core025_final_dual_lab_daily_split__2026-04-05_v8_exact_source_isolated_labs_rolling_window'


def _make_uploaded_like(data: bytes, name: str):
    buf = io.BytesIO(data)
    buf.name = name
    return buf


def _load_any_table_from_bytes(data: bytes, name: str) -> pd.DataFrame:
    return read_uploaded_table(_make_uploaded_like(data, name))


def _normalize_stream_merge(x: object) -> str:
    s = '' if x is None or pd.isna(x) else str(x).strip()
    s = re.sub(r'\s*\|\s*', '|', s)
    return s


def _trim_raw_history_window(raw_df: pd.DataFrame, locked_total_rows: int) -> pd.DataFrame:
    if raw_df is None:
        return raw_df
    if locked_total_rows is None or int(locked_total_rows) <= 0:
        return raw_df.copy()
    if len(raw_df) <= int(locked_total_rows):
        return raw_df.copy().reset_index(drop=True)
    return raw_df.tail(int(locked_total_rows)).reset_index(drop=True)


def _winner_default_params() -> Dict[str, float]:
    return {
        'run_mode': 'LAB Walk-Forward',
        'min_stream_history': 20.0,
        'per_rule_cap': 2.50,
        'total_boost_cap': 10.0,
        'diminishing_return_factor': 0.35,
        'rule_count_norm_factor': 1.50,
        'max_rules_per_member': 5.0,
        'compression_alpha': 0.45,
        'exclusivity_rule_bonus': 0.08,
        'exclusivity_boost_bonus': 0.20,
        'exclusivity_cap': 0.35,
        'min_compression_factor': 0.30,
        'dominant_gap_strict': 0.65,
        'dominant_ratio_max_strict': 0.65,
        'dominant_exclusivity_min': 0.24,
        'dominant_rule_gap_min': 3.00,
        'dominant_alignment_min': 0.58,
        'contested_gap_max': 0.12,
        'contested_ratio_min': 0.97,
        'top2_ratio_trigger': 0.97,
        'top2_gap_trigger': 0.08,
        'top2_alignment_ceiling': 0.62,
        'top2_exclusivity_ceiling': 0.22,
        'm0025_boost_gap_min': 0.60,
        'm0025_alignment_min': 0.60,
        'm0025_top2_score_max': 1.75,
        'm0225_boost_gap_min': 0.45,
        'm0225_alignment_min': 0.58,
        'm0225_ratio_max': 0.88,
        'm0255_boost_gap_min': 0.40,
        'm0255_alignment_min': 0.55,
        'm0255_gap_min': 0.18,
        'm0025_penalty_top2_score_min': 1.70,
        'm0025_penalty_alignment_max': 0.58,
        'm0025_penalty_multiplier_top2': 0.88,
        'm0025_penalty_multiplier_align': 0.90,
        'm0225_boost_alignment_min': 0.60,
        'm0225_boost_multiplier': 1.05,
        'm0255_boost_multiplier_gap': 1.30,
        'm0255_boost_multiplier_align': 1.24,
        'weak_top1_score_floor': 0.20,
        'rows_to_show': 50,
        'lab_max_events': 0,
    }


def _skip_default_params() -> Dict[str, float]:
    return {
        'min_trait_support': 12,
        'top_negative_traits_to_use': 15,
        'rung_count': 50,
        'target_retention_pct': 0.75,
        'rows_to_show': 25,
    }


def collect_final_sidebar_controls(current_full_rows: Optional[int]) -> Tuple[Dict[str, float], Dict[str, float], int]:
    winner_defaults = _winner_default_params()
    skip_defaults = _skip_default_params()
    with st.sidebar:
        st.markdown(f'**{FINAL_BUILD_MARKER}**')
        st.header('Global Controls')
        if current_full_rows is not None and 'locked_total_history_rows' not in st.session_state:
            st.session_state['locked_total_history_rows'] = int(current_full_rows)
        if 'locked_total_history_rows' not in st.session_state:
            st.session_state['locked_total_history_rows'] = 0 if current_full_rows is None else int(current_full_rows)

        if current_full_rows is not None:
            st.caption(f'Current uploaded full-history rows: {int(current_full_rows):,}')
            if st.button('Reset locked row window to current uploaded full-history size'):
                st.session_state['locked_total_history_rows'] = int(current_full_rows)

        locked_total_rows = st.number_input(
            'Locked total history rows to use (rolling window)',
            min_value=0,
            step=1,
            key='locked_total_history_rows',
            help='The app reads the full uploaded history, then automatically uses only the most recent locked number of rows. As new rows are added in future uploads, the oldest used rows drop off so the used row count stays constant.',
        )

        st.header('Winner Engine Controls')
        min_stream_history = st.number_input('Minimum stream history for baseline fallback', min_value=0, value=int(winner_defaults['min_stream_history']), step=5)
        st.subheader('Scaling controls')
        per_rule_cap = st.slider('Per-rule cap', min_value=0.10, max_value=5.00, value=float(winner_defaults['per_rule_cap']), step=0.05)
        total_boost_cap = st.slider('Total boost cap per member', min_value=0.50, max_value=20.00, value=float(winner_defaults['total_boost_cap']), step=0.10)
        diminishing_return_factor = st.slider('Diminishing return factor', min_value=0.00, max_value=3.00, value=float(winner_defaults['diminishing_return_factor']), step=0.01)
        rule_count_norm_factor = st.slider('Rule-count normalization factor', min_value=0.00, max_value=3.00, value=float(winner_defaults['rule_count_norm_factor']), step=0.01)
        max_rules_per_member = st.number_input('Max fired rules per member', min_value=1, max_value=100, value=int(winner_defaults['max_rules_per_member']), step=1)

        st.subheader('Cross-member compression')
        compression_alpha = st.slider('Base compression alpha', min_value=0.05, max_value=1.00, value=float(winner_defaults['compression_alpha']), step=0.01)
        exclusivity_rule_bonus = st.slider('Exclusivity bonus per rule-gap', min_value=0.00, max_value=0.50, value=float(winner_defaults['exclusivity_rule_bonus']), step=0.01)
        exclusivity_boost_bonus = st.slider('Exclusivity bonus per boost-gap', min_value=0.00, max_value=1.00, value=float(winner_defaults['exclusivity_boost_bonus']), step=0.01)
        exclusivity_cap = st.slider('Exclusivity cap', min_value=0.00, max_value=1.00, value=float(winner_defaults['exclusivity_cap']), step=0.01)
        min_compression_factor = st.slider('Minimum compression factor', min_value=0.05, max_value=1.00, value=float(winner_defaults['min_compression_factor']), step=0.01)

        st.subheader('Dominance thresholds')
        dominant_gap_strict = st.slider('Strict dominant gap threshold', min_value=0.00, max_value=2.00, value=float(winner_defaults['dominant_gap_strict']), step=0.01)
        dominant_ratio_max_strict = st.slider('Strict dominant max ratio', min_value=0.50, max_value=1.00, value=float(winner_defaults['dominant_ratio_max_strict']), step=0.01)
        dominant_exclusivity_min = st.slider('Strict dominant min exclusivity', min_value=0.00, max_value=1.00, value=float(winner_defaults['dominant_exclusivity_min']), step=0.01)
        dominant_rule_gap_min = st.slider('Strict dominant min rule-gap', min_value=0.00, max_value=10.00, value=float(winner_defaults['dominant_rule_gap_min']), step=0.10)
        dominant_alignment_min = st.slider('Strict dominant min alignment', min_value=0.00, max_value=1.00, value=float(winner_defaults['dominant_alignment_min']), step=0.01)

        st.subheader('Top2 widening controls')
        contested_gap_max = st.slider('Contested gap max', min_value=0.00, max_value=2.00, value=float(winner_defaults['contested_gap_max']), step=0.01)
        contested_ratio_min = st.slider('Contested ratio min', min_value=0.50, max_value=1.00, value=float(winner_defaults['contested_ratio_min']), step=0.01)
        top2_ratio_trigger = st.slider('Top2 widen ratio trigger', min_value=0.50, max_value=1.00, value=float(winner_defaults['top2_ratio_trigger']), step=0.01)
        top2_gap_trigger = st.slider('Top2 widen gap trigger', min_value=0.00, max_value=2.00, value=float(winner_defaults['top2_gap_trigger']), step=0.01)
        top2_alignment_ceiling = st.slider('Top2 widen max alignment', min_value=0.00, max_value=1.00, value=float(winner_defaults['top2_alignment_ceiling']), step=0.01)
        top2_exclusivity_ceiling = st.slider('Top2 widen max exclusivity', min_value=0.00, max_value=1.00, value=float(winner_defaults['top2_exclusivity_ceiling']), step=0.01)

        st.subheader('Member-specific Top1 gates')
        m0025_boost_gap_min = st.slider('0025 min boost gap', min_value=0.00, max_value=2.00, value=float(winner_defaults['m0025_boost_gap_min']), step=0.01)
        m0025_alignment_min = st.slider('0025 min alignment', min_value=0.00, max_value=1.00, value=float(winner_defaults['m0025_alignment_min']), step=0.01)
        m0025_top2_score_max = st.slider('0025 max Top2 score', min_value=0.00, max_value=5.00, value=float(winner_defaults['m0025_top2_score_max']), step=0.01)
        m0225_boost_gap_min = st.slider('0225 min boost gap', min_value=0.00, max_value=2.00, value=float(winner_defaults['m0225_boost_gap_min']), step=0.01)
        m0225_alignment_min = st.slider('0225 min alignment', min_value=0.00, max_value=1.00, value=float(winner_defaults['m0225_alignment_min']), step=0.01)
        m0225_ratio_max = st.slider('0225 max ratio', min_value=0.50, max_value=1.00, value=float(winner_defaults['m0225_ratio_max']), step=0.01)
        m0255_boost_gap_min = st.slider('0255 min boost gap', min_value=0.00, max_value=2.00, value=float(winner_defaults['m0255_boost_gap_min']), step=0.01)
        m0255_alignment_min = st.slider('0255 min alignment', min_value=0.00, max_value=1.00, value=float(winner_defaults['m0255_alignment_min']), step=0.01)
        m0255_gap_min = st.slider('0255 min gap', min_value=0.00, max_value=2.00, value=float(winner_defaults['m0255_gap_min']), step=0.01)

        st.subheader('Score-level member calibration')
        m0025_penalty_top2_score_min = st.slider('0025 penalty if score above', min_value=0.00, max_value=5.00, value=float(winner_defaults['m0025_penalty_top2_score_min']), step=0.01)
        m0025_penalty_alignment_max = st.slider('0025 penalty if alignment below', min_value=0.00, max_value=1.00, value=float(winner_defaults['m0025_penalty_alignment_max']), step=0.01)
        m0025_penalty_multiplier_top2 = st.slider('0025 score multiplier on high competition', min_value=0.50, max_value=1.20, value=float(winner_defaults['m0025_penalty_multiplier_top2']), step=0.01)
        m0025_penalty_multiplier_align = st.slider('0025 score multiplier on weak alignment', min_value=0.50, max_value=1.20, value=float(winner_defaults['m0025_penalty_multiplier_align']), step=0.01)
        m0225_boost_alignment_min = st.slider('0225 boost if alignment at least', min_value=0.00, max_value=1.00, value=float(winner_defaults['m0225_boost_alignment_min']), step=0.01)
        m0225_boost_multiplier = st.slider('0225 score multiplier on clean pocket', min_value=0.80, max_value=1.30, value=float(winner_defaults['m0225_boost_multiplier']), step=0.01)
        m0255_boost_multiplier_gap = st.slider('0255 score multiplier on boost-gap signal', min_value=0.80, max_value=1.50, value=float(winner_defaults['m0255_boost_multiplier_gap']), step=0.01)
        m0255_boost_multiplier_align = st.slider('0255 score multiplier on alignment signal', min_value=0.80, max_value=1.50, value=float(winner_defaults['m0255_boost_multiplier_align']), step=0.01)

        st.subheader('Weak-row control')
        weak_top1_score_floor = st.slider('Weak Top1 score floor', min_value=0.00, max_value=5.00, value=float(winner_defaults['weak_top1_score_floor']), step=0.01)
        winner_rows_to_show = st.number_input('Winner rows to display', min_value=5, value=int(winner_defaults['rows_to_show']), step=5)
        lab_max_events = st.number_input('Winner LAB max events (0 = all)', min_value=0, value=int(winner_defaults['lab_max_events']), step=50)

        st.header('Skip Ladder Controls')
        min_trait_support = st.number_input('Minimum trait support', min_value=3, value=int(skip_defaults['min_trait_support']), step=1)
        top_negative_traits_to_use = st.number_input('Top negative traits to use for scoring', min_value=1, value=int(skip_defaults['top_negative_traits_to_use']), step=1)
        rung_count = st.number_input('Ladder rung count', min_value=5, value=int(skip_defaults['rung_count']), step=5)
        target_retention_pct = st.slider('Target hit retention', min_value=0.50, max_value=0.99, value=float(skip_defaults['target_retention_pct']), step=0.01)
        skip_rows_to_show = st.number_input('Skip rows to display', min_value=5, value=int(skip_defaults['rows_to_show']), step=5)

    winner_params = {
        'run_mode': 'LAB Walk-Forward',
        'min_stream_history': float(min_stream_history),
        'per_rule_cap': float(per_rule_cap),
        'total_boost_cap': float(total_boost_cap),
        'diminishing_return_factor': float(diminishing_return_factor),
        'rule_count_norm_factor': float(rule_count_norm_factor),
        'max_rules_per_member': float(max_rules_per_member),
        'compression_alpha': float(compression_alpha),
        'exclusivity_rule_bonus': float(exclusivity_rule_bonus),
        'exclusivity_boost_bonus': float(exclusivity_boost_bonus),
        'exclusivity_cap': float(exclusivity_cap),
        'min_compression_factor': float(min_compression_factor),
        'dominant_gap_strict': float(dominant_gap_strict),
        'dominant_ratio_max_strict': float(dominant_ratio_max_strict),
        'dominant_exclusivity_min': float(dominant_exclusivity_min),
        'dominant_rule_gap_min': float(dominant_rule_gap_min),
        'dominant_alignment_min': float(dominant_alignment_min),
        'contested_gap_max': float(contested_gap_max),
        'contested_ratio_min': float(contested_ratio_min),
        'top2_ratio_trigger': float(top2_ratio_trigger),
        'top2_gap_trigger': float(top2_gap_trigger),
        'top2_alignment_ceiling': float(top2_alignment_ceiling),
        'top2_exclusivity_ceiling': float(top2_exclusivity_ceiling),
        'm0025_boost_gap_min': float(m0025_boost_gap_min),
        'm0025_alignment_min': float(m0025_alignment_min),
        'm0025_top2_score_max': float(m0025_top2_score_max),
        'm0225_boost_gap_min': float(m0225_boost_gap_min),
        'm0225_alignment_min': float(m0225_alignment_min),
        'm0225_ratio_max': float(m0225_ratio_max),
        'm0255_boost_gap_min': float(m0255_boost_gap_min),
        'm0255_alignment_min': float(m0255_alignment_min),
        'm0255_gap_min': float(m0255_gap_min),
        'm0025_penalty_top2_score_min': float(m0025_penalty_top2_score_min),
        'm0025_penalty_alignment_max': float(m0025_penalty_alignment_max),
        'm0025_penalty_multiplier_top2': float(m0025_penalty_multiplier_top2),
        'm0025_penalty_multiplier_align': float(m0025_penalty_multiplier_align),
        'm0225_boost_alignment_min': float(m0225_boost_alignment_min),
        'm0225_boost_multiplier': float(m0225_boost_multiplier),
        'm0255_boost_multiplier_gap': float(m0255_boost_multiplier_gap),
        'm0255_boost_multiplier_align': float(m0255_boost_multiplier_align),
        'weak_top1_score_floor': float(weak_top1_score_floor),
        'rows_to_show': int(winner_rows_to_show),
        'lab_max_events': int(lab_max_events),
    }
    skip_params = {
        'min_trait_support': int(min_trait_support),
        'top_negative_traits_to_use': int(top_negative_traits_to_use),
        'rung_count': int(rung_count),
        'target_retention_pct': float(target_retention_pct),
        'rows_to_show': int(skip_rows_to_show),
    }
    return winner_params, skip_params, int(locked_total_rows)


@st.cache_data(show_spinner=False)
def cached_parse_history_raw(data: bytes, name: str) -> pd.DataFrame:
    return _load_any_table_from_bytes(data, name)


@st.cache_data(show_spinner=False)
def cached_parse_separator_rules(data: bytes, name: str):
    sep_df = _load_any_table_from_bytes(data, name)
    return load_separator_library(sep_df)


@st.cache_data(show_spinner=False)
def cached_winner_lab(history_bytes: bytes, history_name: str, separator_bytes: bytes, separator_name: str, locked_total_rows: int, winner_params: dict):
    raw_history = cached_parse_history_raw(history_bytes, history_name)
    raw_used = _trim_raw_history_window(raw_history, locked_total_rows)
    hist = prep_history(raw_used)
    separator_rules = cached_parse_separator_rules(separator_bytes, separator_name)
    per_event, per_date, per_stream, by_mode, summary = run_lab_walkforward(hist, separator_rules, winner_params, progress_bar=None)
    if int(winner_params.get('lab_max_events', 0)) > 0:
        per_event = per_event.head(int(winner_params['lab_max_events'])).copy()
        non_core = int(summary.loc[summary['metric'] == 'non_core025_transitions_skipped', 'value'].iloc[0]) if not summary.empty and (summary['metric'] == 'non_core025_transitions_skipped').any() else 0
        total_seen = int(summary.loc[summary['metric'] == 'total_transitions_seen', 'value'].iloc[0]) if not summary.empty and (summary['metric'] == 'total_transitions_seen').any() else 0
        per_date, per_stream, by_mode, summary = summarize_lab(per_event, total_seen, non_core)
    return {
        'raw_history_rows': len(raw_history),
        'used_history_rows': len(raw_used),
        'hist_prepped_rows': len(hist),
        'per_event': per_event,
        'per_date': per_date,
        'per_stream': per_stream,
        'by_mode': by_mode,
        'summary': summary,
    }


@st.cache_data(show_spinner=False)
def cached_skip_lab(history_bytes: bytes, history_name: str, last24_bytes: Optional[bytes], last24_name: Optional[str], locked_total_rows: int, skip_params: dict):
    raw_history = cached_parse_history_raw(history_bytes, history_name)
    raw_used = _trim_raw_history_window(raw_history, locked_total_rows)
    last24_raw = None
    if last24_bytes is not None and last24_name is not None:
        last24_raw = _load_any_table_from_bytes(last24_bytes, last24_name)
    results = run_pipeline(
        main_raw_df=raw_used,
        last24_raw_df=last24_raw,
        min_trait_support=int(skip_params['min_trait_support']),
        top_negative_traits_to_use=int(skip_params['top_negative_traits_to_use']),
        rung_count=int(skip_params['rung_count']),
        target_retention_pct=float(skip_params['target_retention_pct']),
    )
    results['raw_history_rows'] = len(raw_history)
    results['used_history_rows'] = len(raw_used)
    return results


@st.cache_data(show_spinner=False)
def cached_daily(history_bytes: bytes, history_name: str, separator_bytes: bytes, separator_name: str, last24_bytes: Optional[bytes], last24_name: Optional[str], locked_total_rows: int, winner_params: dict, skip_params: dict):
    raw_history = cached_parse_history_raw(history_bytes, history_name)
    raw_used = _trim_raw_history_window(raw_history, locked_total_rows)
    hist = prep_history(raw_used)
    separator_rules = cached_parse_separator_rules(separator_bytes, separator_name)
    last24_raw = None
    last24_history = None
    if last24_bytes is not None and last24_name is not None:
        last24_raw = _load_any_table_from_bytes(last24_bytes, last24_name)
        last24_history = prepare_history(last24_raw)

    skip_results = run_pipeline(
        main_raw_df=raw_used,
        last24_raw_df=last24_raw,
        min_trait_support=int(skip_params['min_trait_support']),
        top_negative_traits_to_use=int(skip_params['top_negative_traits_to_use']),
        rung_count=int(skip_params['rung_count']),
        target_retention_pct=float(skip_params['target_retention_pct']),
    )

    current_df = current_seed_rows(skip_results['main_history'], last24_history)
    survivors = current_df.copy()
    survivors['stream'] = survivors['jurisdiction'].astype(str) + '|' + survivors['game'].astype(str)
    survivors = prep_survivors(survivors[['stream', 'seed']].copy())
    playlist, playlist_summary = run_regular_playlist(hist, survivors, separator_rules, winner_params)

    skip_current = skip_results['current_scored_streams'].copy()
    playlist = playlist.copy()
    playlist['stream_merge'] = playlist['stream'].apply(_normalize_stream_merge)
    skip_current['stream_merge'] = skip_current['stream_id'].apply(_normalize_stream_merge)
    merged = playlist.merge(skip_current, on=['stream_merge', 'seed'], how='left', suffixes=('', '_skip'))
    merged['action'] = np.where(merged['skip_class'].fillna('PLAY') == 'SKIP', 'STRIP', 'KEEP')
    merged['stream_id'] = merged.get('stream_id', merged['stream'])
    keep_df = merged[merged['action'] == 'KEEP'].reset_index(drop=True)
    strip_df = merged[merged['action'] == 'STRIP'].reset_index(drop=True)

    return {
        'raw_history_rows': len(raw_history),
        'used_history_rows': len(raw_used),
        'hist_prepped_rows': len(hist),
        'current_seed_rows': current_df,
        'playlist': playlist,
        'playlist_summary': playlist_summary,
        'skip_results': skip_results,
        'merged': merged,
        'keep': keep_df,
        'strip': strip_df,
    }


def build_side_by_side_metrics(winner_per_event: pd.DataFrame, skip_scored_events: pd.DataFrame, recommended_cutoff_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if winner_per_event is None or skip_scored_events is None or len(winner_per_event) == 0 or len(skip_scored_events) == 0:
        metrics = pd.DataFrame([
            {'metric': 'winner_events', 'value': 0},
            {'metric': 'skip_events', 'value': 0},
            {'metric': 'overlap_rows', 'value': 0},
        ])
        return metrics, pd.DataFrame()

    cutoff = float(recommended_cutoff_df.iloc[0]['max_skip_score_included']) if recommended_cutoff_df is not None and len(recommended_cutoff_df) else float(DEFAULT_SKIP_SCORE_CUTOFF)
    w = winner_per_event.copy()
    s = skip_scored_events.copy()
    w['stream_merge'] = w['stream'].apply(_normalize_stream_merge)
    s['stream_merge'] = s['stream_id'].apply(_normalize_stream_merge)

    candidate_date_cols = ['transition_date', 'event_date']
    w_date = next((c for c in candidate_date_cols if c in w.columns), None)
    s_date = next((c for c in candidate_date_cols if c in s.columns), None)
    if w_date is not None:
        w[w_date] = pd.to_datetime(w[w_date], errors='coerce')
    if s_date is not None:
        s[s_date] = pd.to_datetime(s[s_date], errors='coerce')

    join_cols = ['stream_merge', 'seed']
    if w_date is not None and s_date is not None:
        s = s.rename(columns={s_date: '_join_date'})
        w = w.rename(columns={w_date: '_join_date'})
        join_cols.append('_join_date')

    overlap = w.merge(s, on=join_cols, how='inner', suffixes=('_winner', '_skip'))
    if len(overlap):
        overlap['skip_class_recomputed'] = np.where(overlap['skip_score'] >= float(cutoff), 'SKIP', 'PLAY')
        overlap['winner_survived_skip'] = np.where(overlap['skip_class_recomputed'] == 'PLAY', 1, 0)
        overlap['winner_would_be_stripped'] = np.where(overlap['skip_class_recomputed'] == 'SKIP', 1, 0)
    else:
        overlap['winner_survived_skip'] = pd.Series(dtype='int64')
        overlap['winner_would_be_stripped'] = pd.Series(dtype='int64')

    metrics_rows = [
        {'metric': 'winner_events', 'value': int(len(winner_per_event))},
        {'metric': 'skip_events', 'value': int(len(skip_scored_events))},
        {'metric': 'overlap_rows', 'value': int(len(overlap))},
    ]
    if len(overlap):
        if 'top1_hit' in overlap.columns:
            metrics_rows.append({'metric': 'overlap_top1_hits', 'value': int(overlap['top1_hit'].sum())})
            metrics_rows.append({'metric': 'overlap_top1_hit_pct', 'value': float(overlap['top1_hit'].mean())})
        if 'top2_hit' in overlap.columns:
            metrics_rows.append({'metric': 'overlap_top2_hits', 'value': int(overlap['top2_hit'].sum())})
            metrics_rows.append({'metric': 'overlap_top2_hit_pct', 'value': float(overlap['top2_hit'].mean())})
        metrics_rows.append({'metric': 'winner_survived_skip_pct', 'value': float(overlap['winner_survived_skip'].mean())})
        metrics_rows.append({'metric': 'winner_would_be_stripped_pct', 'value': float(overlap['winner_would_be_stripped'].mean())})
        metrics_rows.append({'metric': 'skip_cutoff_reference', 'value': float(cutoff)})
    return pd.DataFrame(metrics_rows), overlap


def render_winner_lab_results(results: dict, rows_to_show: int) -> None:
    per_event = results['per_event']
    per_date = results['per_date']
    per_stream = results['per_stream']
    by_mode = results['by_mode']
    summary = results['summary']
    c1, c2, c3 = st.columns(3)
    c1.metric('Raw history rows uploaded', f"{int(results['raw_history_rows']):,}")
    c2.metric('Rolling history rows used', f"{int(results['used_history_rows']):,}")
    c3.metric('Prepared winner rows', f"{int(results['hist_prepped_rows']):,}")
    st.subheader('Winner LAB summary')
    st.dataframe(summary, use_container_width=True)
    st.subheader('Recommendation breakdown by bucket')
    st.dataframe(by_mode.head(int(rows_to_show)), use_container_width=True)
    st.subheader('Per-event preview')
    st.dataframe(per_event.head(int(rows_to_show)), use_container_width=True)
    st.subheader('Per-date summary')
    st.dataframe(per_date.head(int(rows_to_show)), use_container_width=True)
    st.subheader('Per-stream summary')
    st.dataframe(per_stream.head(int(rows_to_show)), use_container_width=True)

    st.download_button('Download winner per-event CSV', data=per_event.to_csv(index=False), file_name=f'core025_lab_per_event__{APP_DOWNLOAD_STEM}.csv', mime='text/csv')
    st.download_button('Download winner per-date CSV', data=per_date.to_csv(index=False), file_name=f'core025_lab_per_date__{APP_DOWNLOAD_STEM}.csv', mime='text/csv')
    st.download_button('Download winner per-stream CSV', data=per_stream.to_csv(index=False), file_name=f'core025_lab_per_stream__{APP_DOWNLOAD_STEM}.csv', mime='text/csv')
    st.download_button('Download winner by-mode CSV', data=by_mode.to_csv(index=False), file_name=f'core025_lab_by_mode__{APP_DOWNLOAD_STEM}.csv', mime='text/csv')
    st.download_button('Download winner summary CSV', data=summary.to_csv(index=False), file_name=f'core025_lab_summary__{APP_DOWNLOAD_STEM}.csv', mime='text/csv')


def render_skip_lab_results(results: dict, rows_to_show: int) -> None:
    transitions_df = results['transitions']
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Raw history rows uploaded', f"{int(results['raw_history_rows']):,}")
    c2.metric('Rolling history rows used', f"{int(results['used_history_rows']):,}")
    c3.metric('Transition events', f"{len(transitions_df):,}")
    c4.metric('Core025 hits', f"{int(transitions_df['next_is_core025_hit'].sum()) if len(transitions_df) else 0:,}")
    st.subheader('Summary')
    st.text_area('Summary text', results['summary_text'], height=340)
    st.download_button('Download skip summary TXT', data=results['summary_text'].encode('utf-8'), file_name=f'core025_skip_ladder_summary__{APP_DOWNLOAD_STEM}.txt', mime='text/plain')
    st.subheader('Recommended cutoff')
    st.dataframe(results['recommended_cutoff'], use_container_width=True)
    st.download_button('Download recommended cutoff CSV', data=df_to_csv_bytes(results['recommended_cutoff']), file_name=f'core025_skip_ladder_recommended_cutoff__{APP_DOWNLOAD_STEM}.csv', mime='text/csv')
    st.subheader('Retention ladder')
    st.dataframe(safe_display_df(results['retention_ladder'], int(rows_to_show)), use_container_width=True)
    st.download_button('Download retention ladder CSV', data=df_to_csv_bytes(results['retention_ladder']), file_name=f'core025_skip_ladder_retention_ladder__{APP_DOWNLOAD_STEM}.csv', mime='text/csv')
    st.subheader('Current scored streams')
    st.dataframe(safe_display_df(results['current_scored_streams'], int(rows_to_show)), use_container_width=True)
    st.download_button('Download current scored streams CSV', data=df_to_csv_bytes(results['current_scored_streams']), file_name=f'core025_skip_ladder_current_scored_streams__{APP_DOWNLOAD_STEM}.csv', mime='text/csv')
    st.subheader('Top negative traits')
    st.dataframe(safe_display_df(results['negative_traits'], int(rows_to_show)), use_container_width=True)
    st.download_button('Download negative traits CSV', data=df_to_csv_bytes(results['negative_traits']), file_name=f'core025_skip_ladder_negative_traits__{APP_DOWNLOAD_STEM}.csv', mime='text/csv')
    st.subheader('Scored events')
    st.dataframe(safe_display_df(results['scored_events'], int(rows_to_show)), use_container_width=True)
    st.download_button('Download scored events CSV', data=df_to_csv_bytes(results['scored_events']), file_name=f'core025_skip_ladder_scored_events__{APP_DOWNLOAD_STEM}.csv', mime='text/csv')


def render_daily_results(results: dict, rows_to_show: int) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Raw history rows uploaded', f"{int(results['raw_history_rows']):,}")
    c2.metric('Rolling history rows used', f"{int(results['used_history_rows']):,}")
    c3.metric('Current seeds scored', f"{len(results['current_seed_rows']):,}")
    c4.metric('Rows stripped', f"{len(results['strip']):,}")

    st.subheader('Daily merged board')
    st.dataframe(results['merged'].head(int(rows_to_show)), use_container_width=True)
    st.subheader('KEEP list')
    st.dataframe(results['keep'].head(int(rows_to_show)), use_container_width=True)
    st.subheader('STRIP list')
    st.dataframe(results['strip'].head(int(rows_to_show)), use_container_width=True)

    st.download_button('Download daily merged board CSV', data=results['merged'].to_csv(index=False), file_name=f'core025_daily_merged_board__{APP_DOWNLOAD_STEM}.csv', mime='text/csv')
    st.download_button('Download daily KEEP list CSV', data=results['keep'].to_csv(index=False), file_name=f'core025_daily_keep_list__{APP_DOWNLOAD_STEM}.csv', mime='text/csv')
    st.download_button('Download daily STRIP list CSV', data=results['strip'].to_csv(index=False), file_name=f'core025_daily_strip_list__{APP_DOWNLOAD_STEM}.csv', mime='text/csv')
    st.download_button('Download daily playlist CSV', data=results['playlist'].to_csv(index=False), file_name=f'core025_daily_playlist__{APP_DOWNLOAD_STEM}.csv', mime='text/csv')
    st.download_button('Download daily playlist summary CSV', data=results['playlist_summary'].to_csv(index=False), file_name=f'core025_daily_playlist_summary__{APP_DOWNLOAD_STEM}.csv', mime='text/csv')


def main() -> None:
    st.set_page_config(page_title='Core025 Final Dual LAB + Daily', layout='wide')
    st.title('Core025 Final Dual LAB + Daily')
    st.caption('One app, isolated Winner LAB and Skip LAB, plus Daily KEEP/STRIP. Winner scoring logic and skip scoring logic remain separate.')
    st.code(FINAL_BUILD_MARKER, language='text')

    st.subheader('Upload files')
    hist_file = st.file_uploader('Upload FULL history file', type=['txt','tsv','csv','xlsx','xls'], key='core025_hist_file_final_v8')
    sep_file = st.file_uploader('Upload promoted separator library CSV', type=['csv','txt','tsv','xlsx','xls'], key='core025_sep_file_final_v8')
    last24_file = st.file_uploader('Optional last 24 file (for Skip LAB and Daily)', type=['txt','tsv','csv','xlsx','xls'], key='core025_last24_file_final_v8')

    current_full_rows = None
    hist_bytes = hist_name = None
    sep_bytes = sep_name = None
    last24_bytes = last24_name = None

    if hist_file is not None:
        hist_bytes = hist_file.getvalue()
        hist_name = hist_file.name
        try:
            raw_preview = cached_parse_history_raw(hist_bytes, hist_name)
            current_full_rows = len(raw_preview)
        except Exception as e:
            st.exception(e)
            return
    if sep_file is not None:
        sep_bytes = sep_file.getvalue()
        sep_name = sep_file.name
    if last24_file is not None:
        last24_bytes = last24_file.getvalue()
        last24_name = last24_file.name

    winner_params, skip_params, locked_total_rows = collect_final_sidebar_controls(current_full_rows)

    if hist_file is not None:
        st.info(f'Rolling history window is locked to the most recent {int(locked_total_rows):,} raw rows. The full uploaded history remains the source, but rows older than this rolling window are not used.')

    tabs = st.tabs(['Winner LAB', 'Skip LAB', 'Daily KEEP / STRIP'])

    with tabs[0]:
        st.subheader('Blind walk-forward winner lab')
        if hist_file is None or sep_file is None:
            st.info('Upload the full history file and the promoted separator library CSV to run Winner LAB.')
        else:
            if st.button('Run Winner Engine LAB', type='primary', key='run_winner_lab_v8'):
                try:
                    with st.spinner('Running full no-lookahead Winner LAB...'):
                        st.session_state['winner_lab_results_v8'] = cached_winner_lab(hist_bytes, hist_name, sep_bytes, sep_name, locked_total_rows, winner_params)
                except Exception as e:
                    st.exception(e)
            if st.session_state.get('winner_lab_results_v8') is not None:
                render_winner_lab_results(st.session_state['winner_lab_results_v8'], int(winner_params['rows_to_show']))

    with tabs[1]:
        st.subheader('Skip LAB')
        if hist_file is None:
            st.info('Upload the full history file to run Skip LAB.')
        else:
            if st.button('Run Skip LAB', type='primary', key='run_skip_lab_v8'):
                try:
                    with st.spinner('Scoring skip ladder and computing retention cutoffs...'):
                        st.session_state['skip_lab_results_v8'] = cached_skip_lab(hist_bytes, hist_name, last24_bytes, last24_name, locked_total_rows, skip_params)
                except Exception as e:
                    st.exception(e)
            if st.session_state.get('skip_lab_results_v8') is not None:
                render_skip_lab_results(st.session_state['skip_lab_results_v8'], int(skip_params['rows_to_show']))
                if st.session_state.get('winner_lab_results_v8') is not None:
                    st.subheader('Optional side-by-side metrics from existing stored LAB outputs')
                    if st.button('Build side-by-side metrics from stored LAB results', key='build_side_by_side_v8'):
                        metrics_df, overlap_df = build_side_by_side_metrics(
                            st.session_state['winner_lab_results_v8']['per_event'],
                            st.session_state['skip_lab_results_v8']['scored_events'],
                            st.session_state['skip_lab_results_v8']['recommended_cutoff'],
                        )
                        st.session_state['side_by_side_metrics_v8'] = metrics_df
                        st.session_state['side_by_side_overlap_v8'] = overlap_df
                if st.session_state.get('side_by_side_metrics_v8') is not None:
                    st.dataframe(st.session_state['side_by_side_metrics_v8'], use_container_width=True)
                    st.download_button('Download side-by-side metrics CSV', data=st.session_state['side_by_side_metrics_v8'].to_csv(index=False), file_name=f'core025_lab_side_by_side_metrics__{APP_DOWNLOAD_STEM}.csv', mime='text/csv')
                if st.session_state.get('side_by_side_overlap_v8') is not None:
                    st.dataframe(st.session_state['side_by_side_overlap_v8'].head(int(max(winner_params['rows_to_show'], skip_params['rows_to_show']))), use_container_width=True)
                    st.download_button('Download side-by-side overlap CSV', data=st.session_state['side_by_side_overlap_v8'].to_csv(index=False), file_name=f'core025_lab_side_by_side_overlap__{APP_DOWNLOAD_STEM}.csv', mime='text/csv')

    with tabs[2]:
        st.subheader('Daily KEEP / STRIP')
        st.caption('Daily runs the real winner playlist first and then applies skip as post-process only.')
        if hist_file is None or sep_file is None:
            st.info('Upload the full history file and promoted separator library CSV to run Daily KEEP / STRIP.')
        else:
            if st.button('Run Daily KEEP / STRIP', type='primary', key='run_daily_v8'):
                try:
                    with st.spinner('Building current playlist and applying skip filter...'):
                        st.session_state['daily_results_v8'] = cached_daily(hist_bytes, hist_name, sep_bytes, sep_name, last24_bytes, last24_name, locked_total_rows, winner_params, skip_params)
                except Exception as e:
                    st.exception(e)
            if st.session_state.get('daily_results_v8') is not None:
                render_daily_results(st.session_state['daily_results_v8'], int(max(winner_params['rows_to_show'], skip_params['rows_to_show'])))


if __name__ == '__main__':
    main()

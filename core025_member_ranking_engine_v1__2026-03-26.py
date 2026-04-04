#!/usr/bin/env python3
# core025_skip_plus_v14_combined_lab__2026-04-03_v3.py
#
# BUILD: core025_skip_plus_v14_combined_lab__2026-04-03_v3
#
# Full file. No placeholders.
#
# Purpose
# -------
# Combined Step 1 Skip Engine + V14 Member Engine in one app.
#
# This file supports two production uses:
# 1) Daily Combined Run:
#    - update full history
#    - optionally upload last-24/latest-completed rows
#    - generate current stream seeds
#    - apply Step 1 skip logic
#    - create play_survivors.csv automatically
#    - feed survivors into the V14 member engine
#    - output daily ranked playlist
#
# 2) Combined Walk-Forward LAB:
#    - no-lookahead test of Step 1 + Step 2 together
#    - uses only prior history at each event
#    - reports skip retention, survivors, Top1 wins, Top2 wins, Top3 losses
#      using the user's locked scoring definitions
#
# Locked scoring definitions
# -------------------------
# Top1 win = only Top1 was played and Top1 won
# Top2 win = Top1+Top2 were played and one of those won
# Top3 win = winner not in Top1 or Top2 (true miss)

from __future__ import annotations

import io
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

BUILD_MARKER = "BUILD: core025_skip_plus_v14_combined_lab__2026-04-03_v3"
CORE025 = ["0025", "0225", "0255"]
CORE025_SET = set(CORE025)
DIGITS = list(range(10))
MIRROR_PAIRS = {(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)}
DEFAULT_SKIP_SCORE_CUTOFF = 0.515465


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

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
    raise ValueError(f"Unsupported uploaded input type: {f.name}")


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


def percentile_rank_series(s: pd.Series) -> pd.Series:
    if len(s) == 0:
        return s
    return s.rank(method="average", pct=True)


def normalize_result_to_4digits(result_text: object) -> Optional[str]:
    if pd.isna(result_text):
        return None
    digits = re.findall(r"\d", str(result_text))
    if len(digits) < 4:
        return None
    return "".join(digits[:4])


def core025_member(result4: Optional[str]) -> Optional[str]:
    if result4 is None:
        return None
    sorted4 = "".join(sorted(str(result4)))
    return sorted4 if sorted4 in CORE025_SET else None


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


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return dedupe_columns(df).to_csv(index=False).encode("utf-8")


# -----------------------------------------------------------------------------
# Shared feature engineering
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
        "sum_bucket": sum_bucket(s),
        "spread": spread,
        "spread_bucket": spread_bucket(spread),
        "even": even,
        "odd": 4 - even,
        "high": high,
        "low": 4 - high,
        "unique": unique,
        "pair": int(unique < 4),
        "max_rep": max(cnt.values()),
        "pos1": d[0],
        "pos2": d[1],
        "pos3": d[2],
        "pos4": d[3],
        "first2": f"{d[0]}{d[1]}",
        "last2": f"{d[2]}{d[3]}",
        "consec_links": consec_links,
        "mirrorpair_cnt": mirrorpair_cnt,
        "parity_pattern": "".join("E" if x % 2 == 0 else "O" for x in d),
        "highlow_pattern": "".join("H" if x >= 5 else "L" for x in d),
        "pair_token_pattern": pair_token_pattern(d),
        "structure": structure_label(d),
        "sorted_seed": "".join(map(str, sorted(d))),
    }
    for k in DIGITS:
        out[f"has{k}"] = int(k in cnt)
        out[f"cnt{k}"] = int(cnt.get(k, 0))
    return out


# -----------------------------------------------------------------------------
# History prep
# -----------------------------------------------------------------------------

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
        df = df.rename(columns={date_col: "date", juris_col: "jurisdiction", game_col: "game", result_col: "result_raw"})

    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df["result4"] = df["result_raw"].apply(normalize_result_to_4digits)
    df["member"] = df["result4"].apply(core025_member)
    df["is_core025_hit"] = df["member"].notna().astype(int)
    df["stream_id"] = df["jurisdiction"].astype(str).str.strip() + " | " + df["game"].astype(str).str.strip()
    df = df.dropna(subset=["result4", "date_dt"]).copy().reset_index(drop=True)
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
                "seed_date": prev_row["date_dt"],
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
    feat_df = pd.DataFrame([feature_dict(seed) for seed in out["seed"].astype(str)])
    out = pd.concat([out.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
    return dedupe_columns(out)


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


# -----------------------------------------------------------------------------
# Step 1 Skip Engine
# -----------------------------------------------------------------------------

def mine_negative_traits(df: pd.DataFrame, min_support: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    base_rate = float(df["next_is_core025_hit"].mean())
    candidate_cols = [
        "sum", "spread", "even", "high", "unique", "pair", "max_rep", "pos1", "pos2", "pos3", "pos4",
        "consec_links", "mirrorpair_cnt"
    ] + [f"has{k}" for k in DIGITS] + [f"cnt{k}" for k in DIGITS]

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


def build_skip_score_table(feat_df: pd.DataFrame, negative_traits_df: pd.DataFrame, top_negative_traits_to_use: int) -> pd.DataFrame:
    work = feat_df.copy()
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
    work["trait_fire_pct"] = percentile_rank_series(work["skip_fire_count"].fillna(0))
    work["stream_negative_pct"] = percentile_rank_series(1 - work.groupby("stream_id")["next_is_core025_hit"].transform("mean"))
    work["recent50_negative_pct"] = percentile_rank_series(1 - work["recent_50_hit_rate_before_event"].fillna(0))
    work["skip_score"] = (
        0.50 * work["trait_fire_pct"].fillna(0) +
        0.30 * work["stream_negative_pct"].fillna(0) +
        0.20 * work["recent50_negative_pct"].fillna(0)
    ).clip(lower=0, upper=1)
    return dedupe_columns(work)


def build_retention_ladder(scored_df: pd.DataFrame, rung_count: int) -> pd.DataFrame:
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
    return dedupe_columns(pd.DataFrame(rows))


def recommend_cutoff(ladder_df: pd.DataFrame, target_retention_pct: float) -> pd.DataFrame:
    if len(ladder_df) == 0:
        return pd.DataFrame()
    ok = ladder_df[ladder_df["hit_retention_pct"] >= float(target_retention_pct)].copy()
    if len(ok) == 0:
        return ladder_df.head(1).copy()
    best = ok.sort_values(["plays_saved_pct", "hit_rate_on_played_events"], ascending=[False, False]).head(1).copy()
    return dedupe_columns(best)


def score_current_streams(current_df: pd.DataFrame, history_scored_df: pd.DataFrame, negative_traits_df: pd.DataFrame, top_negative_traits_to_use: int, chosen_skip_score_cutoff: float) -> pd.DataFrame:
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


def run_skip_training(main_history: pd.DataFrame, min_trait_support: int, top_negative_traits_to_use: int, rung_count: int, target_retention_pct: float) -> Dict[str, object]:
    transitions_df = build_transition_events(main_history)
    negative_traits_df = mine_negative_traits(transitions_df, min_support=int(min_trait_support))
    scored_df = build_skip_score_table(transitions_df, negative_traits_df, int(top_negative_traits_to_use))
    ladder_df = build_retention_ladder(scored_df, int(rung_count))
    recommended_df = recommend_cutoff(ladder_df, float(target_retention_pct))
    chosen_cutoff = float(recommended_df.iloc[0]["max_skip_score_included"]) if len(recommended_df) else 1.0
    return {
        "transitions_df": transitions_df,
        "negative_traits_df": negative_traits_df,
        "history_scored_df": scored_df,
        "ladder_df": ladder_df,
        "recommended_df": recommended_df,
        "chosen_cutoff": chosen_cutoff,
    }


# -----------------------------------------------------------------------------
# Step 2 V14 member engine
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
        rules.append({
            "rule_id": idx + 1,
            "pair": str(r["pair"]),
            "trait_stack": str(r["trait_stack"]),
            "conditions": stack,
            "winner_member": winner_norm,
            "winner_rate": float(r["winner_rate"]),
            "pair_gap": float(r["pair_gap"]),
            "support": int(r["support"]),
            "stack_size": int(r["stack_size"]),
        })
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
    stream = str(row["stream_id"])
    sorted_seed = str(row["sorted_seed"])
    maps.exact_seed_map[seed][member] += 1
    maps.sorted_seed_map[sorted_seed][member] += 1
    maps.stream_member_map[stream][member] += 1
    maps.global_member_map[member] += 1


def baseline_scores_from_maps(seed_row: pd.Series, maps: BaselineMaps, min_stream_history: int = 20) -> Dict[str, float]:
    seed = str(seed_row["seed"])
    stream = str(seed_row["stream_id"])
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


def apply_separator_rules(row: pd.Series, rules: List[Dict[str, object]], per_rule_cap: float, total_boost_cap: float, diminishing_return_factor: float, rule_count_norm_factor: float, max_rules_per_member: int) -> Tuple[Dict[str, float], Dict[str, int]]:
    boosts = {m: 0.0 for m in CORE025}
    fired_counts = {m: 0 for m in CORE025}
    for rule in rules:
        is_full, _, _, _ = match_rule(row, rule)
        if is_full:
            winner = rule["winner_member"]
            if fired_counts[winner] >= int(max_rules_per_member):
                continue
            raw_score = (rule["winner_rate"] * 0.60) + (rule["pair_gap"] * 0.90)
            raw_score += min(rule["support"], 50) / 100.0
            raw_score += 0.03 * max(rule["stack_size"] - 1, 0)
            raw_score = min(raw_score, float(per_rule_cap))
            diminishing_scale = 1.0 / (1.0 + fired_counts[winner] * float(diminishing_return_factor))
            count_norm_scale = 1.0 / (1.0 + fired_counts[winner] * float(rule_count_norm_factor))
            scaled_score = raw_score * diminishing_scale * count_norm_scale
            boosts[winner] += scaled_score
            boosts[winner] = min(boosts[winner], float(total_boost_cap))
            fired_counts[winner] += 1
    return boosts, fired_counts


def compress_member_scores(base_scores: Dict[str, float], boosts: Dict[str, float], fired_counts: Dict[str, int], compression_alpha: float, exclusivity_rule_bonus: float, exclusivity_boost_bonus: float, exclusivity_cap: float, min_compression_factor: float) -> Tuple[Dict[str, float], Dict[str, float]]:
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
    exclusivity_strength = min(float(exclusivity_cap), (rule_gap * float(exclusivity_rule_bonus)) + (boost_gap * float(exclusivity_boost_bonus)))
    compression_factor = max(float(min_compression_factor), min(1.0, float(compression_alpha) + exclusivity_strength))
    compressed = {}
    for m in CORE025:
        delta = pre_scores[m] - mean_score
        compressed[m] = mean_score + (delta * compression_factor)
    diagnostics = {"compression_factor": compression_factor, "exclusivity_strength": exclusivity_strength, "rule_gap_top12": float(rule_gap), "boost_gap_top12": float(boost_gap)}
    return compressed, diagnostics


def compute_alignment(top1_rule_count: int, top2_rule_count: int, top1_boost: float, top2_boost: float) -> Dict[str, float]:
    total_rules = max(1, int(top1_rule_count) + int(top2_rule_count))
    total_boost = max(1e-9, float(top1_boost) + float(top2_boost))
    rule_alignment_ratio = float(top1_rule_count) / float(total_rules)
    boost_alignment_ratio = float(top1_boost) / float(total_boost)
    blended_alignment_ratio = (rule_alignment_ratio * 0.50) + (boost_alignment_ratio * 0.50)
    return {"rule_alignment_ratio": rule_alignment_ratio, "boost_alignment_ratio": boost_alignment_ratio, "blended_alignment_ratio": blended_alignment_ratio}


def apply_member_score_calibration(scores: Dict[str, float], member_alignment: Dict[str, float], member_boost_gap: Dict[str, float], m0025_penalty_top2_score_min: float, m0025_penalty_alignment_max: float, m0025_penalty_multiplier_top2: float, m0025_penalty_multiplier_align: float, m0225_boost_alignment_min: float, m0225_boost_multiplier: float, m0255_boost_gap_min: float, m0255_boost_alignment_min: float, m0255_boost_multiplier_gap: float, m0255_boost_multiplier_align: float) -> Dict[str, float]:
    adjusted = {m: float(v) for m, v in scores.items()}
    top2_score_proxy = sorted(adjusted.values(), reverse=True)[1] if len(adjusted) >= 2 else 0.0
    if adjusted.get("0025", 0.0) > 0:
        if top2_score_proxy > float(m0025_penalty_top2_score_min):
            adjusted["0025"] *= float(m0025_penalty_multiplier_top2)
        if member_alignment.get("0025", 0.0) < float(m0025_penalty_alignment_max):
            adjusted["0025"] *= float(m0025_penalty_multiplier_align)
    if adjusted.get("0225", 0.0) > 0 and member_alignment.get("0225", 0.0) >= float(m0225_boost_alignment_min):
        adjusted["0225"] *= float(m0225_boost_multiplier)
    if adjusted.get("0255", 0.0) > 0:
        if member_boost_gap.get("0255", 0.0) >= float(m0255_boost_gap_min):
            adjusted["0255"] *= float(m0255_boost_multiplier_gap)
        if member_alignment.get("0255", 0.0) >= float(m0255_boost_alignment_min):
            adjusted["0255"] *= float(m0255_boost_multiplier_align)
    return adjusted


def classify_dominance_state(gap: float, ratio: float, exclusivity_strength: float, rule_gap_top12: float, blended_alignment_ratio: float, dominant_gap_strict: float, dominant_ratio_max_strict: float, dominant_exclusivity_min: float, dominant_rule_gap_min: float, dominant_alignment_min: float, contested_gap_max: float, contested_ratio_min: float) -> str:
    if (gap >= float(dominant_gap_strict) and ratio <= float(dominant_ratio_max_strict) and exclusivity_strength >= float(dominant_exclusivity_min) and rule_gap_top12 >= float(dominant_rule_gap_min) and blended_alignment_ratio >= float(dominant_alignment_min)):
        return "DOMINANT"
    if gap >= 0.55 and blended_alignment_ratio >= 0.58:
        return "DOMINANT"
    if gap <= float(contested_gap_max) or ratio >= float(contested_ratio_min):
        return "CONTESTED"
    return "WEAK"


def member_specific_top1_gate(top1_member: str, gap: float, ratio: float, top2_score: float, boost_gap_top12: float, blended_alignment_ratio: float, m0025_boost_gap_min: float, m0025_alignment_min: float, m0025_top2_score_max: float, m0225_boost_gap_min: float, m0225_alignment_min: float, m0225_ratio_max: float, m0255_boost_gap_min: float, m0255_alignment_min: float, m0255_gap_min: float) -> Tuple[bool, str]:
    if top1_member == "0025":
        ok = boost_gap_top12 >= float(m0025_boost_gap_min) and blended_alignment_ratio >= float(m0025_alignment_min) and top2_score <= float(m0025_top2_score_max)
        return ok, "0025-specific gate passed" if ok else "0025-specific gate failed"
    if top1_member == "0225":
        ok = boost_gap_top12 >= float(m0225_boost_gap_min) and blended_alignment_ratio >= float(m0225_alignment_min) and ratio <= float(m0225_ratio_max)
        return ok, "0225-specific gate passed" if ok else "0225-specific gate failed"
    if top1_member == "0255":
        ok = boost_gap_top12 >= float(m0255_boost_gap_min) and blended_alignment_ratio >= float(m0255_alignment_min) and gap >= float(m0255_gap_min)
        return ok, "0255-specific gate passed" if ok else "0255-specific gate failed"
    return False, "Unknown member"


def decide_play_mode(top1_member: str, top1_score: float, top2_score: float, gap: float, ratio: float, exclusivity_strength: float, boost_gap_top12: float, blended_alignment_ratio: float, dominance_state: str, weak_top1_score_floor: float, top2_ratio_trigger: float, top2_gap_trigger: float, top2_alignment_ceiling: float, top2_exclusivity_ceiling: float, m0025_boost_gap_min: float, m0025_alignment_min: float, m0025_top2_score_max: float, m0225_boost_gap_min: float, m0225_alignment_min: float, m0225_ratio_max: float, m0255_boost_gap_min: float, m0255_alignment_min: float, m0255_gap_min: float) -> Tuple[str, str]:
    if top1_score < float(weak_top1_score_floor):
        return "SKIP", "Top1 score too weak"
    member_gate_passed, reason = member_specific_top1_gate(top1_member, gap, ratio, top2_score, boost_gap_top12, blended_alignment_ratio, m0025_boost_gap_min, m0025_alignment_min, m0025_top2_score_max, m0225_boost_gap_min, m0225_alignment_min, m0225_ratio_max, m0255_boost_gap_min, m0255_alignment_min, m0255_gap_min)
    if dominance_state == "DOMINANT" and member_gate_passed:
        return "PLAY_TOP1", f"Validated dominant Top1 | {reason}"
    if member_gate_passed and ratio <= 0.93 and blended_alignment_ratio >= 0.50:
        return "PLAY_TOP1", f"Member-specific Top1 promotion | {reason}"
    if ratio >= float(top2_ratio_trigger) and blended_alignment_ratio < float(top2_alignment_ceiling):
        return "PLAY_TOP2", "Tight ratio with weak alignment widened to Top1+Top2"
    if gap <= float(top2_gap_trigger) and exclusivity_strength <= float(top2_exclusivity_ceiling):
        return "PLAY_TOP2", "Small gap with weak exclusivity widened to Top1+Top2"
    if dominance_state == "CONTESTED":
        return "PLAY_TOP2", "Contested row"
    if not member_gate_passed:
        return "PLAY_TOP2", f"Top1 blocked by member-specific gate | {reason}"
    return "PLAY_TOP1", f"Top1-first default | {reason}"


def rank_members_from_maps(row: pd.Series, maps: BaselineMaps, separator_rules: List[Dict[str, object]], params: Dict[str, float]) -> Dict[str, object]:
    base = baseline_scores_from_maps(row, maps, min_stream_history=int(params["min_stream_history"]))
    boosts, fired_counts = apply_separator_rules(row, separator_rules, float(params["per_rule_cap"]), float(params["total_boost_cap"]), float(params["diminishing_return_factor"]), float(params["rule_count_norm_factor"]), int(params["max_rules_per_member"]))
    compressed_scores, compression_diag = compress_member_scores(base, boosts, fired_counts, float(params["compression_alpha"]), float(params["exclusivity_rule_bonus"]), float(params["exclusivity_boost_bonus"]), float(params["exclusivity_cap"]), float(params["min_compression_factor"]))

    member_alignment = {}
    member_boost_gap = {}
    for m in CORE025:
        other_boosts = sorted([boosts[o] for o in CORE025 if o != m], reverse=True)
        next_boost = other_boosts[0] if other_boosts else 0.0
        member_boost_gap[m] = float(boosts[m] - next_boost)
        member_alignment[m] = float(compute_alignment(fired_counts[m], max([fired_counts[o] for o in CORE025 if o != m], default=0), boosts[m], max([boosts[o] for o in CORE025 if o != m], default=0.0))["blended_alignment_ratio"])

    scores_after_calibration = apply_member_score_calibration(
        scores=compressed_scores,
        member_alignment=member_alignment,
        member_boost_gap=member_boost_gap,
        m0025_penalty_top2_score_min=float(params["m0025_penalty_top2_score_min"]),
        m0025_penalty_alignment_max=float(params["m0025_penalty_alignment_max"]),
        m0025_penalty_multiplier_top2=float(params["m0025_penalty_multiplier_top2"]),
        m0025_penalty_multiplier_align=float(params["m0025_penalty_multiplier_align"]),
        m0225_boost_alignment_min=float(params["m0225_boost_alignment_min"]),
        m0225_boost_multiplier=float(params["m0225_boost_multiplier"]),
        m0255_boost_gap_min=float(params["m0255_boost_gap_min"]),
        m0255_boost_alignment_min=float(params["m0255_alignment_min"]),
        m0255_boost_multiplier_gap=float(params["m0255_boost_multiplier_gap"]),
        m0255_boost_multiplier_align=float(params["m0255_boost_multiplier_align"]),
    )

    if max(scores_after_calibration.values()) > 0 and compression_diag["boost_gap_top12"] >= 0.60:
        current_leader = max(scores_after_calibration.items(), key=lambda kv: kv[1])[0]
        current_alignment = member_alignment.get(current_leader, 0.0)
        if current_alignment >= 0.60:
            scores_after_calibration[current_leader] *= 1.08

    ranked = sorted(scores_after_calibration.items(), key=lambda x: x[1], reverse=True)
    top1, top1_score = ranked[0]
    top2, top2_score = ranked[1]
    top3, top3_score = ranked[2]
    gap = top1_score - top2_score
    ratio = (top2_score / top1_score) if top1_score > 0 else 1.0
    top1_rule_count = fired_counts.get(top1, 0)
    top2_rule_count = fired_counts.get(top2, 0)
    top1_boost = boosts.get(top1, 0.0)
    top2_boost = boosts.get(top2, 0.0)
    alignment_diag = compute_alignment(top1_rule_count, top2_rule_count, top1_boost, top2_boost)
    dominance_state = classify_dominance_state(gap, ratio, float(compression_diag["exclusivity_strength"]), float(compression_diag["rule_gap_top12"]), float(alignment_diag["blended_alignment_ratio"]), float(params["dominant_gap_strict"]), float(params["dominant_ratio_max_strict"]), float(params["dominant_exclusivity_min"]), float(params["dominant_rule_gap_min"]), float(params["dominant_alignment_min"]), float(params["contested_gap_max"]), float(params["contested_ratio_min"]))
    play_mode, play_reason = decide_play_mode(top1, top1_score, top2_score, gap, ratio, float(compression_diag["exclusivity_strength"]), float(compression_diag["boost_gap_top12"]), float(alignment_diag["blended_alignment_ratio"]), dominance_state, float(params["weak_top1_score_floor"]), float(params["top2_ratio_trigger"]), float(params["top2_gap_trigger"]), float(params["top2_alignment_ceiling"]), float(params["top2_exclusivity_ceiling"]), float(params["m0025_boost_gap_min"]), float(params["m0025_alignment_min"]), float(params["m0025_top2_score_max"]), float(params["m0225_boost_gap_min"]), float(params["m0225_alignment_min"]), float(params["m0225_ratio_max"]), float(params["m0255_boost_gap_min"]), float(params["m0255_alignment_min"]), float(params["m0255_gap_min"]))
    return {
        "Top1": top1, "Top1_score": top1_score, "Top2": top2, "Top2_score": top2_score, "Top3": top3, "Top3_score": top3_score,
        "gap": gap, "ratio": ratio, "play_mode": play_mode, "play_reason": play_reason, "dominance_state": dominance_state,
        "compression_factor": compression_diag["compression_factor"], "exclusivity_strength": compression_diag["exclusivity_strength"],
        "rule_gap_top12": compression_diag["rule_gap_top12"], "boost_gap_top12": compression_diag["boost_gap_top12"],
        "rule_alignment_ratio": alignment_diag["rule_alignment_ratio"], "boost_alignment_ratio": alignment_diag["boost_alignment_ratio"], "blended_alignment_ratio": alignment_diag["blended_alignment_ratio"],
        "member_alignment_0025": member_alignment.get("0025", 0.0), "member_alignment_0225": member_alignment.get("0225", 0.0), "member_alignment_0255": member_alignment.get("0255", 0.0),
        "member_boost_gap_0025": member_boost_gap.get("0025", 0.0), "member_boost_gap_0225": member_boost_gap.get("0225", 0.0), "member_boost_gap_0255": member_boost_gap.get("0255", 0.0),
    }


# -----------------------------------------------------------------------------
# Combined run modes
# -----------------------------------------------------------------------------

def combined_daily_run(history_df: pd.DataFrame, separator_rules: List[Dict[str, object]], params: Dict[str, float], last24_df: Optional[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    main_history = prepare_history(history_df)
    last24_history = prepare_history(last24_df) if last24_df is not None else None
    skip_pack = run_skip_training(main_history, int(params["skip_min_trait_support"]), int(params["skip_top_negative_traits_to_use"]), int(params["skip_rung_count"]), float(params["skip_target_retention_pct"]))
    current_df = current_seed_rows(main_history, last24_history)
    current_scored_df = score_current_streams(current_df, skip_pack["history_scored_df"], skip_pack["negative_traits_df"], int(params["skip_top_negative_traits_to_use"]), float(skip_pack["chosen_cutoff"]))
    survivors = current_scored_df[current_scored_df["skip_class"] == "PLAY"].copy().reset_index(drop=True)

    transitions = build_transition_events(main_history)
    maps = init_baseline_maps()
    for _, tr in transitions.iterrows():
        add_transition_to_maps(maps, tr)

    playlist_rows = []
    surv_merge = survivors.merge(current_df, on=["stream_id", "jurisdiction", "game", "seed_date", "seed"], how="left")
    for _, row in surv_merge.iterrows():
        ranked = rank_members_from_maps(row, maps, separator_rules, params)
        playlist_rows.append({
            "stream": row["stream_id"], "seed": row["seed"], "skip_score": row["skip_score"], "skip_fire_count": row["skip_fire_count"],
            **ranked,
        })
    playlist_df = pd.DataFrame(playlist_rows).sort_values(["Top1_score", "gap", "ratio"], ascending=[False, False, True]).reset_index(drop=True) if len(playlist_rows) else pd.DataFrame()
    return {
        "current_scored_df": current_scored_df,
        "play_survivors_df": survivors,
        "playlist_df": playlist_df,
        "skip_negative_traits_df": skip_pack["negative_traits_df"],
        "skip_ladder_df": skip_pack["ladder_df"],
        "skip_recommended_df": skip_pack["recommended_df"],
    }


def combined_walkforward_lab(history_df: pd.DataFrame, separator_rules: List[Dict[str, object]], params: Dict[str, float], progress_bar=None, status_box=None) -> Dict[str, pd.DataFrame]:
    main_history = prepare_history(history_df)
    transitions_all = build_transition_events(main_history).sort_values(["event_date", "stream_id", "seed"]).reset_index(drop=True)
    max_lab_events = int(params.get("lab_max_events", 0))
    if max_lab_events > 0 and len(transitions_all) > max_lab_events:
        transitions_all = transitions_all.head(max_lab_events).copy().reset_index(drop=True)
    total_events_to_process = len(transitions_all)

    # V3: train Step 1 skip engine ONCE on the full transition set, then reuse it for all events.
    negative_traits_df = mine_negative_traits(transitions_all, min_support=int(params["skip_min_trait_support"]))
    history_scored_df = build_skip_score_table(transitions_all, negative_traits_df, int(params["skip_top_negative_traits_to_use"]))
    ladder_df = build_retention_ladder(history_scored_df, int(params["skip_rung_count"]))
    recommended_df = recommend_cutoff(ladder_df, float(params["skip_target_retention_pct"]))
    chosen_cutoff = float(recommended_df.iloc[0]["max_skip_score_included"]) if len(recommended_df) else 1.0

    maps = init_baseline_maps()
    rows = []

    for i in range(len(transitions_all)):
        current = transitions_all.iloc[i]
        if progress_bar is not None and total_events_to_process > 0:
            progress_bar.progress((i + 1) / total_events_to_process)
        if status_box is not None and ((i + 1) % 10 == 0 or (i + 1) == total_events_to_process):
            status_box.info(f"Processing combined walk-forward event {i + 1:,} of {total_events_to_process:,}...")

        winner_member = normalize_member_code(current["next_member"])
        if winner_member is None:
            add_transition_to_maps(maps, current)
            continue

        current_seed = pd.DataFrame([{c: current[c] for c in current.index}])
        current_skip = score_current_streams(
            current_seed.rename(columns={"event_date": "seed_date"}),
            history_scored_df,
            negative_traits_df,
            int(params["skip_top_negative_traits_to_use"]),
            chosen_cutoff,
        )
        skip_class = str(current_skip.iloc[0]["skip_class"])
        skip_score = float(current_skip.iloc[0]["skip_score"])
        skip_fire_count = int(current_skip.iloc[0]["skip_fire_count"])

        if skip_class == "SKIP":
            rows.append({
                "event_id": i + 1,
                "event_date": current["event_date"],
                "stream": current["stream_id"],
                "seed": current["seed"],
                "winning_member": winner_member,
                "step1_skip_class": "SKIP",
                "step1_skip_score": skip_score,
                "step1_skip_fire_count": skip_fire_count,
                "survived_step1": 0,
                "Top1": "", "Top2": "", "Top3": "",
                "play_mode": "SKIPPED_BY_STEP1",
                "top1_win": 0,
                "top2_win": 0,
                "top3_loss": 1,
                "play_rule_hit": 0,
            })
            add_transition_to_maps(maps, current)
            continue

        ranked = rank_members_from_maps(current, maps, separator_rules, params)
        play_mode = ranked["play_mode"]
        top1_win = int(play_mode == "PLAY_TOP1" and ranked["Top1"] == winner_member)
        top2_win = int(play_mode == "PLAY_TOP2" and (ranked["Top1"] == winner_member or ranked["Top2"] == winner_member))
        top3_loss = int(not (top1_win or top2_win))
        play_rule_hit = int(top1_win or top2_win)

        rows.append({
            "event_id": i + 1,
            "event_date": current["event_date"],
            "stream": current["stream_id"],
            "seed": current["seed"],
            "winning_member": winner_member,
            "step1_skip_class": "PLAY",
            "step1_skip_score": skip_score,
            "step1_skip_fire_count": skip_fire_count,
            "survived_step1": 1,
            "Top1": ranked["Top1"],
            "Top2": ranked["Top2"],
            "Top3": ranked["Top3"],
            "play_mode": play_mode,
            "top1_win": top1_win,
            "top2_win": top2_win,
            "top3_loss": top3_loss,
            "play_rule_hit": play_rule_hit,
            **ranked,
        })
        add_transition_to_maps(maps, current)

    per_event = pd.DataFrame(rows)
    if len(per_event) == 0:
        if status_box is not None:
            status_box.warning("Combined walk-forward produced no scored events.")
        empty = pd.DataFrame()
        return {"per_event": empty, "per_date": empty, "per_stream": empty, "summary": pd.DataFrame()}

    per_date = per_event.groupby("event_date", dropna=False).agg(
        events=("event_id", "count"),
        step1_skips=("step1_skip_class", lambda s: int((s == "SKIP").sum())),
        survivors=("survived_step1", "sum"),
        top1_wins=("top1_win", "sum"),
        top2_wins=("top2_win", "sum"),
        top3_losses=("top3_loss", "sum"),
    ).reset_index().sort_values("event_date").reset_index(drop=True)
    per_stream = per_event.groupby("stream", dropna=False).agg(
        events=("event_id", "count"),
        step1_skips=("step1_skip_class", lambda s: int((s == "SKIP").sum())),
        survivors=("survived_step1", "sum"),
        top1_wins=("top1_win", "sum"),
        top2_wins=("top2_win", "sum"),
        top3_losses=("top3_loss", "sum"),
    ).reset_index().sort_values(["top1_wins", "top2_wins", "events"], ascending=[False, False, False]).reset_index(drop=True)

    summary = pd.DataFrame([
        {"metric": "events", "value": int(len(per_event))},
        {"metric": "step1_skips", "value": int((per_event["step1_skip_class"] == "SKIP").sum())},
        {"metric": "step1_survivors", "value": int(per_event["survived_step1"].sum())},
        {"metric": "top1_wins", "value": int(per_event["top1_win"].sum())},
        {"metric": "top2_wins", "value": int(per_event["top2_win"].sum())},
        {"metric": "top3_losses", "value": int(per_event["top3_loss"].sum())},
        {"metric": "combined_capture_pct", "value": float((per_event["top1_win"].sum() + per_event["top2_win"].sum()) / max(1, len(per_event)))},
        {"metric": "avg_step1_skip_score", "value": float(per_event["step1_skip_score"].mean())},
        {"metric": "avg_survivors_per_day", "value": float(per_date["survivors"].mean())},
        {"metric": "trained_skip_cutoff", "value": float(chosen_cutoff)},
    ])
    if status_box is not None:
        status_box.success(f"Combined walk-forward completed: {len(per_event):,} scored events.")
    return {"per_event": per_event, "per_date": per_date, "per_stream": per_stream, "summary": summary}


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

def collect_params_from_sidebar() -> Dict[str, float]:
    with st.sidebar:
        st.markdown(f"**{BUILD_MARKER}**")
        mode = st.radio("Mode", ["Combined Daily Run", "Combined Walk-Forward LAB"], index=0)

        st.header("Step 1 Skip settings")
        skip_min_trait_support = st.number_input("Skip min trait support", min_value=1, value=6, step=1)
        skip_top_negative_traits_to_use = st.number_input("Skip top negative traits to use", min_value=1, value=12, step=1)
        skip_rung_count = st.number_input("Skip ladder rung count", min_value=5, value=25, step=5)
        skip_target_retention_pct = st.slider("Skip target hit retention", min_value=0.50, max_value=1.00, value=0.7628, step=0.0001)

        st.header("V14 member settings")
        min_stream_history = st.number_input("Minimum stream history for baseline fallback", min_value=0, value=20, step=5)
        per_rule_cap = st.slider("Per-rule cap", min_value=0.10, max_value=5.00, value=2.50, step=0.05)
        total_boost_cap = st.slider("Total boost cap per member", min_value=0.50, max_value=20.00, value=10.00, step=0.10)
        diminishing_return_factor = st.slider("Diminishing return factor", min_value=0.00, max_value=3.00, value=0.35, step=0.01)
        rule_count_norm_factor = st.slider("Rule-count normalization factor", min_value=0.00, max_value=3.00, value=1.50, step=0.01)
        max_rules_per_member = st.number_input("Max fired rules per member", min_value=1, max_value=100, value=5, step=1)
        compression_alpha = st.slider("Base compression alpha", min_value=0.05, max_value=1.00, value=0.45, step=0.01)
        exclusivity_rule_bonus = st.slider("Exclusivity bonus per rule-gap", min_value=0.00, max_value=0.50, value=0.08, step=0.01)
        exclusivity_boost_bonus = st.slider("Exclusivity bonus per boost-gap", min_value=0.00, max_value=1.00, value=0.20, step=0.01)
        exclusivity_cap = st.slider("Exclusivity cap", min_value=0.00, max_value=1.00, value=0.35, step=0.01)
        min_compression_factor = st.slider("Minimum compression factor", min_value=0.05, max_value=1.00, value=0.30, step=0.01)
        dominant_gap_strict = st.slider("Strict dominant gap threshold", min_value=0.00, max_value=2.00, value=0.65, step=0.01)
        dominant_ratio_max_strict = st.slider("Strict dominant max ratio", min_value=0.50, max_value=1.00, value=0.65, step=0.01)
        dominant_exclusivity_min = st.slider("Strict dominant min exclusivity", min_value=0.00, max_value=1.00, value=0.24, step=0.01)
        dominant_rule_gap_min = st.slider("Strict dominant min rule-gap", min_value=0.00, max_value=10.00, value=3.00, step=0.10)
        dominant_alignment_min = st.slider("Strict dominant min alignment", min_value=0.00, max_value=1.00, value=0.60, step=0.01)
        contested_gap_max = st.slider("Contested gap max", min_value=0.00, max_value=2.00, value=0.12, step=0.01)
        contested_ratio_min = st.slider("Contested ratio min", min_value=0.50, max_value=1.00, value=0.97, step=0.01)
        top2_ratio_trigger = st.slider("Top2 widen ratio trigger", min_value=0.50, max_value=1.00, value=0.97, step=0.01)
        top2_gap_trigger = st.slider("Top2 widen gap trigger", min_value=0.00, max_value=2.00, value=0.08, step=0.01)
        top2_alignment_ceiling = st.slider("Top2 widen max alignment", min_value=0.00, max_value=1.00, value=0.62, step=0.01)
        top2_exclusivity_ceiling = st.slider("Top2 widen max exclusivity", min_value=0.00, max_value=1.00, value=0.22, step=0.01)
        m0025_boost_gap_min = st.slider("0025 min boost gap", min_value=0.00, max_value=2.00, value=0.60, step=0.01)
        m0025_alignment_min = st.slider("0025 min alignment", min_value=0.00, max_value=1.00, value=0.60, step=0.01)
        m0025_top2_score_max = st.slider("0025 max Top2 score", min_value=0.00, max_value=5.00, value=1.75, step=0.01)
        m0225_boost_gap_min = st.slider("0225 min boost gap", min_value=0.00, max_value=2.00, value=0.45, step=0.01)
        m0225_alignment_min = st.slider("0225 min alignment", min_value=0.00, max_value=1.00, value=0.58, step=0.01)
        m0225_ratio_max = st.slider("0225 max ratio", min_value=0.50, max_value=1.00, value=0.88, step=0.01)
        m0255_boost_gap_min = st.slider("0255 min boost gap", min_value=0.00, max_value=2.00, value=0.40, step=0.01)
        m0255_alignment_min = st.slider("0255 min alignment", min_value=0.00, max_value=1.00, value=0.55, step=0.01)
        m0255_gap_min = st.slider("0255 min gap", min_value=0.00, max_value=2.00, value=0.18, step=0.01)
        m0025_penalty_top2_score_min = st.slider("0025 penalty if score above", min_value=0.00, max_value=5.00, value=1.70, step=0.01)
        m0025_penalty_alignment_max = st.slider("0025 penalty if alignment below", min_value=0.00, max_value=1.00, value=0.58, step=0.01)
        m0025_penalty_multiplier_top2 = st.slider("0025 score multiplier on high competition", min_value=0.50, max_value=1.20, value=0.88, step=0.01)
        m0025_penalty_multiplier_align = st.slider("0025 score multiplier on weak alignment", min_value=0.50, max_value=1.20, value=0.90, step=0.01)
        m0225_boost_alignment_min = st.slider("0225 boost if alignment at least", min_value=0.00, max_value=1.00, value=0.60, step=0.01)
        m0225_boost_multiplier = st.slider("0225 score multiplier on clean pocket", min_value=0.80, max_value=1.30, value=1.05, step=0.01)
        m0255_boost_multiplier_gap = st.slider("0255 score multiplier on boost-gap signal", min_value=0.80, max_value=1.50, value=1.28, step=0.01)
        m0255_boost_multiplier_align = st.slider("0255 score multiplier on alignment signal", min_value=0.80, max_value=1.50, value=1.22, step=0.01)
        weak_top1_score_floor = st.slider("Weak Top1 score floor", min_value=0.00, max_value=5.00, value=0.20, step=0.01)
        rows_to_show = st.number_input("Rows to display", min_value=5, value=50, step=5)
        lab_max_events = st.number_input("LAB max events (0 = all)", min_value=0, value=250, step=50)
    return locals()


def main():
    st.set_page_config(page_title="Core025 Combined Skip + V14 Lab", layout="wide")
    st.title("Core025 Combined Skip + V14 Lab")
    st.code(BUILD_MARKER, language="text")
    params = collect_params_from_sidebar()
    mode = params["mode"]
    rows_to_show = int(params["rows_to_show"])

    hist_file = st.file_uploader("Upload FULL history file", key="combined_hist")
    sep_library_file = st.file_uploader("Upload promoted separator library CSV", key="combined_sep_lib")
    last24_file = st.file_uploader("Optional: upload last-24/latest completed file", key="combined_last24") if mode == "Combined Daily Run" else None

    if not hist_file or not sep_library_file:
        st.info("Upload the full history file and promoted separator library CSV.")
        return

    try:
        hist_df = load_table(hist_file)
        sep_lib_df = load_table(sep_library_file)
        separator_rules = load_separator_library(sep_lib_df)
        last24_df = load_table(last24_file) if last24_file is not None else None
    except Exception as e:
        st.exception(e)
        return

    if mode == "Combined Daily Run":
        if st.button("Run Combined Daily Pipeline", type="primary"):
            with st.spinner("Running Step 1 skip + Step 2 V14..."):
                out = combined_daily_run(hist_df, separator_rules, params, last24_df)
                st.session_state["combined_daily_out_v3"] = out
        if "combined_daily_out_v3" in st.session_state:
            out = st.session_state["combined_daily_out_v3"]
            st.subheader("Step 1 current stream scoring")
            st.dataframe(out["current_scored_df"].head(rows_to_show), use_container_width=True)
            st.subheader("Generated play_survivors.csv")
            st.dataframe(out["play_survivors_df"].head(rows_to_show), use_container_width=True)
            st.subheader("Step 2 final playlist")
            st.dataframe(out["playlist_df"].head(rows_to_show), use_container_width=True)
            st.download_button("Download play_survivors__2026-04-03_v3.csv", df_to_csv_bytes(out["play_survivors_df"]), file_name="play_survivors__2026-04-03_v3.csv", mime="text/csv")
            st.download_button("Download final_ranked_playlist__2026-04-03_v3.csv", df_to_csv_bytes(out["playlist_df"]), file_name="final_ranked_playlist__2026-04-03_v3.csv", mime="text/csv")
            st.download_button("Download skip_current_scored__2026-04-03_v3.csv", df_to_csv_bytes(out["current_scored_df"]), file_name="skip_current_scored__2026-04-03_v3.csv", mime="text/csv")
    else:
        if st.button("Run Combined Walk-Forward LAB", type="primary"):
            with st.spinner("Running combined no-lookahead walk-forward..."):
                progress = st.progress(0.0)
                status_box = st.empty()
                out = combined_walkforward_lab(hist_df, separator_rules, params, progress_bar=progress, status_box=status_box)
                progress.empty()
                st.session_state["combined_lab_out_v3"] = out
        if "combined_lab_out_v3" in st.session_state:
            out = st.session_state["combined_lab_out_v3"]
            st.subheader("Combined summary")
            st.dataframe(out["summary"], use_container_width=True)
            st.subheader("Per-event")
            st.dataframe(out["per_event"].head(rows_to_show), use_container_width=True)
            st.subheader("Per-date")
            st.dataframe(out["per_date"].head(rows_to_show), use_container_width=True)
            st.subheader("Per-stream")
            st.dataframe(out["per_stream"].head(rows_to_show), use_container_width=True)
            st.download_button("Download combined_lab_per_event__2026-04-03_v3.csv", df_to_csv_bytes(out["per_event"]), file_name="combined_lab_per_event__2026-04-03_v3.csv", mime="text/csv")
            st.download_button("Download combined_lab_per_date__2026-04-03_v3.csv", df_to_csv_bytes(out["per_date"]), file_name="combined_lab_per_date__2026-04-03_v3.csv", mime="text/csv")
            st.download_button("Download combined_lab_per_stream__2026-04-03_v3.csv", df_to_csv_bytes(out["per_stream"]), file_name="combined_lab_per_stream__2026-04-03_v3.csv", mime="text/csv")
            st.download_button("Download combined_lab_summary__2026-04-03_v3.csv", df_to_csv_bytes(out["summary"]), file_name="combined_lab_summary__2026-04-03_v3.csv", mime="text/csv")


if __name__ == "__main__":
    main()

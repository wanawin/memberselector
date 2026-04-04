#!/usr/bin/env python3
# core025_skip_plus_v14_combined_lab__2026-04-03_v9.py
#
# BUILD: core025_skip_plus_v14_combined_lab__2026-04-03_v9
#
# Full file. No placeholders.
#
# Combined Step 1 Skip Engine + Step 2 V14-like Core025 member engine.
#
# V9 = member calibration upgrade
# - keeps v8 Top1 separation
# - keeps v7 skip-distribution fix
# - keeps v6 skip-training toggle behavior
# - adds member-specific score calibration BEFORE final ranking
# - adds member-specific calibration diagnostics so score physics can be audited
#
# Locked scoring definitions:
# - Top1 win = only Top1 was played and Top1 won
# - Top2 win = Top1+Top2 were played and one of those won
# - Top3 win = winner not in Top1 or Top2

from __future__ import annotations

import io
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

BUILD_MARKER = "BUILD: core025_skip_plus_v14_combined_lab__2026-04-03_v9"
CORE025 = ["0025", "0225", "0255"]
CORE025_SET = set(CORE025)
DIGITS = list(range(10))
DEFAULT_SKIP_SCORE_CUTOFF = 0.515465


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


def feature_dict(seed: str) -> Dict[str, object]:
    d = [int(ch) for ch in str(seed)]
    cnt = Counter(d)
    s = sum(d)
    spread = max(d) - min(d)
    even = sum(x % 2 == 0 for x in d)
    high = sum(x >= 5 for x in d)
    out: Dict[str, object] = {
        "sum": s,
        "spread": spread,
        "even": even,
        "odd": 4 - even,
        "high": high,
        "low": 4 - high,
        "unique": len(cnt),
        "pair": int(len(cnt) < 4),
        "max_rep": max(cnt.values()),
        "sorted_seed": "".join(map(str, sorted(d))),
        "pos1": d[0],
        "pos2": d[1],
        "pos3": d[2],
        "pos4": d[3],
    }
    for k in DIGITS:
        out[f"has{k}"] = int(k in cnt)
        out[f"cnt{k}"] = int(cnt.get(k, 0))
    return out


def prepare_history(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = dedupe_columns(df_raw.copy())
    if len(df.columns) == 4:
        c0, c1, c2, c3 = list(df.columns)
        df = df.rename(columns={c0: "date", c1: "jurisdiction", c2: "game", c3: "result_raw"})
    else:
        date_col = find_col(df, ["date"], required=True)
        juris_col = find_col(df, ["jurisdiction", "state"], required=True)
        game_col = find_col(df, ["game", "stream"], required=True)
        result_col = find_col(df, ["result", "winning result", "draw result"], required=True)
        df = df.rename(columns={date_col: "date", juris_col: "jurisdiction", game_col: "game", result_col: "result_raw"})
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df["result4"] = df["result_raw"].apply(normalize_result_to_4digits)
    df["member"] = df["result4"].apply(core025_member)
    df["is_core025_hit"] = df["member"].notna().astype(int)
    df["stream_id"] = df["jurisdiction"].astype(str).str.strip() + " | " + df["game"].astype(str).str.strip()
    df = df.dropna(subset=["date_dt", "result4"]).copy().reset_index(drop=True)
    df["file_order"] = np.arange(len(df))
    return dedupe_columns(df)


def build_transition_events(history_df: pd.DataFrame) -> pd.DataFrame:
    sort_df = history_df.sort_values(["stream_id", "date_dt", "file_order"], ascending=[True, True, False]).copy()
    rows: List[Dict[str, object]] = []
    for stream_id, g in sort_df.groupby("stream_id", sort=False):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue
        hit_positions: List[int] = []
        for i in range(1, len(g)):
            prev_row = g.iloc[i - 1]
            cur_row = g.iloc[i]
            last_hit_before_prev = hit_positions[-1] if hit_positions else None
            current_gap = (i - 1 - last_hit_before_prev) if last_hit_before_prev is not None else i
            last50 = g.iloc[max(0, i - 50):i]
            recent50 = float(last50["is_core025_hit"].mean()) if len(last50) else 0.0
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
                "current_gap_before_event": int(current_gap),
                "recent_50_hit_rate_before_event": recent50,
                **feature_dict(str(prev_row["result4"])),
            })
            if int(cur_row["is_core025_hit"]) == 1:
                hit_positions.append(i)
    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise ValueError("No usable transitions could be created from the uploaded history.")
    return dedupe_columns(out)


def current_seed_rows(history_df: pd.DataFrame, last24_history: Optional[pd.DataFrame]) -> pd.DataFrame:
    source = last24_history if last24_history is not None and len(last24_history) else history_df
    latest = source.sort_values(["stream_id", "date_dt", "file_order"], ascending=[True, True, False]).groupby("stream_id", as_index=False).tail(1).copy().reset_index(drop=True)
    feat_df = pd.DataFrame([feature_dict(str(x)) for x in latest["result4"]])
    out = pd.concat([
        latest[["stream_id", "jurisdiction", "game", "date_dt", "result4"]].rename(columns={"date_dt": "seed_date", "result4": "seed"}),
        feat_df,
    ], axis=1)
    return dedupe_columns(out)


def mine_negative_traits(df: pd.DataFrame, min_support: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    base_rate = float(df["next_is_core025_hit"].mean())
    candidate_cols = [
        "sum", "spread", "even", "high", "unique", "pair", "max_rep", "pos1", "pos2", "pos3", "pos4"
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
    trait_list = negative_traits_df.head(int(top_negative_traits_to_use))["trait"].tolist()
    fire_counts: List[int] = []
    fired_traits: List[str] = []
    for idx in work.index:
        row_df = work.loc[[idx]]
        fired = [t for t in trait_list if bool(eval_single_trait(row_df, t).iloc[0])]
        fire_counts.append(len(fired))
        fired_traits.append(" | ".join(fired))
    work["skip_fire_count"] = fire_counts
    work["fired_skip_traits"] = fired_traits
    work["trait_fire_pct"] = percentile_rank_series(work["skip_fire_count"].fillna(0))
    work["stream_negative_pct"] = percentile_rank_series(1 - work.groupby("stream_id")["next_is_core025_hit"].transform("mean"))
    work["recent50_negative_pct"] = percentile_rank_series(1 - work["recent_50_hit_rate_before_event"].fillna(0))
    work["skip_score"] = (
        0.50 * work["trait_fire_pct"].fillna(0)
        + 0.30 * work["stream_negative_pct"].fillna(0)
        + 0.20 * work["recent50_negative_pct"].fillna(0)
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
        rows.append({
            "ladder_rank": len(rows) + 1,
            "events_marked_skip": int(len(skipped)),
            "plays_saved_pct": float(len(skipped) / total_events) if total_events else 0.0,
            "hits_skipped": int(skipped["next_is_core025_hit"].sum()) if len(skipped) else 0,
            "hits_kept": int(played["next_is_core025_hit"].sum()) if len(played) else 0,
            "hit_retention_pct": float(played["next_is_core025_hit"].sum() / total_hits) if total_hits else 0.0,
            "hit_rate_on_played_events": float(played["next_is_core025_hit"].mean()) if len(played) else 0.0,
            "max_skip_score_included": float(skipped["skip_score"].min()) if len(skipped) else np.nan,
            "next_score_after_cutoff": float(played["skip_score"].max()) if len(played) else np.nan,
        })
    return dedupe_columns(pd.DataFrame(rows))


def recommend_cutoff(ladder_df: pd.DataFrame, target_retention_pct: float) -> pd.DataFrame:
    if len(ladder_df) == 0:
        return pd.DataFrame()
    ok = ladder_df[ladder_df["hit_retention_pct"] >= float(target_retention_pct)].copy()
    if len(ok) == 0:
        return ladder_df.head(1).copy()
    return dedupe_columns(ok.sort_values(["plays_saved_pct", "hit_rate_on_played_events"], ascending=[False, False]).head(1))


def score_current_streams(current_df: pd.DataFrame, history_scored_df: pd.DataFrame, negative_traits_df: pd.DataFrame, top_negative_traits_to_use: int, chosen_skip_score_cutoff: float) -> pd.DataFrame:
    work = current_df.copy()
    trait_list = negative_traits_df.head(int(top_negative_traits_to_use))["trait"].tolist()
    fire_counts: List[int] = []
    fired_traits: List[str] = []
    for idx in work.index:
        row_df = work.loc[[idx]]
        fired = [t for t in trait_list if bool(eval_single_trait(row_df, t).iloc[0])]
        fire_counts.append(len(fired))
        fired_traits.append(" | ".join(fired))
    work["skip_fire_count"] = fire_counts
    work["fired_skip_traits"] = fired_traits

    stream_hist = history_scored_df.groupby("stream_id")["next_is_core025_hit"].mean().rename("stream_hit_rate")
    stream_hist_recent = history_scored_df.groupby("stream_id")["recent_50_hit_rate_before_event"].mean().rename("stream_recent50")
    work = work.merge(stream_hist, on="stream_id", how="left")
    work = work.merge(stream_hist_recent, on="stream_id", how="left")

    max_train_fire = float(max(1.0, history_scored_df["skip_fire_count"].max())) if len(history_scored_df) else 1.0
    fallback_hit_rate = float(history_scored_df["next_is_core025_hit"].mean()) if len(history_scored_df) else 0.0
    fallback_recent50 = float(history_scored_df["recent_50_hit_rate_before_event"].mean()) if len(history_scored_df) else 0.0

    work["trait_fire_pct"] = (work["skip_fire_count"].fillna(0).astype(float) / max_train_fire).clip(lower=0, upper=1)
    work["stream_negative_pct"] = (1 - work["stream_hit_rate"].fillna(fallback_hit_rate).astype(float)).clip(lower=0, upper=1)
    work["recent50_negative_pct"] = (1 - work["stream_recent50"].fillna(fallback_recent50).astype(float)).clip(lower=0, upper=1)
    work["skip_score"] = (
        0.50 * work["trait_fire_pct"].fillna(0)
        + 0.30 * work["stream_negative_pct"].fillna(0)
        + 0.20 * work["recent50_negative_pct"].fillna(0)
    ).clip(lower=0, upper=1)
    work["skip_class"] = np.where(work["skip_score"] >= float(chosen_skip_score_cutoff), "SKIP", "PLAY")
    out = work[[
        "stream_id", "jurisdiction", "game", "seed_date", "seed", "skip_fire_count", "fired_skip_traits",
        "trait_fire_pct", "stream_negative_pct", "recent50_negative_pct", "skip_score", "skip_class"
    ]].copy()
    return dedupe_columns(out.sort_values(["skip_score", "skip_fire_count"], ascending=[False, False]).reset_index(drop=True))


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
            "conditions": stack,
            "winner_member": winner_norm,
            "winner_rate": float(r["winner_rate"]),
            "pair_gap": float(r["pair_gap"]),
            "support": int(r["support"]),
            "stack_size": int(r["stack_size"]),
        })
    return rules


def match_rule(row: pd.Series, rule: Dict[str, object]) -> bool:
    for col, val in rule["conditions"]:
        if col not in row.index:
            return False
        if normalize_scalar(row[col]) != val:
            return False
    return True


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


@dataclass
class CalibrationMaps:
    stream_member_rate: Dict[str, Dict[str, float]]
    global_member_rate: Dict[str, float]


def init_baseline_maps() -> BaselineMaps:
    return BaselineMaps(defaultdict(Counter), defaultdict(Counter), defaultdict(Counter), Counter())


def build_calibration_maps(transitions_df: pd.DataFrame) -> CalibrationMaps:
    winners = transitions_df[transitions_df["next_member"].apply(lambda x: normalize_member_code(x) is not None)].copy()
    if len(winners) == 0:
        return CalibrationMaps(stream_member_rate={}, global_member_rate={m: 1 / 3 for m in CORE025})
    winners["winner_norm"] = winners["next_member"].apply(normalize_member_code)
    global_counts = winners["winner_norm"].value_counts().to_dict()
    global_total = max(1, int(sum(global_counts.values())))
    global_member_rate = {m: global_counts.get(m, 0) / global_total for m in CORE025}
    stream_member_rate: Dict[str, Dict[str, float]] = {}
    for stream_id, g in winners.groupby("stream_id", sort=False):
        counts = g["winner_norm"].value_counts().to_dict()
        total = max(1, int(sum(counts.values())))
        stream_member_rate[stream_id] = {m: counts.get(m, 0) / total for m in CORE025}
    return CalibrationMaps(stream_member_rate=stream_member_rate, global_member_rate=global_member_rate)


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
    g = counter_to_probs(maps.global_member_map)
    for m in CORE025:
        score_accum[m] += g[m] * 0.25
    if sum(maps.stream_member_map[stream].values()) >= int(min_stream_history):
        s = counter_to_probs(maps.stream_member_map[stream])
        for m in CORE025:
            score_accum[m] += s[m] * 1.20
    if seed in maps.exact_seed_map and sum(maps.exact_seed_map[seed].values()) > 0:
        e = counter_to_probs(maps.exact_seed_map[seed])
        for m in CORE025:
            score_accum[m] += e[m] * 1.50
    sorted_key = str(seed_row["sorted_seed"])
    if sorted_key in maps.sorted_seed_map and sum(maps.sorted_seed_map[sorted_key].values()) > 0:
        sk = counter_to_probs(maps.sorted_seed_map[sorted_key])
        for m in CORE025:
            score_accum[m] += sk[m] * 1.10
    total = sum(score_accum.values())
    if total <= 0:
        return {m: 1 / 3 for m in CORE025}
    return {m: score_accum[m] / total for m in CORE025}


def apply_separator_rules(row: pd.Series, rules: List[Dict[str, object]], params: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, int]]:
    boosts = {m: 0.0 for m in CORE025}
    fired_counts = {m: 0 for m in CORE025}
    for rule in rules:
        if not match_rule(row, rule):
            continue
        winner = rule["winner_member"]
        if fired_counts[winner] >= int(params["max_rules_per_member"]):
            continue
        raw_score = (rule["winner_rate"] * 0.60) + (rule["pair_gap"] * 0.90)
        raw_score += min(rule["support"], 50) / 100.0
        raw_score += 0.03 * max(rule["stack_size"] - 1, 0)
        raw_score = min(raw_score, float(params["per_rule_cap"]))
        diminishing_scale = 1.0 / (1.0 + fired_counts[winner] * float(params["diminishing_return_factor"]))
        count_norm_scale = 1.0 / (1.0 + fired_counts[winner] * float(params["rule_count_norm_factor"]))
        boosts[winner] = min(float(params["total_boost_cap"]), boosts[winner] + raw_score * diminishing_scale * count_norm_scale)
        fired_counts[winner] += 1
    return boosts, fired_counts


def apply_member_calibration(base_scores: Dict[str, float], boosts: Dict[str, float], fired_counts: Dict[str, int], stream_id: str, calibration_maps: CalibrationMaps, params: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    adjusted = {m: base_scores[m] + boosts[m] for m in CORE025}
    diagnostics: Dict[str, float] = {}
    global_rates = calibration_maps.global_member_rate
    stream_rates = calibration_maps.stream_member_rate.get(stream_id, global_rates)
    total_rules = max(1, sum(fired_counts.values()))
    max_boost = max(1e-9, max(boosts.values()) if boosts else 0.0)

    for m in CORE025:
        calibration = 1.0
        stream_bias = stream_rates.get(m, global_rates.get(m, 1 / 3)) - global_rates.get(m, 1 / 3)
        calibration += float(params[f"cal_{m}_stream_bias_weight"]) * stream_bias
        calibration += float(params[f"cal_{m}_global_bias_weight"]) * (global_rates.get(m, 1 / 3) - (1 / 3))
        calibration += float(params[f"cal_{m}_rule_align_weight"]) * (fired_counts[m] / total_rules)
        calibration += float(params[f"cal_{m}_boost_align_weight"]) * (boosts[m] / max_boost if max_boost > 0 else 0.0)
        calibration = max(float(params["calibration_floor"]), min(float(params["calibration_cap"]), calibration))
        adjusted[m] *= calibration
        diagnostics[f"calibration_{m}"] = calibration
        diagnostics[f"stream_rate_{m}"] = stream_rates.get(m, global_rates.get(m, 1 / 3))
        diagnostics[f"global_rate_{m}"] = global_rates.get(m, 1 / 3)
    return adjusted, diagnostics


def rank_members_from_maps(row: pd.Series, maps: BaselineMaps, separator_rules: List[Dict[str, object]], params: Dict[str, float], calibration_maps: CalibrationMaps) -> Dict[str, object]:
    base = baseline_scores_from_maps(row, maps, int(params["min_stream_history"]))
    boosts, fired_counts = apply_separator_rules(row, separator_rules, params)
    calibrated_scores, calibration_diag = apply_member_calibration(base, boosts, fired_counts, str(row["stream_id"]), calibration_maps, params)

    separation_pressure = float(params["top1_separation_pressure"])
    top1_commit_gap = float(params["top1_commit_gap"])
    top1_commit_ratio_max = float(params["top1_commit_ratio_max"])
    top2_ratio_trigger = float(params["top2_ratio_trigger"])
    top2_gap_trigger = float(params["top2_gap_trigger"])

    scores = dict(calibrated_scores)
    first_pass = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    fp_top1, fp_s1 = first_pass[0]
    fp_top2, fp_s2 = first_pass[1]
    fp_gap = fp_s1 - fp_s2
    fp_ratio = (fp_s2 / fp_s1) if fp_s1 > 0 else 1.0

    total_rule_fires = max(1, sum(fired_counts.values()))
    top1_rule_share = fired_counts[fp_top1] / total_rule_fires
    total_boost = max(1e-9, sum(boosts.values()))
    top1_boost_share = boosts[fp_top1] / total_boost if total_boost > 0 else 0.0
    top1_confidence = 0.55 * top1_rule_share + 0.45 * top1_boost_share

    if fp_gap >= top1_commit_gap or (fp_ratio <= top1_commit_ratio_max and top1_confidence >= 0.52):
        scores[fp_top1] *= (1.0 + separation_pressure)
    elif fp_ratio >= max(0.90, top2_ratio_trigger - 0.01) and fp_gap <= max(0.10, top2_gap_trigger + 0.02):
        scores[fp_top1] *= 0.97

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top1, s1 = ranked[0]
    top2, s2 = ranked[1]
    top3, s3 = ranked[2]
    gap = s1 - s2
    ratio = (s2 / s1) if s1 > 0 else 1.0

    top1_rules = fired_counts[top1]
    top2_rules = fired_counts[top2]
    rule_gap = top1_rules - top2_rules
    total_rule_fires_final = max(1, sum(fired_counts.values()))
    top1_rule_share_final = top1_rules / total_rule_fires_final
    total_boost_final = max(1e-9, sum(boosts.values()))
    top1_boost_share_final = boosts[top1] / total_boost_final if total_boost_final > 0 else 0.0
    confidence = 0.55 * top1_rule_share_final + 0.45 * top1_boost_share_final

    if s1 < float(params["weak_top1_score_floor"]):
        play_mode = "SKIP"
        reason = "Top1 score too weak"
    elif gap >= top1_commit_gap and ratio <= top1_commit_ratio_max:
        play_mode = "PLAY_TOP1"
        reason = "Strong Top1 commitment"
    elif confidence >= float(params["top1_confidence_min"]) and rule_gap >= int(params["top1_min_rule_gap"]) and ratio <= float(params["top1_ratio_soft_max"]):
        play_mode = "PLAY_TOP1"
        reason = "Confidence-backed Top1"
    elif ratio >= top2_ratio_trigger or gap <= top2_gap_trigger:
        play_mode = "PLAY_TOP2"
        reason = "Tight top1/top2"
    else:
        play_mode = "PLAY_TOP1"
        reason = "Top1-first default"

    return {
        "Top1": top1, "Top1_score": s1,
        "Top2": top2, "Top2_score": s2,
        "Top3": top3, "Top3_score": s3,
        "gap": gap, "ratio": ratio,
        "play_mode": play_mode, "play_reason": reason,
        "rules_0025": fired_counts["0025"], "rules_0225": fired_counts["0225"], "rules_0255": fired_counts["0255"],
        "boost_0025": boosts["0025"], "boost_0225": boosts["0225"], "boost_0255": boosts["0255"],
        "top1_rule_share": top1_rule_share_final,
        "top1_boost_share": top1_boost_share_final,
        "top1_confidence": confidence,
        "top1_rule_gap": rule_gap,
        **calibration_diag,
    }


def train_skip_engine(transitions_df: pd.DataFrame, params: Dict[str, float]) -> Dict[str, object]:
    negative_traits_df = mine_negative_traits(transitions_df, min_support=int(params["skip_min_trait_support"]))
    history_scored_df = build_skip_score_table(transitions_df, negative_traits_df, int(params["skip_top_negative_traits_to_use"]))
    ladder_df = build_retention_ladder(history_scored_df, int(params["skip_rung_count"]))
    recommended_df = recommend_cutoff(ladder_df, float(params["skip_target_retention_pct"]))
    chosen_cutoff = DEFAULT_SKIP_SCORE_CUTOFF if bool(params["use_locked_skip_cutoff"]) else (float(recommended_df.iloc[0]["max_skip_score_included"]) if len(recommended_df) else 1.0)
    return {
        "negative_traits_df": negative_traits_df,
        "history_scored_df": history_scored_df,
        "ladder_df": ladder_df,
        "recommended_df": recommended_df,
        "chosen_cutoff": chosen_cutoff,
    }


def combined_daily_run(history_df: pd.DataFrame, separator_rules: List[Dict[str, object]], params: Dict[str, float], last24_df: Optional[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    main_history = prepare_history(history_df)
    last24_history = prepare_history(last24_df) if last24_df is not None else None
    transitions_full = build_transition_events(main_history)
    calibration_maps = build_calibration_maps(transitions_full)
    skip_pack = train_skip_engine(transitions_full, params)
    current_df = current_seed_rows(main_history, last24_history)
    current_scored_df = score_current_streams(current_df, skip_pack["history_scored_df"], skip_pack["negative_traits_df"], int(params["skip_top_negative_traits_to_use"]), float(skip_pack["chosen_cutoff"]))
    survivors = current_scored_df[current_scored_df["skip_class"] == "PLAY"].copy().reset_index(drop=True)
    maps = init_baseline_maps()
    for _, tr in transitions_full.iterrows():
        add_transition_to_maps(maps, tr)
    playlist_rows = []
    surv_merge = survivors.merge(current_df, on=["stream_id", "jurisdiction", "game", "seed_date", "seed"], how="left")
    for _, row in surv_merge.iterrows():
        ranked = rank_members_from_maps(row, maps, separator_rules, params, calibration_maps)
        playlist_rows.append({"stream": row["stream_id"], "seed": row["seed"], "skip_score": row["skip_score"], **ranked})
    playlist_df = pd.DataFrame(playlist_rows).sort_values(["Top1_score", "top1_confidence", "gap", "ratio"], ascending=[False, False, False, True]).reset_index(drop=True) if playlist_rows else pd.DataFrame()
    return {
        "current_scored_df": current_scored_df,
        "play_survivors_df": survivors,
        "playlist_df": playlist_df,
        "skip_negative_traits_df": skip_pack["negative_traits_df"],
        "skip_ladder_df": skip_pack["ladder_df"],
        "skip_recommended_df": skip_pack["recommended_df"],
        "skip_chosen_cutoff_df": pd.DataFrame([{"skip_cutoff": skip_pack["chosen_cutoff"], "locked_cutoff_mode": int(bool(params["use_locked_skip_cutoff"]))}]),
    }


def combined_walkforward_lab(history_df: pd.DataFrame, separator_rules: List[Dict[str, object]], params: Dict[str, float], progress_bar=None, status_box=None) -> Dict[str, pd.DataFrame]:
    main_history = prepare_history(history_df)
    transitions_full = build_transition_events(main_history).sort_values(["event_date", "stream_id", "seed"]).reset_index(drop=True)
    calibration_maps = build_calibration_maps(transitions_full)

    train_on_core025_only = bool(params.get("lab_train_skip_on_core025_only", False))
    skip_train_df = transitions_full.copy()
    if train_on_core025_only:
        skip_train_df = skip_train_df[skip_train_df["next_member"].apply(lambda x: normalize_member_code(x) is not None)].copy().reset_index(drop=True)

    skip_pack = train_skip_engine(skip_train_df, params)

    score_events = transitions_full[transitions_full["next_member"].apply(lambda x: normalize_member_code(x) is not None)].copy().reset_index(drop=True)
    max_lab_events = int(params.get("lab_max_events", 0))
    if max_lab_events > 0 and len(score_events) > max_lab_events:
        score_events = score_events.tail(max_lab_events).copy().reset_index(drop=True)

    total_events_to_process = len(score_events)
    maps = init_baseline_maps()
    rows = []
    event_lookup = {tuple(row[["event_date", "stream_id", "seed"]]): idx for idx, row in score_events.iterrows()}

    for _, current in transitions_full.iterrows():
        key = (current["event_date"], current["stream_id"], current["seed"])
        in_scored_window = key in event_lookup
        winner_member = normalize_member_code(current["next_member"])
        if not in_scored_window:
            add_transition_to_maps(maps, current)
            continue

        position = event_lookup[key] + 1
        if progress_bar is not None and total_events_to_process > 0:
            progress_bar.progress(position / total_events_to_process)
        if status_box is not None and (position % 10 == 0 or position == total_events_to_process):
            status_box.info(f"Processing combined walk-forward event {position:,} of {total_events_to_process:,}...")

        current_seed = pd.DataFrame([{c: current[c] for c in current.index}])
        current_skip = score_current_streams(current_seed.rename(columns={"event_date": "seed_date"}), skip_pack["history_scored_df"], skip_pack["negative_traits_df"], int(params["skip_top_negative_traits_to_use"]), float(skip_pack["chosen_cutoff"]))
        skip_class = str(current_skip.iloc[0]["skip_class"])
        skip_score = float(current_skip.iloc[0]["skip_score"])
        skip_fire_count = int(current_skip.iloc[0]["skip_fire_count"])

        if skip_class == "SKIP":
            rows.append({
                "event_id": position,
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
                "top1_win": 0, "top2_win": 0, "top3_loss": 1,
                "play_rule_hit": 0,
            })
            add_transition_to_maps(maps, current)
            continue

        ranked = rank_members_from_maps(current, maps, separator_rules, params, calibration_maps)
        play_mode = ranked["play_mode"]
        top1_win = int(play_mode == "PLAY_TOP1" and ranked["Top1"] == winner_member)
        top2_win = int(play_mode == "PLAY_TOP2" and (ranked["Top1"] == winner_member or ranked["Top2"] == winner_member))
        top3_loss = int(not (top1_win or top2_win))
        play_rule_hit = int(top1_win or top2_win)
        rows.append({
            "event_id": position,
            "event_date": current["event_date"],
            "stream": current["stream_id"],
            "seed": current["seed"],
            "winning_member": winner_member,
            "step1_skip_class": "PLAY",
            "step1_skip_score": skip_score,
            "step1_skip_fire_count": skip_fire_count,
            "survived_step1": 1,
            "Top1": ranked["Top1"], "Top2": ranked["Top2"], "Top3": ranked["Top3"],
            "play_mode": play_mode,
            "top1_win": top1_win, "top2_win": top2_win, "top3_loss": top3_loss,
            "play_rule_hit": play_rule_hit,
            **ranked,
        })
        add_transition_to_maps(maps, current)

    per_event = pd.DataFrame(rows)
    if len(per_event) == 0:
        if status_box is not None:
            status_box.warning("Combined walk-forward produced no scored events.")
        empty = pd.DataFrame()
        summary = pd.DataFrame([
            {"metric": "events", "value": 0},
            {"metric": "reason", "value": "No Core025 winner events in selected LAB window"},
        ])
        return {"per_event": empty, "per_date": empty, "per_stream": empty, "summary": summary}

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
        {"metric": "trained_skip_cutoff", "value": float(skip_pack["chosen_cutoff"])},
        {"metric": "locked_cutoff_mode", "value": int(bool(params.get("use_locked_skip_cutoff", True)))},
        {"metric": "lab_train_skip_on_core025_only", "value": int(train_on_core025_only)},
    ])
    if status_box is not None:
        status_box.success(f"Combined walk-forward completed: {len(per_event):,} scored events.")
    return {"per_event": per_event, "per_date": per_date, "per_stream": per_stream, "summary": summary}


def collect_params_from_sidebar() -> Dict[str, float]:
    with st.sidebar:
        st.markdown(f"**{BUILD_MARKER}**")
        mode = st.radio("Mode", ["Combined Daily Run", "Combined Walk-Forward LAB"], index=0)
        st.header("Step 1 Skip settings")
        skip_min_trait_support = st.number_input("Skip min trait support", min_value=1, value=12, step=1)
        skip_top_negative_traits_to_use = st.number_input("Skip top negative traits to use", min_value=1, value=15, step=1)
        skip_rung_count = st.number_input("Skip ladder rung count", min_value=5, value=50, step=5)
        skip_target_retention_pct = st.slider("Skip target hit retention", min_value=0.50, max_value=1.00, value=0.7500, step=0.0001)
        use_locked_skip_cutoff = st.checkbox("Use locked standalone skip cutoff (0.515465)", value=True)
        lab_train_skip_on_core025_only = st.checkbox("LAB: train skip on Core025-only winner transitions", value=False)

        st.header("Step 2 Member settings")
        min_stream_history = st.number_input("Minimum stream history for baseline fallback", min_value=0, value=20, step=5)
        per_rule_cap = st.slider("Per-rule cap", min_value=0.10, max_value=5.00, value=2.50, step=0.05)
        total_boost_cap = st.slider("Total boost cap per member", min_value=0.50, max_value=20.00, value=10.00, step=0.10)
        diminishing_return_factor = st.slider("Diminishing return factor", min_value=0.00, max_value=3.00, value=0.35, step=0.01)
        rule_count_norm_factor = st.slider("Rule-count normalization factor", min_value=0.00, max_value=3.00, value=1.50, step=0.01)
        max_rules_per_member = st.number_input("Max fired rules per member", min_value=1, max_value=100, value=5, step=1)
        dominant_gap_strict = st.slider("Strict dominant gap threshold", min_value=0.00, max_value=2.00, value=0.65, step=0.01)
        dominant_ratio_max_strict = st.slider("Strict dominant max ratio", min_value=0.50, max_value=1.00, value=0.65, step=0.01)
        top2_ratio_trigger = st.slider("Top2 widen ratio trigger", min_value=0.50, max_value=1.00, value=0.98, step=0.01)
        top2_gap_trigger = st.slider("Top2 widen gap trigger", min_value=0.00, max_value=2.00, value=0.06, step=0.01)
        weak_top1_score_floor = st.slider("Weak Top1 score floor", min_value=0.00, max_value=5.00, value=0.20, step=0.01)
        top1_separation_pressure = st.slider("Top1 separation pressure", min_value=0.00, max_value=0.50, value=0.12, step=0.01)
        top1_commit_gap = st.slider("Top1 commit gap", min_value=0.00, max_value=2.00, value=0.45, step=0.01)
        top1_commit_ratio_max = st.slider("Top1 commit max ratio", min_value=0.50, max_value=1.00, value=0.86, step=0.01)
        top1_confidence_min = st.slider("Top1 confidence minimum", min_value=0.00, max_value=1.00, value=0.56, step=0.01)
        top1_min_rule_gap = st.number_input("Top1 minimum rule gap", min_value=0, max_value=10, value=1, step=1)
        top1_ratio_soft_max = st.slider("Top1 soft max ratio", min_value=0.50, max_value=1.00, value=0.90, step=0.01)

        st.header("V9 member calibration")
        calibration_floor = st.slider("Calibration floor", min_value=0.50, max_value=1.00, value=0.80, step=0.01)
        calibration_cap = st.slider("Calibration cap", min_value=1.00, max_value=1.50, value=1.20, step=0.01)
        cal_0025_stream_bias_weight = st.slider("0025 stream-bias weight", min_value=-1.00, max_value=1.00, value=-0.05, step=0.01)
        cal_0025_global_bias_weight = st.slider("0025 global-bias weight", min_value=-1.00, max_value=1.00, value=-0.06, step=0.01)
        cal_0025_rule_align_weight = st.slider("0025 rule-align weight", min_value=0.00, max_value=1.00, value=0.10, step=0.01)
        cal_0025_boost_align_weight = st.slider("0025 boost-align weight", min_value=0.00, max_value=1.00, value=0.08, step=0.01)
        cal_0225_stream_bias_weight = st.slider("0225 stream-bias weight", min_value=-1.00, max_value=1.00, value=0.00, step=0.01)
        cal_0225_global_bias_weight = st.slider("0225 global-bias weight", min_value=-1.00, max_value=1.00, value=0.02, step=0.01)
        cal_0225_rule_align_weight = st.slider("0225 rule-align weight", min_value=0.00, max_value=1.00, value=0.12, step=0.01)
        cal_0225_boost_align_weight = st.slider("0225 boost-align weight", min_value=0.00, max_value=1.00, value=0.10, step=0.01)
        cal_0255_stream_bias_weight = st.slider("0255 stream-bias weight", min_value=-1.00, max_value=1.00, value=0.05, step=0.01)
        cal_0255_global_bias_weight = st.slider("0255 global-bias weight", min_value=-1.00, max_value=1.00, value=0.06, step=0.01)
        cal_0255_rule_align_weight = st.slider("0255 rule-align weight", min_value=0.00, max_value=1.00, value=0.14, step=0.01)
        cal_0255_boost_align_weight = st.slider("0255 boost-align weight", min_value=0.00, max_value=1.00, value=0.12, step=0.01)

        rows_to_show = st.number_input("Rows to display", min_value=5, value=50, step=5)
        lab_max_events = st.number_input("LAB max events (0 = all)", min_value=0, value=250, step=50)
    return locals()


def main() -> None:
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
            with st.spinner("Running Step 1 skip + Step 2 ranking..."):
                out = combined_daily_run(hist_df, separator_rules, params, last24_df)
                st.session_state["combined_daily_out_v9"] = out
        if "combined_daily_out_v9" in st.session_state:
            out = st.session_state["combined_daily_out_v9"]
            st.subheader("Step 1 current stream scoring")
            st.dataframe(out["current_scored_df"].head(rows_to_show), use_container_width=True)
            st.subheader("Generated play_survivors.csv")
            st.dataframe(out["play_survivors_df"].head(rows_to_show), use_container_width=True)
            st.subheader("Step 2 final playlist")
            st.dataframe(out["playlist_df"].head(rows_to_show), use_container_width=True)
            st.subheader("Skip cutoff used")
            st.dataframe(out["skip_chosen_cutoff_df"], use_container_width=True)
            st.download_button("Download play_survivors__2026-04-03_v9.csv", df_to_csv_bytes(out["play_survivors_df"]), file_name="play_survivors__2026-04-03_v9.csv", mime="text/csv")
            st.download_button("Download final_ranked_playlist__2026-04-03_v9.csv", df_to_csv_bytes(out["playlist_df"]), file_name="final_ranked_playlist__2026-04-03_v9.csv", mime="text/csv")
            st.download_button("Download skip_current_scored__2026-04-03_v9.csv", df_to_csv_bytes(out["current_scored_df"]), file_name="skip_current_scored__2026-04-03_v9.csv", mime="text/csv")
    else:
        if st.button("Run Combined Walk-Forward LAB", type="primary"):
            with st.spinner("Running combined no-lookahead walk-forward..."):
                progress = st.progress(0.0)
                status_box = st.empty()
                out = combined_walkforward_lab(hist_df, separator_rules, params, progress_bar=progress, status_box=status_box)
                progress.empty()
                st.session_state["combined_lab_out_v9"] = out
        if "combined_lab_out_v9" in st.session_state:
            out = st.session_state["combined_lab_out_v9"]
            st.subheader("Combined summary")
            st.dataframe(out["summary"], use_container_width=True)
            st.subheader("Per-event")
            st.dataframe(out["per_event"].head(rows_to_show), use_container_width=True)
            st.subheader("Per-date")
            st.dataframe(out["per_date"].head(rows_to_show), use_container_width=True)
            st.subheader("Per-stream")
            st.dataframe(out["per_stream"].head(rows_to_show), use_container_width=True)
            st.download_button("Download combined_lab_per_event__2026-04-03_v9.csv", df_to_csv_bytes(out["per_event"]), file_name="combined_lab_per_event__2026-04-03_v9.csv", mime="text/csv")
            st.download_button("Download combined_lab_per_date__2026-04-03_v9.csv", df_to_csv_bytes(out["per_date"]), file_name="combined_lab_per_date__2026-04-03_v9.csv", mime="text/csv")
            st.download_button("Download combined_lab_per_stream__2026-04-03_v9.csv", df_to_csv_bytes(out["per_stream"]), file_name="combined_lab_per_stream__2026-04-03_v9.csv", mime="text/csv")
            st.download_button("Download combined_lab_summary__2026-04-03_v9.csv", df_to_csv_bytes(out["summary"]), file_name="combined_lab_summary__2026-04-03_v9.csv", mime="text/csv")


if __name__ == "__main__":
    main()

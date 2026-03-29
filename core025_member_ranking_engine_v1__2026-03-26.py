#!/usr/bin/env python3
# core025_deep_separator_library_builder_v1__2026-03-28.py
#
# BUILD: core025_deep_separator_library_builder_v1__2026-03-28
#
# Full file. No placeholders.
#
# Purpose
# -------
# Build a deep separator library for Core025 using broad-to-specific stacked traits.
#
# This builder is meant to stop the repeated shallow-mine / retest / remine cycle by:
# 1. generating a large master library of candidate separators
# 2. scoring them automatically
# 3. promoting the strongest broad/stable separators
# 4. showing which winners remain uncovered
#
# Search strategy
# ---------------
# - mines single traits
# - expands promising singles into 2-trait stacks
# - expands promising 2-trait stacks into 3-trait stacks
# - keeps the full library
# - exports promoted subsets and uncovered cases
#
# Outputs
# -------
# - core025_deep_separator_library_builder_v1__2026-03-28__master_library.csv
# - core025_deep_separator_library_builder_v1__2026-03-28__promoted_library.csv
# - core025_deep_separator_library_builder_v1__2026-03-28__uncovered_winners.csv
# - core025_deep_separator_library_builder_v1__2026-03-28__pair_summary.csv

from __future__ import annotations

import io
import re
from collections import Counter
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

CORE025 = ["0025", "0225", "0255"]
BUILD_MARKER = "BUILD: core025_deep_separator_library_builder_v1__2026-03-28"


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
                "next_member": next_member,
                "is_core025_hit": int(next_member is not None),
                **feat,
            })
    out = pd.DataFrame(rows)
    return out.sort_values(["event_date", "stream", "seed"]).reset_index(drop=True)


def stability_stats(sub: pd.DataFrame) -> Dict[str, int]:
    ym_col = next((c for c in ["year_month", "year_month_x", "year_month_y"] if c in sub.columns), None)
    if ym_col is None:
        raise KeyError("No year_month column found in subset.")
    return {
        "stream_count": int(sub["stream"].nunique()),
        "month_count": int(sub[ym_col].nunique()),
    }


def mask_for_conditions(df: pd.DataFrame, conditions: Sequence[Tuple[str, str]]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, val in conditions:
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        cur = df[col].map(lambda x: "" if pd.isna(x) else (str(int(x)) if isinstance(x, float) and float(x).is_integer() else str(x)))
        mask &= (cur == val)
    return mask


def condition_str(conditions: Sequence[Tuple[str, str]]) -> str:
    return " && ".join([f"{c}={v}" for c, v in conditions])


def score_candidate(
    pair_df: pd.DataFrame,
    left_member: str,
    right_member: str,
    conditions: Sequence[Tuple[str, str]],
) -> Optional[Dict[str, object]]:
    sub = pair_df[mask_for_conditions(pair_df, conditions)].copy()
    support = len(sub)
    if support == 0:
        return None

    left_count = int((sub["next_member"] == left_member).sum())
    right_count = int((sub["next_member"] == right_member).sum())
    left_rate = left_count / support
    right_rate = right_count / support
    winner = left_member if left_rate >= right_rate else right_member
    loser = right_member if winner == left_member else left_member
    winner_rate = max(left_rate, right_rate)
    loser_rate = min(left_rate, right_rate)
    gap = winner_rate - loser_rate
    stable = stability_stats(sub)
    covered_keys = sorted((sub["event_date"].astype(str) + "|" + sub["stream"].astype(str)).unique().tolist())

    return {
        "pair": f"{left_member}_vs_{right_member}",
        "trait_stack": condition_str(conditions),
        "stack_size": len(conditions),
        "support": support,
        "winner_member": winner,
        "loser_member": loser,
        "winner_rate": winner_rate,
        "loser_rate": loser_rate,
        "pair_gap": gap,
        "stream_count": stable["stream_count"],
        "month_count": stable["month_count"],
        f"rate_{left_member}": left_rate,
        f"rate_{right_member}": right_rate,
        "left_count": left_count,
        "right_count": right_count,
        "covered_keys": covered_keys,
    }


def candidate_priority(row: Dict[str, object]) -> Tuple[float, float, int, int, int]:
    # broader/stabler/stronger first
    return (
        float(row["winner_rate"]),
        float(row["pair_gap"]),
        int(row["support"]),
        int(row["stream_count"]),
        int(row["month_count"]),
    )


def build_base_candidates(pair_df: pd.DataFrame) -> List[Tuple[str, str]]:
    candidates: List[Tuple[str, str]] = []
    for col in miner_feature_columns():
        if col not in pair_df.columns:
            continue
        vals = pair_df[col].dropna().unique().tolist()
        try:
            vals = sorted(vals)
        except Exception:
            pass
        for val in vals:
            candidates.append((col, str(int(val)) if isinstance(val, float) and float(val).is_integer() else str(val)))
    return candidates


def overlaps_too_much(a_keys: set[str], b_keys: set[str], max_jaccard: float) -> bool:
    if not a_keys or not b_keys:
        return False
    inter = len(a_keys & b_keys)
    union = len(a_keys | b_keys)
    if union == 0:
        return False
    return (inter / union) > max_jaccard


def promote_rules(
    master_df: pd.DataFrame,
    min_promoted_support: int,
    min_promoted_winner_rate: float,
    min_promoted_gap: float,
    min_promoted_streams: int,
    min_promoted_months: int,
    max_jaccard_overlap: float,
) -> pd.DataFrame:
    if len(master_df) == 0:
        return master_df.copy()

    work = master_df.copy()
    work = work[
        (work["support"] >= int(min_promoted_support)) &
        (work["winner_rate"] >= float(min_promoted_winner_rate)) &
        (work["pair_gap"] >= float(min_promoted_gap)) &
        (work["stream_count"] >= int(min_promoted_streams)) &
        (work["month_count"] >= int(min_promoted_months))
    ].copy()

    if len(work) == 0:
        return work

    work = work.sort_values(
        ["winner_rate", "pair_gap", "support", "stream_count", "month_count", "stack_size"],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)

    promoted_rows = []
    kept_key_sets: List[set[str]] = []
    for _, r in work.iterrows():
        key_set = set(str(r["covered_keys"]).split("||")) if "||" in str(r["covered_keys"]) else set(eval(r["covered_keys"])) if str(r["covered_keys"]).startswith("[") else set()
        # safer parse fallback
        if not key_set:
            try:
                import ast
                key_set = set(ast.literal_eval(r["covered_keys"]))
            except Exception:
                key_set = set()
        redundant = any(overlaps_too_much(key_set, prior, float(max_jaccard_overlap)) for prior in kept_key_sets)
        if redundant:
            continue
        promoted_rows.append(r.to_dict())
        kept_key_sets.append(key_set)

    return pd.DataFrame(promoted_rows)


def uncovered_winners(core_hits: pd.DataFrame, promoted_df: pd.DataFrame) -> pd.DataFrame:
    all_hits = core_hits.copy()
    all_hits["event_key"] = all_hits["event_date"].astype(str) + "|" + all_hits["stream"].astype(str)

    covered = set()
    if len(promoted_df):
        import ast
        for _, r in promoted_df.iterrows():
            try:
                keys = ast.literal_eval(r["covered_keys"])
                covered.update(keys)
            except Exception:
                pass

    out = all_hits[~all_hits["event_key"].isin(covered)].copy()
    cols = ["event_date", "stream", "seed", "next_member", "sorted_seed", "pair_token_pattern", "parity_pattern", "highlow_pattern", "structure", "sum_bucket", "spread_bucket"]
    keep_cols = [c for c in cols if c in out.columns]
    return out[keep_cols].reset_index(drop=True)


def deep_mine_pair(
    core_hits: pd.DataFrame,
    left_member: str,
    right_member: str,
    min_single_support: int,
    min_single_winner_rate: float,
    min_single_gap: float,
    min_stack_support: int,
    min_stack_winner_rate: float,
    min_stack_gap: float,
    min_streams: int,
    min_months: int,
    beam_width: int,
    max_stack_size: int,
) -> pd.DataFrame:
    pair_df = core_hits[core_hits["next_member"].isin([left_member, right_member])].copy()
    base_candidates = build_base_candidates(pair_df)

    accepted_by_level: List[List[Tuple[Tuple[str, str], ...]]] = []
    master_rows: List[Dict[str, object]] = []

    # Level 1
    level1 = []
    for cond in base_candidates:
        row = score_candidate(pair_df, left_member, right_member, [cond])
        if row is None:
            continue
        if row["support"] < int(min_single_support):
            continue
        if row["winner_rate"] < float(min_single_winner_rate):
            continue
        if row["pair_gap"] < float(min_single_gap):
            continue
        if row["stream_count"] < int(min_streams):
            continue
        if row["month_count"] < int(min_months):
            continue
        master_rows.append(row)
        level1.append(((cond[0], cond[1]),))

    level1_scored = []
    for conds in level1:
        row = score_candidate(pair_df, left_member, right_member, list(conds))
        if row is not None:
            level1_scored.append((candidate_priority(row), conds))
    level1_scored = sorted(level1_scored, reverse=True)[: int(beam_width)]
    accepted_by_level.append([conds for _, conds in level1_scored])

    # Levels 2..max_stack_size
    for stack_size in range(2, int(max_stack_size) + 1):
        prev_level = accepted_by_level[-1]
        if not prev_level:
            break

        next_candidates = set()
        for conds in prev_level:
            used_cols = {c for c, _ in conds}
            for extra in base_candidates:
                if extra[0] in used_cols:
                    continue
                merged = tuple(sorted(list(conds) + [extra], key=lambda x: x[0]))
                next_candidates.add(merged)

        scored_next = []
        for conds in next_candidates:
            row = score_candidate(pair_df, left_member, right_member, list(conds))
            if row is None:
                continue
            if row["support"] < int(min_stack_support):
                continue
            if row["winner_rate"] < float(min_stack_winner_rate):
                continue
            if row["pair_gap"] < float(min_stack_gap):
                continue
            if row["stream_count"] < int(min_streams):
                continue
            if row["month_count"] < int(min_months):
                continue
            master_rows.append(row)
            scored_next.append((candidate_priority(row), conds))

        scored_next = sorted(scored_next, reverse=True)[: int(beam_width)]
        accepted_by_level.append([conds for _, conds in scored_next])

    out = pd.DataFrame(master_rows)
    if len(out):
        out["covered_keys"] = out["covered_keys"].apply(lambda x: repr(x))
        out = out.sort_values(
            ["winner_rate", "pair_gap", "support", "stream_count", "month_count", "stack_size"],
            ascending=[False, False, False, False, False, True],
        ).reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=[
            "pair", "trait_stack", "stack_size", "support", "winner_member", "loser_member",
            "winner_rate", "loser_rate", "pair_gap", "stream_count", "month_count",
            f"rate_{left_member}", f"rate_{right_member}", "left_count", "right_count", "covered_keys"
        ])
    return out


def pair_summary(master_df: pd.DataFrame, promoted_df: pd.DataFrame) -> pd.DataFrame:
    if len(master_df) == 0:
        return pd.DataFrame(columns=["pair", "master_rules", "promoted_rules"])
    ms = master_df.groupby("pair").size().rename("master_rules")
    if len(promoted_df):
        ps = promoted_df.groupby("pair").size().rename("promoted_rules")
        out = pd.concat([ms, ps], axis=1).fillna(0).reset_index()
    else:
        out = ms.reset_index()
        out["promoted_rules"] = 0
    out["master_rules"] = out["master_rules"].astype(int)
    out["promoted_rules"] = out["promoted_rules"].astype(int)
    return out.sort_values("pair").reset_index(drop=True)


def main():
    st.set_page_config(page_title="Core025 Deep Separator Library Builder v1", layout="wide")
    st.title("Core025 Deep Separator Library Builder v1")
    st.caption("Over-mine broad-to-specific separator stacks, keep the whole library, and promote only the strongest stable rules.")
    st.code(BUILD_MARKER, language="text")

    with st.sidebar:
        st.markdown(f"**{BUILD_MARKER}**")
        st.header("Search depth")
        beam_width = st.number_input("Beam width per level", min_value=10, value=120, step=10)
        max_stack_size = st.number_input("Maximum stack size", min_value=1, max_value=3, value=3, step=1)

        st.header("Single-trait gate")
        min_single_support = st.number_input("Minimum single support", min_value=5, value=12, step=1)
        min_single_winner_rate = st.slider("Minimum single winner rate", min_value=0.50, max_value=0.95, value=0.58, step=0.01)
        min_single_gap = st.slider("Minimum single pair gap", min_value=0.00, max_value=0.50, value=0.06, step=0.01)

        st.header("Stack gate")
        min_stack_support = st.number_input("Minimum stacked support", min_value=3, value=8, step=1)
        min_stack_winner_rate = st.slider("Minimum stacked winner rate", min_value=0.50, max_value=0.99, value=0.67, step=0.01)
        min_stack_gap = st.slider("Minimum stacked pair gap", min_value=0.00, max_value=0.80, value=0.18, step=0.01)

        st.header("Stability gate")
        min_streams = st.number_input("Minimum distinct streams", min_value=1, value=2, step=1)
        min_months = st.number_input("Minimum distinct months", min_value=1, value=2, step=1)

        st.header("Promotion filter")
        min_promoted_support = st.number_input("Minimum promoted support", min_value=3, value=10, step=1)
        min_promoted_winner_rate = st.slider("Minimum promoted winner rate", min_value=0.50, max_value=0.99, value=0.68, step=0.01)
        min_promoted_gap = st.slider("Minimum promoted gap", min_value=0.00, max_value=0.80, value=0.20, step=0.01)
        min_promoted_streams = st.number_input("Minimum promoted streams", min_value=1, value=2, step=1)
        min_promoted_months = st.number_input("Minimum promoted months", min_value=1, value=2, step=1)
        max_jaccard_overlap = st.slider("Maximum overlap between promoted rules", min_value=0.10, max_value=1.00, value=0.85, step=0.01)

        rows_to_show = st.number_input("Rows to display", min_value=5, value=30, step=5)

    hist_file = st.file_uploader("Upload FULL history file", key="deep_sep_hist")
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

    if st.button("Run Deep Separator Library Builder", type="primary"):
        try:
            with st.spinner("Building deep separator library..."):
                p1 = deep_mine_pair(
                    core_hits=core_hits,
                    left_member="0025",
                    right_member="0225",
                    min_single_support=int(min_single_support),
                    min_single_winner_rate=float(min_single_winner_rate),
                    min_single_gap=float(min_single_gap),
                    min_stack_support=int(min_stack_support),
                    min_stack_winner_rate=float(min_stack_winner_rate),
                    min_stack_gap=float(min_stack_gap),
                    min_streams=int(min_streams),
                    min_months=int(min_months),
                    beam_width=int(beam_width),
                    max_stack_size=int(max_stack_size),
                )
                p2 = deep_mine_pair(
                    core_hits=core_hits,
                    left_member="0225",
                    right_member="0255",
                    min_single_support=int(min_single_support),
                    min_single_winner_rate=float(min_single_winner_rate),
                    min_single_gap=float(min_single_gap),
                    min_stack_support=int(min_stack_support),
                    min_stack_winner_rate=float(min_stack_winner_rate),
                    min_stack_gap=float(min_stack_gap),
                    min_streams=int(min_streams),
                    min_months=int(min_months),
                    beam_width=int(beam_width),
                    max_stack_size=int(max_stack_size),
                )
                p3 = deep_mine_pair(
                    core_hits=core_hits,
                    left_member="0025",
                    right_member="0255",
                    min_single_support=int(min_single_support),
                    min_single_winner_rate=float(min_single_winner_rate),
                    min_single_gap=float(min_single_gap),
                    min_stack_support=int(min_stack_support),
                    min_stack_winner_rate=float(min_stack_winner_rate),
                    min_stack_gap=float(min_stack_gap),
                    min_streams=int(min_streams),
                    min_months=int(min_months),
                    beam_width=int(beam_width),
                    max_stack_size=int(max_stack_size),
                )

                master_library = pd.concat([p1, p2, p3], ignore_index=True) if (len(p1) or len(p2) or len(p3)) else pd.DataFrame()

                promoted_library = promote_rules(
                    master_df=master_library,
                    min_promoted_support=int(min_promoted_support),
                    min_promoted_winner_rate=float(min_promoted_winner_rate),
                    min_promoted_gap=float(min_promoted_gap),
                    min_promoted_streams=int(min_promoted_streams),
                    min_promoted_months=int(min_promoted_months),
                    max_jaccard_overlap=float(max_jaccard_overlap),
                )

                uncovered = uncovered_winners(core_hits=core_hits, promoted_df=promoted_library)
                pair_sum = pair_summary(master_library, promoted_library)

            st.session_state["deep_sep_results"] = {
                "master_library": master_library,
                "promoted_library": promoted_library,
                "uncovered_winners": uncovered,
                "pair_summary": pair_sum,
            }
        except Exception as e:
            st.exception(e)
            return

    if "deep_sep_results" not in st.session_state or st.session_state["deep_sep_results"] is None:
        return

    results = st.session_state["deep_sep_results"]

    st.subheader("Pair summary")
    st.dataframe(results["pair_summary"], use_container_width=True)
    st.download_button(
        "Download pair summary CSV",
        df_to_csv_bytes(results["pair_summary"]),
        "core025_deep_separator_library_builder_v1__2026-03-28__pair_summary.csv",
        "text/csv",
    )

    st.subheader("Promoted separator library")
    st.dataframe(safe_display_df(results["promoted_library"], int(rows_to_show)), use_container_width=True)
    st.download_button(
        "Download promoted separator library CSV",
        df_to_csv_bytes(results["promoted_library"]),
        "core025_deep_separator_library_builder_v1__2026-03-28__promoted_library.csv",
        "text/csv",
    )

    st.subheader("Master separator library")
    st.dataframe(safe_display_df(results["master_library"], int(rows_to_show)), use_container_width=True)
    st.download_button(
        "Download master separator library CSV",
        df_to_csv_bytes(results["master_library"]),
        "core025_deep_separator_library_builder_v1__2026-03-28__master_library.csv",
        "text/csv",
    )

    st.subheader("Uncovered winners after promotion")
    st.dataframe(safe_display_df(results["uncovered_winners"], int(rows_to_show)), use_container_width=True)
    st.download_button(
        "Download uncovered winners CSV",
        df_to_csv_bytes(results["uncovered_winners"]),
        "core025_deep_separator_library_builder_v1__2026-03-28__uncovered_winners.csv",
        "text/csv",
    )


if __name__ == "__main__":
    if "deep_sep_results" not in st.session_state:
        st.session_state["deep_sep_results"] = None
    main()

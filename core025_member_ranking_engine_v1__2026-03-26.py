#!/usr/bin/env python3
# core025_separator_ranking_engine_v1_debugfix3__2026-03-29.py
#
# BUILD: core025_separator_ranking_engine_v1_debugfix3__2026-03-29
#
# Full file. No placeholders.
#
# Purpose
# -------
# Debug-fixed separator ranking engine for Core025.
#
# What this version fixes
# -----------------------
# - robust separator library loading
# - exact normalization for trait values before matching
# - winner-member normalization ("25" -> "0025", etc.)
# - output ranking normalization so Top1/Top2/Top3 always use:
#   0025 / 0225 / 0255
# - rule-level diagnostics
# - near-match diagnostics
# - summary tables for:
#   * rules loaded
#   * rows with fired rules
#   * top failed columns
#   * top near-miss rules
#
# Outputs
# -------
# - core025_separator_ranked_playlist_v1_debugfix3__2026-03-29.csv
# - core025_separator_rule_debug_summary_v1_debugfix3__2026-03-29.csv
# - core025_separator_rule_nearmiss_v1_debugfix3__2026-03-29.csv

from __future__ import annotations

import io
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

CORE025 = ["0025", "0225", "0255"]
BUILD_MARKER = "BUILD: core025_separator_ranking_engine_v1_debugfix3__2026-03-29"


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
    if pd.isna(raw):
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
    out = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
    return out


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
    out = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
    return out


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
                "seed": seed,
                "seed_date": g.loc[i - 1, "date"],
                "transition_date": g.loc[i, "date"],
                "next_member": next_member,
                **feat,
            })
    return pd.DataFrame(rows).sort_values(["transition_date", "stream", "seed"]).reset_index(drop=True)


def counter_to_probs(c: Counter) -> Dict[str, float]:
    total = sum(c.values())
    if total <= 0:
        return {m: 1 / 3 for m in CORE025}
    return {m: c.get(m, 0) / total for m in CORE025}


def build_baseline_maps(transitions: pd.DataFrame):
    exact_seed_map = defaultdict(Counter)
    sorted_seed_map = defaultdict(Counter)
    stream_member_map = defaultdict(Counter)
    global_member_map = Counter()
    for _, r in transitions.iterrows():
        member = r["next_member"]
        if member is None or pd.isna(member):
            continue
        exact_seed_map[str(r["seed"])][member] += 1
        sorted_seed_map[str(r["sorted_seed"])][member] += 1
        stream_member_map[str(r["stream"])][member] += 1
        global_member_map[member] += 1
    return exact_seed_map, sorted_seed_map, stream_member_map, global_member_map


def baseline_scores(seed_row: pd.Series, transitions: pd.DataFrame, min_stream_history: int = 20) -> Dict[str, float]:
    exact_seed_map, sorted_seed_map, stream_member_map, global_member_map = build_baseline_maps(transitions)
    seed = str(seed_row["seed"])
    stream = str(seed_row["stream"])
    score_accum = {m: 0.0 for m in CORE025}

    global_probs = counter_to_probs(global_member_map)
    for m in CORE025:
        score_accum[m] += global_probs[m] * 0.25

    if stream is not None and sum(stream_member_map[str(stream)].values()) >= int(min_stream_history):
        stream_probs = counter_to_probs(stream_member_map[str(stream)])
        for m in CORE025:
            score_accum[m] += stream_probs[m] * 1.20

    if seed in exact_seed_map and sum(exact_seed_map[seed].values()) > 0:
        exact_probs = counter_to_probs(exact_seed_map[seed])
        for m in CORE025:
            score_accum[m] += exact_probs[m] * 1.50

    sorted_key = str(seed_row["sorted_seed"])
    if sorted_key in sorted_seed_map and sum(sorted_seed_map[sorted_key].values()) > 0:
        sorted_probs = counter_to_probs(sorted_seed_map[sorted_key])
        for m in CORE025:
            score_accum[m] += sorted_probs[m] * 1.10

    total = sum(score_accum.values())
    if total <= 0:
        return {m: 1 / 3 for m in CORE025}
    return {m: score_accum[m] / total for m in CORE025}


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
    skipped_invalid_winner = 0
    for idx, r in df.iterrows():
        stack = parse_trait_stack(str(r["trait_stack"]))
        if not stack:
            continue
        winner_norm = normalize_member_code(r["winner_member"])
        if winner_norm is None:
            skipped_invalid_winner += 1
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
    st.session_state["separator_rules_loaded_count"] = len(rules)
    st.session_state["separator_rules_skipped_invalid_winner"] = skipped_invalid_winner
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


def apply_separator_rules(row: pd.Series, rules: List[Dict[str, object]]) -> Tuple[Dict[str, float], List[str], List[Dict[str, object]], Counter]:
    boosts = {m: 0.0 for m in CORE025}
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
            support_bonus = min(rule["support"], 50) / 100.0
            stack_bonus = 0.03 * max(rule["stack_size"] - 1, 0)
            score = (rule["winner_rate"] * 0.60) + (rule["pair_gap"] * 0.90) + support_bonus + stack_bonus
            boosts[winner] += score
            fired_rules.append(
                f"RID{rule['rule_id']} | {rule['pair']} | {rule['trait_stack']} | winner={winner} | wr={rule['winner_rate']:.3f} | gap={rule['pair_gap']:.3f} | sup={rule['support']}"
            )
        else:
            for fc in failed_cols:
                fail_counter[fc.split(":")[0]] += 1
            if matched > 0:
                near_misses.append({
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
                })

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

    return boosts, fired_rules, near_misses, fail_counter


def rank_members(row: pd.Series, transitions: pd.DataFrame, separator_rules: List[Dict[str, object]], min_stream_history: int) -> Dict[str, object]:
    base = baseline_scores(row, transitions, min_stream_history=int(min_stream_history))
    boosts, fired, near_misses, fail_counter = apply_separator_rules(row, separator_rules)

    final_scores = {m: base[m] + boosts[m] for m in CORE025}

    # Safety normalization before ranking so outputs always use 0025 / 0225 / 0255
    normalized_scores: Dict[str, float] = {}
    for k, v in final_scores.items():
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
        "boost_0025": boosts["0025"],
        "boost_0225": boosts["0225"],
        "boost_0255": boosts["0255"],
        "final_0025": normalized_scores["0025"],
        "final_0225": normalized_scores["0225"],
        "final_0255": normalized_scores["0255"],
        "Top1": top1,
        "Top1_score": top1_score,
        "Top2": top2,
        "Top2_score": top2_score,
        "Top3": top3,
        "Top3_score": top3_score,
        "Top1_minus_Top2": top1_score - top2_score,
        "fired_rule_count": len(fired),
        "fired_rules": " || ".join(fired[:25]),
        "near_miss_rule_count": len(near_misses),
        "near_miss_rules": near_text,
        "top_failed_columns": fail_top,
    }


def main():
    st.set_page_config(page_title="Core025 Separator Ranking Engine v1 DebugFix3", layout="wide")
    st.title("Core025 Separator Ranking Engine v1 DebugFix3")
    st.caption("Patched separator engine with winner normalization, output normalization, and exact matcher diagnostics.")
    st.code(BUILD_MARKER, language="text")

    with st.sidebar:
        st.markdown(f"**{BUILD_MARKER}**")
        min_stream_history = st.number_input("Minimum stream history for baseline fallback", min_value=0, value=20, step=5)
        rows_to_show = st.number_input("Rows to display", min_value=5, value=50, step=5)

    hist_file = st.file_uploader("Upload FULL history file", key="sep_rank_hist_dbg3")
    surv_file = st.file_uploader("Upload PLAY survivors file", key="sep_rank_surv_dbg3")
    sep_library_file = st.file_uploader("Upload promoted separator library CSV", key="sep_rank_lib_dbg3")

    if not all([hist_file, surv_file, sep_library_file]):
        st.info("Upload the full history file, play survivors file, and promoted separator library CSV.")
        return

    try:
        hist = prep_history(load_table(hist_file))
        surv = prep_survivors(load_table(surv_file))
        sep_lib_df = load_table(sep_library_file)
        separator_rules = load_separator_library(sep_lib_df)
    except Exception as e:
        st.exception(e)
        return

    transitions = build_transitions(hist)

    rows = []
    progress = st.progress(0.0)
    total = len(surv)

    for i, (_, row) in enumerate(surv.iterrows(), start=1):
        ranked = rank_members(row, transitions, separator_rules, min_stream_history=int(min_stream_history))
        rows.append({
            "stream": row["stream"],
            "seed": row["seed"],
            **ranked,
        })
        progress.progress(i / total if total else 1.0)

    progress.empty()

    out = pd.DataFrame(rows).sort_values(
        ["fired_rule_count", "Top1_score", "Top1_minus_Top2", "near_miss_rule_count"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    debug_summary = pd.DataFrame([
        {"metric": "separator_rules_loaded", "value": len(separator_rules)},
        {"metric": "separator_rules_skipped_invalid_winner", "value": int(st.session_state.get("separator_rules_skipped_invalid_winner", 0))},
        {"metric": "playlist_rows", "value": len(out)},
        {"metric": "rows_with_fired_rules", "value": int((out["fired_rule_count"] > 0).sum())},
        {"metric": "rows_with_near_misses", "value": int((out["near_miss_rule_count"] > 0).sum())},
        {"metric": "max_fired_rule_count", "value": int(out["fired_rule_count"].max() if len(out) else 0)},
        {"metric": "max_near_miss_rule_count", "value": int(out["near_miss_rule_count"].max() if len(out) else 0)},
    ])

    nearmiss = out[["stream", "seed", "fired_rule_count", "near_miss_rule_count", "near_miss_rules", "top_failed_columns"]].copy()

    st.subheader("Debug summary")
    st.dataframe(debug_summary, use_container_width=True)

    st.subheader("Separator-ranked playlist preview")
    st.dataframe(out.head(int(rows_to_show)), use_container_width=True)

    st.subheader("Rows with fired rules")
    st.dataframe(out[out["fired_rule_count"] > 0].head(int(rows_to_show)), use_container_width=True)

    st.subheader("Rows with near misses")
    st.dataframe(nearmiss[nearmiss["near_miss_rule_count"] > 0].head(int(rows_to_show)), use_container_width=True)

    st.download_button(
        "Download core025_separator_ranked_playlist_v1_debugfix3__2026-03-29.csv",
        data=out.to_csv(index=False),
        file_name="core025_separator_ranked_playlist_v1_debugfix3__2026-03-29.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download core025_separator_rule_debug_summary_v1_debugfix3__2026-03-29.csv",
        data=debug_summary.to_csv(index=False),
        file_name="core025_separator_rule_debug_summary_v1_debugfix3__2026-03-29.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download core025_separator_rule_nearmiss_v1_debugfix3__2026-03-29.csv",
        data=nearmiss.to_csv(index=False),
        file_name="core025_separator_rule_nearmiss_v1_debugfix3__2026-03-29.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# core025_trait_driven_member_engine_v2_4_hybrid_no_member_force__2026-03-28.py
#
# BUILD: core025_trait_driven_member_engine_v2_4_hybrid_no_member_force__2026-03-28
#
# Full file. No placeholders.
#
# Purpose
# -------
# Hybrid Step 5 engine with member forcing removed.
#
# Key change
# ----------
# - Member-specific and separation traits are recorded for diagnostics only.
# - They do NOT force Top1 anymore.
# - Top2-needed and skip-danger still qualify buckets with score confirmation.
# - Scoring decides Top1 / Top2 ordering.

from __future__ import annotations

import io
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

CORE025 = ["0025", "0225", "0255"]
BUILD_MARKER = "BUILD: core025_trait_driven_member_engine_v2_4_hybrid_no_member_force__2026-03-28"


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
    digs = [int(x) for x in d[:4]]
    cnt = Counter(digs)
    pair_tokens = []
    for i in range(4):
        for j in range(i + 1, 4):
            pair_tokens.append("".join(sorted([str(digs[i]), str(digs[j])])))
    unique_sorted = sorted(set(digs))
    consec_links = 0
    for a, b in zip(unique_sorted[:-1], unique_sorted[1:]):
        if b - a == 1:
            consec_links += 1
    feat = {
        "sum": sum(digs),
        "spread": max(digs) - min(digs),
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
        "pair_token_pattern": "|".join(sorted(pair_tokens)),
    }
    for k in range(10):
        feat[f"has{k}"] = int(k in cnt)
        feat[f"cnt{k}"] = int(cnt.get(k, 0))
    return feat


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


def similarity(a: Dict[str, object], b: pd.Series) -> float:
    score = 0.0
    if a["sorted_seed"] == b["sorted_seed"]:
        score += 6
    if a["pair_token_pattern"] == b["pair_token_pattern"]:
        score += 4
    if a["parity_pattern"] == b["parity_pattern"]:
        score += 2
    if a["highlow_pattern"] == b["highlow_pattern"]:
        score += 2
    if a["unique"] == b["unique"]:
        score += 1
    if a["max_rep"] == b["max_rep"]:
        score += 1
    return float(score)


def counter_to_probs(c: Counter) -> Dict[str, float]:
    total = sum(c.values())
    if total <= 0:
        return {m: 1 / 3 for m in CORE025}
    return {m: c.get(m, 0) / total for m in CORE025}


def build_transition_maps(transitions: pd.DataFrame) -> Dict[str, object]:
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
    return {
        "exact_seed_map": exact_seed_map,
        "sorted_seed_map": sorted_seed_map,
        "stream_member_map": stream_member_map,
        "global_member_map": global_member_map,
    }


def score_seed_v3(seed: str, stream: str, transitions: pd.DataFrame, maps: Dict[str, object],
                  min_stream_history: int, stream_bias_weight: float, exact_seed_weight: float,
                  sorted_seed_weight: float, similarity_weight: float) -> List[Tuple[str, float]]:
    seed_feat = features(seed)
    if seed_feat is None:
        return [(m, 1 / 3) for m in CORE025]

    exact_seed_map = maps["exact_seed_map"]
    sorted_seed_map = maps["sorted_seed_map"]
    stream_member_map = maps["stream_member_map"]
    global_member_map = maps["global_member_map"]

    score_accum = {m: 0.0 for m in CORE025}
    global_probs = counter_to_probs(global_member_map)
    for m in CORE025:
        score_accum[m] += global_probs[m] * 0.25

    if stream is not None and sum(stream_member_map[str(stream)].values()) >= int(min_stream_history):
        stream_probs = counter_to_probs(stream_member_map[str(stream)])
        for m in CORE025:
            score_accum[m] += stream_probs[m] * float(stream_bias_weight)

    if str(seed) in exact_seed_map and sum(exact_seed_map[str(seed)].values()) > 0:
        exact_probs = counter_to_probs(exact_seed_map[str(seed)])
        for m in CORE025:
            score_accum[m] += exact_probs[m] * float(exact_seed_weight)

    sorted_key = str(seed_feat["sorted_seed"])
    if sorted_key in sorted_seed_map and sum(sorted_seed_map[sorted_key].values()) > 0:
        sorted_probs = counter_to_probs(sorted_seed_map[sorted_key])
        for m in CORE025:
            score_accum[m] += sorted_probs[m] * float(sorted_seed_weight)

    pool = transitions.copy()
    if stream is not None:
        stream_subset = transitions[transitions["stream"] == str(stream)].copy()
        if len(stream_subset) >= int(min_stream_history):
            pool = stream_subset

    if len(pool):
        pool = pool.sort_values("transition_date").reset_index(drop=True)
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
                score_accum[m] += (sim_scores[m] / total_sim) * float(similarity_weight)

    total = sum(score_accum.values())
    if total <= 0:
        return [(m, 1 / 3) for m in CORE025]
    probs = {m: score_accum[m] / total for m in CORE025}
    return sorted(probs.items(), key=lambda x: x[1], reverse=True)


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


def apply_engine(
    surv: pd.DataFrame, transitions: pd.DataFrame, maps: Dict[str, object],
    sep_traits: pd.DataFrame, top2_needed_traits: pd.DataFrame, skip_danger_traits: pd.DataFrame,
    traits_0025: Optional[pd.DataFrame], traits_0225: Optional[pd.DataFrame], traits_0255: Optional[pd.DataFrame],
    min_stream_history: int, stream_bias_weight: float, exact_seed_weight: float,
    sorted_seed_weight: float, similarity_weight: float, top1_only_threshold: float,
    top2_threshold: float, weak_skip_threshold: float, skip_danger_min_rate: float,
    top2_needed_min_rate: float, top2_gap_max: float, skip_score_max: float
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in surv.iterrows():
        seed = str(row["seed"])
        stream = str(row["stream"])

        ranked = score_seed_v3(
            seed=seed,
            stream=stream,
            transitions=transitions,
            maps=maps,
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

        matched_sep = matched_traits_for_row(row, sep_traits)
        matched_top2 = matched_traits_for_row(row, top2_needed_traits)
        matched_skip = matched_traits_for_row(row, skip_danger_traits)
        matched_0025 = matched_traits_for_row(row, traits_0025) if traits_0025 is not None else pd.DataFrame()
        matched_0225 = matched_traits_for_row(row, traits_0225) if traits_0225 is not None else pd.DataFrame()
        matched_0255 = matched_traits_for_row(row, traits_0255) if traits_0255 is not None else pd.DataFrame()

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
                decision_reason = f"Top2 trait + tight gap: {best_top2_trait}"

        rows.append({
            "stream": stream,
            "seed": seed,
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
            "matched_sep_traits": len(matched_sep),
            "matched_member_specific_traits": int(len(matched_0025) + len(matched_0225) + len(matched_0255)),
            "matched_top2_needed_traits": len(matched_top2),
            "best_top2_trait": best_top2_trait,
            "best_top2_rate": best_top2_rate,
            "top2_gap_gate_passed": top2_gap_gate_passed,
            "matched_skip_danger_traits": len(matched_skip),
            "best_skip_trait": best_skip_trait,
            "best_skip_rate": best_skip_rate,
            "skip_score_gate_passed": skip_score_gate_passed,
            "play_top1": int(recommendation in ["Top1 only", "Top1 + Top2"]),
            "play_top2": int(recommendation == "Top1 + Top2"),
            "play_top3": 0,
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["play_top1", "play_top2", "Top1_score", "Top1_minus_Top2"],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)
    return out


def main():
    st.set_page_config(page_title="Core025 Trait-Driven Member Engine v2.4 (No Member Force)", layout="wide")
    st.title("Core025 Trait-Driven Member Engine v2.4 (No Member Force)")
    st.caption("Hybrid engine with member forcing removed. Scoring orders the members; Top2/Skip use score-confirmed traits.")
    st.code(BUILD_MARKER, language="text")

    with st.sidebar:
        st.markdown(f"**{BUILD_MARKER}**")
        st.header("Scoring weights")
        stream_bias_weight = st.slider("Stream-bias weight", min_value=0.0, max_value=3.0, value=1.20, step=0.05)
        exact_seed_weight = st.slider("Exact-seed weight", min_value=0.0, max_value=3.0, value=1.50, step=0.05)
        sorted_seed_weight = st.slider("Sorted-seed weight", min_value=0.0, max_value=3.0, value=1.10, step=0.05)
        similarity_weight = st.slider("Similarity weight", min_value=0.0, max_value=3.0, value=1.80, step=0.05)
        min_stream_history = st.number_input("Minimum stream history", min_value=0, value=20, step=5)

        st.header("Trait thresholds")
        skip_danger_min_rate = st.slider("Skip-danger minimum rate", min_value=0.05, max_value=0.95, value=0.30, step=0.01)
        top2_needed_min_rate = st.slider("Top2-needed minimum rate", min_value=0.05, max_value=0.95, value=0.25, step=0.01)

        st.header("Score confirmation rules")
        top1_only_threshold = st.slider("Top1-only threshold", min_value=0.33, max_value=0.95, value=0.48, step=0.005)
        top2_threshold = st.slider("Top1+Top2 threshold", min_value=0.33, max_value=0.95, value=0.36, step=0.005)
        weak_skip_threshold = st.slider("Weak-score skip threshold", min_value=0.00, max_value=0.50, value=0.36, step=0.001)
        top2_gap_max = st.slider("Top2 max gap for Top2-needed trait", min_value=0.00, max_value=0.30, value=0.08, step=0.005)
        skip_score_max = st.slider("Top1 score max for skip trait to force skip", min_value=0.00, max_value=0.60, value=0.45, step=0.005)
        rows_to_show = st.number_input("Rows to display", min_value=5, value=50, step=5)

    hist_file = st.file_uploader("Upload FULL history file", key="hist_no_force")
    surv_file = st.file_uploader("Upload PLAY survivors file", key="surv_no_force")
    sep_file = st.file_uploader("Upload strict separation traits CSV", key="sep_no_force")
    top2_needed_file = st.file_uploader("Upload strict Top2-needed traits CSV", key="top2_no_force")
    skip_danger_file = st.file_uploader("Upload strict skip-danger traits CSV", key="skip_no_force")
    t0025_file = st.file_uploader("Optional: upload strict 0025 traits CSV", key="t0025_no_force")
    t0225_file = st.file_uploader("Optional: upload strict 0225 traits CSV", key="t0225_no_force")
    t0255_file = st.file_uploader("Optional: upload strict 0255 traits CSV", key="t0255_no_force")

    if not all([hist_file, surv_file, sep_file, top2_needed_file, skip_danger_file]):
        st.info("Upload the full history file, survivors file, strict separation traits, Top2-needed traits, and skip-danger traits to begin.")
        return

    try:
        hist = prep_history(load_table(hist_file))
        surv = prep_survivors(load_table(surv_file))
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
    maps = build_transition_maps(transitions)

    results = apply_engine(
        surv=surv,
        transitions=transitions,
        maps=maps,
        sep_traits=sep_traits,
        top2_needed_traits=top2_needed_traits,
        skip_danger_traits=skip_danger_traits,
        traits_0025=traits_0025,
        traits_0225=traits_0225,
        traits_0255=traits_0255,
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
        top2_gap_max=float(top2_gap_max),
        skip_score_max=float(skip_score_max),
    )

    st.subheader("Recommendation summary")
    st.dataframe(results["recommendation"].value_counts(dropna=False).rename_axis("recommendation").reset_index(name="count"), use_container_width=True)

    st.subheader("Decision source summary")
    st.dataframe(results["decision_source"].value_counts(dropna=False).rename_axis("decision_source").reset_index(name="count"), use_container_width=True)

    st.subheader("Playlist preview")
    st.dataframe(results.head(int(rows_to_show)), use_container_width=True)

    st.download_button(
        "Download hybrid_trait_driven_member_playlist_v2_4_no_member_force.csv",
        data=results.to_csv(index=False),
        file_name="hybrid_trait_driven_member_playlist_v2_4_no_member_force.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# core025_pairwise_separator_miner_v1__2026-03-28.py
#
# BUILD: core025_pairwise_separator_miner_v1__2026-03-28

from __future__ import annotations

import io
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

CORE025 = ["0025", "0225", "0255"]
BUILD_MARKER = "BUILD: core025_pairwise_separator_miner_v1__2026-03-28"


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
        "sum_bucket", "spread_bucket", "even", "odd", "high", "low", "unique",
        "pair", "max_rep", "sorted_seed", "first2", "last2", "consec_links",
        "parity_pattern", "highlow_pattern", "pair_token_pattern", "structure",
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
    return pd.DataFrame(rows).sort_values(["event_date", "stream", "seed"]).reset_index(drop=True)


def stability_stats(sub: pd.DataFrame) -> Dict[str, int]:
    ym_col = next((c for c in ["year_month", "year_month_x", "year_month_y"] if c in sub.columns), None)
    if ym_col is None:
        raise KeyError("No year_month column found in subset.")
    return {"stream_count": int(sub["stream"].nunique()), "month_count": int(sub[ym_col].nunique())}


def build_pairwise_separator_traits(
    core_hits: pd.DataFrame, left_member: str, right_member: str, min_support: int,
    preferred_support: int, min_dom_rate: float, min_gap: float, min_streams: int, min_months: int
) -> pd.DataFrame:
    pair_df = core_hits[core_hits["next_member"].isin([left_member, right_member])].copy()
    rows = []
    for col in miner_feature_columns():
        if col not in pair_df.columns:
            continue
        vals = pair_df[col].dropna().unique().tolist()
        try:
            vals = sorted(vals)
        except Exception:
            pass
        for val in vals:
            sub = pair_df[pair_df[col] == val].copy()
            support = len(sub)
            if support < int(min_support):
                continue
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
            if winner_rate < float(min_dom_rate):
                continue
            if gap < float(min_gap):
                continue
            if stable["stream_count"] < int(min_streams):
                continue
            if stable["month_count"] < int(min_months):
                continue
            rows.append({
                "pair": f"{left_member}_vs_{right_member}",
                "trait": f"{col}={val}",
                "support": support,
                "preferred_support_met": int(support >= int(preferred_support)),
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
            })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["preferred_support_met", "winner_rate", "pair_gap", "support", "stream_count", "month_count"],
            ascending=[False, False, False, False, False, False]
        ).reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=[
            "pair", "trait", "support", "preferred_support_met", "winner_member", "loser_member",
            "winner_rate", "loser_rate", "pair_gap", "stream_count", "month_count",
            f"rate_{left_member}", f"rate_{right_member}", "left_count", "right_count"
        ])
    return out


def main():
    st.set_page_config(page_title="Core025 Pairwise Separator Miner v1", layout="wide")
    st.title("Core025 Pairwise Separator Miner v1")
    st.caption("Mines stronger between-member separator traits so more true winners become Top1.")
    st.code(BUILD_MARKER, language="text")

    with st.sidebar:
        st.markdown(f"**{BUILD_MARKER}**")
        st.header("Pairwise separator filters")
        min_support = st.number_input("Minimum support", min_value=10, value=20, step=1)
        preferred_support = st.number_input("Preferred support", min_value=10, value=35, step=1)
        min_dom_rate = st.slider("Minimum winner rate", min_value=0.50, max_value=0.95, value=0.62, step=0.01)
        min_gap = st.slider("Minimum pairwise gap", min_value=0.00, max_value=0.50, value=0.12, step=0.01)
        min_streams = st.number_input("Minimum distinct streams", min_value=1, value=3, step=1)
        min_months = st.number_input("Minimum distinct months", min_value=1, value=2, step=1)
        rows_to_show = st.number_input("Rows to display", min_value=5, value=25, step=5)

    hist_file = st.file_uploader("Upload FULL history file", key="pairwise_hist")
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

    if st.button("Run Pairwise Separator Miner", type="primary"):
        try:
            with st.spinner("Mining pairwise separator traits..."):
                pair_0025_0225 = build_pairwise_separator_traits(core_hits, "0025", "0225", int(min_support), int(preferred_support), float(min_dom_rate), float(min_gap), int(min_streams), int(min_months))
                pair_0225_0255 = build_pairwise_separator_traits(core_hits, "0225", "0255", int(min_support), int(preferred_support), float(min_dom_rate), float(min_gap), int(min_streams), int(min_months))
                pair_0025_0255 = build_pairwise_separator_traits(core_hits, "0025", "0255", int(min_support), int(preferred_support), float(min_dom_rate), float(min_gap), int(min_streams), int(min_months))
                all_pairwise = pd.concat([pair_0025_0225, pair_0225_0255, pair_0025_0255], ignore_index=True) if (len(pair_0025_0225) or len(pair_0225_0255) or len(pair_0025_0255)) else pd.DataFrame()
            st.session_state["pairwise_separator_results"] = {
                "pair_0025_0225": pair_0025_0225,
                "pair_0225_0255": pair_0225_0255,
                "pair_0025_0255": pair_0025_0255,
                "all_pairwise": all_pairwise,
            }
        except Exception as e:
            st.exception(e)
            return

    if "pairwise_separator_results" not in st.session_state or st.session_state["pairwise_separator_results"] is None:
        return

    results = st.session_state["pairwise_separator_results"]

    st.subheader("0025 vs 0225 separators")
    st.dataframe(safe_display_df(results["pair_0025_0225"], int(rows_to_show)), use_container_width=True)
    st.download_button("Download 0025 vs 0225 separators CSV", df_to_csv_bytes(results["pair_0025_0225"]), "core025_pairwise_separator_miner_v1__2026-03-28__0025_vs_0225.csv", "text/csv")

    st.subheader("0225 vs 0255 separators")
    st.dataframe(safe_display_df(results["pair_0225_0255"], int(rows_to_show)), use_container_width=True)
    st.download_button("Download 0225 vs 0255 separators CSV", df_to_csv_bytes(results["pair_0225_0255"]), "core025_pairwise_separator_miner_v1__2026-03-28__0225_vs_0255.csv", "text/csv")

    st.subheader("0025 vs 0255 separators")
    st.dataframe(safe_display_df(results["pair_0025_0255"], int(rows_to_show)), use_container_width=True)
    st.download_button("Download 0025 vs 0255 separators CSV", df_to_csv_bytes(results["pair_0025_0255"]), "core025_pairwise_separator_miner_v1__2026-03-28__0025_vs_0255.csv", "text/csv")

    st.subheader("All pairwise separators")
    st.dataframe(safe_display_df(results["all_pairwise"], int(rows_to_show)), use_container_width=True)
    st.download_button("Download all pairwise separators CSV", df_to_csv_bytes(results["all_pairwise"]), "core025_pairwise_separator_miner_v1__2026-03-28__all_pairwise.csv", "text/csv")


if __name__ == "__main__":
    if "pairwise_separator_results" not in st.session_state:
        st.session_state["pairwise_separator_results"] = None
    main()

#!/usr/bin/env python3
"""
BUILD: core025_batch_overlay_tester__2026-04-16_v28_autofeature

Full replacement file.
- Accepts raw per-event separator export directly
- Auto-derives required seed features
- Tests multiple overlay variants in one run
- Compares results to locked baseline
- Exports self-naming result files
"""

from __future__ import annotations

import io
import re
from collections import Counter
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

BUILD_MARKER = "BUILD: core025_batch_overlay_tester__2026-04-16_v28_autofeature"
BUILD_SLUG = BUILD_MARKER.replace("BUILD: ", "")

LOCKED_BASELINE = {
    "App Top1": 52,
    "App Top2": 177,
    "Misses": 83,
    "Your Top1": 52,
    "Your Top2": 87,
    "Your Top3": 0,
    "Waste": 90,
    "Needed": 87,
}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def load_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".txt") or name.endswith(".tsv"):
        data = uploaded_file.getvalue()
        try:
            return pd.read_csv(io.BytesIO(data), sep="\t")
        except Exception:
            return pd.read_csv(io.BytesIO(data), sep=None, engine="python")
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    cols = list(df.columns)
    nmap = {_norm(c): c for c in cols}
    for cand in candidates:
        key = _norm(cand)
        if key in nmap:
            return nmap[key]
    for cand in candidates:
        key = _norm(cand)
        for nkey, col in nmap.items():
            if key and key in nkey:
                return col
    if required:
        raise KeyError(f"Required column not found. Tried {candidates}. Available columns: {cols}")
    return None


def canonical_seed(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = re.sub(r"\D", "", str(x))
    return s[:4] if len(s) >= 4 else None


def coerce_member_text(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    nums = re.findall(r"\d+", s)
    if nums:
        for token in reversed(nums):
            v = token.zfill(4)
            if v in {"0025", "0225", "0255"}:
                return v
            if token in {"25", "225", "255"}:
                return {"25": "0025", "225": "0225", "255": "0255"}[token]
    s_up = s.upper()
    return s_up if s_up in {"0025", "0225", "0255"} else None


def compute_seed_features(seed: str) -> Dict[str, object]:
    d = [int(ch) for ch in seed]
    cnt = Counter(d)
    spread = max(d) - min(d)
    parity = "".join("E" if x % 2 == 0 else "O" for x in d)
    highlow = "".join("H" if x >= 5 else "L" for x in d)
    out: Dict[str, object] = {
        "seed_spread": spread,
        "seed_parity_pattern": parity,
        "seed_highlow_pattern": highlow,
        "seed_pos1": d[0],
        "seed_pos2": d[1],
        "seed_pos3": d[2],
        "seed_pos4": d[3],
    }
    for k in range(10):
        out[f"seed_cnt{k}"] = int(cnt.get(k, 0))
    return out


def prepare_per_event(df_raw: pd.DataFrame) -> pd.DataFrame:
    seed_col = find_col(df_raw, ["seed", "PrevSeed", "prev_seed"], required=True)
    top1_col = find_col(df_raw, ["Top1", "top1"], required=True)
    top2_col = find_col(df_raw, ["Top2", "top2"], required=True)
    winning_member_col = find_col(df_raw, ["winning_member", "WinningMember"], required=True)

    out = pd.DataFrame()
    out["seed"] = df_raw[seed_col].apply(canonical_seed)
    out["Top1"] = df_raw[top1_col].apply(coerce_member_text)
    out["Top2"] = df_raw[top2_col].apply(coerce_member_text)
    out["WinningMember"] = df_raw[winning_member_col].apply(coerce_member_text)

    out = out.dropna(subset=["seed", "Top1", "Top2", "WinningMember"]).reset_index(drop=True)
    feat_df = out["seed"].apply(compute_seed_features).apply(pd.Series)
    out = pd.concat([out.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
    out["BuildMarker"] = BUILD_SLUG
    return out


# ------------------------------------------------------------
# Overlay evaluation
# ------------------------------------------------------------
def overlay_decision(row: pd.Series, variant: Dict[str, object]) -> tuple[str, str, int]:
    # MISS block
    miss_hit = False
    if variant["miss_mode"] == "strict":
        miss_hit = (row["seed_spread"] == 6) and (row["seed_parity_pattern"] in ["EOOE", "OEEO"])
    elif variant["miss_mode"] == "loose":
        miss_hit = (row["seed_spread"] == 6) or (row["seed_parity_pattern"] in ["EOOE", "OEEO"])
    elif variant["miss_mode"] == "off":
        miss_hit = False

    if miss_hit:
        return "PLAY_TOP2", "MISS_BLOCK", 0

    # NEEDED protect
    needed_hit = False
    if variant["needed_mode"] == "strict":
        needed_hit = (row["seed_parity_pattern"] == "OOOO") and (row["seed_cnt1"] == 2)
    elif variant["needed_mode"] == "loose":
        needed_hit = (row["seed_parity_pattern"] == "OOOO") or (row["seed_cnt1"] == 2)
    elif variant["needed_mode"] == "off":
        needed_hit = False

    if needed_hit:
        return "PLAY_TOP2", "NEEDED_PROTECT", 0

    # WASTE score
    score = 0
    if row["seed_spread"] == 9:
        score += 2 if variant["weighted"] else 1
    if row["seed_highlow_pattern"] == "LHHH":
        score += 2 if variant["weighted"] else 1
    if row["seed_parity_pattern"] == "EEOE":
        score += 1
    if row["seed_pos2"] == 7:
        score += 1
    if row["seed_cnt7"] == 2:
        score += 1

    if variant["spread_block"] and row["seed_spread"] <= 5:
        score = 0

    if score >= int(variant["threshold"]):
        return "PLAY_TOP1", "WASTE_PROMOTE", score

    return "PLAY_TOP2", "DEFAULT_TOP2", score


def run_variant(df: pd.DataFrame, variant: Dict[str, object]) -> tuple[pd.DataFrame, Dict[str, object]]:
    detail = df.copy()
    plays = []
    reasons = []
    scores = []

    app_top1 = 0
    app_top2 = 0
    your_top1 = 0
    your_top2 = 0
    your_top3 = 0
    misses = 0
    waste = 0
    needed = 0

    for _, row in detail.iterrows():
        play, reason, score = overlay_decision(row, variant)
        plays.append(play)
        reasons.append(reason)
        scores.append(score)

        winner = row["WinningMember"]

        if play == "PLAY_TOP1":
            if row["Top1"] == winner:
                app_top1 += 1
                your_top1 += 1
            else:
                misses += 1
        else:  # PLAY_TOP2
            if row["Top1"] == winner:
                app_top2 += 1
                waste += 1
            elif row["Top2"] == winner:
                app_top2 += 1
                your_top2 += 1
                needed += 1
            else:
                misses += 1

    detail["overlay_play"] = plays
    detail["overlay_reason"] = reasons
    detail["overlay_score"] = scores
    detail["variant_name"] = variant["name"]
    detail["BuildMarker"] = BUILD_SLUG

    summary = {
        "Variant": variant["name"],
        "App Top1": app_top1,
        "App Top2": app_top2,
        "Misses": misses,
        "Your Top1": your_top1,
        "Your Top2": your_top2,
        "Your Top3": your_top3,
        "Waste": waste,
        "Needed": needed,
        "Δ App Top1": app_top1 - LOCKED_BASELINE["App Top1"],
        "Δ App Top2": app_top2 - LOCKED_BASELINE["App Top2"],
        "Δ Misses": misses - LOCKED_BASELINE["Misses"],
        "Δ Your Top1": your_top1 - LOCKED_BASELINE["Your Top1"],
        "Δ Your Top2": your_top2 - LOCKED_BASELINE["Your Top2"],
        "Δ Waste": waste - LOCKED_BASELINE["Waste"],
        "Δ Needed": needed - LOCKED_BASELINE["Needed"],
        "BuildMarker": BUILD_SLUG,
    }
    return detail, summary


def build_baseline_table() -> pd.DataFrame:
    return pd.DataFrame([{
        "Variant": "LOCKED_BASELINE",
        "App Top1": LOCKED_BASELINE["App Top1"],
        "App Top2": LOCKED_BASELINE["App Top2"],
        "Misses": LOCKED_BASELINE["Misses"],
        "Your Top1": LOCKED_BASELINE["Your Top1"],
        "Your Top2": LOCKED_BASELINE["Your Top2"],
        "Your Top3": LOCKED_BASELINE["Your Top3"],
        "Waste": LOCKED_BASELINE["Waste"],
        "Needed": LOCKED_BASELINE["Needed"],
        "BuildMarker": BUILD_SLUG,
    }])


# ------------------------------------------------------------
# App
# ------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Core025 Batch Overlay Tester", layout="wide")
    st.title("Core025 Batch Overlay Tester")
    st.caption(BUILD_MARKER)

    if "batch_outputs_v28" not in st.session_state:
        st.session_state["batch_outputs_v28"] = None

    with st.sidebar:
        st.write(BUILD_MARKER)
        uploaded = st.file_uploader("Upload raw per-event CSV", type=["csv", "txt", "tsv", "xlsx", "xls"])
        rows_to_show = st.number_input("Rows to preview", min_value=20, max_value=500, value=100, step=20)
        run_btn = st.button("Run batch tester", type="primary", use_container_width=True)

    if run_btn:
        if uploaded is None:
            st.error("Upload a per-event file first.")
            st.session_state["batch_outputs_v28"] = None
        else:
            try:
                raw = load_table(uploaded)
                work = prepare_per_event(raw)

                variants = [
                    {"name": "v26_like", "threshold": 1, "miss_mode": "strict", "needed_mode": "strict", "spread_block": False, "weighted": False},
                    {"name": "v27_weighted_balanced", "threshold": 2, "miss_mode": "strict", "needed_mode": "strict", "spread_block": True, "weighted": True},
                    {"name": "aggressive_loose", "threshold": 1, "miss_mode": "loose", "needed_mode": "strict", "spread_block": False, "weighted": True},
                    {"name": "defensive_needed", "threshold": 2, "miss_mode": "strict", "needed_mode": "loose", "spread_block": True, "weighted": True},
                    {"name": "waste_only", "threshold": 2, "miss_mode": "off", "needed_mode": "off", "spread_block": True, "weighted": True},
                ]

                summaries = []
                details = {}
                for v in variants:
                    detail, summary = run_variant(work, v)
                    summaries.append(summary)
                    details[v["name"]] = detail

                summary_df = pd.DataFrame(summaries).sort_values(
                    ["Δ Misses", "Δ App Top1", "Δ Waste"],
                    ascending=[True, False, True]
                ).reset_index(drop=True)

                st.session_state["batch_outputs_v28"] = {
                    "source_name": uploaded.name,
                    "prepared_df": work,
                    "baseline_df": build_baseline_table(),
                    "summary_df": summary_df,
                    "detail_map": details,
                }

            except Exception as e:
                st.session_state["batch_outputs_v28"] = None
                st.error(f"Failed to run batch tester: {e}")

    outputs = st.session_state.get("batch_outputs_v28")
    if outputs is None:
        st.info("Upload the same per-event CSV and click Run batch tester.")
        return

    st.success("Batch testing complete")
    st.write(f"**Source file:** {outputs['source_name']}")
    st.write(f"**Prepared rows:** {len(outputs['prepared_df'])}")

    st.subheader("Locked baseline")
    st.dataframe(outputs["baseline_df"], use_container_width=True, hide_index=True)

    st.subheader("Variant comparison")
    st.dataframe(outputs["summary_df"], use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "Download variant comparison",
            data=df_to_csv_bytes(outputs["summary_df"]),
            file_name=f"variant_comparison__{BUILD_SLUG}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_summary_v28",
        )
    with c2:
        st.download_button(
            "Download prepared feature table",
            data=df_to_csv_bytes(outputs["prepared_df"]),
            file_name=f"prepared_feature_table__{BUILD_SLUG}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_prepared_v28",
        )
    with c3:
        st.download_button(
            "Download locked baseline",
            data=df_to_csv_bytes(outputs["baseline_df"]),
            file_name=f"locked_baseline__{BUILD_SLUG}.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_baseline_v28",
        )

    st.markdown("---")
    st.subheader("Variant detail previews")
    variant_names = list(outputs["detail_map"].keys())
    picked = st.selectbox("Choose a variant to preview / download", variant_names)
    detail_df = outputs["detail_map"][picked]

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            f"Download {picked} detail rows",
            data=df_to_csv_bytes(detail_df),
            file_name=f"{picked}__detail_rows__{BUILD_SLUG}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"dl_detail_{picked}",
        )
    with d2:
        overlay_counts = (
            detail_df["overlay_reason"]
            .value_counts(dropna=False)
            .rename_axis("overlay_reason")
            .reset_index(name="count")
        )
        st.download_button(
            f"Download {picked} overlay reason counts",
            data=df_to_csv_bytes(overlay_counts),
            file_name=f"{picked}__overlay_reason_counts__{BUILD_SLUG}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"dl_reasons_{picked}",
        )

    tab1, tab2 = st.tabs(["Detail rows preview", "Overlay reason counts"])
    with tab1:
        st.dataframe(detail_df.head(int(rows_to_show)), use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(overlay_counts, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

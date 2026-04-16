#!/usr/bin/env python3
"""
BUILD: core025_batch_overlay_tester__2026-04-16_v27

PURPOSE:
- Run multiple overlay variants in ONE run
- Compare vs baseline automatically
- No manual code edits per test
"""

import pandas as pd
import numpy as np
import streamlit as st

BUILD = "core025_batch_overlay_tester__2026-04-16_v27"

st.set_page_config(layout="wide")
st.title("Core025 Batch Overlay Tester")
st.write(f"BUILD: {BUILD}")

# =========================
# Upload
# =========================
file = st.file_uploader("Upload per-event CSV", type=["csv"])

if file is None:
    st.stop()

df = pd.read_csv(file)

# =========================
# Required columns check
# =========================
required = ["seed_spread","seed_parity_pattern","seed_highlow_pattern",
            "seed_cnt1","seed_cnt7","seed_pos2",
            "Top1","Top2","WinningMember"]

missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# =========================
# Overlay Variants
# =========================
def run_variant(df, variant):

    results = {
        "Top1":0,
        "Top2":0,
        "Miss":0,
        "Waste":0,
        "Needed":0
    }

    for _, row in df.iterrows():

        play = "TOP2"  # default conservative

        # -------- MISS --------
        if variant["miss_mode"] == "strict":
            if row["seed_spread"] == 6 and row["seed_parity_pattern"] in ["EOOE","OEEO"]:
                play = "TOP2"

        elif variant["miss_mode"] == "loose":
            if row["seed_spread"] == 6 or row["seed_parity_pattern"] in ["EOOE","OEEO"]:
                play = "TOP2"

        # -------- NEEDED --------
        if variant["needed_mode"] == "strict":
            if row["seed_parity_pattern"] == "OOOO" and row["seed_cnt1"] == 2:
                play = "TOP2"

        elif variant["needed_mode"] == "loose":
            if row["seed_parity_pattern"] == "OOOO" or row["seed_cnt1"] == 2:
                play = "TOP2"

        # -------- WASTE SCORE --------
        score = 0
        if row["seed_spread"] == 9: score += 2
        if row["seed_highlow_pattern"] == "LHHH": score += 2
        if row["seed_parity_pattern"] == "EEOE": score += 1
        if row["seed_pos2"] == 7: score += 1
        if row["seed_cnt7"] == 2: score += 1

        # -------- SAFETY --------
        if variant["spread_block"]:
            if row["seed_spread"] <= 5:
                score = 0

        # -------- PROMOTION --------
        if score >= variant["threshold"]:
            play = "TOP1"

        # =========================
        # Evaluate outcome
        # =========================
        winner = row["WinningMember"]

        if play == "TOP1":
            if row["Top1"] == winner:
                results["Top1"] += 1
            else:
                results["Miss"] += 1

        else:  # TOP2
            if row["Top1"] == winner:
                results["Waste"] += 1
            elif row["Top2"] == winner:
                results["Top2"] += 1
            else:
                results["Miss"] += 1

    return results

# =========================
# Variants to test
# =========================
variants = [
    {"name":"v26_like","threshold":1,"miss_mode":"strict","needed_mode":"strict","spread_block":False},
    {"name":"v27_balanced","threshold":2,"miss_mode":"strict","needed_mode":"strict","spread_block":True},
    {"name":"aggressive","threshold":1,"miss_mode":"loose","needed_mode":"strict","spread_block":False},
    {"name":"defensive","threshold":2,"miss_mode":"strict","needed_mode":"loose","spread_block":True},
]

# =========================
# Run all
# =========================
rows = []

for v in variants:
    r = run_variant(df, v)
    row = {"Variant":v["name"], **r}
    rows.append(row)

out = pd.DataFrame(rows)

st.subheader("Variant Comparison")
st.dataframe(out, use_container_width=True)

# =========================
# Download
# =========================
csv = out.to_csv(index=False).encode()
st.download_button("Download Results", csv, f"batch_results__{BUILD}.csv")

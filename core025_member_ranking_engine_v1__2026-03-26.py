#!/usr/bin/env python3
# BUILD: core025_separator_engine_plus_lab_walkforward__2026-04-01_v5

import pandas as pd
import streamlit as st
from collections import Counter, defaultdict
import re

CORE025 = ["0025", "0225", "0255"]

# -------------------------
# Helpers
# -------------------------

def normalize_member(x):
    d = "".join(re.findall(r"\d", str(x)))
    if d in ["25","025","0025"]: return "0025"
    if d in ["225","0225"]: return "0225"
    if d in ["255","0255"]: return "0255"
    return None

def features(seed):
    d = [int(x) for x in re.findall(r"\d", str(seed))[:4]]
    if len(d) < 4: return None
    return {
        "sum": sum(d),
        "spread": max(d) - min(d),
        "even": sum(x%2==0 for x in d),
        "odd": sum(x%2!=0 for x in d),
    }

# -------------------------
# Baseline
# -------------------------

def baseline_probs(seed, maps):
    s = maps["seed"].get(seed, Counter())
    total = sum(s.values())
    if total == 0:
        return {m:1/3 for m in CORE025}
    return {m:s[m]/total for m in CORE025}

# -------------------------
# Rule Matching
# -------------------------

def match_rule(row, rule):
    for k,v in rule["conditions"].items():
        if str(row.get(k)) != str(v):
            return False
    return True

# -------------------------
# Rule Alignment
# -------------------------

def compute_rule_alignment(top1_rules, top2_rules):
    total = top1_rules + top2_rules
    if total == 0: return 0
    return top1_rules / total

# -------------------------
# Core Ranking Engine
# -------------------------

def rank_row(row, maps, rules):

    base = baseline_probs(row["seed"], maps)

    boosts = {m:0 for m in CORE025}
    rule_counts = {m:0 for m in CORE025}

    for r in rules:
        if match_rule(row, r):
            m = r["winner"]
            boosts[m] += r["score"]
            rule_counts[m] += 1

    scores = {m: base[m] + boosts[m] for m in CORE025}

    ranked = sorted(scores.items(), key=lambda x:x[1], reverse=True)

    top1, s1 = ranked[0]
    top2, s2 = ranked[1]
    top3, s3 = ranked[2]

    gap = s1 - s2
    ratio = s2 / s1 if s1 > 0 else 1

    exclusivity = abs(rule_counts[top1] - rule_counts[top2])
    rule_gap = rule_counts[top1] - rule_counts[top2]

    rule_alignment = compute_rule_alignment(rule_counts[top1], rule_counts[top2])

    # -------------------------
    # DOMINANCE (STRICT v5)
    # -------------------------

    if (
        gap >= 0.75
        and ratio <= 0.65
        and exclusivity >= 2
        and rule_gap >= 3
        and rule_alignment >= 0.65
    ):
        dominance = "DOMINANT"
    elif gap <= 0.15 or ratio >= 0.93:
        dominance = "CONTESTED"
    else:
        dominance = "WEAK"

    # HARD DEMOTION
    if dominance == "DOMINANT" and gap < 0.75:
        dominance = "WEAK"

    # -------------------------
    # PLAY DECISION (v5 FINAL)
    # -------------------------

    # 🔥 NEW: CONFIDENCE RECOVERY (final step)
    if (
        dominance == "WEAK"
        and exclusivity >= 2
        and ratio <= 0.75
        and rule_alignment >= 0.70
    ):
        play = "PLAY_TOP1"

    # FALSE DOMINANCE GUARD
    elif (
        gap >= 0.20
        and ratio < 0.80
        and exclusivity < 2
    ):
        play = "PLAY_TOP2"

    # CONTESTED
    elif dominance == "CONTESTED":
        play = "PLAY_TOP2"

    # TIGHT SIGNAL
    elif ratio >= 0.92 or gap <= 0.18:
        play = "PLAY_TOP2"

    else:
        play = "PLAY_TOP1"

    return {
        "Top1": top1,
        "Top2": top2,
        "Top3": top3,
        "Top1_score": s1,
        "Top2_score": s2,
        "gap": gap,
        "ratio": ratio,
        "rule_alignment": rule_alignment,
        "dominance": dominance,
        "play": play
    }

# -------------------------
# Walk Forward
# -------------------------

def run_lab(df, rules):

    maps = {"seed": defaultdict(Counter)}
    results = []

    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        cur = df.iloc[i]

        seed = prev["result"]
        winner = normalize_member(cur["result"])

        row = {"seed": seed}
        row.update(features(seed) or {})

        ranked = rank_row(row, maps, rules)

        top1_hit = int(ranked["Top1"] == winner)
        top2_hit = int(winner in [ranked["Top1"], ranked["Top2"]])

        if ranked["play"] == "PLAY_TOP1":
            play_hit = top1_hit
        else:
            play_hit = top2_hit

        results.append({
            **ranked,
            "winner": winner,
            "top1_hit": top1_hit,
            "top2_hit": top2_hit,
            "play_hit": play_hit
        })

        maps["seed"][seed][winner] += 1

    return pd.DataFrame(results)

# -------------------------
# UI
# -------------------------

def main():
    st.title("Core025 v5 Final Engine")

    hist_file = st.file_uploader("Upload history CSV")

    if hist_file:
        df = pd.read_csv(hist_file)

        rules = []  # assume pre-loaded or extended later

        if st.button("Run LAB"):
            out = run_lab(df, rules)

            st.dataframe(out.head(50))

            total = len(out)

            st.write("Top1:", out["top1_hit"].sum())
            st.write("Top2:", out["top2_hit"].sum())
            st.write("Play Capture:", out["play_hit"].sum())
            st.write("Misses:", total - out["play_hit"].sum())

if __name__ == "__main__":
    main()

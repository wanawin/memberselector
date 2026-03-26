#!/usr/bin/env python3
# core025_member_ranking_engine_v1__2026-03-26.py

import pandas as pd
import numpy as np
import re
import streamlit as st
from collections import Counter, defaultdict

CORE025 = ["0025","0225","0255"]

# -----------------------
# Helpers
# -----------------------

def norm(r):
    d = re.findall(r"\d", str(r))
    return "".join(d[:4]) if len(d)>=4 else None

def to_member(r):
    if r is None: return None
    s = "".join(sorted(r))
    return s if s in CORE025 else None

def load(f):
    name = f.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(f)
    if name.endswith(".txt") or name.endswith(".tsv"):
        return pd.read_csv(f, sep="\t", header=None)
    if name.endswith(".xlsx"):
        return pd.read_excel(f)

# -----------------------
# Prepare history
# -----------------------

def prep(df):
    if len(df.columns)==4:
        df.columns=["date","jurisdiction","game","result"]

    df["date"]=pd.to_datetime(df["date"], errors="coerce")
    df["r4"]=df["result"].apply(norm)
    df["member"]=df["r4"].apply(to_member)
    df["stream"]=df["jurisdiction"].astype(str)+"|"+df["game"].astype(str)

    return df.dropna(subset=["r4"]).reset_index(drop=True)

# -----------------------
# Build transition table
# -----------------------

def build_transitions(df):
    rows=[]
    for s,g in df.groupby("stream"):
        g=g.sort_values("date").reset_index(drop=True)
        for i in range(1,len(g)):
            rows.append({
                "stream":s,
                "seed":g.loc[i-1,"r4"],
                "next_member":g.loc[i,"member"]
            })
    return pd.DataFrame(rows)

# -----------------------
# Build probability model
# -----------------------

def build_model(tr):
    model = defaultdict(lambda: Counter())

    for _,r in tr.iterrows():
        if r["next_member"] is not None:
            model[r["seed"]][r["next_member"]] += 1

    return model

# -----------------------
# Score a seed
# -----------------------

def score_seed(seed, model):
    counts = model.get(seed, Counter())

    total = sum(counts.values())

    scores = {}
    for m in CORE025:
        if total > 0:
            scores[m] = counts[m] / total
        else:
            scores[m] = 0.0

    # fallback: if unseen seed
    if total == 0:
        for m in CORE025:
            scores[m] = 1/3

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return ranked

# -----------------------
# Apply to survivors
# -----------------------

def apply_survivors(survivors, model):
    rows=[]

    for _,r in survivors.iterrows():
        seed = r["seed"]

        ranked = score_seed(seed, model)

        rows.append({
            "stream": r["stream_id"] if "stream_id" in r else r["stream"],
            "seed": seed,
            "Top1": ranked[0][0],
            "Top1_score": ranked[0][1],
            "Top2": ranked[1][0],
            "Top2_score": ranked[1][1],
            "Top3": ranked[2][0],
            "Top3_score": ranked[2][1]
        })

    return pd.DataFrame(rows)

# -----------------------
# App
# -----------------------

def app():
    st.title("Member Ranking Engine v1")

    history_file = st.file_uploader("Upload FULL history file")
    survivor_file = st.file_uploader("Upload PLAY survivors (from skip ladder)")

    if not history_file or not survivor_file:
        return

    hist = prep(load(history_file))
    surv = load(survivor_file)

    # normalize survivor format
    if "stream_id" in surv.columns:
        surv["stream"] = surv["stream_id"]
    if "seed" not in surv.columns:
        st.error("Survivor file must contain 'seed'")
        return

    tr = build_transitions(hist)
    model = build_model(tr)

    results = apply_survivors(surv, model)

    st.subheader("Member Rankings")
    st.dataframe(results)

    st.download_button(
        "Download rankings",
        results.to_csv(index=False),
        "member_rankings.csv"
    )

if __name__=="__main__":
    app()

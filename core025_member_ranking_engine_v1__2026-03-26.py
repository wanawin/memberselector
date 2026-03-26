#!/usr/bin/env python3
# core025_member_engine_v2__2026-03-26.py

import pandas as pd
import numpy as np
import re
import streamlit as st
from collections import Counter

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
# Feature builder
# -----------------------

def features(seed):
    d = [int(x) for x in seed]
    return {
        "sum": sum(d),
        "spread": max(d)-min(d),
        "even": sum(x%2==0 for x in d),
        "high": sum(x>=5 for x in d),
        "unique": len(set(d)),
        "pair": int(len(set(d))<4),
        "pos1": d[0],
        "pos2": d[1],
        "pos3": d[2],
        "pos4": d[3],
    }

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

    df=df.dropna(subset=["r4"]).reset_index(drop=True)

    feats = df["r4"].apply(features).apply(pd.Series)
    return pd.concat([df,feats],axis=1)

# -----------------------
# Build transitions
# -----------------------

def build_transitions(df):
    rows=[]
    for s,g in df.groupby("stream"):
        g=g.sort_values("date").reset_index(drop=True)
        for i in range(1,len(g)):
            rows.append({
                "seed":g.loc[i-1,"r4"],
                "next_member":g.loc[i,"member"],
                **features(g.loc[i-1,"r4"])
            })
    return pd.DataFrame(rows)

# -----------------------
# Similarity scoring
# -----------------------

def similarity(a,b):
    score=0

    if a["sum"]==b["sum"]: score+=2
    if abs(a["sum"]-b["sum"])<=2: score+=1

    if a["spread"]==b["spread"]: score+=2
    if abs(a["spread"]-b["spread"])<=1: score+=1

    if a["even"]==b["even"]: score+=1
    if a["high"]==b["high"]: score+=1
    if a["unique"]==b["unique"]: score+=1
    if a["pair"]==b["pair"]: score+=1

    # position weighting
    if a["pos1"]==b["pos1"]: score+=1
    if a["pos2"]==b["pos2"]: score+=1

    return score

# -----------------------
# Score seed
# -----------------------

def score_seed(seed, transitions):
    seed_feat = features(seed)

    scores = {m:0 for m in CORE025}
    total_weight = 0

    for _,r in transitions.iterrows():
        if r["next_member"] is None:
            continue

        sim = similarity(seed_feat, r)

        if sim > 0:
            scores[r["next_member"]] += sim
            total_weight += sim

    if total_weight == 0:
        return [(m,1/3) for m in CORE025]

    probs = {m: scores[m]/total_weight for m in CORE025}
    ranked = sorted(probs.items(), key=lambda x:x[1], reverse=True)

    return ranked

# -----------------------
# Apply to survivors
# -----------------------

def apply_survivors(surv, transitions):
    rows=[]

    for _,r in surv.iterrows():
        seed = str(r["seed"])

        ranked = score_seed(seed, transitions)

        rows.append({
            "stream": r.get("stream_id", r.get("stream")),
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
    st.title("Member Ranking Engine v2 (Trait-Based)")

    hist_file = st.file_uploader("Upload FULL history file")
    surv_file = st.file_uploader("Upload PLAY survivors")

    if not hist_file or not surv_file:
        return

    hist = prep(load(hist_file))
    surv = load(surv_file)

    if "stream_id" in surv.columns:
        surv["stream"] = surv["stream_id"]

    transitions = build_transitions(hist)

    results = apply_survivors(surv, transitions)

    st.subheader("Member Rankings v2")
    st.dataframe(results)

    st.download_button(
        "Download rankings",
        results.to_csv(index=False),
        "member_rankings_v2.csv"
    )

if __name__=="__main__":
    app()

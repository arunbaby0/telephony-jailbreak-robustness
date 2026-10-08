"""Builds artifacts/data/prompts.parquet."""
import re, io, sys
import pandas as pd
import numpy as np
import urllib.request

OUT = "/data/artifacts/data/prompts.parquet"
import os
os.makedirs("/data/artifacts/data", exist_ok=True)

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "tjr-research"})
    return urllib.request.urlopen(req, timeout=30).read()

def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

rows = []

# HarmBench standard behaviors
hb_url = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
hb = pd.read_csv(io.BytesIO(fetch(hb_url)))
hb = hb[hb["FunctionalCategory"] == "standard"]
hb_texts = hb["Behavior"].tolist()[:200]
for i, t in enumerate(hb_texts):
    rows.append({"id": f"harmbench_{i}", "text": t, "split": "harmful", "source": "harmbench"})

# AdvBench harmful_behaviors.csv
adv_url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
adv = pd.read_csv(io.BytesIO(fetch(adv_url)))
hb_norm = set(norm(t) for t in hb_texts)
kept = []
for t in adv["goal"].tolist():
    n = norm(t)
    if n in hb_norm:
        continue
    # crude fuzzy: skip if shares >85% of tokens with any harmbench prompt
    toks = set(n.split())
    dup = False
    for h in hb_norm:
        htoks = set(h.split())
        if not htoks: continue
        overlap = len(toks & htoks) / max(1, len(toks | htoks))
        if overlap >= 0.85:
            dup = True; break
    if not dup:
        kept.append(t)
    if len(kept) >= 100:
        break
for i, t in enumerate(kept):
    rows.append({"id": f"advbench_{i}", "text": t, "split": "harmful", "source": "advbench"})

harmful_df = pd.DataFrame(rows)
harmful_df = harmful_df[harmful_df["text"].str.split().str.len() <= 40]
print("harmful N =", len(harmful_df))

# XSTest safe prompts
from datasets import load_dataset
xstest = load_dataset("Paul/XSTest", split="train")
xdf = xstest.to_pandas()
label_col = "label" if "label" in xdf.columns else "type"
safe = xdf[xdf[label_col].astype(str).str.contains("safe", case=False)]
safe = safe.sample(n=min(250, len(safe)), random_state=0)
text_col = "prompt" if "prompt" in safe.columns else safe.columns[0]
xrows = [{"id": f"xstest_{i}", "text": t, "split": "benign_borderline", "source": "xstest"}
         for i, t in enumerate(safe[text_col].tolist())]
xstest_df = pd.DataFrame(xrows)
xstest_df = xstest_df[xstest_df["text"].str.split().str.len() <= 40]
print("xstest N =", len(xstest_df))

# VoiceBench alpacaeval (text field only here; already-recorded audio handled at TTS stage where applicable)
vb = load_dataset("hlt-lab/voicebench", "alpacaeval", split="test")
vdf = vb.to_pandas()
vdf = vdf.sample(n=min(300, len(vdf)), random_state=0)
text_col = "prompt" if "prompt" in vdf.columns else ("instruction" if "instruction" in vdf.columns else vdf.columns[0])
vrows = [{"id": f"voicebench_{i}", "text": t, "split": "benign_realspeech", "source": "voicebench"}
         for i, t in enumerate(vdf[text_col].tolist())]
voicebench_df = pd.DataFrame(vrows)
print("voicebench N =", len(voicebench_df))

full = pd.concat([harmful_df, xstest_df, voicebench_df], ignore_index=True)
full = full[full["text"].str.split().str.len() <= 40].reset_index(drop=True)
full.to_parquet(OUT)
print("TOTAL", len(full))
print(full["split"].value_counts())
print("PROMPTS_BUILD_OK")

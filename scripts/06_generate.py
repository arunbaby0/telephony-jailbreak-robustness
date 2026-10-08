"""Phase 5: vLLM generation grid over (model x channel x clip). Resumable.
Runs one target model at a time (T1, T2) against every clip under artifacts/data/channels.
"""
import os, sys, glob, json, time
import numpy as np
import pandas as pd
import soundfile as sf
from vllm import LLM, SamplingParams

ART = "/data/artifacts"
CH = f"{ART}/data/channels"
OUT = f"{ART}/generations.parquet"

TARGETS = {
    "T1": "/data/models/qwen2-audio-7b-instruct",
    "T2": "/data/models/qwen2.5-omni-7b",
}
CONDITIONS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]

def list_all_clips():
    """Walk clips_manifest + pgd_manifest to know (family, speaker, id, split, source),
    then map each to its channel-processed paths."""
    manifest = pd.read_parquet(f"{ART}/data/clips_manifest.parquet")
    if os.path.exists(f"{ART}/data/pgd_manifest.parquet"):
        pgd = pd.read_parquet(f"{ART}/data/pgd_manifest.parquet")
        pgd["family"] = "A5"
        pgd["split"] = "harmful"
        pgd["source"] = "pgd"
        manifest = pd.concat([manifest, pgd[["id", "family", "speaker", "path", "split", "source"]]], ignore_index=True)

    rows = []
    for _, r in manifest.iterrows():
        tag = f"{r['family']}_{r['speaker']}_{r['id']}"
        for cond in CONDITIONS:
            seeds = [0]  # generation and judging use loss seed 0 only
            for s in seeds:
                cpath = f"{CH}/{cond}/{s}/{tag}.wav"
                if os.path.exists(cpath):
                    rows.append({"clip_id": f"{r['family']}_{r['speaker']}_{tag}_{cond}_{s}",
                                 "prompt_id": tag, "family": r["family"], "speaker": r["speaker"],
                                 "channel": cond, "seed": s, "split": r["split"], "source": r["source"],
                                 "audio_path": cpath})
    return pd.DataFrame(rows)

def run_target(model_key, model_path, work):
    from transformers import AutoProcessor
    llm = LLM(model=model_path, dtype="bfloat16", max_model_len=2048,
              limit_mm_per_prompt={"audio": 1}, gpu_memory_utilization=0.85, trust_remote_code=True)
    sp = SamplingParams(temperature=0, max_tokens=256, seed=0)

    # Audio-only user turn through each model's own chat template (no extra text or
    # system prompt beyond the template's default); a bare audio-token string yields
    # empty or non-assistant output on both targets.
    proc = AutoProcessor.from_pretrained(model_path)
    conv = [{"role": "user", "content": [{"type": "audio", "audio_url": "x"}]}]
    chat_prefix = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)

    results = []
    BATCH = 16
    for i in range(0, len(work), BATCH):
        chunk = work.iloc[i:i+BATCH]
        prompts = []
        for _, r in chunk.iterrows():
            d, sr = sf.read(r["audio_path"])
            prompts.append({
                "prompt": chat_prefix,
                "multi_modal_data": {"audio": (d.astype(np.float32), sr)},
            })
        outs = llm.generate(prompts, sp)
        for (_, r), o in zip(chunk.iterrows(), outs):
            results.append({**r.to_dict(), "model": model_key, "response": o.outputs[0].text})
        if i % (BATCH * 20) == 0:
            print(f"[{model_key}] progress {i}/{len(work)}", flush=True)
        if i % (BATCH * 200) == 0 and i > 0:
            pd.DataFrame(results).to_parquet(f"{ART}/generations_{model_key}_partial.parquet")
    del llm
    return pd.DataFrame(results)

def main():
    all_clips = list_all_clips()
    print(f"total (family,channel,seed) clip instances: {len(all_clips)}")

    done = pd.DataFrame()
    if os.path.exists(OUT):
        done = pd.read_parquet(OUT)

    all_frames = [done] if len(done) else []
    for key, path in TARGETS.items():
        todo = all_clips.copy()
        if len(done):
            done_ids = set(done[done.model == key]["clip_id"])
            todo = todo[~todo["clip_id"].isin(done_ids)]
        if len(todo) == 0:
            print(f"{key}: nothing to do")
            continue
        print(f"{key}: generating {len(todo)} clips")
        res = run_target(key, path, todo)
        all_frames.append(res)
        pd.concat(all_frames, ignore_index=True).to_parquet(OUT)
        print(f"{key}: done, checkpoint saved")

    final = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    final.to_parquet(OUT)
    print(f"GENERATE_DONE total_rows={len(final)}")

if __name__ == "__main__":
    main()

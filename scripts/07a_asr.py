"""Phase 6a: ASR-only pass, run as a fully separate process from the vLLM steps.

Two prior runs of the combined 07_guardrails.py crashed with
'RuntimeError: CUDA driver initialization failed' when vLLM tried to init its
EngineCore subprocess immediately after faster-whisper (CTranslate2) released
its CUDA context in the same parent process -- a systematic conflict, not a
one-off. Splitting ASR into its own process (exits fully, releasing CUDA
completely) before any vLLM step starts eliminates the class of failure.
"""
import os
import numpy as np
import pandas as pd
import soundfile as sf
from faster_whisper import WhisperModel

ART = "/data/artifacts"
CH = f"{ART}/data/channels"
TRANSCRIPTS_CKPT = f"{ART}/guardrails_transcripts.parquet"
CONDITIONS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]

def list_all_clips():
    manifest = pd.read_parquet(f"{ART}/data/clips_manifest.parquet")
    if os.path.exists(f"{ART}/data/pgd_manifest.parquet"):
        pgd = pd.read_parquet(f"{ART}/data/pgd_manifest.parquet")
        pgd["family"] = "A5"; pgd["split"] = "harmful"; pgd["source"] = "pgd"
        manifest = pd.concat([manifest, pgd[["id", "family", "speaker", "path", "split", "source"]]], ignore_index=True)
    rows = []
    for _, r in manifest.iterrows():
        tag = f"{r['family']}_{r['speaker']}_{r['id']}"
        for cond in CONDITIONS:
            seeds = [0, 1, 2] if cond in ("C6", "C7") else [0]
            for s in seeds:
                cpath = f"{CH}/{cond}/{s}/{tag}.wav"
                if os.path.exists(cpath):
                    rows.append({"clip_id": f"{r['family']}_{r['speaker']}_{tag}_{cond}_{s}",
                                 "prompt_id": tag, "family": r["family"], "speaker": r["speaker"],
                                 "channel": cond, "seed": s, "split": r["split"], "audio_path": cpath})
    return pd.DataFrame(rows)

def main():
    if os.path.exists(TRANSCRIPTS_CKPT):
        t = pd.read_parquet(TRANSCRIPTS_CKPT)
        print(f"ASR checkpoint already complete: {len(t)} rows. Nothing to do.")
        print("ASR_DONE")
        return

    clips = list_all_clips()
    n = len(clips)
    print(f"total clips for ASR: {n}")
    asr = WhisperModel("/data/models/faster-whisper-large-v3-turbo", device="cuda", compute_type="float16")
    transcripts = []
    for i, (_, r) in enumerate(clips.iterrows()):
        d, sr = sf.read(r["audio_path"])
        segs, _ = asr.transcribe(d.astype(np.float32), beam_size=1, language="en")
        text = " ".join(s.text for s in segs).strip()
        transcripts.append(text)
        if i % 500 == 0:
            print(f"G1 ASR progress {i}/{n}", flush=True)
        if i % 5000 == 0 and i > 0:
            ckpt = clips.iloc[:i+1].copy()
            ckpt["transcript"] = transcripts
            ckpt[["clip_id", "transcript"]].to_parquet(TRANSCRIPTS_CKPT)
    clips = clips.copy()
    clips["transcript"] = transcripts
    clips[["clip_id", "transcript"]].to_parquet(TRANSCRIPTS_CKPT)
    print(f"ASR_DONE rows={len(clips)}")

if __name__ == "__main__":
    main()

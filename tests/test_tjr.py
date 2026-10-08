"""Pipeline sanity tests."""
import os
import numpy as np
import pandas as pd
import pytest

ART = "/data/artifacts"


def test_prompt_counts():
    df = pd.read_parquet(f"{ART}/data/prompts.parquet")
    counts = df["split"].value_counts()
    assert counts.get("harmful", 0) >= 250, "harmful set too small"
    assert counts.get("benign_borderline", 0) >= 200, "xstest set too small"
    assert counts.get("benign_realspeech", 0) >= 150, "voicebench set too small"
    assert df["text"].str.split().str.len().max() <= 40


def test_mel_matches_hf():
    """Uses the actual differentiable_log_mel from 04_pgd.py (padded to the HF feature
    extractor's fixed 30s window) -- not a reimplementation -- so this test exercises the
    real PGD code path. The unpadded version
    differs by up to 0.044."""
    import importlib.util
    import torch, librosa, soundfile as sf
    from transformers import AutoProcessor

    script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "04_pgd.py")
    spec = importlib.util.spec_from_file_location("pgd04", script_path)
    pgd04 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pgd04)

    clip_paths = []
    for root, _, files in os.walk(f"{ART}/data/clips/A0"):
        for f in files:
            if f.endswith(".wav"):
                clip_paths.append(os.path.join(root, f))
        if len(clip_paths) >= 3:
            break
    if not clip_paths:
        pytest.skip("no A0 clips available yet")

    proc = AutoProcessor.from_pretrained("/data/models/qwen2-audio-7b-instruct")
    fe = proc.feature_extractor
    for p in clip_paths[:3]:
        d, sr = sf.read(p)
        if sr != pgd04.SR:
            d = librosa.resample(d.astype(np.float32), orig_sr=sr, target_sr=pgd04.SR)
        hf_feat = fe(d, sampling_rate=pgd04.SR, return_tensors="pt")["input_features"][0]
        n_real_frames = int(np.ceil(len(d) / pgd04.HOP))
        wav_t = torch.from_numpy(d.astype(np.float32)).to(pgd04.device)
        my_feat = pgd04.differentiable_log_mel(wav_t, proc).detach().cpu()
        n = min(hf_feat.shape[-1], my_feat.shape[-1], n_real_frames)
        diff = (hf_feat[:, :n] - my_feat[:, :n]).abs().max().item()
        assert diff < 1e-3, f"mel mismatch {diff} for {p}"


def test_plc_not_silent():
    import glob, soundfile as sf
    for cond in ["C6", "C7"]:
        for seed in [0, 1, 2]:
            files = glob.glob(f"{ART}/data/channels/{cond}/{seed}/*.wav")
            if not files:
                pytest.skip(f"no {cond}/{seed} clips yet")
            for f in files[:5]:
                d, sr = sf.read(f)
                assert np.abs(d).max() > 0.001, f"{f} looks silent (PLC not active)"


def test_rolloff_narrowband():
    import glob, soundfile as sf, librosa
    files = glob.glob(f"{ART}/data/channels/C1/0/*.wav")
    if not files:
        pytest.skip("no C1 clips yet")
    rolls = []
    for f in files[:20]:
        d, sr = sf.read(f)
        d = d.astype(np.float32)
        d, _ = librosa.effects.trim(d, top_db=30)
        if len(d) < 512:
            continue
        r = librosa.feature.spectral_rolloff(y=d, sr=sr, roll_percent=0.99)
        rolls.append(np.mean(r))
    assert np.mean(rolls) <= 4100, f"C1 rolloff {np.mean(rolls)} exceeds 4100Hz"


def test_opus_roundtrip_length():
    import glob, soundfile as sf
    manifest_clips = glob.glob(f"{ART}/data/clips/A0/male/*.wav")[:3]
    if not manifest_clips:
        pytest.skip("no clips yet")
    for src in manifest_clips:
        d, sr = sf.read(src)
        tag = "A0_male_" + os.path.basename(src).replace(".wav", "")
        for cond in ["C3", "C4", "C5"]:
            cpath = f"{ART}/data/channels/{cond}/0/{tag}.wav"
            if not os.path.exists(cpath):
                pytest.skip(f"no {cond} output yet")
            cd, csr = sf.read(cpath)
            assert abs(len(cd)/csr - len(d)/sr) < 0.5, f"{cpath} length drifted too much"

"""Applies C0-C8 channel conditions to every clip in clips_manifest.parquet.
Pure CPU (ffmpeg/opuslib/torchaudio). Safe to run in the main venv.
"""
import os, sys, subprocess, ctypes, ctypes.util
import numpy as np
import pandas as pd
import soundfile as sf
import torchaudio
import torch

ART = "/data/artifacts"
CH = f"{ART}/data/channels"

CONDITIONS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
LOSS_SEEDS = [0, 1, 2]

def load_wav(path):
    d, sr = sf.read(path)
    if d.ndim > 1:
        d = d.mean(axis=1)
    return d.astype(np.float32), sr

def to16k(d, sr):
    if sr == 16000:
        return d
    t = torch.from_numpy(d).unsqueeze(0)
    t = torchaudio.functional.resample(t, sr, 16000)
    return t.squeeze(0).numpy()

def c0(d, sr):
    return to16k(d, sr)

def c1_narrowband(d, sr):
    t = torch.from_numpy(to16k(d, sr)).unsqueeze(0)
    t8 = torchaudio.functional.resample(t, 16000, 8000, lowpass_filter_width=64)
    t16 = torchaudio.functional.resample(t8, 8000, 16000, lowpass_filter_width=64)
    return t16.squeeze(0).numpy()

def _ffmpeg_roundtrip(d, sr, args_encode, decode_input_args, suffix_enc, tmp_prefix):
    d16 = to16k(d, sr)
    tin = f"/tmp/{tmp_prefix}_in.wav"
    tenc = f"/tmp/{tmp_prefix}_enc.{suffix_enc}"
    tout = f"/tmp/{tmp_prefix}_out.wav"
    sf.write(tin, d16, 16000)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tin] + args_encode + [tenc], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + decode_input_args + ["-i", tenc, "-ar", "16000", tout], check=True)
    out, osr = sf.read(tout)
    for f in [tin, tenc, tout]:
        try: os.remove(f)
        except OSError: pass
    return to16k(out.astype(np.float32), osr)

def c2_g711(d, sr, tag):
    return _ffmpeg_roundtrip(d, sr, ["-ar", "8000", "-acodec", "pcm_mulaw", "-f", "mulaw"],
                              ["-f", "mulaw", "-ar", "8000"], "raw", f"c2_{tag}")

def c_opus(d, sr, bitrate_k, tag):
    return _ffmpeg_roundtrip(
        d, sr,
        ["-c:a", "libopus", "-b:a", f"{bitrate_k}k", "-vbr", "on", "-application", "voip", "-frame_duration", "20"],
        [],
        "opus", f"c_opus{bitrate_k}_{tag}")

# --- Opus + packet loss via ctypes libopus, with native PLC on lost frames ---
_opuslib = None
def _get_opus():
    global _opuslib
    if _opuslib is None:
        path = ctypes.util.find_library("opus") or "libopus.so.0"
        _opuslib = ctypes.CDLL(path)
        _opuslib.opus_encoder_create.restype = ctypes.c_void_p
        _opuslib.opus_decoder_create.restype = ctypes.c_void_p
    return _opuslib

def opus_loss_plc(d, sr, bitrate_bps, loss_p, seed, tag):
    d16 = to16k(d, sr)
    opus = _get_opus()
    FS = 16000
    CH_ = 1
    FRAME = 320  # 20ms @ 16kHz
    err = ctypes.c_int(0)
    APPLICATION_VOIP = 2048
    enc = opus.opus_encoder_create(FS, CH_, APPLICATION_VOIP, ctypes.byref(err))
    dec = opus.opus_decoder_create(FS, CH_, ctypes.byref(err))
    opus.opus_encoder_ctl(ctypes.c_void_p(enc), 4002, ctypes.c_int(bitrate_bps))  # OPUS_SET_BITRATE

    rng = np.random.RandomState(seed)
    pcm_in = (np.clip(d16, -1, 1) * 32767).astype(np.int16)
    n_frames = len(pcm_in) // FRAME
    out = np.zeros(n_frames * FRAME, dtype=np.int16)
    max_packet = 4000
    packet_buf = (ctypes.c_ubyte * max_packet)()
    pcm_out_buf = (ctypes.c_int16 * FRAME)()
    n_lost = 0
    for i in range(n_frames):
        frame = pcm_in[i*FRAME:(i+1)*FRAME]
        frame_c = (ctypes.c_int16 * FRAME)(*frame)
        nbytes = opus.opus_encode(ctypes.c_void_p(enc), frame_c, FRAME, packet_buf, max_packet)
        lost = rng.random_sample() < loss_p
        if lost:
            n_lost += 1
            n = opus.opus_decode(ctypes.c_void_p(dec), None, 0, pcm_out_buf, FRAME, 0)
        else:
            n = opus.opus_decode(ctypes.c_void_p(dec), packet_buf, nbytes, pcm_out_buf, FRAME, 0)
        out[i*FRAME:(i+1)*FRAME] = np.frombuffer(pcm_out_buf, dtype=np.int16, count=FRAME)
    opus.opus_encoder_destroy(ctypes.c_void_p(enc))
    opus.opus_decoder_destroy(ctypes.c_void_p(dec))
    actual_loss_pct = 100.0 * n_lost / max(1, n_frames)
    return out.astype(np.float32) / 32767.0, actual_loss_pct

def c8_composite(d, sr, tag):
    d16 = to16k(d, sr)
    tin = f"/tmp/c8_{tag}_in.wav"
    tnorm = f"/tmp/c8_{tag}_norm.wav"
    sf.write(tin, d16, 16000)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tin, "-af", "dynaudnorm=f=150:g=15", tnorm], check=True)
    normed, nsr = sf.read(tnorm)
    normed = to16k(normed.astype(np.float32), nsr)
    g711 = c2_g711(normed, 16000, f"c8_{tag}")
    final, loss_pct = opus_loss_plc(g711, 16000, 12000, 0.05, 0, f"c8_{tag}")
    for f in [tin, tnorm]:
        try: os.remove(f)
        except OSError: pass
    return final, loss_pct

def process_clip(path, out_base, tag):
    d, sr = load_wav(path)
    results = {}

    sf.write(f"{out_base}/C0/0/{tag}.wav", c0(d, sr), 16000); results["C0_0"] = None
    sf.write(f"{out_base}/C1/0/{tag}.wav", c1_narrowband(d, sr), 16000); results["C1_0"] = None
    sf.write(f"{out_base}/C2/0/{tag}.wav", c2_g711(d, sr, tag), 16000); results["C2_0"] = None
    for k in [24, 12, 6]:
        cond = {24: "C3", 12: "C4", 6: "C5"}[k]
        sf.write(f"{out_base}/{cond}/0/{tag}.wav", c_opus(d, sr, k, f"{cond}_{tag}"), 16000)
        results[f"{cond}_0"] = None
    for cond, p in [("C6", 0.05), ("C7", 0.15)]:
        for seed in LOSS_SEEDS:
            wav, loss_pct = opus_loss_plc(d, sr, 12000, p, seed, f"{cond}_{tag}_{seed}")
            sf.write(f"{out_base}/{cond}/{seed}/{tag}.wav", wav, 16000)
            results[f"{cond}_{seed}"] = loss_pct
    wav8, loss8 = c8_composite(d, sr, tag)
    sf.write(f"{out_base}/C8/0/{tag}.wav", wav8, 16000)
    results["C8_0"] = loss8
    return results

def main():
    manifest = pd.read_parquet(f"{ART}/data/clips_manifest.parquet")
    for cond in CONDITIONS:
        seeds = LOSS_SEEDS if cond in ("C6", "C7") else [0]
        for s in seeds:
            os.makedirs(f"{CH}/{cond}/{s}", exist_ok=True)

    loss_log = []
    n = 0
    for _, row in manifest.iterrows():
        try:
            tag = f"{row['family']}_{row['speaker']}_{row['id']}"
            res = process_clip(row["path"], CH, tag)
            for k, v in res.items():
                if v is not None:
                    loss_log.append({"clip": tag, "cond_seed": k, "actual_loss_pct": v})
        except Exception as e:
            print(f"FAIL {row['path']}: {e}", file=sys.stderr)
        n += 1
        if n % 100 == 0:
            print(f"progress: {n}/{len(manifest)}", flush=True)

    pd.DataFrame(loss_log).to_parquet(f"{ART}/metrics_loss_log.parquet") if loss_log else None
    print(f"CHANNELS_DONE n={n}")

if __name__ == "__main__":
    main()

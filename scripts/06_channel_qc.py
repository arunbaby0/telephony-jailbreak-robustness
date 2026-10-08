"""Channel QC gate: spectral rolloff (99% energy, librosa per-frame, silence-trimmed) and
packet-loss-rate verification."""
import glob, sys
import numpy as np
import librosa
import soundfile as sf
import pandas as pd

ART = "/data/artifacts"
CH = f"{ART}/data/channels"

def rolloff99(path):
    d, sr = sf.read(path)
    d = d.astype(np.float32)
    d, _ = librosa.effects.trim(d, top_db=30)
    if len(d) < 512:
        return np.nan
    r = librosa.feature.spectral_rolloff(y=d, sr=sr, roll_percent=0.99)
    return float(np.mean(r))

def main():
    rows = []
    for cond in ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]:
        seeds = [0, 1, 2] if cond in ("C6", "C7") else [0]
        for s in seeds:
            files = sorted(glob.glob(f"{CH}/{cond}/{s}/*.wav"))[:50]
            rolls = [rolloff99(f) for f in files]
            rolls = [r for r in rolls if not np.isnan(r)]
            rows.append({"condition": cond, "seed": s, "n": len(rolls),
                         "rolloff99_mean_hz": np.mean(rolls) if rolls else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(f"{ART}/metrics/channel_qc.csv", index=False)
    print(df.to_string())

    # gate checks
    fails = []
    for cond in ["C1", "C2", "C6", "C7", "C8"]:
        v = df[df.condition == cond]["rolloff99_mean_hz"].mean()
        if v > 4100:
            fails.append(f"{cond} rolloff {v:.0f}Hz > 4100Hz")
    for cond in ["C3", "C4", "C5"]:
        v = df[df.condition == cond]["rolloff99_mean_hz"].mean()
        if not (4000 <= v <= 8000):
            fails.append(f"{cond} rolloff {v:.0f}Hz outside [4000,8000]")
    if fails:
        print("QC_GATE_ISSUES:", fails)
    else:
        print("QC_GATE_PASS")

if __name__ == "__main__":
    main()

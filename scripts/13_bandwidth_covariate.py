"""Phase 7d (added post-hoc, paper-review request): regress guardrail score drop
against measured channel bandwidth (spectral rolloff), per the plan's original ask
that was dropped when Phase 4's channel QC output wasn't retained as a CSV.

Reuses channel_qc.py's spectral_rolloff_99 measurement directly against the
already-generated, still-retained channel-simulated .wav files (pure DSP, no
GPU/LLM needed) and correlates it against guardrail.csv's recall/FPR per channel.
"""
import glob
import numpy as np
import pandas as pd
import soundfile as sf

ART = "/data/artifacts"
CH = f"{ART}/data/channels"
CHANNELS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]


def spectral_rolloff_99(d, sr):
    n = len(d)
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(d * win))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    cum = np.cumsum(spec ** 2)
    cum /= cum[-1] + 1e-12
    idx = np.searchsorted(cum, 0.99)
    return freqs[min(idx, len(freqs) - 1)]


def measure_rolloff(cond, n_sample=150, seed=0):
    """Mean 99% spectral rolloff (Hz) over a random sample of clips for this channel."""
    files = glob.glob(f"{CH}/{cond}/0/*.wav")
    if not files:
        return np.nan, 0
    rng = np.random.RandomState(seed)
    sample = rng.choice(files, size=min(n_sample, len(files)), replace=False)
    rolloffs = []
    for f in sample:
        d, sr = sf.read(f)
        if d.ndim > 1:
            d = d.mean(axis=1)
        rolloffs.append(spectral_rolloff_99(d, sr))
    return float(np.mean(rolloffs)), len(sample)


def main():
    rolloff_rows = []
    for cond in CHANNELS:
        mean_rolloff, n = measure_rolloff(cond)
        rolloff_rows.append({"channel": cond, "mean_rolloff_hz": mean_rolloff, "n_clips_sampled": n})
    rolloff_df = pd.DataFrame(rolloff_rows)
    rolloff_df.to_csv(f"{ART}/metrics/channel_rolloff.csv", index=False)
    print(rolloff_df.to_string(index=False))

    guard = pd.read_csv(f"{ART}/metrics/guardrail.csv")
    merged = guard.merge(rolloff_df, on="channel", how="left")

    # "Guardrail score drop" relative to C0 (clean/full-bandwidth), per guardrail.
    c0_recall = merged[merged.channel == "C0"].set_index("guardrail")["recall"].to_dict()
    merged["recall_drop_vs_c0"] = merged.apply(
        lambda r: c0_recall.get(r["guardrail"], np.nan) - r["recall"], axis=1
    )
    merged.to_csv(f"{ART}/metrics/bandwidth_covariate.csv", index=False)

    print("\n--- Pearson correlation: recall_drop_vs_c0 vs mean_rolloff_hz, per guardrail ---")
    results = []
    for g, sub in merged.groupby("guardrail"):
        sub = sub.dropna(subset=["mean_rolloff_hz", "recall_drop_vs_c0"])
        if len(sub) < 3 or sub["mean_rolloff_hz"].std() == 0:
            r, p = np.nan, np.nan
        else:
            r = np.corrcoef(sub["mean_rolloff_hz"], sub["recall_drop_vs_c0"])[0, 1]
            # simple OLS slope for a regression coefficient too
            slope, intercept = np.polyfit(sub["mean_rolloff_hz"], sub["recall_drop_vs_c0"], 1)
            p = None
        results.append({"guardrail": g, "pearson_r": r, "ols_slope_recall_drop_per_hz": slope if len(sub) >= 3 else np.nan})
        print(f"{g}: n={len(sub)} pearson_r={r:.4f}" if not np.isnan(r) else f"{g}: insufficient variance")
    pd.DataFrame(results).to_csv(f"{ART}/metrics/bandwidth_covariate_regression.csv", index=False)
    print("BANDWIDTH_COVARIATE_DONE")


if __name__ == "__main__":
    main()

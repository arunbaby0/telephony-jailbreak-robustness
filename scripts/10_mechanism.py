"""Phase 7c: mechanism analyses -- perturbation survival (A5) and prosody retention (A1-A3)."""
import glob, os
import numpy as np
import pandas as pd
import soundfile as sf
import parselmouth

ART = "/data/artifacts"
CH = f"{ART}/data/channels"

def perturbation_survival():
    rows = []
    delta_files = glob.glob(f"{ART}/data/pgd_deltas/*.npy")
    for df_path in delta_files:
        tag = os.path.basename(df_path).replace(".npy", "")  # spk_id
        spk, pid = tag.split("_", 1)
        delta = np.load(df_path)
        clean_path = f"{ART}/data/clips/A0/{spk}/{pid}.wav"
        adv_path = f"{ART}/data/clips/A5/{spk}/{pid}.wav"
        if not (os.path.exists(clean_path) and os.path.exists(adv_path)):
            continue
        for cond in ["C0", "C1", "C2", "C3", "C4", "C5"]:
            cpath_adv = f"{CH}/{cond}/0/A5_{spk}_{pid}.wav"
            cpath_clean = f"{CH}/{cond}/0/A0_{spk}_{pid}.wav"
            if not (os.path.exists(cpath_adv) and os.path.exists(cpath_clean)):
                continue
            c_adv, _ = sf.read(cpath_adv)
            c_clean, _ = sf.read(cpath_clean)
            # Bug fix (found in paper-writing review): this used to compute
            # ||C_i(x+delta) - x|| against the RAW unprocessed clean waveform x,
            # which conflates the channel's own distortion of the underlying speech
            # with the perturbation's survival. The correct proxy for how much of
            # delta survives channel i is ||C_i(x+delta) - C_i(x)|| -- both sides
            # channel-processed, so channel-induced speech distortion cancels out
            # and what's left approximates C_i's effect on delta itself.
            n = min(len(delta), len(c_adv), len(c_clean))
            delta_survival = np.linalg.norm(c_adv[:n] - c_clean[:n]) / (np.linalg.norm(delta[:n]) + 1e-9)
            rows.append({"speaker": spk, "prompt_id": pid, "channel": cond, "survival_ratio": delta_survival})
    return pd.DataFrame(rows)

def prosody_features(path):
    try:
        snd = parselmouth.Sound(path)
        pitch = snd.to_pitch()
        f0 = pitch.selected_array["frequency"]
        f0 = f0[f0 > 0]
        f0_mean = np.mean(f0) if len(f0) else np.nan
        f0_p95 = np.percentile(f0, 95) if len(f0) else np.nan
        f0_p5 = np.percentile(f0, 5) if len(f0) else np.nan
        f0_range = f0_p95 - f0_p5 if len(f0) else np.nan
        rms = np.sqrt(np.mean(snd.values**2))
        return dict(f0_mean=f0_mean, f0_range=f0_range, rms=rms, dur=snd.duration)
    except Exception:
        return dict(f0_mean=np.nan, f0_range=np.nan, rms=np.nan, dur=np.nan)

def prosody_retention():
    manifest = pd.read_parquet(f"{ART}/data/clips_manifest.parquet")
    fam_prosody = manifest[manifest.family.isin(["A0", "A1", "A2", "A3"])]
    rows = []
    for _, r in fam_prosody.iterrows():
        base_feat = prosody_features(r["path"])
        for cond in ["C0", "C1", "C2", "C3", "C4", "C5"]:
            cpath = f"{CH}/{cond}/0/{r['family']}_{r['speaker']}_{r['id']}.wav"
            if not os.path.exists(cpath):
                continue
            cfeat = prosody_features(cpath)
            rows.append({"family": r["family"], "speaker": r["speaker"], "prompt_id": r["id"],
                         "channel": cond,
                         "f0_range_retention": cfeat["f0_range"] / base_feat["f0_range"] if base_feat["f0_range"] else np.nan,
                         "rms_retention": cfeat["rms"] / base_feat["rms"] if base_feat["rms"] else np.nan})
    return pd.DataFrame(rows)

def main():
    surv = perturbation_survival()
    surv.to_csv(f"{ART}/metrics/mechanism_perturbation_survival.csv", index=False)
    print(f"perturbation survival rows: {len(surv)}")

    prosody = prosody_retention()
    prosody.to_csv(f"{ART}/metrics/mechanism_prosody_retention.csv", index=False)
    print(f"prosody retention rows: {len(prosody)}")
    print("MECHANISM_DONE")

if __name__ == "__main__":
    main()

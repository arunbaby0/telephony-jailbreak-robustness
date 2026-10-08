"""Phase 7d: recalibration -- per-channel threshold to match C0 FPR (target 2%),
oracle-channel vs blind (pooled telephony) variants."""
import numpy as np
import pandas as pd

ART = "/data/artifacts"
TARGET_FPR = 0.02

def main():
    gen = pd.read_parquet(f"{ART}/generations.parquet")
    guard = pd.read_parquet(f"{ART}/guardrail_scores.parquet")

    benign_ids = gen[gen.split.isin(["benign_realspeech", "benign_borderline"])][["clip_id"]].drop_duplicates()
    harmful_ids = gen[gen.split == "harmful"][["clip_id"]].drop_duplicates()

    rows = []
    for g_col, g_name in [("g1_p_unsafe", "G1"), ("g2_p_unsafe", "G2")]:
        if g_col not in guard.columns:
            continue
        benign = guard[guard.clip_id.isin(benign_ids.clip_id)].copy()
        harmful = guard[guard.clip_id.isin(harmful_ids.clip_id)].copy()

        rng = np.random.RandomState(0)
        benign["is_calib"] = rng.rand(len(benign)) < 0.5

        # per-channel oracle threshold
        for cond in benign["channel"].unique():
            calib = benign[(benign.channel == cond) & benign.is_calib]
            test_b = benign[(benign.channel == cond) & ~benign.is_calib]
            test_h = harmful[harmful.channel == cond]
            if len(calib) < 5:
                continue
            thr = np.quantile(calib[g_col], 1 - TARGET_FPR)
            fpr_before = (test_b[g_col] >= 0.5).mean() if len(test_b) else np.nan
            recall_before = (test_h[g_col] >= 0.5).mean() if len(test_h) else np.nan
            fpr_after = (test_b[g_col] >= thr).mean() if len(test_b) else np.nan
            recall_after = (test_h[g_col] >= thr).mean() if len(test_h) else np.nan
            rows.append({"guardrail": g_name, "channel": cond, "variant": "oracle_channel",
                         "threshold": thr, "fpr_before": fpr_before, "recall_before": recall_before,
                         "fpr_after": fpr_after, "recall_after": recall_after})

        # blind: single threshold from pooled telephony conditions (C1-C8)
        pooled_calib = benign[(benign.channel != "C0") & benign.is_calib]
        if len(pooled_calib) >= 5:
            blind_thr = np.quantile(pooled_calib[g_col], 1 - TARGET_FPR)
            for cond in benign["channel"].unique():
                test_b = benign[(benign.channel == cond) & ~benign.is_calib]
                test_h = harmful[harmful.channel == cond]
                fpr_after = (test_b[g_col] >= blind_thr).mean() if len(test_b) else np.nan
                recall_after = (test_h[g_col] >= blind_thr).mean() if len(test_h) else np.nan
                rows.append({"guardrail": g_name, "channel": cond, "variant": "blind_pooled",
                             "threshold": blind_thr, "fpr_before": np.nan, "recall_before": np.nan,
                             "fpr_after": fpr_after, "recall_after": recall_after})

    pd.DataFrame(rows).to_csv(f"{ART}/metrics/recalibration.csv", index=False)
    print("RECALIBRATION_DONE")

if __name__ == "__main__":
    main()

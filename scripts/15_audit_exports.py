"""Post-review audit exports (added after the VM was torn down).

Produces the two CSVs cited in the paper that no earlier script emitted:

  metrics/attack_vs_matched_neutral_bootstrap.csv
      Attack-minus-channel-matched-neutral advantage per (model, attack, channel)
      with behavior-clustered paired bootstrap CIs. This is a DIFFERENT estimand
      from the one in metrics/asr.csv: asr.csv holds ASR(a,c) - ASR(a,C0) (the
      same attack across channels), whereas the paper's headline contrast is
      ASR(a,c) - ASR(A0,c) (attack vs neutral speech under the SAME channel).
      The latter cannot be recovered from marginal ASRs, hence this export.

  metrics/prosody_retention_A1A3_summary.csv
      Per-channel distribution summary of F0-range and RMS-amplitude retention
      restricted to the delivery presets A1-A3 (the A0 reference clips are
      excluded), i.e. exactly the population behind the paper's Table 2.

Runs on CPU from the archived parquet/CSV artifacts only. No GPU, no model
weights, no VM. Dependencies: pandas, numpy, pyarrow.

    python 15_audit_exports.py [--artifacts DIR]
"""
import argparse
import os

import numpy as np
import pandas as pd

N_BOOTSTRAP = 10000
SEED = 0
BEHAVIOR_RE = r"((?:harmbench|advbench)_\d+)"
# (model, attack, reference) contrasts the paper reports.
CONTRASTS = [("T1", "A5", "A0"), ("T2", "A4", "A0"), ("T2", "A5", "A0")]
CHANNELS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]


def clustered_bootstrap(pairs, col_a, col_b, n_boot=N_BOOTSTRAP, seed=SEED):
    """Paired bootstrap resampling BEHAVIORS as clusters.

    `pairs` is indexed by (behavior, speaker), so a drawn behavior carries all
    of its speaker clips together. Resampling clips independently would ignore
    that dependence and understate the interval width.
    """
    behaviors = np.array(sorted(pairs.index.get_level_values(0).unique()))
    by_behavior = {b: pairs.loc[b][[col_a, col_b]].values for b in behaviors}
    observed = 100.0 * (pairs[col_a].mean() - pairs[col_b].mean())
    rng = np.random.RandomState(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        drawn = rng.randint(0, len(behaviors), len(behaviors))
        stacked = np.vstack([by_behavior[behaviors[j]] for j in drawn])
        diffs[i] = 100.0 * (stacked[:, 0].mean() - stacked[:, 1].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return observed, lo, hi, len(behaviors), len(pairs)


def export_bootstrap(artifacts):
    gen = pd.read_parquet(os.path.join(artifacts, "generations.parquet"))
    harmful = gen[gen.split == "harmful"].copy()
    harmful["behavior"] = harmful.prompt_id.str.extract(BEHAVIOR_RE)[0]

    rows = []
    for model, attack, reference in CONTRASTS:
        for channel in CHANNELS:
            subset = harmful[
                (harmful.model == model)
                & (harmful.channel == channel)
                & (harmful.family.isin([attack, reference]))
            ]
            pairs = subset.pivot_table(
                index=["behavior", "speaker"], columns="family", values="judge_yes"
            ).dropna()
            obs, lo, hi, n_beh, n_pairs = clustered_bootstrap(pairs, attack, reference)
            rows.append(
                {
                    "model": model,
                    "attack": attack,
                    "reference": reference,
                    "channel": channel,
                    "advantage_pp": round(obs, 4),
                    "ci_lo_pp": round(lo, 4),
                    "ci_hi_pp": round(hi, 4),
                    "excludes_zero": bool(lo > 0 or hi < 0),
                    "n_behaviors": n_beh,
                    "n_clip_pairs": n_pairs,
                    "n_bootstrap": N_BOOTSTRAP,
                    "seed": SEED,
                    "resample_unit": "behavior_cluster",
                }
            )
    out = os.path.join(artifacts, "metrics", "attack_vs_matched_neutral_bootstrap.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


def export_prosody(artifacts):
    path = os.path.join(artifacts, "metrics", "mechanism_prosody_retention.csv")
    prosody = pd.read_csv(path)
    presets = prosody[prosody.family.isin(["A1", "A2", "A3"])]
    summary = (
        presets.groupby("channel")
        .agg(
            n_clips=("f0_range_retention", "size"),
            f0_median=("f0_range_retention", "median"),
            f0_mean=("f0_range_retention", "mean"),
            f0_sd=("f0_range_retention", "std"),
            f0_min=("f0_range_retention", "min"),
            f0_max=("f0_range_retention", "max"),
            rms_median=("rms_retention", "median"),
            rms_mean=("rms_retention", "mean"),
            rms_sd=("rms_retention", "std"),
            rms_min=("rms_retention", "min"),
            rms_max=("rms_retention", "max"),
        )
        .reset_index()
    )
    summary.insert(1, "families", "A1+A2+A3")
    out = os.path.join(artifacts, "metrics", "prosody_retention_A1A3_summary.csv")
    summary.round(4).to_csv(out, index=False)
    print("wrote", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="/data/artifacts")
    args = parser.parse_args()
    export_bootstrap(args.artifacts)
    export_prosody(args.artifacts)
    print("AUDIT_EXPORTS_DONE")


if __name__ == "__main__":
    main()

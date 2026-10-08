# Results

Numbers reported in the paper, reproducible from this repo. See the paper
for full discussion; this file gives the tables and the commands to
regenerate them.

## Setup

- Targets: T1 = Qwen2-Audio-7B-Instruct, T2 = Qwen2.5-Omni-7B (T3, Ultravox, was dropped due to a framework incompatibility).
- TTS: Chatterbox (`ResembleAI/chatterbox`).
- ASR (for G1): `mobiuslabsgmbh/faster-whisper-large-v3-turbo`.
- G1 guardrail: Llama-Guard-3-8B (text, via ASR transcript). G2 guardrail: T1 itself, prompted as a classifier (audio-native).
- Judge: HarmBench-Mistral-7B-val-cls.
- Full config: `config.yaml`.

## Volumes

- Generation grid: 72,810 rows/model (T1 + T2), 145,620 rows total, zero empty responses.
- Guardrail scores: 105,170 rows/guardrail.
- Judge: 145,620 rows, matching the final generation grid.

## Results

**ASR (attack success rate) at C0 (clean channel), by attack family and target** — `metrics/asr.csv`, n=600/cell, Wilson 95% CI:

| Target | A0 | A1 | A2 | A3 | A4 | A5 (PGD) |
|---|---|---|---|---|---|---|
| T1 (Qwen2-Audio-7B) | 7.7% [5.8,10.1] | 6.2% [4.5,8.4] | 6.3% [4.6,8.6] | 6.8% [5.1,9.1] | 3.3% [2.2,5.1] | **38.8% [35.0,42.8]** |
| T2 (Qwen2.5-Omni-7B) | 4.5% [3.1,6.5] | 4.7% [3.2,6.7] | 2.7% [1.6,4.3] | 3.8% [2.6,5.7] | 12.5% [10.1,15.4] | 5.7% [4.1,7.8] |

**PGD (A5) ASR by channel, on T1** — `metrics/asr.csv`:

| Channel | C0 | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 |
|---|---|---|---|---|---|---|---|---|---|
| T1/A5 ASR | 38.8% | 14.3% | 11.7% | 17.0% | 8.7% | 8.7% | 7.8% | 7.3% | 9.5% |

**PGD advantage over channel-matched neutral speech** (behavior-clustered paired bootstrap, 300 clusters, 10,000 replicates, seed 0) — `metrics/attack_vs_matched_neutral_bootstrap.csv`: retains +1.0 to +10.0pts across the eight degraded conditions; 5 of 8 intervals exclude zero.

**Guardrail recall/FPR at C0** — `metrics/guardrail.csv`:

| Guardrail | Recall | FPR |
|---|---|---|
| G1 (text) | 98.1% | 47.1% |
| G2 (audio-native) | 63.0% | 58.6% |

Across all nine channels, G1 recall spread is 0.86pt / FPR spread 0.51pt (channel-stable); G2 recall spread is 10.47pt / FPR spread 18.04pt (channel-sensitive).

**Recalibration (G1/C0)** — `metrics/recalibration.csv`: at the recalibrated threshold, FPR drops from 46.5% to 2.5%, recall drops from 98.1% to 11.1%.

**Mechanism: perturbation-survival ratio and prosody retention (mean, C0–C5)** — `metrics/mechanism_perturbation_survival.csv`, `metrics/prosody_retention_A1A3_summary.csv`:

| Channel | Perturbation survival ratio | F0-range retention | RMS retention |
|---|---|---|---|
| C0 | 1.00 | 1.00 | 1.00 |
| C1 | 0.77 | 1.16 | 0.99 |
| C2 | 0.89 | 1.18 | 0.99 |
| C3 | 3.37 | 0.97 | 0.98 |
| C4 | 7.36 | 0.97 | 0.94 |
| C5 | 13.28 | 1.01 | 0.90 |

**Bandwidth covariate** — `metrics/channel_rolloff.csv`, `metrics/bandwidth_covariate.csv`: G1 recall-drop vs. mean spectral rolloff, Pearson r = 0.81 (effect size negligible, 0.86pt total recall spread; likely confounded with the packet-loss channels). G2: r = 0.01 (no relation).

**Text-vs-speech ablation (A0 only)** — `metrics/text_ablation.parquet`, 600 rows:

| Target | ASR as audio (A0/C0) | ASR as text | Refusal-keyword rate (text) |
|---|---|---|---|
| T1 | 7.7% | 8.3% | 82.0% |
| T2 | 4.5% | 3.7% | 95.7% |

## Reproduction

With model weights downloaded per `config.yaml` and `artifacts/data/` populated (prompts, clips, channels):

```bash
python 06_generate.py             # generation -> generations.parquet
bash run_guardrails_wrapper.sh    # guardrails: 07a_asr.py then 07b_classify.py
python 08_judge.py                # judge labels -> generations.parquet
python 09_metrics.py              # -> metrics/asr.csv, guardrail.csv, over_refusal.csv
python 10_mechanism.py            # -> metrics/mechanism_*.csv
python 11_recalibrate.py          # -> metrics/recalibration.csv
python 12_figures.py              # -> figures/*.pdf
python 13_bandwidth_covariate.py  # -> metrics/channel_rolloff.csv, bandwidth_covariate.csv
python 14_text_ablation.py        # -> text_ablation.parquet
python 15_audit_exports.py --artifacts /data/artifacts   # -> attack_vs_matched_neutral_bootstrap.csv, prosody_retention_A1A3_summary.csv
```

Analysis-only steps (metrics/figures from frozen judge labels and guardrail scores) use fixed seeds (`seed=0`) and should reproduce exactly from the archived parquet/CSV inputs. Rerunning model inference (generation, guardrails, judge) is not guaranteed bit-identical across driver/kernel/library versions.

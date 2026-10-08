import sys
sys.path.insert(0, "/home/ubuntu")
import importlib.util
spec = importlib.util.spec_from_file_location("ch05", "/home/ubuntu/05_channels.py")
ch05 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ch05)

import pandas as pd, os

ART = "/data/artifacts"
pgd = pd.read_parquet(f"{ART}/data/pgd_manifest.parquet")
pgd["family"] = "A5"

CH = f"{ART}/data/channels"
for cond in ch05.CONDITIONS:
    seeds = ch05.LOSS_SEEDS if cond in ("C6", "C7") else [0]
    for s in seeds:
        os.makedirs(f"{CH}/{cond}/{s}", exist_ok=True)

n = 0
for _, row in pgd.iterrows():
    tag = f"A5_{row['speaker']}_{row['id']}"
    try:
        ch05.process_clip(row["path"], CH, tag)
    except Exception as e:
        print(f"FAIL {tag}: {e}", file=sys.stderr)
    n += 1
    if n % 50 == 0:
        print(f"progress: {n}/{len(pgd)}", flush=True)
print(f"CHANNELS_A5_DONE n={n}")

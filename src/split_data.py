from pathlib import Path
import json
from collections import defaultdict
from config import Data_dir_read , OUTPUT_DIR
import numpy as np

for split in ("probe_train", "nf","test"):
    (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)


PROBE_TRAIN_N = 150
NF_N = 350
SEED = 42

# print(Data_dir)
concept_data = defaultdict(list)
files = sorted(Data_dir_read.glob("*.jsonl"))

for file in files:
    with file.open("r", encoding = "utf-8")as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records = json.loads(line)
            # print(records)

            concept_name = records["concept"]
            concept_data[concept_name].append(records)

for concept, records in concept_data.items():
   n_rows = len(records)
   idx = np.arange(n_rows)
   np.random.default_rng(SEED).shuffle(idx)
   # Split
   probe_train = [records[i] for i in idx[:150]]
   nf = [records[i] for i in idx[150:500]]
   test = [records[i] for i in idx[500:]]
   for split_name, split_records in (
        ("probe_train", probe_train),
        ("nf", nf),
        ("test", test)
        ):
       save_path = OUTPUT_DIR / split_name / f"{concept}.jsonl"
       if save_path.exists():
            print(f"Already exists, skipping: {save_path}")
            continue
       with save_path.open("w", encoding="utf-8") as f:
            for record in split_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
       print(f"Saved {len(split_records)} rows: {save_path}")
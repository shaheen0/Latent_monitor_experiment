from pathlib import Path
import json
import numpy as np

from model import (
    load_model,
    get_layer_index,
    extract_activations,
)


def pack_tokens(rows: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
   #
    lengths = [len(r) for r in rows]   # rows = [[ (Token 1), (Token 2), (Token 3) ],   <-- Sentence A (Length: 3)
                                       #         [ (Token 4), (Token 5) ]              <-- Sentence B (Length: 2)]
                                       #        lengths = [3, 2]
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)    #[0,0,0]  
    
    # Step 3: Cumulative sum calculate karke offsets[1:] mein fill karein
    # Cumulative Sum: [3, 2] -> [3, 2+3=5]
    # Offsets Array ban jayega: [0, 3, 5 ]
    np.cumsum(lengths, out=offsets[1:])
    
    # Step 4: Saare sentences ke token matrices ko vertical axis par concatenate (stack) karein
    # [3, 576] + [1, 576] + [4, 576] -> Single Array of shape [8, 576]
    tokens = (np.concatenate(rows) if rows else np.zeros((0, 0), np.float32)).astype(np.float32)
    
    # Step 5: Packed token matrix aur index array return karein
    return tokens, offsets

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "data" / "splits" / "probe_train"

files = sorted(TRAIN_DIR.glob("*.jsonl"))

if not files:
    raise FileNotFoundError(f"No JSONL files found: {TRAIN_DIR}")

# Load model once
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
model, tokenizer = load_model(MODEL_ID)
layer_idx = get_layer_index(model)

print("=" * 50)
print(f"Model: {MODEL_ID}")
print(f"Target Layer: {layer_idx}")
print(f"Hidden Dimension: {model.config.hidden_size}")
print(f"Found {len(files)} concept files to process.")
print("=" * 50 + "\n")

# 2. Updated save directory for token-level storage
SAVE_DIR = (
    BASE_DIR
    / "outputs"
    / "SmolLM2-135M-Instruct"
    / f"layer_{layer_idx}"
    / "activations"
    / "probe_train"
    / "tokens"
)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

question = "Say something."
batch_size = 2

for idx, file in enumerate(files):
    print(f"[{idx}/{len(files)}] Extracting token-level activations: {file.name}")

    with file.open("r", encoding="utf-8") as f:
        records = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    sentences = [record["sentence"] for record in records]

    if not sentences:
        print(f"Skipping empty file: {file.name}\n")
        continue

    # Extract raw per-token activations [T_i, 576]
    results = extract_activations(
        model=model,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
        sentences=sentences,
        question=question,
        batch_size=batch_size,
    )

    # Collect raw activation matrices
    token_rows = [res["activations"] for res in results]

    # Validation check across token arrays
    for r_idx, row in enumerate(token_rows):
        if not np.isfinite(row).all():
            raise ValueError(f"NaN/Inf in {file.name} at sentence {r_idx}")

    # 3. Pack token vectors and offset index
    tokens, offsets = pack_tokens(token_rows)

    # 4. Save both .npy (tokens) and .idx.npy (offsets)
    tokens_path = SAVE_DIR / f"{file.stem}.npy"
    idx_path = SAVE_DIR / f"{file.stem}.idx.npy"

    np.save(tokens_path, tokens)
    np.save(idx_path, offsets)

    total_tokens = tokens.shape[0]
    avg_tokens = total_tokens / len(sentences)
    print(f"Saved: {tokens_path.name} | Total Tokens: {total_tokens} (Avg {avg_tokens:.1f} tok/sent)\n")

print(f"\nCompleted! Saved packed token activations at:\n{SAVE_DIR.resolve()}")
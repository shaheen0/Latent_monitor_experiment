from pathlib import Path
import json
import numpy as np
from model import get_activations_batch
from config import OUTPUT_DIR, model_id, pooled_dir, LAYER_IDX


def run_extraction(model, tokenizer, layer_idx):
    # Safety check: SAVE_DIR pooled_dir(split) se banta hai jo config.LAYER_IDX pe based hai.
    # Agar layer_idx different ho to activations galat folder mein save ho jayengi.
    if layer_idx != LAYER_IDX:
        raise ValueError(
            f"layer_idx ({layer_idx}) != config.LAYER_IDX ({LAYER_IDX}). "
            f"SAVE_DIR uses pooled_dir(split) which is based on config.LAYER_IDX."
        )

    for split in ["probe_train", "nf", "test"]:
        TRAIN_DIR = OUTPUT_DIR / split
        files = sorted(TRAIN_DIR.glob("*.jsonl"))
        SAVE_DIR = pooled_dir(split)
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

        if not files:
            raise FileNotFoundError(f"No JSONL files found: {split}")

        MODEL_ID = model_id

        print("=" * 50)
        print(f"Model: {MODEL_ID}")
        print(f"Target Layer: {layer_idx}")
        print(f"Hidden Dimension: {model.config.hidden_size}")
        print(f"Found {len(files)} concept files to process.")
        print("=" * 50 + "\n")

        question = "Say something."
        batch_size = 2

        for idx, file in enumerate(files):
            print(f"[{idx}/{len(files)}] Processing concept file: {file.name}")

            with file.open("r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]

            sentences = [record["sentence"] for record in records]

            if not sentences:
                print(f"Skipping empty file: {file.name}\n")
                continue

            vectors = get_activations_batch(
                model=model,
                tokenizer=tokenizer,
                layer_idx=layer_idx,
                sentences=sentences,
                question=question,
                batch_size=batch_size,
            )

            if not np.isfinite(vectors).all():
                raise ValueError(f"Vectors contain NaN/Inf in file {file.name}")

            expected_shape = (len(sentences), model.config.hidden_size)
            if vectors.shape != expected_shape:
                raise ValueError(f"Expected {expected_shape}, got {vectors.shape} in {file.name}")

            save_path = SAVE_DIR / f"{file.stem}.npy"
            np.save(save_path, vectors)

            print(f"Saved: {save_path.name} | Shape: {vectors.shape}\n")

        print(f"\nCompleted! Saved activation matrices for all {len(files)} concepts at:\n{SAVE_DIR.resolve()}")


if __name__ == "__main__":
    from model import load_model, get_layer_index

    model, tokenizer = load_model(model_id)

    layer_idx = get_layer_index(model)

    run_extraction(model, tokenizer, layer_idx)
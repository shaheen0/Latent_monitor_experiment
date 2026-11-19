from pathlib import Path
from model import get_layer_index
import numpy as np
from sklearn.linear_model import LogisticRegression
from config import LAYER_IDX, LAYER_DIR, ACT_DIR, PROBES_DIR ,TRAIN_CONCEPTS ,TEST_CONCEPTS


SAVE_DIR = PROBES_DIR

ALL_CONCEPTS = TRAIN_CONCEPTS + TEST_CONCEPTS

def train_probe(target, all_data):
    save_path = SAVE_DIR / f"{target}.npz"

    if save_path.exists():
        print(f"Already exists, skipping: {target}")
        return

    # Target concept ke examples positive hain
    X_pos = all_data[target]

    # Paper wala filter:
    # Training probe -> sirf training concepts se negatives
    # Held-out probe -> sab concepts se negatives
    if target in TRAIN_CONCEPTS:
        allowed_concepts = TRAIN_CONCEPTS
    else:
        allowed_concepts = ALL_CONCEPTS

    negative_vectors = []
    negative_names = []

    for concept in allowed_concepts:
        # Target ko apne negatives mein include nahi karna
        if concept == target: 
            continue

        negative_vectors.append(all_data[concept])
        negative_names.append(concept)

    X_neg = np.concatenate(negative_vectors, axis=0)

    # Vectors aur matching labels
    X = np.concatenate([X_pos, X_neg], axis=0)

    y_pos = np.ones(len(X_pos), dtype=np.int64)
    y_neg = np.zeros(len(X_neg), dtype=np.int64)
    y = np.concatenate([y_pos, y_neg])

    print(f"\nTraining probe: {target}")
    print("Negative concepts:", negative_names)
    print("Positive rows:", len(X_pos))
    print("Negative rows:", len(X_neg))

    probe = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=0,
    )

    probe.fit(X, y)

    # Learned weights ko unit length do
    w = probe.coef_[0]
    norm = np.linalg.norm(w)

    if norm == 0:
        raise ValueError(f"Zero weight vector: {target}")

    w = w / norm

    # Fitted intercept discard: raw score = vector @ w
    pos_scores = X_pos @ w
    neg_scores = X_neg @ w

    threshold = (
        pos_scores.mean() + neg_scores.mean()
    ) / 2

    scores = np.concatenate([pos_scores, neg_scores])
    scale = max(float(scores.std()), 1e-6)

    np.savez(
        save_path,
        w=w,
        threshold=threshold,
        scale=scale,
        negative_concepts=np.array(negative_names),
    )

    print(f"Saved: {save_path}")


if __name__ == "__main__":
 
    all_data = {}

    for concept in ALL_CONCEPTS:
        file = ACT_DIR / f"{concept}.npy"
        vectors = np.load(file)

        if vectors.ndim != 2 or not np.isfinite(vectors).all():
            raise ValueError(f"Invalid activation matrix: {file}")

        all_data[concept] = vectors

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Har concept ka separate probe train karo
    for target in ALL_CONCEPTS:
        train_probe(target, all_data)
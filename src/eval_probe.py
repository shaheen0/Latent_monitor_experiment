# eval_probe.py  — Phase A (fixed)
from config import PROBES_DIR, pooled_dir ,TRAIN_CONCEPTS,TEST_CONCEPTS
import numpy as np 
from sklearn.metrics import roc_auc_score

ALL_CONCEPTS = TRAIN_CONCEPTS + TEST_CONCEPTS

EVAL_N = 100   # ("probe kitni achi hai?" check karne ke liye)
EVAL_SPLIT = "nf"   # ⚠️ NOT probe_train


def eval_concept(target):
    with np.load(PROBES_DIR / f"{target}.npz") as p:
        w = p["w"]

    act = pooled_dir(EVAL_SPLIT)
    X_pos = np.load(act / f"{target}.npy")[:EVAL_N]   # first 100
    # X_pos = np.load(act / f"{target}.npy")[250:350]    

    if target in TRAIN_CONCEPTS:
        peers = [c for c in TRAIN_CONCEPTS if c != target]   # sirf training peers
    else:
        peers = [c for c in ALL_CONCEPTS if c != target]     # saare
    # peers = [c for c in ALL_CONCEPTS if c != target]
    X_neg = np.concatenate([np.load(act / f"{c}.npy") for c in peers], axis=0)

    pos_scores = X_pos @ w
    neg_scores = X_neg @ w
    tau = np.percentile(neg_scores, 99)

    tpr = float((pos_scores > tau).mean())
    auroc = float(roc_auc_score(
        np.r_[np.ones(len(X_pos)), np.zeros(len(X_neg))],
        np.r_[pos_scores, neg_scores],
    ))
    print(f"{target:20s}  TPR@1%FPR={tpr:.3f}  AUROC={auroc:.4f}")
    return tpr, auroc


if __name__ == "__main__":
    for c in ALL_CONCEPTS:
        eval_concept(c)
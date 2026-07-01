#!/usr/bin/env python3
"""
split_data.py
--------------
Découpe le corpus nettoyé en train/val/test (90/5/5 par défaut), de façon
reproductible (seed fixe), et écrit 3 CSV séparés.

Usage:
    python scripts/split_data.py --input data/processed/corpus_clean.csv \
        --outdir data/processed --seed 42 --val-ratio 0.05 --test-ratio 0.05
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8")
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    n = len(df)
    n_val = max(1, int(n * args.val_ratio))
    n_test = max(1, int(n * args.test_ratio))
    n_train = n - n_val - n_test

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    args.outdir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(args.outdir / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(args.outdir / "val.csv", index=False, encoding="utf-8")
    test_df.to_csv(args.outdir / "test.csv", index=False, encoding="utf-8")

    print(f"Total: {n} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Fichiers écrits dans {args.outdir}")


if __name__ == "__main__":
    main()

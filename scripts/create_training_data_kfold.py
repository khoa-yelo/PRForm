"""
Create genus-blocked K-fold splits from training_data_template_v2.csv for OOF scoring.

K=5 folds, each fold rotation: train=60%, val=20%, test=20% of genera.
- Genera are shuffled and divided into 5 groups.
- For fold k: test=group[k], val=group[(k+1)%5], train=remaining 3 groups.
- Rows with NaN genus_merged always go to train.

Output: one directory per fold under training_data/genus_block_kfold/
  fold_0/training_data.csv  ... fold_4/training_data.csv
"""

import os
import pandas as pd
import numpy as np

INPUT_PATH = "/projects/m000151/khoa/repos/PRForm/training_data/training_data_template_v2.csv"
OUTPUT_DIR = "/projects/m000151/khoa/repos/PRForm/training_data/genus_block_kfold"
K = 5
SEED = 42

df = pd.read_csv(INPUT_PATH)
print(f"Loaded {len(df)} rows, {df['genus_merged'].nunique()} unique genera")

# Shuffle all genera for fold assignment
all_genera = df["genus_merged"].dropna().unique()
foldable_genera = np.array(list(all_genera))
rng = np.random.RandomState(SEED)
rng.shuffle(foldable_genera)

# Split genera into K roughly equal groups
genus_groups = np.array_split(foldable_genera, K)
print(f"Genera per fold group: {[len(g) for g in genus_groups]}")

# Count labeled genera per group for sanity check
labeled_genera = set(
    df.loc[df["prf_type"].isin([-1.0, 1.0]), "genus_merged"].dropna().unique()
)
for i, grp in enumerate(genus_groups):
    n_labeled = sum(1 for g in grp if g in labeled_genera)
    print(f"  Group {i}: {len(grp)} genera, {n_labeled} with labels")

os.makedirs(OUTPUT_DIR, exist_ok=True)

for fold_idx in range(K):
    test_genera = set(genus_groups[fold_idx])
    val_genera = set(genus_groups[(fold_idx + 1) % K])
    train_genera = set()
    for j in range(K):
        if j != fold_idx and j != (fold_idx + 1) % K:
            train_genera.update(genus_groups[j])

    # Assign splits: NaN genus → train
    split = np.where(
        df["genus_merged"].isna() | df["genus_merged"].isin(train_genera), "train",
        np.where(df["genus_merged"].isin(val_genera), "val", "test")
    )

    df_fold = df.copy()
    df_fold["split"] = split

    # Print stats
    counts = df_fold["split"].value_counts()
    labeled_counts = df_fold[df_fold["prf_type"].isin([-1.0, 1.0])]["split"].value_counts()
    print(f"\nFold {fold_idx}:")
    print(f"  All:     train={counts.get('train',0):>6}  val={counts.get('val',0):>5}  test={counts.get('test',0):>5}")
    print(f"  Labeled: train={labeled_counts.get('train',0):>6}  val={labeled_counts.get('val',0):>5}  test={labeled_counts.get('test',0):>5}")

    fold_dir = os.path.join(OUTPUT_DIR, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)
    df_fold.to_csv(os.path.join(fold_dir, "training_data.csv"), index=False)

print(f"\nDone. Wrote {K} folds to {OUTPUT_DIR}")

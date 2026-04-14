import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

path = "/farmshare/user_data/khoang99/repos/PRForm/prform/model_data_full_string2.csv"
df = pd.read_csv(path)
# rename prf_direction to prf_type
df_drop = df.sort_values('prf_direction', key=lambda x: x.map({-1: 0, 1: 1}).fillna(20)).drop_duplicates(subset='cluster_id', keep='first')
df_drop.rename(columns={"prf_direction": "prf_type"}, inplace=True)

species_ids = df_drop[~df_drop.prf_type.isin([-1.0, 1.0, np.nan])].species_taxid.unique()
df_selected = df_drop[~df_drop.species_taxid.isin(species_ids)].copy()
df_selected["prf_cds_strand"] = df_selected["strand"]
df_selected["strand"] = "+"
df_selected.prf_cds_strand = df_selected.prf_cds_strand.map({1.0: "+", -1.0: "-"})
df_selected = df_selected[["accession_id", "record_id", "cluster_id", "prf_position", "prf_type", "strand", "prf_cds_strand",
             "species_taxid", "genus_taxid", "species_name", "genus_name", "sequence"]]
df_selected["sample_weight"] = df_selected.species_taxid.map(1 / np.sqrt(df_selected.species_taxid.value_counts()))
df_selected.reset_index(drop=True, inplace=True)

OUTPUT_DIR = "/farmshare/user_data/khoang99/repos/PRForm/training_data"


def assign_split_labels(index, train_idx, val_idx, test_idx):
    split = pd.Series("train", index=index)
    split.iloc[val_idx] = "val"
    split.iloc[test_idx] = "test"
    return split


# ── 1. Random split 60/20/20 ─────────────────────────────────────────────────
def make_random_split(df):
    df = df.copy()
    idx = np.arange(len(df))
    train_idx, temp_idx = train_test_split(idx, test_size=0.4, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    df["split"] = assign_split_labels(df.index, train_idx, val_idx, test_idx)
    return df


PINNED_SPECIES_NAME = "Alphainfluenzavirus influenzae"

# ── 2. Species blocking split 60/20/20 ───────────────────────────────────────
# Test and val sets contain only species NOT seen in train.
# PINNED_SPECIES_NAME is always assigned to train.
def make_species_block_split(df):
    df = df.copy()
    pinned_sp = set(df.loc[df["species_name"] == PINNED_SPECIES_NAME, "species_taxid"].unique())
    species = np.array([s for s in df["species_taxid"].unique() if s not in pinned_sp])
    np.random.seed(42)
    np.random.shuffle(species)
    n = len(species)
    train_sp = set(species[:int(n * 0.6)]) | pinned_sp
    val_sp   = set(species[int(n * 0.6):int(n * 0.8)])
    test_sp  = set(species[int(n * 0.8):])

    split = np.where(
        df["species_taxid"].isin(train_sp), "train",
        np.where(df["species_taxid"].isin(val_sp), "val", "test")
    )
    df["split"] = split
    return df


# ── 3. Genus blocking split 60/20/20 ─────────────────────────────────────────
# Test and val sets contain only genera NOT seen in train.
# The genus of PINNED_SPECIES_NAME is always assigned to train.
def make_genus_block_split(df):
    df = df.copy()
    pinned_ge = set(df.loc[df["species_name"] == PINNED_SPECIES_NAME, "genus_taxid"].dropna().unique())
    genera = np.array([g for g in df["genus_taxid"].dropna().unique() if g not in pinned_ge])
    np.random.seed(42)
    np.random.shuffle(genera)
    n = len(genera)
    train_ge = set(genera[:int(n * 0.6)]) | pinned_ge
    val_ge   = set(genera[int(n * 0.6):int(n * 0.8)])
    test_ge  = set(genera[int(n * 0.8):])

    # Rows with NaN genus_taxid go to train
    split = np.where(
        df["genus_taxid"].isna() | df["genus_taxid"].isin(train_ge), "train",
        np.where(df["genus_taxid"].isin(val_ge), "val", "test")
    )
    df["split"] = split
    return df


df_template = df_selected.copy()
df_template["split"] = np.nan
df_template.to_csv(f"{OUTPUT_DIR}/training_data_template.csv", index=False)

df_random  = make_random_split(df_selected)
df_species = make_species_block_split(df_selected)
df_genus   = make_genus_block_split(df_selected)

df_random.to_csv(f"{OUTPUT_DIR}/training_data_random.csv",         index=False)
df_species.to_csv(f"{OUTPUT_DIR}/training_data_species_block.csv", index=False)
df_genus.to_csv(f"{OUTPUT_DIR}/training_data_genus_block.csv",     index=False)

for name, d in [("random", df_random), ("species_block", df_species), ("genus_block", df_genus)]:
    print(f"\n── {name} ──")
    print(d["split"].value_counts())

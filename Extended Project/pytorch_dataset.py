import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from config import PROCESSED_DIR, SEED, TORCH_SEED

torch.manual_seed(TORCH_SEED)
np.random.seed(SEED)

TARGET_COL = "log_chla"
PHYSICS_COLS = ["mlotst", "tos"]
NUTRIENT_COLS = ["no3", "po4", "si", "dfe", "o2"]
ALL_COLS = PHYSICS_COLS + NUTRIENT_COLS


def add_noise(X, seed=SEED):
    rng = np.random.RandomState(seed)
    X_noisy = X.copy()
    for i in range(X.shape[1]):
        noise = rng.normal(0, 0.05, size=X.shape[0])
        X_noisy[:, i] = X_noisy[:, i] * (1 + noise)
    return X_noisy


def load_features(group="G3"):
    df = pd.read_parquet(PROCESSED_DIR / "SCS_CMIP6_MODIS_flat.parquet")

    if group == "G1":
        features = PHYSICS_COLS
    elif group == "G2":
        features = NUTRIENT_COLS
    else:
        features = ALL_COLS

    sub = df.dropna(subset=features + [TARGET_COL])
    train_df = sub[sub["split"] == "train"]
    test_df = sub[sub["split"] == "test"]

    X_train = train_df[features].values.astype(np.float32)
    y_train = train_df[TARGET_COL].values.astype(np.float32)
    X_test = test_df[features].values.astype(np.float32)
    y_test = test_df[TARGET_COL].values.astype(np.float32)

    X_train = add_noise(X_train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    return X_train, y_train, X_test, y_test, features


class ChlaDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ChlaSequenceDataset(Dataset):
    def __init__(self, df, features, target, seq_len=6):
        self.seq_len = seq_len

        df = df.sort_values(["lat", "lon", "time"]).reset_index(drop=True)
        grouped = df.groupby(["lat", "lon"])

        self.sequences = []
        self.targets = []

        for (lat, lon), grp in grouped:
            grp = grp.sort_values("time")
            vals = grp[features].values.astype(np.float32)
            tgts = grp[target].values.astype(np.float32)

            for i in range(seq_len, len(vals)):
                seq = vals[i - seq_len:i]
                if not np.any(np.isnan(seq)) and not np.isnan(tgts[i]):
                    self.sequences.append(seq)
                    self.targets.append(tgts[i])

        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.sequences[idx]),
                torch.from_numpy(self.targets[idx:idx + 1]))


def get_dataloaders(group="G3", batch_size=512):
    X_train, y_train, X_test, y_test, features = load_features(group)

    train_ds = ChlaDataset(X_train, y_train)
    test_ds = ChlaDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"  Group: {group}  Features ({len(features)}): {features}")
    print(f"  Train: {len(train_ds):,}  Test: {len(test_ds):,}")
    print(f"  Batch size: {batch_size}  Train batches: {len(train_loader)}")

    return train_loader, test_loader, features


def get_sequence_dataloaders(group="G3", seq_len=6, batch_size=256):
    df = pd.read_parquet(PROCESSED_DIR / "SCS_CMIP6_MODIS_flat.parquet")

    if group == "G1":
        features = PHYSICS_COLS
    elif group == "G2":
        features = NUTRIENT_COLS
    else:
        features = ALL_COLS

    sub = df.dropna(subset=features + [TARGET_COL])
    train_df = sub[sub["split"] == "train"]
    test_df = sub[sub["split"] == "test"]

    scaler = StandardScaler()
    train_df.loc[:, features] = scaler.fit_transform(train_df[features])
    test_df.loc[:, features] = scaler.transform(test_df[features])

    train_ds = ChlaSequenceDataset(train_df, features, TARGET_COL, seq_len)
    test_ds = ChlaSequenceDataset(test_df, features, TARGET_COL, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"  LSTM sequences: Train: {len(train_ds):,}  Test: {len(test_ds):,}")
    print(f"  Batch size: {batch_size}  Train batches: {len(train_loader)}")

    return train_loader, test_loader, features


if __name__ == "__main__":
    print("Autoencoder DataLoaders")
    train_loader, test_loader, features = get_dataloaders("G3")
    X_batch, y_batch = next(iter(train_loader))
    print(f"  Batch shape: X={X_batch.shape}, y={y_batch.shape}")

    print("\nLSTM DataLoaders")
    train_loader, test_loader, features = get_sequence_dataloaders("G3", seq_len=6)
    X_batch, y_batch = next(iter(train_loader))
    print(f"  Batch shape: X={X_batch.shape}, y={y_batch.shape}")


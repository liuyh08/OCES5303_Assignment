import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use("TkAgg")

from config import PROCESSED_DIR, FIGURES_DIR, SEED, TORCH_SEED
from pytorch_dataset import get_sequence_dataloaders

torch.manual_seed(TORCH_SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 12,
    "figure.dpi": 300,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.unicode_minus": False,
})


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.head(last_hidden).squeeze(-1)


def train_lstm(model, train_loader, test_loader, epochs=100, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    train_losses, test_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).squeeze()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        train_losses.append(epoch_loss / len(train_loader.dataset))

        model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).squeeze()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                epoch_loss += loss.item() * X_batch.size(0)
        test_losses.append(epoch_loss / len(test_loader.dataset))

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch + 1:3d}/{epochs}  "
                  f"Train: {train_losses[-1]:.6f}  Test: {test_losses[-1]:.6f}")

    return train_losses, test_losses


def train_lstm_quiet(model, train_loader, epochs=100, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).squeeze()
            loss = criterion(model(X_batch), y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def evaluate_model(model, test_loader):
    model.eval()
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch).cpu().numpy()
            y_true_all.append(y_batch.numpy().squeeze())
            y_pred_all.append(y_pred)

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print(f"R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    return y_true, y_pred, {"R2": r2, "RMSE": rmse, "MAE": mae}


def evaluate_quiet(model, test_loader):
    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch).cpu().numpy()
            y_true_all.append(y_batch.numpy().squeeze())
            y_pred_all.append(y_pred)
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    return r2_score(y_true, y_pred)


def plot_training(train_losses, test_losses, group, seq_len):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train")
    ax.plot(test_losses, label="Test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title(f"LSTM Training: {group} (seq_len={seq_len})")
    ax.legend()
    fig.tight_layout()
    p = FIGURES_DIR / f"fig_lstm_training_{group}_seq{seq_len}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  {p.name}")
    plt.close(fig)


def plot_scatter_subplot(ax, y_true, y_pred, group, r2):
    ax.scatter(y_true, y_pred, s=1, alpha=0.3, c="#2c7bb6")
    lims = [min(y_true.min(), y_pred.min()) - 0.1,
            max(y_true.max(), y_pred.max()) + 0.1]
    ax.plot(lims, lims, "r--", lw=1, label="1:1 line")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Observed log_chla")
    ax.set_ylabel("Predicted log_chla")
    ax.set_title(f"{group} (R$^2$={r2:.4f})")
    ax.legend(fontsize=9, loc="upper left")


def plot_scatter_combined(seq_len, scatter_data):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, (group, (y_true, y_pred, r2)) in zip(axes, scatter_data.items()):
        plot_scatter_subplot(ax, y_true, y_pred, group, r2)
    fig.tight_layout()
    p = FIGURES_DIR / f"fig_lstm_scatter_seq{seq_len}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  {p.name}")
    plt.close(fig)


def main():
    seq_lens = [6, 12]
    groups = ["G1", "G2", "G3"]

    print("Single-Seed Training")

    all_metrics = {}

    for seq_len in seq_lens:
        scatter_data = {}

        for group in groups:
            tag = f"{group}_seq{seq_len}"
            print(f"\n  {tag}")

            train_loader, test_loader, features = get_sequence_dataloaders(
                group, seq_len=seq_len, batch_size=256
            )
            input_dim = len(features)

            model = LSTMRegressor(input_dim).to(device)
            train_losses, test_losses = train_lstm(
                model, train_loader, test_loader, epochs=100, lr=1e-3
            )

            y_true, y_pred, metrics = evaluate_model(model, test_loader)
            all_metrics[tag] = metrics

            # Save model
            model_path = PROCESSED_DIR / f"model_LSTM_{tag}.pt"
            torch.save(model.state_dict(), model_path)
            print(f"  Saved: {model_path.name}")

            plot_training(train_losses, test_losses, group, seq_len)
            scatter_data[group] = (y_true, y_pred, metrics["R2"])

        plot_scatter_combined(seq_len, scatter_data)

    print("\nLSTM Single-Seed Results:")
    results_df = pd.DataFrame(all_metrics).T
    results_df.index.name = "Config"
    print(results_df.round(4).to_string())
    results_df.to_csv(PROCESSED_DIR / "pytorch_lstm_results.csv")

    multi_seeds = [42, 123, 456, 789, 1024]
    multi_results = {g: [] for g in groups}

    for seed in multi_seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        for group in groups:
            train_loader, test_loader, features = get_sequence_dataloaders(
                group, seq_len=6, batch_size=256
            )
            model = LSTMRegressor(len(features)).to(device)
            model = train_lstm_quiet(model, train_loader, epochs=100, lr=1e-3)
            r2 = evaluate_quiet(model, test_loader)
            multi_results[group].append(r2)
            print(f"  seed={seed:5d}  {group}  R2={r2:.4f}")


    multi_seeds = [42, 123, 456, 789, 1024]
    multi_results = {}

    for seq_len in [6, 12]:
        multi_results[seq_len] = {g: [] for g in groups}
        for seed in multi_seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            for group in groups:
                train_loader, test_loader, features = get_sequence_dataloaders(
                    group, seq_len=seq_len, batch_size=256
                )
                model = LSTMRegressor(len(features)).to(device)
                model = train_lstm_quiet(model, train_loader, epochs=100, lr=1e-3)
                r2 = evaluate_quiet(model, test_loader)
                multi_results[seq_len][group].append(r2)
                print(f"  seed={seed:5d}  seq={seq_len:2d}  {group}  R2={r2:.4f}")

    print("\nLSTM Multi-Seed Summary:")
    print(f"  {'Seq':>4s}  {'Group':8s}  {'R2_mean':>8s}  {'R2_std':>8s}  {'R2_min':>8s}  {'R2_max':>8s}")
    for seq_len in [6, 12]:
        for group in groups:
            vals = multi_results[seq_len][group]
            print(f"  {seq_len:4d}  {group:8s}  {np.mean(vals):8.4f}  {np.std(vals):8.4f}  "
                  f"{min(vals):8.4f}  {max(vals):8.4f}")

    records = []
    for seq_len in [6, 12]:
        for group in groups:
            vals = multi_results[seq_len][group]
            records.append({
                "Method": "LSTM", "Group": group, "Seq_len": seq_len,
                "R2_mean": np.mean(vals), "R2_std": np.std(vals),
                "R2_min": min(vals), "R2_max": max(vals),
            })
    multi_df = pd.DataFrame(records)
    multi_df.to_csv(PROCESSED_DIR / "lstm_multi_seed_results.csv", index=False)
    print(f"\n  Saved to {PROCESSED_DIR / 'lstm_multi_seed_results.csv'}")


if __name__ == "__main__":
    main()

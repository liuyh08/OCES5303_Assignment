import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pysindy as ps
import matplotlib
matplotlib.use("TkAgg")

from config import PROCESSED_DIR, FIGURES_DIR, SEED

np.random.seed(SEED)

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

df = pd.read_parquet(PROCESSED_DIR / "SCS_CMIP6_MODIS_flat.parquet")

FEATURE_COLS = ["no3", "po4", "si", "dfe", "o2", "mlotst", "tos"]
TARGET_COL = "log_chla"

sub = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
train_df = sub[sub["split"] == "train"]
test_df = sub[sub["split"] == "test"]

X_train = train_df[FEATURE_COLS].values.astype(np.float64)
y_train = train_df[TARGET_COL].values.astype(np.float64)
X_test = test_df[FEATURE_COLS].values.astype(np.float64)
y_test = test_df[TARGET_COL].values.astype(np.float64)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"Train: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}")
print(f"Features: {FEATURE_COLS}")


degrees = [1, 2]
all_metrics = {}
all_formulas = {}

for degree in degrees:
    tag = f"degree{degree}"
    print(f"SINDy  degree={degree}  STLSQ (threshold=0.01)")

    library = ps.PolynomialLibrary(degree=degree, include_interaction=True)
    library.fit(X_train_sc)

    Theta_train = library.transform(X_train_sc)
    Theta_test = library.transform(X_test_sc)
    feature_names = library.get_feature_names(input_features=FEATURE_COLS)

    print(f"  Library features: {len(feature_names)}")
    print(f"  Theta shape: {Theta_train.shape}")

    optimizer = ps.STLSQ(threshold=0.01, alpha=0.5, max_iter=100)
    optimizer.fit(Theta_train, y_train)
    coef = optimizer.coef_.flatten()

    y_pred = np.asarray(Theta_test) @ coef

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    bias = np.mean(y_pred - y_test)
    print(f"  R2={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}  Bias={bias:.4f}")

    all_metrics[tag] = {"degree": degree, "R2": r2, "RMSE": rmse, "MAE": mae, "Bias": bias}

    print(f"\nDiscovered formula (degree={degree}):")
    print(f"log_chla = ")
    formula_lines = []
    for name, c in zip(feature_names, coef):
        if abs(c) > 0.001:
            line = f"    {c:+.4f} * {name}"
            print(line)
            formula_lines.append({"feature": name, "coef": c})
    all_formulas[tag] = formula_lines

    fig, ax = plt.subplots(figsize=(8, max(5, len(feature_names) * 0.3)))
    mask = np.abs(coef) > 0.001
    names_sel = [feature_names[i] for i in range(len(coef)) if mask[i]]
    coef_sel = coef[mask]

    if len(coef_sel) == 0:
        print(f"No nonzero coefficients for degree={degree}, skipping plots.")
        plt.close(fig)
        continue

    sorted_idx = np.argsort(np.abs(coef_sel))
    colors = ["#d73027" if c > 0 else "#4575b4" for c in coef_sel[sorted_idx]]
    ax.barh(
        [names_sel[i] for i in sorted_idx],
        coef_sel[sorted_idx],
        color=colors
    )
    ax.set_xlabel("Coefficient")
    ax.set_title(f"SINDy STLSQ Coefficients (degree={degree})")
    ax.axvline(0, color="black", lw=0.5)
    fig.tight_layout()
    p = FIGURES_DIR / f"fig_sindy_coefficients_{tag}.png"
    # fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"{p.name}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, s=1, alpha=0.3, c="#2c7bb6")
    lims = [min(y_test.min(), y_pred.min()) - 0.1,
            max(y_test.max(), y_pred.max()) + 0.1]
    ax.plot(lims, lims, "r--", lw=1, label="1:1 line")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Observed log_chla")
    ax.set_ylabel("Predicted log_chla")
    ax.set_title(f"SINDy degree={degree} (R²={r2:.4f})")
    ax.legend()
    fig.tight_layout()
    p = FIGURES_DIR / f"fig_sindy_scatter_{tag}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"{p.name}")
    plt.close(fig)


print("SINDy RESULTS COMPARISON")
results_df = pd.DataFrame(all_metrics).T
results_df.index.name = "Config"
print(results_df.round(4).to_string())
results_df.to_csv(PROCESSED_DIR / "sindy_results.csv")
print(f"sindy_results.csv")

formula_rows = []
for tag, lines in all_formulas.items():
    for item in lines:
        formula_rows.append({"config": tag, "feature": item["feature"], "coef": item["coef"]})
formula_df = pd.DataFrame(formula_rows)
formula_df.to_csv(PROCESSED_DIR / "sindy_formulas.csv", index=False)
print(f"sindy_formulas.csv")


fig, ax = plt.subplots(figsize=(8, 4))
configs = list(all_metrics.keys())
r2_vals = [all_metrics[c]["R2"] for c in configs]
colors = ["#4575b4" if "degree1" in c else "#d73027" for c in configs]
bars = ax.barh(configs, r2_vals, color=colors, height=0.5)

for bar, val in zip(bars, r2_vals):
    ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=11)

ax.set_xlabel("R²")
ax.set_title("SINDy: degree=1 (blue) vs degree=2 (red)")
ax.set_xlim(0, min(1.0, max(r2_vals) + 0.15))
fig.tight_layout()
p = FIGURES_DIR / "fig_sindy_degree_comparison.png"
fig.savefig(p, dpi=300, bbox_inches="tight")
print(f"{p.name}")
plt.close(fig)

print("\nSINDy done")

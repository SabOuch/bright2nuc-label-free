from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import StratifiedGroupKFold


# =====================================================================
# PATHS
# =====================================================================

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data/raw/TF_prediction"
PROCESSED = ROOT / "data/processed"
OUT = ROOT / "outputs/m0"

PROCESSED.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

POS_PATH = RAW / "Positions_Classes.csv"
MORPH_PATH = RAW / "Features.csv"
BF_PATH = RAW / "Features_Brightfield.csv"

FOLDS_PATH = PROCESSED / "folds.csv"
CULTURE_FOLDS_PATH = PROCESSED / "culture_folds.csv"


# =====================================================================
# HELPERS
# =====================================================================

def compute_metrics(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "pearson": pearsonr(y_true, y_pred).statistic,
        "spearman": spearmanr(y_true, y_pred).statistic,
        "abs_error_std": np.std(
            np.abs(y_true - y_pred),
            ddof=1,
        ),
    }


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


# =====================================================================
# 1. LOAD METADATA
# =====================================================================

pos = pd.read_csv(POS_PATH)

if pos["filename"].duplicated().any():
    raise RuntimeError("Duplicate filename in Positions_Classes.csv")

pos["culture"] = pos["filename"].str.replace(
    r"\.tif\d+\.tif$",
    ".tif",
    regex=True,
)

pos["dataset_tag"] = pos["culture"].str.extract(
    r"(Day\d+|EndPoint)",
    expand=False,
)

if pos["culture"].nunique() != 48:
    raise RuntimeError(
        f"Expected 48 published cultures, got "
        f"{pos['culture'].nunique()}"
    )

if len(pos) != 30877:
    raise RuntimeError(
        f"Expected 30,877 nuclei, got {len(pos)}"
    )


# =====================================================================
# 2. REPRODUCE PREVIEW 3 EXACTLY AND FREEZE IT
# =====================================================================

culture = (
    pos.groupby("culture")
    .agg(
        dataset_tag=("dataset_tag", "first"),
        n_nuclei=("filename", "size"),
        dl_mean=("TF_class", "mean"),
        dl_std=("TF_class", "std"),
        dl_median=("TF_class", "median"),
    )
    .reset_index()
)

culture["culture_dl_bin"] = pd.qcut(
    culture["dl_mean"],
    q=5,
    labels=False,
    duplicates="drop",
)

sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

culture["fold"] = -1

for fold, (_, test_idx) in enumerate(
    sgkf.split(
        X=np.zeros(len(culture)),
        y=culture["culture_dl_bin"],
        groups=culture["culture"],
    )
):
    culture.loc[test_idx, "fold"] = fold

if (culture["fold"] < 0).any():
    raise RuntimeError("Unassigned culture.")

if culture["culture"].duplicated().any():
    raise RuntimeError("Duplicate culture.")

fold_map = dict(
    zip(
        culture["culture"],
        culture["fold"],
    )
)

pos["fold"] = pos["culture"].map(fold_map)

if pos["fold"].isna().any():
    raise RuntimeError("Some nuclei have no fold.")

if (
    pos.groupby("culture")["fold"]
    .nunique()
    .max()
    != 1
):
    raise RuntimeError("Culture leakage detected.")


# =====================================================================
# 3. WRITE FINAL FOLDS
# =====================================================================

culture_out = culture[
    [
        "culture",
        "dataset_tag",
        "n_nuclei",
        "dl_mean",
        "dl_std",
        "dl_median",
        "culture_dl_bin",
        "fold",
    ]
].sort_values(
    ["fold", "culture"]
)

folds_out = pos[
    [
        "filename",
        "culture",
        "dataset_tag",
        "TF_class",
        "fold",
    ]
].sort_values("filename")

culture_out.to_csv(
    CULTURE_FOLDS_PATH,
    index=False,
)

folds_out.to_csv(
    FOLDS_PATH,
    index=False,
)

folds_hash = sha256_file(FOLDS_PATH)

print("=" * 80)
print("FROZEN FOLDS")
print("=" * 80)

print("Nuclei   :", len(folds_out))
print("Cultures :", culture_out["culture"].nunique())
print("SHA256   :", folds_hash)

print("\nSummary:")

print(
    folds_out.groupby("fold")
    .agg(
        n_nuclei=("filename", "size"),
        n_cultures=("culture", "nunique"),
        dl_mean=("TF_class", "mean"),
        dl_std=("TF_class", "std"),
    )
    .to_string()
)

print("\nCultures by DL stratum:")

print(
    pd.crosstab(
        culture_out["culture_dl_bin"],
        culture_out["fold"],
    ).to_string()
)


# =====================================================================
# 4. LOAD FEATURES
# =====================================================================

morph = pd.read_csv(MORPH_PATH)
bf = pd.read_csv(BF_PATH)

if morph["filename"].duplicated().any():
    raise RuntimeError("Duplicates in Features.csv")

if bf["filename"].duplicated().any():
    raise RuntimeError("Duplicates in Features_Brightfield.csv")

morph_cols_original = [
    c for c in morph.columns
    if c != "filename"
]

bf_cols_original = [
    c for c in bf.columns
    if c != "filename"
]

if len(morph_cols_original) != 63:
    raise RuntimeError(
        f"Expected 63 morph columns, got "
        f"{len(morph_cols_original)}"
    )

if len(bf_cols_original) != 56:
    raise RuntimeError(
        f"Expected 56 BF columns, got "
        f"{len(bf_cols_original)}"
    )

morph = morph.rename(
    columns={
        c: f"morph__{c}"
        for c in morph_cols_original
    }
)

bf = bf.rename(
    columns={
        c: f"bf__{c}"
        for c in bf_cols_original
    }
)

morph_cols = [
    c for c in morph.columns
    if c != "filename"
]

bf_cols = [
    c for c in bf.columns
    if c != "filename"
]


# =====================================================================
# 5. MERGE EVERYTHING BY FILENAME
# =====================================================================

data = (
    pos[
        [
            "filename",
            "culture",
            "dataset_tag",
            "TF_class",
            "boarder_distance",
            "fold",
        ]
    ]
    .merge(
        morph,
        on="filename",
        how="inner",
        validate="one_to_one",
    )
    .merge(
        bf,
        on="filename",
        how="inner",
        validate="one_to_one",
    )
)

if len(data) != 30877:
    raise RuntimeError(
        f"Merge produced {len(data)} rows "
        f"instead of 30,877."
    )

feature_sets = {
    "M0_119": morph_cols + bf_cols,
    "M0_120_border": (
        morph_cols
        + bf_cols
        + ["boarder_distance"]
    ),
}

assert len(feature_sets["M0_119"]) == 119
assert len(feature_sets["M0_120_border"]) == 120


# =====================================================================
# 6. CHECK NUMERICAL VALUES
# =====================================================================

for model_name, columns in feature_sets.items():

    X = data[columns].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(X).all():

        bad = np.argwhere(~np.isfinite(X))

        row, col = bad[0]

        raise RuntimeError(
            f"{model_name}: non-finite value at "
            f"row={row}, feature={columns[col]}"
        )

y = data["TF_class"].to_numpy(
    dtype=np.float64
)

if not np.isfinite(y).all():
    raise RuntimeError(
        "Non-finite target detected."
    )


# =====================================================================
# 7. TRAIN TWO M0 VARIANTS
# =====================================================================

all_predictions = data[
    [
        "filename",
        "culture",
        "dataset_tag",
        "TF_class",
        "fold",
    ]
].copy()

metrics_rows = []
importance_rows = []

print("\n" + "=" * 80)
print("M0 TRAINING")
print("=" * 80)

print(
    "RF parameters: "
    "n_estimators=1000, "
    "max_depth=10, "
    "random_state=42"
)

for model_name, columns in feature_sets.items():

    print("\n" + "#" * 80)
    print(model_name)
    print("#" * 80)

    X = data[columns].to_numpy(
        dtype=np.float32
    )

    oof = np.full(
        len(data),
        np.nan,
        dtype=np.float64,
    )

    fold_importances = []

    for fold in range(5):

        test_mask = (
            data["fold"].to_numpy()
            == fold
        )

        train_mask = ~test_mask

        X_train = X[train_mask]
        y_train = y[train_mask]

        X_test = X[test_mask]
        y_test = y[test_mask]

        model = RandomForestRegressor(
            n_estimators=1000,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            criterion="squared_error",
            bootstrap=True,
            max_features=1.0,
        )

        model.fit(
            X_train,
            y_train,
        )

        pred = model.predict(
            X_test
        )

        oof[test_mask] = pred

        fm = compute_metrics(
            y_test,
            pred,
        )

        metrics_rows.append({
            "model": model_name,
            "level": "nucleus_fold",
            "fold": fold,
            "n": int(test_mask.sum()),
            **fm,
        })

        fold_importances.append(
            model.feature_importances_.copy()
        )

        print(
            f"Fold {fold} | "
            f"n={test_mask.sum():5d} | "
            f"R2={fm['r2']:.4f} | "
            f"MAE={fm['mae']:.4f} | "
            f"Pearson={fm['pearson']:.4f}"
        )

    if np.isnan(oof).any():
        raise RuntimeError(
            f"{model_name}: incomplete OOF predictions"
        )

    all_predictions[
        f"pred_{model_name}"
    ] = oof

    # ---------------------------------------------------------
    # POOLED NUCLEUS METRICS
    # ---------------------------------------------------------

    pooled = compute_metrics(
        y,
        oof,
    )

    metrics_rows.append({
        "model": model_name,
        "level": "nucleus_oof_pooled",
        "fold": -1,
        "n": len(y),
        **pooled,
    })

    # ---------------------------------------------------------
    # CULTURE-LEVEL METRICS
    # ---------------------------------------------------------

    culture_eval = (
        pd.DataFrame({
            "culture": data["culture"],
            "true": y,
            "pred": oof,
        })
        .groupby("culture")
        .agg(
            true=("true", "mean"),
            pred=("pred", "mean"),
        )
        .reset_index()
    )

    culture_metrics = compute_metrics(
        culture_eval["true"].to_numpy(),
        culture_eval["pred"].to_numpy(),
    )

    metrics_rows.append({
        "model": model_name,
        "level": "culture_oof",
        "fold": -1,
        "n": len(culture_eval),
        **culture_metrics,
    })

    print("\nPOOLED OOF — NUCLEI")
    print(
        f"R2       : {pooled['r2']:.6f}\n"
        f"MAE      : {pooled['mae']:.6f}\n"
        f"MAE SD   : {pooled['abs_error_std']:.6f}\n"
        f"RMSE     : {pooled['rmse']:.6f}\n"
        f"Pearson  : {pooled['pearson']:.6f}\n"
        f"Spearman : {pooled['spearman']:.6f}"
    )

    print("\nOOF — MEAN BY CULTURE")
    print(
        f"R2       : {culture_metrics['r2']:.6f}\n"
        f"MAE      : {culture_metrics['mae']:.6f}\n"
        f"Pearson  : {culture_metrics['pearson']:.6f}\n"
        f"Spearman : {culture_metrics['spearman']:.6f}"
    )

    # ---------------------------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------------------------

    imp = np.stack(
        fold_importances,
        axis=0,
    )

    imp_mean = imp.mean(axis=0)
    imp_std = imp.std(axis=0, ddof=1)

    imp_df = pd.DataFrame({
        "model": model_name,
        "feature": columns,
        "importance_mean": imp_mean,
        "importance_std": imp_std,
    })

    imp_df["group"] = np.where(
        imp_df["feature"].str.startswith(
            "bf__"
        ),
        "brightfield",
        np.where(
            imp_df["feature"].str.startswith(
                "morph__"
            ),
            "morphology",
            "border_distance",
        )
    )

    importance_rows.append(
        imp_df
    )

    print("\nIMPORTANCE BY GROUP")

    group_imp = (
        imp_df.groupby("group")
        ["importance_mean"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    for group, value in group_imp.items():
        print(
            f"{group:18s}: "
            f"{100 * value:6.2f} %"
        )

    print("\nTOP 15 FEATURES")

    top = (
        imp_df.sort_values(
            "importance_mean",
            ascending=False,
        )
        .head(15)
    )

    for _, row in top.iterrows():
        print(
            f"{row['feature']:35s} "
            f"{row['importance_mean']:.6f}"
        )


# =====================================================================
# 8. SAVE RESULTS
# =====================================================================

metrics = pd.DataFrame(
    metrics_rows
)

importance = pd.concat(
    importance_rows,
    ignore_index=True,
)

all_predictions.to_csv(
    OUT / "m0_oof_predictions.csv",
    index=False,
)

metrics.to_csv(
    OUT / "m0_metrics.csv",
    index=False,
)

importance.to_csv(
    OUT / "m0_feature_importance.csv",
    index=False,
)


# =====================================================================
# 9. FINAL SUMMARY
# =====================================================================

print("\n" + "=" * 80)
print("FINAL COMPARISON")
print("=" * 80)

pooled_table = (
    metrics[
        metrics["level"]
        == "nucleus_oof_pooled"
    ][
        [
            "model",
            "r2",
            "mae",
            "abs_error_std",
            "pearson",
            "spearman",
        ]
    ]
)

print(
    pooled_table.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print("\nPAPER REFERENCE")
print("Published R2 : 0.48")
print("Published MAE: 0.19")
print(
    "WARNING: external comparison only "
    "(26,213 nuclei + different split protocol)."
)

print("\n" + "=" * 80)
print("FILES")
print("=" * 80)

print(FOLDS_PATH)
print(CULTURE_FOLDS_PATH)
print(OUT / "m0_oof_predictions.csv")
print(OUT / "m0_metrics.csv")
print(OUT / "m0_feature_importance.csv")

print("\nFOLDS SHA256:")
print(folds_hash)

print("\nDONE")

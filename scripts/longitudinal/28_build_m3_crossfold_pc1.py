"""
Build fold-specific M3 latent-PC1 calibrators without held-out-culture leakage.

Purpose
-------
For each outer fold, extract M3 embeddings from the development cultures,
fit a one-component PCA using development data only, orient the PC1 sign using
the development differentiation target, standardize the resulting score, and
save the fold-specific projection parameters required by longitudinal M3
inference.

The held-out fold is projected only after all PCA parameters, sign orientation,
and score standardization statistics have been fixed from development data.

Inputs
------
- ``data/processed/m1_bf16_uint8.npy``
- ``outputs/m3/m3_oof_predictions.csv``
- ``outputs/m3/fold{0..4}/best_model.pt``
- ``outputs/m3/oof_embeddings/metadata.csv``
- ``outputs/m3/oof_embeddings/fold{0..4}_embeddings_tta8.npz``

Outputs
-------
Written under ``outputs/m3/latent_pc1/``:
- ``fold{0..4}_pc1_calibrator.npz``
- ``m3_oof_latent_pc1.csv``
- ``m3_pc1_per_fold.csv``
- ``m3_pc1_per_culture.csv``
- ``summary.txt``

Calibrator contract
-------------------
Each ``fold*_pc1_calibrator.npz`` contains at least:
- ``center``
- ``component``
- ``score_mean``
- ``score_std``

These are the fields consumed by ``32_run_m3_longitudinal.py``.

Interpretation
--------------
PC1 is a latent representation axis. It is not biological time and is not a
validated pseudotime. Longitudinal data are never used to fit or orient PC1.

The frozen calibration procedure was performed with CUDA; CUDA remains required
here to preserve the validated reproduction path.
"""

from pathlib import Path

import gc
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import timm

from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr


# =====================================================================
# CONFIG
# =====================================================================

ROOT = Path(__file__).resolve().parents[2]

CACHE_PATH = (
    ROOT
    / "data/processed/m1_bf16_uint8.npy"
)

M3_DIR = (
    ROOT
    / "outputs/m3"
)

OOF_PATH = (
    M3_DIR
    / "m3_oof_predictions.csv"
)

EMBED_DIR = (
    M3_DIR
    / "oof_embeddings"
)

OUT_DIR = (
    M3_DIR
    / "latent_pc1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_OOF = (
    OUT_DIR
    / "m3_oof_latent_pc1.csv"
)

OUT_FOLD = (
    OUT_DIR
    / "m3_pc1_per_fold.csv"
)

OUT_CULTURE = (
    OUT_DIR
    / "m3_pc1_per_culture.csv"
)

OUT_SUMMARY = (
    OUT_DIR
    / "summary.txt"
)

EXPECTED_N = 30877
EXPECTED_CULTURES = 48

BATCH_SIZE = 64
NUM_WORKERS = 4
SEED = 42


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable."
    )


device = torch.device("cuda")


# =====================================================================
# MODEL
# =====================================================================

class M3ConvNeXtTiny(nn.Module):
    """ConvNeXt-Tiny architecture matching the frozen M3 checkpoints."""

    def __init__(self):

        super().__init__()

        self.encoder = timm.create_model(
            "convnext_tiny",
            pretrained=False,
            in_chans=16,
            num_classes=0,
            global_pool="avg",
        )

        self.n_features = int(
            self.encoder.num_features
        )

        self.head_dl = nn.Linear(
            self.n_features,
            1,
        )

        self.head_markers = nn.Linear(
            self.n_features,
            3,
        )


    def encode(self, x):
        """Resize one 16-channel crop batch and return pooled encoder features."""

        x = F.interpolate(
            x,
            size=(128, 128),
            mode="bilinear",
            align_corners=False,
        )

        return self.encoder(x)


# =====================================================================
# DATASET
# =====================================================================

class EmbeddingDataset(Dataset):
    """Read selected brightfield crops from the frozen memory-mapped cache."""

    def __init__(
        self,
        metadata,
        cache_path,
    ):

        self.metadata = (
            metadata
            .reset_index(drop=True)
            .copy()
        )

        self.cache_path = cache_path
        self.cache = None


    def _ensure_cache(self):

        if self.cache is None:

            self.cache = np.load(
                self.cache_path,
                mmap_mode="r",
            )


    def __len__(self):

        return len(self.metadata)


    def __getitem__(self, i):

        self._ensure_cache()

        cache_index = int(
            self.metadata
            .iloc[i]["cache_index"]
        )

        image = np.array(
            self.cache[cache_index],
            dtype=np.uint8,
            copy=True,
        )

        return torch.from_numpy(image)


# =====================================================================
# TTA x8 EMBEDDINGS
# =====================================================================

@torch.no_grad()
def extract_embeddings(
    model,
    loader,
    bf_mean,
    bf_std,
):
    """Extract TTA8-averaged M3 embeddings for one development subset."""

    model.eval()

    outputs = []


    for batch_number, x in enumerate(
        loader,
        start=1,
    ):

        x = x.to(
            device,
            non_blocking=True,
        )

        x = (
            x.float()
            / 255.0
        )

        x = (
            x - bf_mean
        ) / bf_std


        tta = []


        for k in range(4):

            xr = torch.rot90(
                x,
                k=k,
                dims=(-2, -1),
            )


            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
            ):

                e = model.encode(xr)


            tta.append(
                e.float()
            )


            xf = torch.flip(
                xr,
                dims=(-1,),
            )


            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
            ):

                e = model.encode(xf)


            tta.append(
                e.float()
            )


        emb = (
            torch.stack(
                tta,
                dim=0,
            )
            .mean(dim=0)
        )


        outputs.append(
            emb.cpu().numpy()
        )


        if (
            batch_number == 1
            or batch_number % 50 == 0
            or batch_number == len(loader)
        ):

            print(
                f"  batch "
                f"{batch_number:4d}/"
                f"{len(loader):4d}"
            )


    return np.concatenate(
        outputs,
        axis=0,
    )


# =====================================================================
# LOAD GLOBAL OOF TABLE
# =====================================================================

meta = pd.read_csv(
    OOF_PATH
)


# =====================================================================
# RESTORE CACHE_INDEX FROM EMBEDDING METADATA
# =====================================================================

if "cache_index" not in meta.columns:

    embedding_meta_path = (
        EMBED_DIR
        / "metadata.csv"
    )

    if not embedding_meta_path.exists():
        raise FileNotFoundError(
            embedding_meta_path
        )

    embedding_meta = pd.read_csv(
        embedding_meta_path,
        usecols=[
            "filename",
            "cache_index",
        ],
    )

    if embedding_meta["filename"].duplicated().any():
        raise RuntimeError(
            "Duplicate filename in embedding metadata."
        )

    meta = meta.merge(
        embedding_meta,
        on="filename",
        how="left",
        validate="one_to_one",
    )

    if meta["cache_index"].isna().any():
        raise RuntimeError(
            "Some OOF filenames have no cache_index."
        )

    meta["cache_index"] = (
        meta["cache_index"]
        .astype("int64")
    )

    print(
        "cache_index restored from:",
        embedding_meta_path,
    )


required = {
    "cache_index",
    "filename",
    "culture",
    "fold",
    "DL",
    "m3_tta8",
}


missing = (
    required
    - set(meta.columns)
)


if missing:
    raise RuntimeError(
        f"Missing columns: {missing}"
    )


if len(meta) != EXPECTED_N:
    raise RuntimeError(
        f"Expected {EXPECTED_N} nuclei, "
        f"got {len(meta)}."
    )


if meta["filename"].duplicated().any():
    raise RuntimeError(
        "Duplicate filename."
    )


print("=" * 88)
print("M3 — CROSS-FOLD LATENT PC1 AXIS CONSTRUCTION")
print("=" * 88)

print(
    "Nuclei   :",
    len(meta),
)

print(
    "Cultures :",
    meta["culture"].nunique(),
)

print(
    "GPU      :",
    torch.cuda.get_device_name(0),
)


# =====================================================================
# PROCESS EACH FOLD
# =====================================================================

oof_parts = []
fold_rows = []


for fold in range(5):

    print()
    print("=" * 88)
    print(
        f"FOLD {fold}"
    )
    print("=" * 88)


    dev = (
        meta[
            meta["fold"] != fold
        ]
        .copy()
        .reset_index(drop=True)
    )


    test = (
        meta[
            meta["fold"] == fold
        ]
        .copy()
        .reset_index(drop=True)
    )


    print(
        "Development:",
        len(dev),
        "nuclei /",
        dev["culture"].nunique(),
        "cultures",
    )

    print(
        "Held-out test:",
        len(test),
        "nuclei /",
        test["culture"].nunique(),
        "cultures",
    )


    checkpoint_path = (
        M3_DIR
        / f"fold{fold}"
        / "best_model.pt"
    )


    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )


    bf_mean = float(
        checkpoint["bf_mean"]
    )

    bf_std = float(
        checkpoint["bf_std"]
    )


    model = (
        M3ConvNeXtTiny()
        .to(device)
    )


    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )


    # =============================================================
    # DEVELOPMENT EMBEDDINGS
    # =============================================================

    print()
    print(
        "Extracting development embeddings..."
    )


    dev_dataset = EmbeddingDataset(
        dev,
        CACHE_PATH,
    )


    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )


    dev_embeddings = extract_embeddings(
        model,
        dev_loader,
        bf_mean,
        bf_std,
    )


    if dev_embeddings.shape != (
        len(dev),
        768,
    ):

        raise RuntimeError(
            f"Unexpected development shape: "
            f"{dev_embeddings.shape}"
        )


    # =============================================================
    # PCA FIT ON DEVELOPMENT ONLY
    # =============================================================

    print()
    print(
        "Fitting PCA on development only..."
    )


    pca = PCA(
        n_components=1,
        svd_solver="randomized",
        random_state=SEED,
    )


    dev_pc1_raw = (
        pca.fit_transform(
            dev_embeddings
        )
        .ravel()
    )


    # =============================================================
    # PRE-SPECIFIED SIGN ORIENTATION
    #
    # The sign is fixed using DEVELOPMENT DL only.
    # Never using the test fold.
    # Never using longitudinal data.
    # =============================================================

    dev_r_raw = pearsonr(
        dev_pc1_raw,
        dev["DL"],
    ).statistic


    sign = (
        1.0
        if dev_r_raw >= 0
        else -1.0
    )


    component = (
        pca.components_[0]
        .astype(np.float32)
        * sign
    )


    center = (
        pca.mean_
        .astype(np.float32)
    )


    dev_pc1 = (
        dev_pc1_raw
        * sign
    )


    dev_score_mean = float(
        np.mean(dev_pc1)
    )

    dev_score_std = float(
        np.std(
            dev_pc1,
            ddof=0,
        )
    )


    if (
        not np.isfinite(dev_score_std)
        or dev_score_std <= 0
    ):
        raise RuntimeError(
            f"Fold {fold}: "
            "invalid PC1 standard deviation."
        )


    dev_z = (
        dev_pc1
        - dev_score_mean
    ) / dev_score_std


    dev_pearson = pearsonr(
        dev_z,
        dev["DL"],
    ).statistic


    dev_spearman = spearmanr(
        dev_z,
        dev["DL"],
    ).statistic


    explained = float(
        pca.explained_variance_ratio_[0]
    )


    print(
        "Explained PC1 variance:",
        f"{explained:.6f}",
    )

    print(
        "Orientation sign       :",
        f"{sign:+.0f}",
    )

    print(
        "Dev Pearson PC1~DL     :",
        f"{dev_pearson:.6f}",
    )

    print(
        "Dev Spearman PC1~DL    :",
        f"{dev_spearman:.6f}",
    )


    # =============================================================
    # SAVE CALIBRATOR FOR FUTURE LONGITUDINAL PROJECTION
    # =============================================================

    calibrator_path = (
        OUT_DIR
        / f"fold{fold}_pc1_calibrator.npz"
    )


    np.savez_compressed(
        calibrator_path,

        center=center,

        component=component,

        score_mean=np.float32(
            dev_score_mean
        ),

        score_std=np.float32(
            dev_score_std
        ),

        explained_variance_ratio=np.float32(
            explained
        ),

        orientation_sign=np.float32(
            sign
        ),

        bf_mean=np.float32(
            bf_mean
        ),

        bf_std=np.float32(
            bf_std
        ),
    )


    # =============================================================
    # LOAD ALREADY EXTRACTED HELD-OUT TEST EMBEDDINGS
    # =============================================================

    test_embedding_path = (
        EMBED_DIR
        / f"fold{fold}_embeddings_tta8.npz"
    )


    z = np.load(
        test_embedding_path,
        allow_pickle=True,
    )


    test_embeddings_file = (
        z["embeddings"]
        .astype(
            np.float32,
            copy=False,
        )
    )


    test_filenames_file = (
        z["filenames"]
        .astype(str)
    )


    filename_to_row = {
        name: i
        for i, name
        in enumerate(
            test_filenames_file
        )
    }


    try:

        order = np.array(
            [
                filename_to_row[name]
                for name
                in test["filename"]
                .astype(str)
            ],
            dtype=np.int64,
        )

    except KeyError as e:

        raise RuntimeError(
            f"Fold {fold}: "
            f"filename missing from embeddings: {e}"
        )


    test_embeddings = (
        test_embeddings_file[
            order
        ]
    )


    # =============================================================
    # PROJECT HELD-OUT TEST
    # =============================================================

    test_pc1 = (
        (
            test_embeddings
            - center
        )
        @ component
    )


    test_z = (
        test_pc1
        - dev_score_mean
    ) / dev_score_std


    test_pearson = pearsonr(
        test_z,
        test["DL"],
    ).statistic


    test_spearman = spearmanr(
        test_z,
        test["DL"],
    ).statistic


    print()
    print(
        "TEST Pearson PC1~DL    :",
        f"{test_pearson:.6f}",
    )

    print(
        "TEST Spearman PC1~DL   :",
        f"{test_spearman:.6f}",
    )


    result = test[
        [
            "cache_index",
            "filename",
            "culture",
            "fold",
            "DL",
            "m3_tta8",
        ]
    ].copy()


    result[
        "latent_pc1"
    ] = test_pc1


    result[
        "latent_pc1_z"
    ] = test_z


    oof_parts.append(
        result
    )


    fold_rows.append({
        "fold":
            fold,

        "n_dev":
            len(dev),

        "n_test":
            len(test),

        "pc1_explained_variance":
            explained,

        "orientation_sign":
            sign,

        "dev_pearson_pc1_DL":
            dev_pearson,

        "dev_spearman_pc1_DL":
            dev_spearman,

        "test_pearson_pc1_DL":
            test_pearson,

        "test_spearman_pc1_DL":
            test_spearman,
    })


    del model
    del dev_embeddings
    del test_embeddings
    del test_embeddings_file

    torch.cuda.empty_cache()

    gc.collect()


# =====================================================================
# COMBINE HELD-OUT PROJECTIONS
# =====================================================================

oof = pd.concat(
    oof_parts,
    ignore_index=True,
)


if len(oof) != EXPECTED_N:

    raise RuntimeError(
        f"Expected {EXPECTED_N} OOF rows, "
        f"got {len(oof)}."
    )


if oof["filename"].duplicated().any():

    raise RuntimeError(
        "Duplicate OOF filename."
    )


if (
    oof["culture"].nunique()
    != EXPECTED_CULTURES
):

    raise RuntimeError(
        "Incorrect number of cultures."
    )


oof = (
    oof
    .sort_values(
        "cache_index"
    )
    .reset_index(
        drop=True
    )
)


fold_df = pd.DataFrame(
    fold_rows
)


# =====================================================================
# OOF NUCLEUS-LEVEL ANALYSIS
# =====================================================================

oof_pearson = pearsonr(
    oof["latent_pc1_z"],
    oof["DL"],
).statistic


oof_spearman = spearmanr(
    oof["latent_pc1_z"],
    oof["DL"],
).statistic


# =====================================================================
# CULTURE LEVEL
# =====================================================================

culture = (
    oof
    .groupby(
        "culture"
    )
    .agg(
        n=(
            "filename",
            "size",
        ),

        DL=(
            "DL",
            "mean",
        ),

        M3_DL_prediction=(
            "m3_tta8",
            "mean",
        ),

        latent_pc1_z=(
            "latent_pc1_z",
            "mean",
        ),
    )
    .reset_index()
)


culture_pearson = pearsonr(
    culture["latent_pc1_z"],
    culture["DL"],
).statistic


culture_spearman = spearmanr(
    culture["latent_pc1_z"],
    culture["DL"],
).statistic


# =====================================================================
# SAVE
# =====================================================================

oof.to_csv(
    OUT_OOF,
    index=False,
)


fold_df.to_csv(
    OUT_FOLD,
    index=False,
)


culture.to_csv(
    OUT_CULTURE,
    index=False,
)


summary_lines = [
    "=" * 88,
    "M3 CROSS-FOLD LATENT PC1 — RESULTS",
    "=" * 88,
    "",
    (
        f"Nuclei   : "
        f"{len(oof)}"
    ),
    (
        f"Cultures : "
        f"{culture.shape[0]}"
    ),
    "",
    (
        "Principle: PCA fitted separately "
        "on the development set of each fold."
    ),
    (
        "The test fold is never used "
        "to fit the PCA."
    ),
    (
        "The PC1 sign is fixed using only "
        "development-set DL."
    ),
    (
        "Longitudinal data are never used "
        "to choose the sign."
    ),
    "",
    "OOF — NUCLEUS LEVEL",
    (
        f"Pearson PC1~DL  : "
        f"{oof_pearson:.6f}"
    ),
    (
        f"Spearman PC1~DL : "
        f"{oof_spearman:.6f}"
    ),
    "",
    "OOF — CULTURE LEVEL",
    (
        f"Pearson PC1~DL  : "
        f"{culture_pearson:.6f}"
    ),
    (
        f"Spearman PC1~DL : "
        f"{culture_spearman:.6f}"
    ),
    "",
    "BY FOLD",
    fold_df.round(6).to_string(
        index=False
    ),
    "",
    (
        "IMPORTANT: this score is a latent representation "
        "axis, not biological time or validated pseudotime."
    ),
]


summary = "\n".join(
    summary_lines
)


OUT_SUMMARY.write_text(
    summary,
    encoding="utf-8",
)


print()
print(
    summary
)


print()
print("=" * 88)
print("OUTPUT FILES")
print("=" * 88)

print(
    OUT_OOF
)

print(
    OUT_FOLD
)

print(
    OUT_CULTURE
)

print(
    OUT_SUMMARY
)


for fold in range(5):

    print(
        OUT_DIR
        / f"fold{fold}_pc1_calibrator.npz"
    )


print()
print("DONE")

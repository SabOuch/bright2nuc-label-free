"""
Extract held-out M3 embeddings for the five frozen culture-wise folds.

Purpose
-------
Re-run each frozen M3 checkpoint on its held-out fold using the same TTA8
inference procedure as the static evaluation, then save the corresponding
ConvNeXt-Tiny embeddings together with explicit sample metadata.

Inputs
------
- ``data/processed/m1_bf16_uint8.npy``
- ``outputs/m3/fold{0..4}/test_predictions.csv``
- ``outputs/m3/fold{0..4}/best_model.pt``

Outputs
-------
Written under ``outputs/m3/oof_embeddings/``:
- ``fold{0..4}_embeddings_tta8.npz``
- ``metadata.csv``
- ``summary.txt``

Validation
----------
For every fold, regenerated TTA8 differentiation predictions are compared with
the predictions stored by the frozen M3 run. A mismatch larger than
``PREDICTION_TOLERANCE`` aborts the script.

Notes
-----
The embeddings are taken before the prediction heads and averaged over the same
eight in-plane TTA views. Fold-specific embeddings are stored separately
because they originate from different trained outer-fold models.

The frozen extraction was performed with CUDA and mixed precision; CUDA remains
required here to preserve the validated reproduction path.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import timm


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

OUT_DIR = (
    M3_DIR
    / "oof_embeddings"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

METADATA_PATH = (
    OUT_DIR
    / "metadata.csv"
)

SUMMARY_PATH = (
    OUT_DIR
    / "summary.txt"
)

EXPECTED_TOTAL = 30877
EXPECTED_CULTURES = 48

BATCH_SIZE = 64
NUM_WORKERS = 4

PREDICTION_TOLERANCE = 1e-3


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable."
    )


device = torch.device(
    "cuda"
)


print("=" * 88)
print("M3 — OOF EMBEDDING EXTRACTION")
print("=" * 88)

print(
    "GPU       :",
    torch.cuda.get_device_name(0),
)

print(
    "PyTorch   :",
    torch.__version__,
)

print(
    "timm      :",
    timm.__version__,
)

print(
    "Cache     :",
    CACHE_PATH,
)


# =====================================================================
# MODEL — EXACT M3 ARCHITECTURE
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

        return self.encoder(
            x
        )


    def forward(self, x):
        """Return differentiation and auxiliary-marker predictions."""

        features = self.encode(
            x
        )

        dl = (
            self.head_dl(
                features
            )
            .squeeze(1)
        )

        markers = self.head_markers(
            features
        )

        return dl, markers


# =====================================================================
# DATASET
# =====================================================================

class EmbeddingDataset(Dataset):
    """Read brightfield crops from the frozen memory-mapped cache."""

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

        return len(
            self.metadata
        )


    def __getitem__(
        self,
        i,
    ):

        self._ensure_cache()

        cache_index = int(
            self.metadata
            .iloc[i][
                "cache_index"
            ]
        )

        image = np.array(
            self.cache[
                cache_index
            ],
            dtype=np.uint8,
            copy=True,
        )

        return (
            torch.from_numpy(
                image
            ),
            torch.tensor(
                cache_index,
                dtype=torch.long,
            ),
        )


# =====================================================================
# EXTRACTION TTA x8
# =====================================================================

@torch.no_grad()
def extract_fold(
    model,
    loader,
    bf_mean,
    bf_std,
):
    """Extract TTA8-averaged embeddings, predictions, and cache indices."""

    model.eval()

    all_embeddings = []
    all_predictions = []
    all_indices = []


    for batch_number, (
        x,
        indices,
    ) in enumerate(
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


        embeddings_tta = []
        predictions_tta = []


        # ---------------------------------------------------------
        # Same D4 TTA as M3:
        # rotations 0 / 90 / 180 / 270
        # each without and then with horizontal mirroring.
        # ---------------------------------------------------------

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

                emb = model.encode(
                    xr
                )

                pred = (
                    model.head_dl(
                        emb
                    )
                    .squeeze(1)
                )


            embeddings_tta.append(
                emb.float()
            )

            predictions_tta.append(
                pred.float()
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

                emb = model.encode(
                    xf
                )

                pred = (
                    model.head_dl(
                        emb
                    )
                    .squeeze(1)
                )


            embeddings_tta.append(
                emb.float()
            )

            predictions_tta.append(
                pred.float()
            )


        embedding_mean = (
            torch.stack(
                embeddings_tta,
                dim=0,
            )
            .mean(
                dim=0
            )
        )


        prediction_mean = (
            torch.stack(
                predictions_tta,
                dim=0,
            )
            .mean(
                dim=0
            )
        )


        all_embeddings.append(
            embedding_mean
            .cpu()
            .numpy()
        )

        all_predictions.append(
            prediction_mean
            .cpu()
            .numpy()
        )

        all_indices.append(
            indices.numpy()
        )


        if (
            batch_number == 1
            or batch_number % 20 == 0
            or batch_number == len(loader)
        ):

            print(
                f"  batch "
                f"{batch_number:4d}/"
                f"{len(loader):4d}"
            )


    return (
        np.concatenate(
            all_embeddings,
            axis=0,
        ),

        np.concatenate(
            all_predictions,
            axis=0,
        ),

        np.concatenate(
            all_indices,
            axis=0,
        ),
    )


# =====================================================================
# FIVE OOF FOLDS
# =====================================================================

metadata_parts = []

embedding_dim = None
total_nuclei = 0

max_prediction_difference_global = 0.0


for fold in range(5):

    print()
    print("=" * 88)
    print(
        f"FOLD {fold}"
    )
    print("=" * 88)


    predictions_path = (
        M3_DIR
        / f"fold{fold}"
        / "test_predictions.csv"
    )

    checkpoint_path = (
        M3_DIR
        / f"fold{fold}"
        / "best_model.pt"
    )


    if not predictions_path.exists():
        raise FileNotFoundError(
            predictions_path
        )


    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )


    meta = pd.read_csv(
        predictions_path
    )


    required_columns = {
        "cache_index",
        "filename",
        "culture",
        "fold",
        "target_DL",
        "prediction_tta8",
    }


    missing = (
        required_columns
        - set(meta.columns)
    )


    if missing:
        raise RuntimeError(
            f"Fold {fold}: "
            f"missing columns "
            f"{missing}"
        )


    if not (
        meta["fold"] == fold
    ).all():

        raise RuntimeError(
            f"Fold {fold}: "
            "inconsistent fold number."
        )


    if meta[
        "filename"
    ].duplicated().any():

        raise RuntimeError(
            f"Fold {fold}: "
            "duplicate filename."
        )


    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )


    required_checkpoint = {
        "model_state_dict",
        "bf_mean",
        "bf_std",
        "test_cultures",
        "fold",
    }


    missing_checkpoint = (
        required_checkpoint
        - set(
            checkpoint.keys()
        )
    )


    if missing_checkpoint:
        raise RuntimeError(
            f"Fold {fold}: "
            f"incomplete checkpoint "
            f"{missing_checkpoint}"
        )


    if int(
        checkpoint[
            "fold"
        ]
    ) != fold:

        raise RuntimeError(
            f"Fold {fold}: "
            "checkpoint from another fold."
        )


    csv_cultures = sorted(
        meta[
            "culture"
        ].unique()
    )


    checkpoint_cultures = sorted(
        checkpoint[
            "test_cultures"
        ]
    )


    if (
        csv_cultures
        != checkpoint_cultures
    ):

        raise RuntimeError(
            f"Fold {fold}: "
            "different test cultures "
            "in the CSV and checkpoint."
        )


    bf_mean = float(
        checkpoint[
            "bf_mean"
        ]
    )

    bf_std = float(
        checkpoint[
            "bf_std"
        ]
    )


    model = (
        M3ConvNeXtTiny()
        .to(device)
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )


    current_dim = int(
        model.n_features
    )


    if embedding_dim is None:

        embedding_dim = current_dim

    elif current_dim != embedding_dim:

        raise RuntimeError(
            "Embedding dimension differs "
            "across folds."
        )


    print(
        "Nuclei        :",
        len(meta),
    )

    print(
        "Cultures      :",
        meta[
            "culture"
        ].nunique(),
    )

    print(
        "Embedding dim :",
        embedding_dim,
    )

    print(
        "BF mean       :",
        f"{bf_mean:.8f}",
    )

    print(
        "BF std        :",
        f"{bf_std:.8f}",
    )


    dataset = EmbeddingDataset(
        meta,
        CACHE_PATH,
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )


    (
        embeddings,
        regenerated_predictions,
        cache_indices,
    ) = extract_fold(
        model,
        loader,
        bf_mean,
        bf_std,
    )


    # -------------------------------------------------------------
    # SANITY CHECK 1: dimensions
    # -------------------------------------------------------------

    if embeddings.shape != (
        len(meta),
        embedding_dim,
    ):

        raise RuntimeError(
            f"Fold {fold}: "
            f"unexpected shape "
            f"{embeddings.shape}"
        )


    # -------------------------------------------------------------
    # SANITY CHECK 2: order
    # -------------------------------------------------------------

    expected_indices = (
        meta[
            "cache_index"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )


    if not np.array_equal(
        cache_indices,
        expected_indices,
    ):

        raise RuntimeError(
            f"Fold {fold}: "
            "misaligned cache_index."
        )


    # -------------------------------------------------------------
    # SANITY CHECK 3 :
    # Predictions regenerated from the checkpoint must match
    # the stored prediction_tta8 values.
    # -------------------------------------------------------------

    stored_predictions = (
        meta[
            "prediction_tta8"
        ]
        .to_numpy(
            dtype=np.float32
        )
    )


    max_prediction_difference = float(
        np.max(
            np.abs(
                regenerated_predictions
                - stored_predictions
            )
        )
    )


    max_prediction_difference_global = max(
        max_prediction_difference_global,
        max_prediction_difference,
    )


    print(
        "Maximum TTA8 prediction difference:",
        f"{max_prediction_difference:.8g}",
    )


    if (
        max_prediction_difference
        > PREDICTION_TOLERANCE
    ):

        raise RuntimeError(
            f"Fold {fold}: "
            "the regenerated predictions "
            "do not match the M3 run."
        )


    if not np.isfinite(
        embeddings
    ).all():

        raise RuntimeError(
            f"Fold {fold}: "
            "NaN or Inf in embeddings."
        )


    # -------------------------------------------------------------
    # SAVE FOLD SEPARATELY
    # -------------------------------------------------------------

    fold_path = (
        OUT_DIR
        / f"fold{fold}_embeddings_tta8.npz"
    )


    np.savez_compressed(
        fold_path,

        embeddings=embeddings.astype(
            np.float32,
            copy=False,
        ),

        cache_indices=expected_indices,

        filenames=(
            meta[
                "filename"
            ]
            .astype(str)
            .to_numpy()
        ),

        cultures=(
            meta[
                "culture"
            ]
            .astype(str)
            .to_numpy()
        ),

        target_DL=(
            meta[
                "target_DL"
            ]
            .to_numpy(
                dtype=np.float32
            )
        ),

        prediction_tta8=(
            stored_predictions
        ),
    )


    fold_meta = meta[
        [
            "cache_index",
            "filename",
            "culture",
            "fold",
            "target_DL",
            "prediction_tta8",
        ]
    ].copy()


    fold_meta[
        "embedding_file"
    ] = (
        fold_path.name
    )


    fold_meta[
        "embedding_row"
    ] = np.arange(
        len(fold_meta),
        dtype=np.int64,
    )


    metadata_parts.append(
        fold_meta
    )


    total_nuclei += len(
        meta
    )


    print(
        "Saved          :",
        fold_path,
    )


    del model
    del embeddings
    del regenerated_predictions

    torch.cuda.empty_cache()


# =====================================================================
# GLOBAL METADATA CHECK
# =====================================================================

metadata = pd.concat(
    metadata_parts,
    ignore_index=True,
)


if len(metadata) != EXPECTED_TOTAL:

    raise RuntimeError(
        f"Expected {EXPECTED_TOTAL} nuclei, "
        f"got {len(metadata)}."
    )


if (
    metadata[
        "filename"
    ].duplicated().any()
):

    raise RuntimeError(
        "Duplicate filename across folds."
    )


if (
    metadata[
        "cache_index"
    ].duplicated().any()
):

    raise RuntimeError(
        "Duplicate cache_index across folds."
    )


if (
    metadata[
        "culture"
    ].nunique()
    != EXPECTED_CULTURES
):

    raise RuntimeError(
        f"Expected {EXPECTED_CULTURES} cultures, "
        f"got "
        f"{metadata['culture'].nunique()}."
    )


metadata = (
    metadata
    .sort_values(
        "cache_index"
    )
    .reset_index(
        drop=True
    )
)


metadata.to_csv(
    METADATA_PATH,
    index=False,
)


# =====================================================================
# SUMMARY
# =====================================================================

summary_lines = [
    "=" * 88,
    "M3 OOF EMBEDDINGS — EXTRACTION COMPLETE",
    "=" * 88,
    "",
    (
        f"Nuclei        : "
        f"{len(metadata)}"
    ),
    (
        f"Cultures      : "
        f"{metadata['culture'].nunique()}"
    ),
    (
        f"Folds         : "
        f"{sorted(metadata['fold'].unique())}"
    ),
    (
        f"Embedding dim : "
        f"{embedding_dim}"
    ),
    (
        "Method        : "
        "ConvNeXt embedding before the heads, "
        "TTA x8 mean"
    ),
    (
        "Storage       : "
        "one separate NPZ file per fold"
    ),
    (
        "Maximum regenerated-prediction difference: "
        f"{max_prediction_difference_global:.8g}"
    ),
    "",
    (
        "No model was retrained."
    ),
    (
        "Fluorescence is never an input."
    ),
    (
        "Biological time is neither an input "
        "nor a target."
    ),
    "",
    (
        "IMPORTANT: raw embeddings from different folds "
        "must not be mixed directly in a single PCA."
    ),
]


summary = "\n".join(
    summary_lines
)


SUMMARY_PATH.write_text(
    summary,
    encoding="utf-8",
)


print()
print(
    summary
)

print()
print("=" * 88)
print("FILES")
print("=" * 88)


for fold in range(5):

    print(
        OUT_DIR
        / f"fold{fold}_embeddings_tta8.npz"
    )


print(
    METADATA_PATH
)

print(
    SUMMARY_PATH
)


print()
print("DONE")

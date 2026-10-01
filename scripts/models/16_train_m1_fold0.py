from pathlib import Path
import os
import hashlib
import math
import random
import time

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr

from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import timm


# =====================================================================
# CONFIGURATION
# =====================================================================

ROOT = Path(__file__).resolve().parents[2]

CACHE_PATH = ROOT / "data/processed/m1_bf16_uint8.npy"
INDEX_PATH = ROOT / "data/processed/m1_bf16_index.csv"
FOLDS_PATH = ROOT / "data/processed/folds.csv"

TEST_FOLD = int(os.environ.get("M1_FOLD", "0"))

if TEST_FOLD not in {0, 1, 2, 3, 4}:
    raise ValueError(f"Invalid M1_FOLD: {TEST_FOLD}")

OUT = ROOT / f"outputs/m1/fold{TEST_FOLD}"
OUT.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = OUT / "best_model.pt"
HISTORY_PATH = OUT / "history.csv"
PREDICTIONS_PATH = OUT / "test_predictions.csv"
SUMMARY_PATH = OUT / "summary.txt"

SEED = 42

MAX_EPOCHS = 30
PATIENCE = 5

MICRO_BATCH = 64
GRAD_ACCUM = 4
EFFECTIVE_BATCH = MICRO_BATCH * GRAD_ACCUM

NUM_WORKERS = 4

ENCODER_LR = 1e-4
HEAD_LR = 3e-4
WEIGHT_DECAY = 0.05

EXPECTED_FOLDS_SHA256 = (
    "8907a4f596a595c11b84ab87f1141f6b"
    "4124100bd8f88f32b4c1b7d11d34cca7"
)

EXPECTED_INDEX_SHA256 = (
    "55de2d34c1bba17e58ebd4ee3c60f5cd"
    "16efb4ea0e04582fc0166bb3835d6635"
)


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )

    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# =====================================================================
# HASH CHECK
# =====================================================================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


folds_hash = sha256_file(FOLDS_PATH)
index_hash = sha256_file(INDEX_PATH)

if folds_hash != EXPECTED_FOLDS_SHA256:
    raise RuntimeError(
        "folds.csv has changed!\n"
        f"Expected: {EXPECTED_FOLDS_SHA256}\n"
        f"Found   : {folds_hash}"
    )

if index_hash != EXPECTED_INDEX_SHA256:
    raise RuntimeError(
        "m1_bf16_index.csv has changed!\n"
        f"Expected: {EXPECTED_INDEX_SHA256}\n"
        f"Found   : {index_hash}"
    )


# =====================================================================
# DEVICE
# =====================================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable. "
        "M1 is not run on CPU."
    )

device = torch.device("cuda")

print("=" * 88)
print(f"M1 — RESNET18 2.5D — FOLD {TEST_FOLD}")
print("=" * 88)

print("GPU                :", torch.cuda.get_device_name(0))
print("PyTorch            :", torch.__version__)
print("timm               :", timm.__version__)
print("Micro batch        :", MICRO_BATCH)
print("Accumulation       :", GRAD_ACCUM)
print("Effective batch    :", EFFECTIVE_BATCH)
print("Folds SHA256       :", folds_hash)
print("Index SHA256       :", index_hash)


# =====================================================================
# METADATA
# =====================================================================

meta = pd.read_csv(INDEX_PATH)

if len(meta) != 30877:
    raise RuntimeError(
        f"Expected 30,877 rows, got {len(meta)}"
    )

if meta["filename"].duplicated().any():
    raise RuntimeError("Duplicate filename.")

if sorted(meta["fold"].unique().tolist()) != [0, 1, 2, 3, 4]:
    raise RuntimeError("Unexpected folds.")

if not np.array_equal(
    meta["cache_index"].to_numpy(),
    np.arange(len(meta)),
):
    raise RuntimeError(
        "cache_index is not sequential."
    )


# =====================================================================
# TEST = SELECTED OUTER FOLD
# =====================================================================

test_meta = meta[
    meta["fold"] == TEST_FOLD
].copy()

development_meta = meta[
    meta["fold"] != TEST_FOLD
].copy()

test_cultures = sorted(
    test_meta["culture"].unique()
)

development_cultures = (
    development_meta
    .groupby("culture")
    .agg(
        dl_mean=("TF_class", "mean"),
        n_nuclei=("filename", "size"),
    )
    .reset_index()
)


# =====================================================================
# INTERNAL VALIDATION, AT CULTURE LEVEL ONLY
#
# No nucleus from the test fold is used here.
# =====================================================================

development_cultures["dl_bin"] = pd.qcut(
    development_cultures["dl_mean"],
    q=5,
    labels=False,
    duplicates="drop",
)

sss = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.15,
    random_state=SEED,
)

train_c_idx, val_c_idx = next(
    sss.split(
        development_cultures,
        development_cultures["dl_bin"],
    )
)

train_cultures = sorted(
    development_cultures
    .iloc[train_c_idx]["culture"]
    .tolist()
)

val_cultures = sorted(
    development_cultures
    .iloc[val_c_idx]["culture"]
    .tolist()
)

train_meta = meta[
    meta["culture"].isin(train_cultures)
].copy()

val_meta = meta[
    meta["culture"].isin(val_cultures)
].copy()


# =====================================================================
# LEAKAGE CHECK
# =====================================================================

assert not (
    set(train_cultures)
    & set(val_cultures)
)

assert not (
    set(train_cultures)
    & set(test_cultures)
)

assert not (
    set(val_cultures)
    & set(test_cultures)
)


print("\n" + "=" * 88)
print("SPLITS")
print("=" * 88)

print(
    f"Train : {len(train_meta):6d} nuclei / "
    f"{len(train_cultures):2d} cultures"
)

print(
    f"Val   : {len(val_meta):6d} nuclei / "
    f"{len(val_cultures):2d} cultures"
)

print(
    f"Test  : {len(test_meta):6d} nuclei / "
    f"{len(test_cultures):2d} cultures"
)

print("\nInternal validation cultures:")

for c in val_cultures:
    print(" -", c)

print(f"\nTest cultures for fold {TEST_FOLD}:")

for c in test_cultures:
    print(" -", c)


# =====================================================================
# TRAIN-ONLY NORMALIZATION
#
# A single global BF mean/std.
# No validation/test information.
# =====================================================================

cache = np.load(
    CACHE_PATH,
    mmap_mode="r",
)

train_indices = train_meta[
    "cache_index"
].to_numpy(dtype=np.int64)

pixel_sum = 0.0
pixel_sq_sum = 0.0
pixel_count = 0

CHUNK = 128

print("\n" + "=" * 88)
print("NORMALIZATION — TRAIN ONLY")
print("=" * 88)

for start in range(
    0,
    len(train_indices),
    CHUNK,
):
    ids = train_indices[
        start:start + CHUNK
    ]

    x = np.asarray(
        cache[ids],
        dtype=np.float32,
    ) / 255.0

    pixel_sum += x.sum(
        dtype=np.float64
    )

    pixel_sq_sum += np.square(
        x,
        dtype=np.float32,
    ).sum(
        dtype=np.float64
    )

    pixel_count += x.size

train_mean = pixel_sum / pixel_count

train_var = (
    pixel_sq_sum / pixel_count
    - train_mean ** 2
)

train_std = float(
    np.sqrt(
        max(train_var, 1e-12)
    )
)

train_mean = float(train_mean)

print(f"BF mean train : {train_mean:.8f}")
print(f"BF std train  : {train_std:.8f}")

del cache


# =====================================================================
# DATASET
# =====================================================================

class BFCacheDataset(Dataset):

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

        row = self.metadata.iloc[i]

        cache_idx = int(
            row["cache_index"]
        )

        # Explicit copy:
        # avoids read-only mmap memory issues.
        x = np.array(
            self.cache[cache_idx],
            copy=True,
            dtype=np.uint8,
        )

        y = np.float32(
            row["TF_class"]
        )

        return (
            torch.from_numpy(x),
            torch.tensor(y),
            torch.tensor(
                cache_idx,
                dtype=torch.long,
            ),
        )


train_dataset = BFCacheDataset(
    train_meta,
    CACHE_PATH,
)

val_dataset = BFCacheDataset(
    val_meta,
    CACHE_PATH,
)

test_dataset = BFCacheDataset(
    test_meta,
    CACHE_PATH,
)


generator = torch.Generator()
generator.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=MICRO_BATCH,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    generator=generator,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)


# =====================================================================
# AUGMENTATIONS
#
# No z-axis reversal.
# Spatial operations are performed only in XY.
# =====================================================================

def augment_train(x):

    # x : uint8 B,C,H,W
    x = x.float() / 255.0

    # One of the four 90° rotations.
    k = int(
        torch.randint(
            0,
            4,
            (1,),
            device=x.device,
        ).item()
    )

    if k:
        x = torch.rot90(
            x,
            k=k,
            dims=(-2, -1),
        )

    # With the flip, this yields the eight square symmetries.
    if torch.rand(
        1,
        device=x.device,
    ).item() < 0.5:
        x = torch.flip(
            x,
            dims=(-1,),
        )

    b = x.shape[0]

    # Contrast ±20%
    contrast = (
        0.8
        + 0.4
        * torch.rand(
            b, 1, 1, 1,
            device=x.device,
        )
    )

    mean = x.mean(
        dim=(1, 2, 3),
        keepdim=True,
    )

    x = (
        (x - mean)
        * contrast
        + mean
    )

    # Brightness ±20%
    brightness = (
        0.8
        + 0.4
        * torch.rand(
            b, 1, 1, 1,
            device=x.device,
        )
    )

    x = x * brightness

    x = x.clamp(
        0.0,
        1.0,
    )

    # Gamma 0.8–1.2
    gamma = (
        0.8
        + 0.4
        * torch.rand(
            b, 1, 1, 1,
            device=x.device,
        )
    )

    x = torch.pow(
        x.clamp(min=1e-6),
        gamma,
    )

    # Light Gaussian noise
    x = x + (
        0.01
        * torch.randn_like(x)
    )

    x = x.clamp(
        0.0,
        1.0,
    )

    return x


def preprocess_eval(x):

    return (
        x.float()
        / 255.0
    )


def normalize(x):

    return (
        x - train_mean
    ) / train_std


# =====================================================================
# MODEL
# =====================================================================

class M1ResNet18(nn.Module):

    def __init__(self):

        super().__init__()

        try:
            self.encoder = timm.create_model(
                "resnet18",
                pretrained=True,
                in_chans=16,
                num_classes=0,
                global_pool="avg",
            )

        except Exception as e:
            raise RuntimeError(
                "Unable to load the ImageNet "
                "ResNet-18 weights.\n"
                "NO silent fallback to random "
                "initialization is performed.\n"
                f"Original error: {e}"
            )

        n_features = int(
            self.encoder.num_features
        )

        self.head = nn.Linear(
            n_features,
            1,
        )

    def forward(self, x):

        # 64x64 -> 128x128 as expected.
        x = F.interpolate(
            x,
            size=(128, 128),
            mode="bilinear",
            align_corners=False,
        )

        features = self.encoder(x)

        pred = self.head(
            features
        ).squeeze(1)

        return pred


model = M1ResNet18().to(device)

print("\n" + "=" * 88)
print("MODEL")
print("=" * 88)

print("Backbone : ResNet-18 ImageNet")
print("Input    : 16 x 64 x 64")
print("Resize   : 16 x 128 x 128")
print("Output   : scalar DL")

n_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Parameters: "
    f"{n_params:,}"
)


# =====================================================================
# OPTIMIZER
# =====================================================================

optimizer = torch.optim.AdamW(
    [
        {
            "params": model.encoder.parameters(),
            "lr": ENCODER_LR,
        },
        {
            "params": model.head.parameters(),
            "lr": HEAD_LR,
        },
    ],
    weight_decay=WEIGHT_DECAY,
)

criterion = nn.SmoothL1Loss()

optimizer_steps_per_epoch = math.ceil(
    len(train_loader)
    / GRAD_ACCUM
)

total_steps = (
    optimizer_steps_per_epoch
    * MAX_EPOCHS
)

warmup_steps = (
    optimizer_steps_per_epoch
)


def lr_factor(step):

    if step < warmup_steps:

        # 10% -> 100% during the first epoch
        return (
            0.1
            + 0.9
            * (step + 1)
            / warmup_steps
        )

    progress = (
        step - warmup_steps
    ) / max(
        1,
        total_steps - warmup_steps
    )

    progress = min(
        max(progress, 0.0),
        1.0,
    )

    return (
        0.5
        * (
            1.0
            + math.cos(
                math.pi
                * progress
            )
        )
    )


scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lr_factor,
)


use_amp = True

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp,
)


# =====================================================================
# METRICS
# =====================================================================

def regression_metrics(
    y_true,
    y_pred,
):

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    return {
        "r2": r2_score(
            y_true,
            y_pred,
        ),
        "mae": mean_absolute_error(
            y_true,
            y_pred,
        ),
        "pearson": pearsonr(
            y_true,
            y_pred,
        ).statistic,
        "spearman": spearmanr(
            y_true,
            y_pred,
        ).statistic,
    }


# =====================================================================
# TRAIN EPOCH
# =====================================================================

def train_one_epoch():

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    running_loss = 0.0
    n_samples = 0

    for step, (
        x,
        y,
        _,
    ) in enumerate(train_loader):

        x = x.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )

        x = augment_train(x)
        x = normalize(x)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):

            pred = model(x)

            loss_full = criterion(
                pred,
                y,
            )

            loss = (
                loss_full
                / GRAD_ACCUM
            )

        scaler.scale(
            loss
        ).backward()

        do_step = (
            (step + 1) % GRAD_ACCUM == 0
            or
            (step + 1) == len(train_loader)
        )

        if do_step:

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

            scheduler.step()

        batch_size = y.shape[0]

        running_loss += (
            float(
                loss_full.detach().cpu()
            )
            * batch_size
        )

        n_samples += batch_size

    return (
        running_loss
        / n_samples
    )


# =====================================================================
# STANDARD EVALUATION
# =====================================================================

@torch.no_grad()
def predict(
    loader,
    use_tta=False,
):

    model.eval()

    all_targets = []
    all_preds = []
    all_indices = []

    for x, y, idx in loader:

        x = x.to(
            device,
            non_blocking=True,
        )

        x = preprocess_eval(x)
        x = normalize(x)

        if not use_tta:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                pred = model(x)

        else:

            tta_preds = []

            # 4 rotations
            for k in range(4):

                xr = torch.rot90(
                    x,
                    k=k,
                    dims=(-2, -1),
                )

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    tta_preds.append(
                        model(xr)
                    )

                # same rotation + XY mirroring
                xf = torch.flip(
                    xr,
                    dims=(-1,),
                )

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    tta_preds.append(
                        model(xf)
                    )

            pred = torch.stack(
                tta_preds,
                dim=0,
            ).mean(dim=0)

        all_targets.append(
            y.numpy()
        )

        all_preds.append(
            pred.float()
            .cpu()
            .numpy()
        )

        all_indices.append(
            idx.numpy()
        )

    return (
        np.concatenate(
            all_targets
        ),
        np.concatenate(
            all_preds
        ),
        np.concatenate(
            all_indices
        ),
    )


# =====================================================================
# TRAINING
# =====================================================================

history = []

best_val_mae = np.inf
best_epoch = -1
epochs_without_improvement = 0

print("\n" + "=" * 88)
print("TRAINING")
print("=" * 88)

start_time = time.time()

for epoch in range(
    1,
    MAX_EPOCHS + 1,
):

    epoch_start = time.time()

    train_loss = train_one_epoch()

    val_y, val_pred, _ = predict(
        val_loader,
        use_tta=False,
    )

    val_metrics = regression_metrics(
        val_y,
        val_pred,
    )

    encoder_lr = optimizer.param_groups[0]["lr"]
    head_lr = optimizer.param_groups[1]["lr"]

    elapsed = (
        time.time()
        - epoch_start
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_r2": val_metrics["r2"],
        "val_mae": val_metrics["mae"],
        "val_pearson": val_metrics["pearson"],
        "val_spearman": val_metrics["spearman"],
        "encoder_lr": encoder_lr,
        "head_lr": head_lr,
        "seconds": elapsed,
    })

    print(
        f"Epoch {epoch:02d} | "
        f"loss={train_loss:.5f} | "
        f"val R2={val_metrics['r2']:+.4f} | "
        f"MAE={val_metrics['mae']:.4f} | "
        f"Pearson={val_metrics['pearson']:.4f} | "
        f"{elapsed:.1f}s"
    )

    # ---------------------------------------------------------
    # EARLY STOPPING — validation only
    # ---------------------------------------------------------

    if (
        val_metrics["mae"]
        < best_val_mae - 1e-5
    ):

        best_val_mae = val_metrics["mae"]
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "epoch":
                    epoch,

                "best_val_mae":
                    best_val_mae,

                "train_mean":
                    train_mean,

                "train_std":
                    train_std,

                "train_cultures":
                    train_cultures,

                "val_cultures":
                    val_cultures,

                "test_cultures":
                    test_cultures,

                "fold":
                    TEST_FOLD,

                "folds_sha256":
                    folds_hash,

                "index_sha256":
                    index_hash,

                "architecture":
                    "resnet18",

                "in_chans":
                    16,
            },
            CHECKPOINT_PATH,
        )

    else:

        epochs_without_improvement += 1

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            "\nEarly stopping: "
            f"{PATIENCE} epochs "
            "without improvement in validation MAE."
        )

        break


pd.DataFrame(
    history
).to_csv(
    HISTORY_PATH,
    index=False,
)


# =====================================================================
# RELOAD BEST CHECKPOINT
# =====================================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print("\n" + "=" * 88)
print("BEST CHECKPOINT")
print("=" * 88)

print("Epoch     :", best_epoch)
print(
    "Val MAE   :",
    f"{best_val_mae:.6f}",
)


# =====================================================================
# TEST — RUN ONCE AFTER SELECTION BY VALIDATION
# =====================================================================

print("\nEvaluating test set without TTA...")

test_y, test_pred, test_idx = predict(
    test_loader,
    use_tta=False,
)

single_metrics = regression_metrics(
    test_y,
    test_pred,
)


print("Evaluating test set with TTA ×8...")

test_y_tta, test_pred_tta, test_idx_tta = predict(
    test_loader,
    use_tta=True,
)

assert np.array_equal(
    test_idx,
    test_idx_tta,
)

assert np.allclose(
    test_y,
    test_y_tta,
)


tta_metrics = regression_metrics(
    test_y_tta,
    test_pred_tta,
)


# =====================================================================
# CULTURE LEVEL — TTA
# =====================================================================

pred_df = pd.DataFrame({
    "cache_index": test_idx,
    "target": test_y,
    "prediction_single": test_pred,
    "prediction_tta8": test_pred_tta,
})

test_metadata = (
    meta[
        [
            "cache_index",
            "filename",
            "culture",
            "dataset_tag",
            "fold",
        ]
    ]
    .merge(
        pred_df,
        on="cache_index",
        how="inner",
        validate="one_to_one",
    )
)

culture_eval = (
    test_metadata
    .groupby("culture")
    .agg(
        target=("target", "mean"),
        prediction=("prediction_tta8", "mean"),
    )
    .reset_index()
)

culture_metrics = regression_metrics(
    culture_eval["target"],
    culture_eval["prediction"],
)


# =====================================================================
# SAVE PREDICTIONS
# =====================================================================

test_metadata.to_csv(
    PREDICTIONS_PATH,
    index=False,
)


# =====================================================================
# M0 REFERENCE FOR THE SELECTED OUTER FOLD
# =====================================================================

m0_path = (
    ROOT
    / "outputs/m0/m0_metrics.csv"
)

m0_r2 = np.nan
m0_mae = np.nan

if m0_path.exists():

    m0 = pd.read_csv(
        m0_path
    )

    row = m0[
        (m0["model"] == "M0_119")
        &
        (m0["level"] == "nucleus_fold")
        &
        (m0["fold"] == TEST_FOLD)
    ]

    if len(row) == 1:
        m0_r2 = float(
            row.iloc[0]["r2"]
        )

        m0_mae = float(
            row.iloc[0]["mae"]
        )


# =====================================================================
# OUT OF RANGE
# =====================================================================

out_low = float(
    np.mean(
        test_pred_tta < 0
    )
)

out_high = float(
    np.mean(
        test_pred_tta > 1
    )
)


# =====================================================================
# FINAL REPORT
# =====================================================================

total_minutes = (
    time.time()
    - start_time
) / 60.0

lines = []

lines.append(
    "=" * 88
)

lines.append(
    f"M1 FOLD {TEST_FOLD} — RESULTS"
)

lines.append(
    "=" * 88
)

lines.append(
    f"Best epoch       : {best_epoch}"
)

lines.append(
    f"Training time    : {total_minutes:.2f} min"
)

lines.append("")

lines.append(
    "TEST WITHOUT TTA"
)

for k, v in single_metrics.items():
    lines.append(
        f"{k:10s}: {v:.6f}"
    )

lines.append("")

lines.append(
    "TEST TTA x8 — PRIMARY RESULT"
)

for k, v in tta_metrics.items():
    lines.append(
        f"{k:10s}: {v:.6f}"
    )

lines.append("")

lines.append(
    "CULTURE LEVEL — TTA x8"
)

for k, v in culture_metrics.items():
    lines.append(
        f"{k:10s}: {v:.6f}"
    )

lines.append("")

lines.append(
    f"Pred < 0 : {100*out_low:.3f} %"
)

lines.append(
    f"Pred > 1 : {100*out_high:.3f} %"
)

lines.append("")

lines.append(
    f"M0_119 REFERENCE — SAME FOLD {TEST_FOLD}"
)

lines.append(
    f"R2  : {m0_r2:.6f}"
)

lines.append(
    f"MAE : {m0_mae:.6f}"
)

lines.append("")

lines.append(
    "Important: the test fold was used neither "
    "for normalization nor for early stopping."
)

summary = "\n".join(
    lines
)

print(
    "\n" + summary
)

SUMMARY_PATH.write_text(
    summary,
    encoding="utf-8",
)

print("\n" + "=" * 88)
print("OUTPUT FILES")
print("=" * 88)

print(CHECKPOINT_PATH)
print(HISTORY_PATH)
print(PREDICTIONS_PATH)
print(SUMMARY_PATH)

print("\nDONE")

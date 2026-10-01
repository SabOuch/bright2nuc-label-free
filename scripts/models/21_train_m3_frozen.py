from pathlib import Path
import hashlib
import math
import os
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
# CONFIG
# =====================================================================

ROOT = Path(__file__).resolve().parents[2]

CACHE_PATH = ROOT / "data/processed/m1_bf16_uint8.npy"
TARGETS_PATH = ROOT / "data/processed/m2_targets_raw.csv"

TEST_FOLD = int(
    os.environ.get("M3_FOLD", "0")
)

if TEST_FOLD not in {0, 1, 2, 3, 4}:
    raise ValueError(
        f"Invalid M3_FOLD: {TEST_FOLD}"
    )

OUT = ROOT / f"outputs/m3/fold{TEST_FOLD}"
OUT.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_PATH = OUT / "best_model.pt"
HISTORY_PATH = OUT / "history.csv"
PREDICTIONS_PATH = OUT / "test_predictions.csv"
SUMMARY_PATH = OUT / "summary.txt"

SEED = 42

MAX_EPOCHS = 30
PATIENCE = 5

MICRO_BATCH = 32
GRAD_ACCUM = 8
EFFECTIVE_BATCH = (
    MICRO_BATCH
    * GRAD_ACCUM
)

NUM_WORKERS = 4

ENCODER_LR = 1e-4
HEAD_LR = 3e-4
WEIGHT_DECAY = 0.05

MARKER_LOSS_WEIGHT = 0.5

EXPECTED_TARGETS_SHA256 = (
    "48f547631a39daed19eeb745bb182950"
    "b58fcc1a29b02671e1eac3c59d0b4aa6"
)


# =====================================================================
# UTILS
# =====================================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for block in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


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


set_seed(SEED)


# =====================================================================
# PROVENANCE
# =====================================================================

targets_hash = sha256_file(
    TARGETS_PATH
)

if (
    targets_hash
    != EXPECTED_TARGETS_SHA256
):
    raise RuntimeError(
        "m2_targets_raw.csv has changed!\n"
        f"Expected: {EXPECTED_TARGETS_SHA256}\n"
        f"Found   : {targets_hash}"
    )


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable."
    )

device = torch.device("cuda")


print("=" * 88)
print(
    f"M3 — CONVNEXT-TINY MULTITASK — FOLD {TEST_FOLD}"
)
print("=" * 88)

print(
    "GPU               :",
    torch.cuda.get_device_name(0),
)

print(
    "PyTorch           :",
    torch.__version__,
)

print(
    "timm              :",
    timm.__version__,
)

print(
    "Effective batch   :",
    EFFECTIVE_BATCH,
)

print(
    "Marker loss weight:",
    MARKER_LOSS_WEIGHT,
)

print(
    "Targets SHA256    :",
    targets_hash,
)


# =====================================================================
# LOAD METADATA
# =====================================================================

meta = pd.read_csv(
    TARGETS_PATH
)


if len(meta) != 30877:
    raise RuntimeError(
        f"Expected 30877 nuclei, "
        f"got {len(meta)}"
    )


if meta["filename"].duplicated().any():
    raise RuntimeError(
        "Duplicate filename."
    )


if not np.array_equal(
    meta["cache_index"].to_numpy(),
    np.arange(30877),
):
    raise RuntimeError(
        "Misaligned cache_index."
    )


# =====================================================================
# OFFICIAL TEST FOLD
# =====================================================================

test_meta = meta[
    meta["fold"] == TEST_FOLD
].copy()

development_meta = meta[
    meta["fold"] != TEST_FOLD
].copy()


test_cultures = sorted(
    test_meta[
        "culture"
    ].unique()
)


# =====================================================================
# INTERNAL VALIDATION
#
# EXACT SAME LOGIC AS M1.
# =====================================================================

development_cultures = (
    development_meta
    .groupby("culture")
    .agg(
        dl_mean=(
            "DL",
            "mean",
        ),
        n_nuclei=(
            "filename",
            "size",
        ),
    )
    .reset_index()
)


development_cultures[
    "dl_bin"
] = pd.qcut(
    development_cultures[
        "dl_mean"
    ],
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
        development_cultures[
            "dl_bin"
        ],
    )
)


train_cultures = sorted(
    development_cultures
    .iloc[train_c_idx][
        "culture"
    ]
    .tolist()
)


val_cultures = sorted(
    development_cultures
    .iloc[val_c_idx][
        "culture"
    ]
    .tolist()
)


train_meta = meta[
    meta["culture"].isin(
        train_cultures
    )
].copy()


val_meta = meta[
    meta["culture"].isin(
        val_cultures
    )
].copy()


# =====================================================================
# VERIFY EXACT SAME CULTURE SPLIT AS M1
# =====================================================================

m1_checkpoint_path = (
    ROOT
    / f"outputs/m1/fold{TEST_FOLD}/best_model.pt"
)


if not m1_checkpoint_path.exists():
    raise FileNotFoundError(
        m1_checkpoint_path
    )


m1_checkpoint = torch.load(
    m1_checkpoint_path,
    map_location="cpu",
    weights_only=False,
)


if (
    sorted(
        m1_checkpoint[
            "train_cultures"
        ]
    )
    != train_cultures
):
    raise RuntimeError(
        "Train cultures M3 != M1"
    )


if (
    sorted(
        m1_checkpoint[
            "val_cultures"
        ]
    )
    != val_cultures
):
    raise RuntimeError(
        "Validation cultures M3 != M1"
    )


if (
    sorted(
        m1_checkpoint[
            "test_cultures"
        ]
    )
    != test_cultures
):
    raise RuntimeError(
        "Test cultures M3 != M1"
    )


print("\n" + "=" * 88)
print("SPLITS")
print("=" * 88)

print(
    f"Train : {len(train_meta):6d} "
    f"nuclei / "
    f"{len(train_cultures):2d} cultures"
)

print(
    f"Val   : {len(val_meta):6d} "
    f"nuclei / "
    f"{len(val_cultures):2d} cultures"
)

print(
    f"Test  : {len(test_meta):6d} "
    f"nuclei / "
    f"{len(test_cultures):2d} cultures"
)

print(
    "\nM1/M3 split match: OK"
)


# =====================================================================
# BF NORMALIZATION — TRAIN ONLY
#
# EXACT SAME PROCEDURE AS M1.
# =====================================================================

cache = np.load(
    CACHE_PATH,
    mmap_mode="r",
)


train_indices = train_meta[
    "cache_index"
].to_numpy(
    dtype=np.int64
)


pixel_sum = 0.0
pixel_sq_sum = 0.0
pixel_count = 0

CHUNK = 128


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


bf_mean = (
    pixel_sum
    / pixel_count
)


bf_var = (
    pixel_sq_sum
    / pixel_count
    - bf_mean ** 2
)


bf_std = float(
    np.sqrt(
        max(
            bf_var,
            1e-12,
        )
    )
)

bf_mean = float(
    bf_mean
)


print("\n" + "=" * 88)
print("BF NORMALIZATION — TRAIN ONLY")
print("=" * 88)

print(
    f"mean : {bf_mean:.8f}"
)

print(
    f"std  : {bf_std:.8f}"
)


del cache


# =====================================================================
# MARKER NORMALIZATION — TRAIN ONLY
#
# percentile 1 -> 0
# percentile 99 -> 1
# outside -> clipped.
# =====================================================================

MARKER_COLS = [
    "OCT4_raw",
    "FOXA2_raw",
    "SOX17_raw",
]


marker_p1 = {}
marker_p99 = {}


print("\n" + "=" * 88)
print("MARKER NORMALIZATION — TRAIN ONLY")
print("=" * 88)


for col in MARKER_COLS:

    values = train_meta[
        col
    ].to_numpy(
        dtype=np.float64
    )

    p1 = float(
        np.percentile(
            values,
            1,
        )
    )

    p99 = float(
        np.percentile(
            values,
            99,
        )
    )

    if p99 <= p1:
        raise RuntimeError(
            f"Invalid percentiles "
            f"for {col}"
        )

    marker_p1[col] = p1
    marker_p99[col] = p99

    print(
        f"{col:10s}: "
        f"p1={p1:.6f} "
        f"p99={p99:.6f}"
    )


marker_low = torch.tensor(
    [
        marker_p1[c]
        for c in MARKER_COLS
    ],
    dtype=torch.float32,
    device=device,
).view(
    1,
    3,
)


marker_high = torch.tensor(
    [
        marker_p99[c]
        for c in MARKER_COLS
    ],
    dtype=torch.float32,
    device=device,
).view(
    1,
    3,
)


def normalize_markers(x):

    x = (
        x - marker_low
    ) / (
        marker_high
        - marker_low
    )

    return torch.clamp(
        x,
        0.0,
        1.0,
    )


# =====================================================================
# DATASET
# =====================================================================

class M3Dataset(Dataset):

    def __init__(
        self,
        metadata,
        cache_path,
    ):

        self.metadata = (
            metadata
            .reset_index(
                drop=True
            )
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

        row = self.metadata.iloc[i]

        cache_idx = int(
            row["cache_index"]
        )

        image = np.array(
            self.cache[
                cache_idx
            ],
            copy=True,
            dtype=np.uint8,
        )

        dl = np.float32(
            row["DL"]
        )

        markers = np.array(
            [
                row["OCT4_raw"],
                row["FOXA2_raw"],
                row["SOX17_raw"],
            ],
            dtype=np.float32,
        )

        return (
            torch.from_numpy(
                image
            ),
            torch.tensor(
                dl
            ),
            torch.from_numpy(
                markers
            ),
            torch.tensor(
                cache_idx,
                dtype=torch.long,
            ),
        )


train_dataset = M3Dataset(
    train_meta,
    CACHE_PATH,
)

val_dataset = M3Dataset(
    val_meta,
    CACHE_PATH,
)

test_dataset = M3Dataset(
    test_meta,
    CACHE_PATH,
)


generator = torch.Generator()
generator.manual_seed(
    SEED
)


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
    batch_size=64,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)


test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)


# =====================================================================
# SAME AUGMENTATIONS AS M1
# =====================================================================

def augment_train(x):

    x = (
        x.float()
        / 255.0
    )

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


    if (
        torch.rand(
            1,
            device=x.device,
        ).item()
        < 0.5
    ):

        x = torch.flip(
            x,
            dims=(-1,),
        )


    b = x.shape[0]


    contrast = (
        0.8
        + 0.4
        * torch.rand(
            b,
            1,
            1,
            1,
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


    brightness = (
        0.8
        + 0.4
        * torch.rand(
            b,
            1,
            1,
            1,
            device=x.device,
        )
    )


    x = (
        x
        * brightness
    )


    x = x.clamp(
        0.0,
        1.0,
    )


    gamma = (
        0.8
        + 0.4
        * torch.rand(
            b,
            1,
            1,
            1,
            device=x.device,
        )
    )


    x = torch.pow(
        x.clamp(
            min=1e-6
        ),
        gamma,
    )


    x = (
        x
        + 0.01
        * torch.randn_like(x)
    )


    return x.clamp(
        0.0,
        1.0,
    )


def preprocess_eval(x):

    return (
        x.float()
        / 255.0
    )


def normalize_bf(x):

    return (
        x - bf_mean
    ) / bf_std


# =====================================================================
# MODEL
# =====================================================================

class M3ConvNeXtTiny(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = timm.create_model(
            "convnext_tiny",
            pretrained=True,
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

        # Same principle as M2:
        # fluorescence is only an auxiliary target.
        self.head_markers = nn.Linear(
            self.n_features,
            3,
        )


    def encode(self, x):

        # Exactly the same resizing as M2.
        x = F.interpolate(
            x,
            size=(128, 128),
            mode="bilinear",
            align_corners=False,
        )

        return self.encoder(x)


    def forward(self, x):

        features = self.encode(x)

        dl = (
            self.head_dl(features)
            .squeeze(1)
        )

        markers = self.head_markers(
            features
        )

        return dl, markers


model = M3ConvNeXtTiny().to(
    device
)


print("\n" + "=" * 88)
print("MODEL")
print("=" * 88)

print(
    "Backbone : ConvNeXt-Tiny ImageNet"
)

print(
    "Input    : BF 16 x 64 x 64"
)

print(
    "Outputs  : DL + OCT4 + FOXA2 + SOX17"
)

print(
    "Biological time: NOT USED"
)

print(
    "Parameters:",
    f"{sum(p.numel() for p in model.parameters()):,}",
)


# =====================================================================
# OPTIMIZER
# =====================================================================

optimizer = torch.optim.AdamW(
    [
        {
            "params":
                model.encoder.parameters(),
            "lr":
                ENCODER_LR,
        },
        {
            "params":
                list(
                    model.head_dl.parameters()
                )
                +
                list(
                    model.head_markers.parameters()
                ),
            "lr":
                HEAD_LR,
        },
    ],
    weight_decay=WEIGHT_DECAY,
)


criterion_dl = nn.SmoothL1Loss()
criterion_markers = nn.SmoothL1Loss()


optimizer_steps_per_epoch = (
    math.ceil(
        len(train_loader)
        / GRAD_ACCUM
    )
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
        total_steps
        - warmup_steps,
    )


    progress = min(
        max(
            progress,
            0.0,
        ),
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


scheduler = (
    torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_factor,
    )
)


use_amp = True


scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp,
)


# =====================================================================
# TRAIN
# =====================================================================

def train_one_epoch():

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    total_loss_sum = 0.0
    dl_loss_sum = 0.0
    marker_loss_sum = 0.0

    n_samples = 0


    for step, (
        x,
        y_dl,
        y_markers_raw,
        _,
    ) in enumerate(
        train_loader
    ):

        x = x.to(
            device,
            non_blocking=True,
        )

        y_dl = y_dl.to(
            device,
            non_blocking=True,
        )

        y_markers_raw = (
            y_markers_raw.to(
                device,
                non_blocking=True,
            )
        )


        y_markers = (
            normalize_markers(
                y_markers_raw
            )
        )


        x = augment_train(x)

        x = normalize_bf(x)


        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):

            pred_dl, pred_markers = (
                model(x)
            )


            loss_dl = criterion_dl(
                pred_dl,
                y_dl,
            )


            loss_markers = (
                criterion_markers(
                    pred_markers,
                    y_markers,
                )
            )


            loss_full = (
                loss_dl
                + MARKER_LOSS_WEIGHT
                * loss_markers
            )


            loss = (
                loss_full
                / GRAD_ACCUM
            )


        scaler.scale(
            loss
        ).backward()


        do_step = (
            (
                (step + 1)
                % GRAD_ACCUM
                == 0
            )
            or
            (
                step + 1
                == len(train_loader)
            )
        )


        if do_step:

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            # With AMP, GradScaler may sometimes skip optimizer.step()
            # when an overflow is detected.
            # In that case, the scheduler must NOT advance.
            scale_before = scaler.get_scale()

            scaler.step(
                optimizer
            )

            scaler.update()

            scale_after = scaler.get_scale()

            optimizer.zero_grad(
                set_to_none=True
            )

            if scale_after >= scale_before:
                scheduler.step()


        batch_size = (
            y_dl.shape[0]
        )


        total_loss_sum += (
            float(
                loss_full.detach().cpu()
            )
            * batch_size
        )


        dl_loss_sum += (
            float(
                loss_dl.detach().cpu()
            )
            * batch_size
        )


        marker_loss_sum += (
            float(
                loss_markers.detach().cpu()
            )
            * batch_size
        )


        n_samples += (
            batch_size
        )


    return {
        "loss":
            total_loss_sum
            / n_samples,

        "dl_loss":
            dl_loss_sum
            / n_samples,

        "marker_loss":
            marker_loss_sum
            / n_samples,
    }


# =====================================================================
# PREDICTION
# =====================================================================

@torch.no_grad()
def predict(
    loader,
    use_tta=False,
):

    model.eval()

    all_dl_true = []
    all_dl_pred = []

    all_marker_true = []
    all_marker_pred = []

    all_indices = []


    for (
        x,
        y_dl,
        y_markers_raw,
        idx,
    ) in loader:

        x = x.to(
            device,
            non_blocking=True,
        )

        y_markers_raw = (
            y_markers_raw.to(
                device,
                non_blocking=True,
            )
        )

        y_markers = (
            normalize_markers(
                y_markers_raw
            )
        )


        x = preprocess_eval(x)

        x = normalize_bf(x)


        if not use_tta:

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):

                pred_dl, pred_markers = (
                    model(x)
                )


        else:

            dl_preds = []
            marker_preds = []


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

                    p_dl, p_m = model(
                        xr
                    )


                dl_preds.append(
                    p_dl
                )

                marker_preds.append(
                    p_m
                )


                xf = torch.flip(
                    xr,
                    dims=(-1,),
                )


                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=use_amp,
                ):

                    p_dl, p_m = model(
                        xf
                    )


                dl_preds.append(
                    p_dl
                )

                marker_preds.append(
                    p_m
                )


            pred_dl = torch.stack(
                dl_preds,
                dim=0,
            ).mean(
                dim=0
            )


            pred_markers = torch.stack(
                marker_preds,
                dim=0,
            ).mean(
                dim=0
            )


        all_dl_true.append(
            y_dl.numpy()
        )

        all_dl_pred.append(
            pred_dl.float()
            .cpu()
            .numpy()
        )

        all_marker_true.append(
            y_markers.float()
            .cpu()
            .numpy()
        )

        all_marker_pred.append(
            pred_markers.float()
            .cpu()
            .numpy()
        )

        all_indices.append(
            idx.numpy()
        )


    return (
        np.concatenate(
            all_dl_true
        ),
        np.concatenate(
            all_dl_pred
        ),
        np.concatenate(
            all_marker_true
        ),
        np.concatenate(
            all_marker_pred
        ),
        np.concatenate(
            all_indices
        ),
    )


# =====================================================================
# TRAINING LOOP
# =====================================================================

history = []

best_val_mae = np.inf
best_epoch = -1

epochs_without_improvement = 0

start_time = time.time()


print("\n" + "=" * 88)
print("TRAINING")
print("=" * 88)


for epoch in range(
    1,
    MAX_EPOCHS + 1,
):

    epoch_start = time.time()


    train_stats = (
        train_one_epoch()
    )


    (
        val_dl,
        val_pred_dl,
        val_markers,
        val_pred_markers,
        _,
    ) = predict(
        val_loader,
        use_tta=False,
    )


    val_metrics = (
        regression_metrics(
            val_dl,
            val_pred_dl,
        )
    )


    marker_mae = np.mean(
        np.abs(
            val_markers
            - val_pred_markers
        ),
        axis=0,
    )


    elapsed = (
        time.time()
        - epoch_start
    )


    history.append({
        "epoch":
            epoch,

        "train_loss":
            train_stats["loss"],

        "train_dl_loss":
            train_stats["dl_loss"],

        "train_marker_loss":
            train_stats[
                "marker_loss"
            ],

        "val_r2":
            val_metrics["r2"],

        "val_mae":
            val_metrics["mae"],

        "val_pearson":
            val_metrics[
                "pearson"
            ],

        "val_OCT4_mae_norm":
            marker_mae[0],

        "val_FOXA2_mae_norm":
            marker_mae[1],

        "val_SOX17_mae_norm":
            marker_mae[2],

        "seconds":
            elapsed,
    })


    print(
        f"Epoch {epoch:02d} | "
        f"loss={train_stats['loss']:.5f} | "
        f"DL={train_stats['dl_loss']:.5f} | "
        f"markers={train_stats['marker_loss']:.5f} | "
        f"val R2={val_metrics['r2']:+.4f} | "
        f"MAE={val_metrics['mae']:.4f} | "
        f"marker MAE="
        f"{marker_mae.mean():.4f} | "
        f"{elapsed:.1f}s"
    )


    # ------------------------------------------------------------
    # CHECKPOINT SELECTION ONLY BY MAIN TASK DL
    # ------------------------------------------------------------

    if (
        val_metrics["mae"]
        < best_val_mae - 1e-5
    ):

        best_val_mae = (
            val_metrics["mae"]
        )

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

                "bf_mean":
                    bf_mean,

                "bf_std":
                    bf_std,

                "marker_p1":
                    marker_p1,

                "marker_p99":
                    marker_p99,

                "train_cultures":
                    train_cultures,

                "val_cultures":
                    val_cultures,

                "test_cultures":
                    test_cultures,

                "fold":
                    TEST_FOLD,

                "targets_sha256":
                    targets_hash,

                "marker_loss_weight":
                    MARKER_LOSS_WEIGHT,
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
            "without improvement in "
            "validation DL MAE."
        )

        break


pd.DataFrame(
    history
).to_csv(
    HISTORY_PATH,
    index=False,
)


# =====================================================================
# LOAD BEST
# =====================================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False,
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


print("\n" + "=" * 88)
print("BEST CHECKPOINT")
print("=" * 88)

print(
    "Epoch   :",
    best_epoch,
)

print(
    "Val MAE :",
    f"{best_val_mae:.6f}",
)


# =====================================================================
# TEST
# =====================================================================

print(
    "\nEvaluating test set without TTA..."
)


(
    test_dl,
    test_pred_single,
    test_markers,
    test_marker_pred_single,
    test_idx,
) = predict(
    test_loader,
    use_tta=False,
)


single_metrics = (
    regression_metrics(
        test_dl,
        test_pred_single,
    )
)


print(
    "Evaluating test set with TTA x8..."
)


(
    test_dl_tta,
    test_pred_tta,
    test_markers_tta,
    test_marker_pred_tta,
    test_idx_tta,
) = predict(
    test_loader,
    use_tta=True,
)


assert np.array_equal(
    test_idx,
    test_idx_tta,
)


tta_metrics = (
    regression_metrics(
        test_dl_tta,
        test_pred_tta,
    )
)


marker_test_mae = np.mean(
    np.abs(
        test_markers_tta
        - test_marker_pred_tta
    ),
    axis=0,
)


# =====================================================================
# CULTURE LEVEL
# =====================================================================

pred_df = pd.DataFrame({
    "cache_index":
        test_idx,

    "target_DL":
        test_dl,

    "prediction_single":
        test_pred_single,

    "prediction_tta8":
        test_pred_tta,

    "target_OCT4_norm":
        test_markers[:, 0],

    "target_FOXA2_norm":
        test_markers[:, 1],

    "target_SOX17_norm":
        test_markers[:, 2],

    "pred_OCT4_norm":
        test_marker_pred_tta[:, 0],

    "pred_FOXA2_norm":
        test_marker_pred_tta[:, 1],

    "pred_SOX17_norm":
        test_marker_pred_tta[:, 2],
})


test_output = (
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
    test_output
    .groupby("culture")
    .agg(
        target=(
            "target_DL",
            "mean",
        ),
        prediction=(
            "prediction_tta8",
            "mean",
        ),
    )
    .reset_index()
)


culture_metrics = (
    regression_metrics(
        culture_eval[
            "target"
        ],
        culture_eval[
            "prediction"
        ],
    )
)


# =====================================================================
# M2 SAME-FOLD REFERENCE
# =====================================================================

m2_predictions_path = (
    ROOT
    / f"outputs/m2/fold{TEST_FOLD}"
    / "test_predictions.csv"
)

if not m2_predictions_path.exists():
    raise FileNotFoundError(
        m2_predictions_path
    )

m2 = pd.read_csv(
    m2_predictions_path
)

required_m2 = {
    "filename",
    "target_DL",
    "prediction_tta8",
}

missing_m2 = (
    required_m2
    - set(m2.columns)
)

if missing_m2:
    raise RuntimeError(
        f"Missing M2 columns: "
        f"{missing_m2}"
    )


m2_aligned = (
    test_output[
        [
            "filename",
            "target_DL",
        ]
    ]
    .merge(
        m2[
            [
                "filename",
                "target_DL",
                "prediction_tta8",
            ]
        ],
        on="filename",
        how="inner",
        suffixes=(
            "_m3",
            "_m2",
        ),
        validate="one_to_one",
    )
)


if len(m2_aligned) != len(test_output):
    raise RuntimeError(
        "The M2 and M3 test nuclei "
        "do not match."
    )


max_target_difference = float(
    np.max(
        np.abs(
            m2_aligned[
                "target_DL_m3"
            ].to_numpy()
            -
            m2_aligned[
                "target_DL_m2"
            ].to_numpy()
        )
    )
)


if max_target_difference > 1e-6:
    raise RuntimeError(
        "The M2/M3 DL targets "
        "do not match."
    )


m2_metrics = regression_metrics(
    m2_aligned[
        "target_DL_m2"
    ],
    m2_aligned[
        "prediction_tta8"
    ],
)

# =====================================================================
# SAVE
# =====================================================================

test_output.to_csv(
    PREDICTIONS_PATH,
    index=False,
)


total_minutes = (
    time.time()
    - start_time
) / 60.0


lines = []

lines.append(
    "=" * 88
)

lines.append(
    f"M3 FOLD {TEST_FOLD} — RESULTS"
)

lines.append(
    "=" * 88
)

lines.append(
    f"Best epoch       : {best_epoch}"
)

lines.append(
    f"Training time    : "
    f"{total_minutes:.2f} min"
)

lines.append("")

lines.append(
    "DL TEST WITHOUT TTA"
)

for k, v in single_metrics.items():

    lines.append(
        f"{k:10s}: {v:.6f}"
    )


lines.append("")

lines.append(
    "DL TEST TTA x8 — PRIMARY RESULT"
)

for k, v in tta_metrics.items():

    lines.append(
        f"{k:10s}: {v:.6f}"
    )


lines.append("")

lines.append(
    "MARKERS — NORMALIZED MAE TTA x8"
)

lines.append(
    f"OCT4  : {marker_test_mae[0]:.6f}"
)

lines.append(
    f"FOXA2 : {marker_test_mae[1]:.6f}"
)

lines.append(
    f"SOX17 : {marker_test_mae[2]:.6f}"
)


lines.append("")

lines.append(
    "CULTURE LEVEL — DL TTA x8"
)

for k, v in culture_metrics.items():

    lines.append(
        f"{k:10s}: {v:.6f}"
    )


lines.append("")

lines.append(
    f"M2 REFERENCE — SAME FOLD {TEST_FOLD}"
)

lines.append(
    f"R2  : {m2_metrics['r2']:.6f}"
)

lines.append(
    f"MAE : {m2_metrics['mae']:.6f}"
)


lines.append("")

lines.append(
    "Delta M3-M2 :"
)

lines.append(
    f"R2  : "
    f"{tta_metrics['r2'] - m2_metrics['r2']:+.6f}"
)

lines.append(
    f"MAE : "
    f"{tta_metrics['mae'] - m2_metrics['mae']:+.6f}"
)


lines.append("")

lines.append(
    "The checkpoint is selected using only "
    "validation DL MAE."
)

lines.append(
    "Fluorescence is never a model input."
)

lines.append(
    "Biological time is neither an input nor a target."
)

lines.append(
    "M3 differs from M2 in its backbone: ResNet18 -> ConvNeXt-Tiny."
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

print(
    CHECKPOINT_PATH
)

print(
    HISTORY_PATH
)

print(
    PREDICTIONS_PATH
)

print(
    SUMMARY_PATH
)

print("\nDONE")

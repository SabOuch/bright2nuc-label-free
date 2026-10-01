from pathlib import Path
import time

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm


# ======================================================================
# CONFIG
# ======================================================================

ROOT = Path(__file__).resolve().parents[2]

CACHE_PATH = (
    ROOT
    / "data/processed/m1_bf16_uint8.npy"
)

META_PATH = (
    ROOT
    / "data/processed/m2_targets_raw.csv"
)

OUT = (
    ROOT
    / "outputs/m4"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

EMBED_PATH = (
    OUT
    / "dinov2_static_embeddings.npy"
)

PROGRESS_PATH = (
    OUT
    / "dinov2_static_progress.txt"
)

META_OUT = (
    OUT
    / "dinov2_static_metadata.csv"
)

MODEL_NAME = (
    "vit_small_patch14_dinov2.lvd142m"
)

BATCH_SIZE = 8
NUM_WORKERS = 4

EXPECTED_N = 30877
EMBED_DIM = 384

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ======================================================================
# METADATA
# ======================================================================

meta = pd.read_csv(
    META_PATH
)

if len(meta) != EXPECTED_N:
    raise RuntimeError(
        f"Expected {EXPECTED_N} metadata rows, "
        f"got {len(meta)}."
    )

if not np.array_equal(
    meta["cache_index"].to_numpy(),
    np.arange(EXPECTED_N),
):
    raise RuntimeError(
        "Metadata cache_index is not sequential."
    )


# ======================================================================
# CACHE
# ======================================================================

cache = np.load(
    CACHE_PATH,
    mmap_mode="r",
)

if cache.shape != (
    EXPECTED_N,
    16,
    64,
    64,
):
    raise RuntimeError(
        f"Unexpected cache shape: {cache.shape}"
    )

if cache.dtype != np.uint8:
    raise RuntimeError(
        f"Unexpected cache dtype: {cache.dtype}"
    )


print("=" * 88)
print("M4 — STATIC DINOv2 EXTRACTION")
print("=" * 88)

print()
print("Device      :", DEVICE)

if DEVICE.type == "cuda":
    print(
        "GPU         :",
        torch.cuda.get_device_name(0),
    )

print("Crops       :", len(cache))
print("Cache       :", cache.shape)
print("Batch       :", BATCH_SIZE)
print("Slices      : 7, 8, 9")
print("Input DINO  : 3 x 518 x 518")
print("Features    :", EMBED_DIM)
print()


# ======================================================================
# DATASET
# ======================================================================

class CropDataset(Dataset):

    def __init__(
        self,
        cache_path,
        start_index,
    ):

        self.cache_path = cache_path
        self.start_index = int(
            start_index
        )

        self.cache = None

        self.length = (
            EXPECTED_N
            - self.start_index
        )


    def _ensure_cache(self):

        if self.cache is None:

            self.cache = np.load(
                self.cache_path,
                mmap_mode="r",
            )


    def __len__(self):

        return self.length


    def __getitem__(
        self,
        i,
    ):

        self._ensure_cache()

        idx = (
            self.start_index
            + int(i)
        )

        crop = np.asarray(
            self.cache[
                idx,
                [7, 8, 9],
                :,
                :
            ],
            dtype=np.uint8,
        ).copy()

        return (
            torch.from_numpy(crop),
            idx,
        )


# ======================================================================
# RESUME EXTRACTION
# ======================================================================

if EMBED_PATH.exists():

    embeddings = np.lib.format.open_memmap(
        EMBED_PATH,
        mode="r+",
    )

    if embeddings.shape != (
        EXPECTED_N,
        EMBED_DIM,
    ):
        raise RuntimeError(
            "Existing embedding file "
            f"has the wrong shape: {embeddings.shape}"
        )

else:

    embeddings = (
        np.lib.format.open_memmap(
            EMBED_PATH,
            mode="w+",
            dtype=np.float32,
            shape=(
                EXPECTED_N,
                EMBED_DIM,
            ),
        )
    )


if PROGRESS_PATH.exists():

    start_index = int(
        PROGRESS_PATH
        .read_text()
        .strip()
    )

else:

    start_index = 0


if not (
    0
    <= start_index
    <= EXPECTED_N
):

    raise RuntimeError(
        f"Invalid progress index: {start_index}"
    )


print(
    "Resume index:",
    start_index,
)


if start_index == EXPECTED_N:

    print(
        "Extraction is already complete."
    )

else:

    dataset = CropDataset(
        CACHE_PATH,
        start_index,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )


    # ==================================================================
    # MODEL
    # ==================================================================

    print()
    print(
        "Loading DINOv2..."
    )

    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=0,
    )

    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    model = model.to(
        DEVICE
    )


    mean = torch.tensor(
        [
            0.485,
            0.456,
            0.406,
        ],
        dtype=torch.float32,
        device=DEVICE,
    ).view(
        1,
        3,
        1,
        1,
    )

    std = torch.tensor(
        [
            0.229,
            0.224,
            0.225,
        ],
        dtype=torch.float32,
        device=DEVICE,
    ).view(
        1,
        3,
        1,
        1,
    )


    # ==================================================================
    # EXTRACTION
    # ==================================================================

    t0 = time.time()

    n_done = start_index


    with torch.inference_mode():

        for batch_id, (
            x,
            indices,
        ) in enumerate(loader):

            x = x.to(
                DEVICE,
                non_blocking=True,
            ).float()

            x = (
                x
                / 255.0
            )

            x = F.interpolate(
                x,
                size=(
                    518,
                    518,
                ),
                mode="bicubic",
                align_corners=False,
            )

            x = (
                x - mean
            ) / std


            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=(
                    DEVICE.type
                    == "cuda"
                ),
            ):

                feat = model(
                    x
                )


            feat = (
                feat
                .float()
                .cpu()
                .numpy()
            )


            if (
                feat.ndim != 2
                or feat.shape[1]
                != EMBED_DIM
            ):
                raise RuntimeError(
                    f"Unexpected embedding shape: "
                    f"{feat.shape}"
                )


            if not np.isfinite(
                feat
            ).all():
                raise RuntimeError(
                    "NaN/Inf detected."
                )


            ids = (
                indices
                .numpy()
                .astype(np.int64)
            )


            embeddings[
                ids
            ] = feat


            n_done = (
                int(
                    ids[-1]
                )
                + 1
            )


            # Save progress regularly.
            if (
                batch_id % 25
                == 0
            ):

                embeddings.flush()

                PROGRESS_PATH.write_text(
                    str(
                        n_done
                    )
                )


            if (
                batch_id % 50
                == 0
                or n_done
                == EXPECTED_N
            ):

                elapsed = (
                    time.time()
                    - t0
                )

                processed = (
                    n_done
                    - start_index
                )

                rate = (
                    processed
                    / max(
                        elapsed,
                        1e-9,
                    )
                )

                remaining = (
                    EXPECTED_N
                    - n_done
                )

                eta_min = (
                    remaining
                    / max(
                        rate,
                        1e-9,
                    )
                    / 60.0
                )

                print(
                    f"{n_done:5d}/{EXPECTED_N} "
                    f"| {100*n_done/EXPECTED_N:6.2f}% "
                    f"| {rate:7.1f} crops/s "
                    f"| ETA {eta_min:6.1f} min"
                )


    embeddings.flush()

    PROGRESS_PATH.write_text(
        str(
            EXPECTED_N
        )
    )


# ======================================================================
# FINAL VALIDATION
# ======================================================================

embeddings = np.load(
    EMBED_PATH,
    mmap_mode="r",
)

print()
print("=" * 88)
print("FINAL VALIDATION")
print("=" * 88)

print(
    "Shape :",
    embeddings.shape,
)

print(
    "dtype :",
    embeddings.dtype,
)

finite = bool(
    np.isfinite(
        embeddings
    ).all()
)

print(
    "Finite:",
    finite,
)

print(
    "Mean  :",
    f"{float(embeddings.mean()):.6f}",
)

print(
    "Std   :",
    f"{float(embeddings.std()):.6f}",
)


norms = np.linalg.norm(
    np.asarray(
        embeddings,
        dtype=np.float32,
    ),
    axis=1,
)

print(
    "Mean norm:",
    f"{norms.mean():.6f}",
)

print(
    "Minimum norm:",
    f"{norms.min():.6f}",
)

print(
    "Maximum norm:",
    f"{norms.max():.6f}",
)


if embeddings.shape != (
    EXPECTED_N,
    EMBED_DIM,
):
    raise RuntimeError(
        "Incorrect final shape."
    )

if not finite:
    raise RuntimeError(
        "Non-finite embeddings."
    )


# ======================================================================
# SAVE METADATA ALIGNMENT
# ======================================================================

keep_columns = [
    c
    for c in [
        "cache_index",
        "filename",
        "culture",
        "dataset_tag",
        "fold",
        "DL",
    ]
    if c in meta.columns
]

meta[
    keep_columns
].to_csv(
    META_OUT,
    index=False,
)


print()
print("=" * 88)
print("FILES")
print("=" * 88)

print(
    EMBED_PATH
)

print(
    META_OUT
)

print(
    PROGRESS_PATH
)

print()
print(
    "✅ M4 EXTRACTION COMPLETE"
)

print("DONE")

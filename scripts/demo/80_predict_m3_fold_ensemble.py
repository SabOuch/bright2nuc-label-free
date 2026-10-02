#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


SEEDS = [42, 43, 44, 45, 46]


class M3ConvNeXtTiny(nn.Module):
    def __init__(self):
        super().__init__()

        # pretrained=False is intentional:
        # the released checkpoint contains the complete trained state_dict.
        self.encoder = timm.create_model(
            "convnext_tiny",
            pretrained=False,
            in_chans=16,
            num_classes=0,
            global_pool="avg",
        )

        self.n_features = int(self.encoder.num_features)

        self.head_dl = nn.Linear(
            self.n_features,
            1,
        )

        self.head_markers = nn.Linear(
            self.n_features,
            3,
        )

    def encode(self, x):
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen five-seed M3 ensemble for one "
            "culture-wise outer fold."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help=(
            "NumPy file containing one crop with shape "
            "(16,64,64) or multiple crops with shape "
            "(N,16,64,64)."
        ),
    )

    parser.add_argument(
        "--weights",
        required=True,
        type=Path,
        help=(
            "Root of the extracted release weights containing "
            "fold0/, ..., fold4/."
        ),
    )

    parser.add_argument(
        "--fold",
        required=True,
        type=int,
        choices=range(5),
        help="Outer culture-wise fold (0 to 4).",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output CSV file.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    return parser.parse_args()


def load_crops(path):
    x = np.load(path)

    if x.ndim == 3:
        x = x[None, ...]

    if x.ndim != 4:
        raise ValueError(
            "Expected shape (16,64,64) or (N,16,64,64), "
            f"got {x.shape}."
        )

    if x.shape[1:] != (16, 64, 64):
        raise ValueError(
            "Expected crop shape (16,64,64), "
            f"got {x.shape[1:]}."
        )

    if x.dtype != np.uint8:
        raise ValueError(
            "Expected uint8 brightfield crops, "
            f"got dtype={x.dtype}."
        )

    return x


def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    required = {
        "model_state_dict",
        "bf_mean",
        "bf_std",
        "fold",
    }

    missing = required - set(checkpoint)

    if missing:
        raise RuntimeError(
            f"{checkpoint_path}: missing checkpoint fields {missing}"
        )

    model = M3ConvNeXtTiny().to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    return (
        model,
        float(checkpoint["bf_mean"]),
        float(checkpoint["bf_std"]),
        int(checkpoint["fold"]),
    )


@torch.no_grad()
def predict_tta8(
    model,
    x,
):
    predictions = []

    for k in range(4):
        xr = torch.rot90(
            x,
            k=k,
            dims=(-2, -1),
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=x.is_cuda,
        ):
            pred, _ = model(xr)

        predictions.append(pred)

        xf = torch.flip(
            xr,
            dims=(-1,),
        )

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=x.is_cuda,
        ):
            pred, _ = model(xf)

        predictions.append(pred)

    return torch.stack(
        predictions,
        dim=0,
    ).mean(
        dim=0,
    )


def predict_checkpoint(
    crops,
    checkpoint_path,
    expected_fold,
    batch_size,
    device,
):
    (
        model,
        bf_mean,
        bf_std,
        checkpoint_fold,
    ) = load_model(
        checkpoint_path,
        device,
    )

    if checkpoint_fold != expected_fold:
        raise RuntimeError(
            f"{checkpoint_path}: checkpoint fold "
            f"{checkpoint_fold} != requested fold "
            f"{expected_fold}."
        )

    predictions = []

    for start in range(
        0,
        len(crops),
        batch_size,
    ):
        stop = min(
            start + batch_size,
            len(crops),
        )

        x = torch.from_numpy(
            np.asarray(
                crops[start:stop],
                dtype=np.uint8,
            )
        ).to(
            device,
            non_blocking=True,
        )

        x = x.float() / 255.0

        x = (
            x - bf_mean
        ) / bf_std

        pred = predict_tta8(
            model,
            x,
        )

        predictions.append(
            pred.float().cpu().numpy()
        )

    del model

    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return np.concatenate(
        predictions
    )


def main():
    args = parse_args()

    crops = load_crops(
        args.input
    )

    fold_dir = (
        args.weights
        / f"fold{args.fold}"
    )

    if not fold_dir.is_dir():
        raise FileNotFoundError(
            fold_dir
        )

    print(
        f"Input crops : {len(crops)}"
    )
    print(
        f"Fold        : {args.fold}"
    )
    print(
        f"Device      : {args.device}"
    )

    member_predictions = []

    for seed in SEEDS:
        checkpoint_path = (
            fold_dir
            / f"seed{seed}.pt"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                checkpoint_path
            )

        print(
            f"Running seed {seed}: "
            f"{checkpoint_path}"
        )

        pred = predict_checkpoint(
            crops=crops,
            checkpoint_path=checkpoint_path,
            expected_fold=args.fold,
            batch_size=args.batch_size,
            device=args.device,
        )

        member_predictions.append(
            pred
        )

    matrix = np.stack(
        member_predictions,
        axis=1,
    )

    result = pd.DataFrame({
        "crop_index":
            np.arange(len(crops)),
    })

    for j, seed in enumerate(SEEDS):
        result[
            f"prediction_seed{seed}"
        ] = matrix[:, j]

    result["ensemble_mean"] = (
        matrix.mean(axis=1)
    )

    result["ensemble_std"] = (
        matrix.std(
            axis=1,
            ddof=1,
        )
    )

    result["ensemble_min"] = (
        matrix.min(axis=1)
    )

    result["ensemble_max"] = (
        matrix.max(axis=1)
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        args.output,
        index=False,
    )

    print()
    print(
        f"Saved: {args.output}"
    )
    print(
        f"Mean prediction  : "
        f"{result['ensemble_mean'].mean():.6f}"
    )
    print(
        f"Mean uncertainty : "
        f"{result['ensemble_std'].mean():.6f}"
    )


if __name__ == "__main__":
    main()

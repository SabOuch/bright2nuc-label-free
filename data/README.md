# Bright2Nuc data

The datasets used by this repository are not redistributed here.

The static prediction pipeline uses data derived from the Bright2Nuc dataset. The original dataset and associated resources should be obtained from the official public source:

- Zenodo DOI: `10.5281/zenodo.7014598`
- Original Bright2Nuc repository: `https://github.com/marrlab/Bright2Nuc`

## Expected external data

For the static experiments, the preprocessing scripts expect the Bright2Nuc transcription-factor prediction resources under:

```text
data/raw/TF_prediction/
├── Positions_Classes.csv
├── Features.csv
├── Features_Brightfield.csv
├── SingleNuclei/
├── SingleNucleiBF/
├── SingleNucleiTF/
└── Rawdata/
    └── TF/
```

`SingleNuclei/` contains the segmented per-nucleus crops used to define the nuclear mask during target reconstruction and validation.

`SingleNucleiBF/` contains the per-nucleus brightfield stacks used as model input. Each sample is expected to have shape `16 × 64 × 64`.

`SingleNucleiTF/` contains the corresponding fluorescence-derived transcription-factor crops used to reconstruct and audit the auxiliary supervision targets. Fluorescence is never used as model input at inference time.

`Rawdata/TF/` contains the original transcription-factor TIFF files inspected by `18_audit_m2_targets.py`. These files are required only for the auxiliary-target audit and are not used as model inputs.

## Frozen culture-wise folds

The repository includes the frozen culture-wise split definitions:

```text
data/processed/folds.csv
data/processed/culture_folds.csv
```

These files define the five-fold culture-wise evaluation used for the reported M0–M4 results.

## Generated processed files

The following files are generated locally and are intentionally excluded from Git because they can be large or reconstructed:

```text
data/processed/m1_bf16_uint8.npy
data/processed/m1_bf16_index.csv
data/processed/m2_targets_raw.csv
```

The main preparation sequence is:

```text
11_validate_target_and_features.py
        |
13_freeze_folds_and_run_m0.py
        |
15_build_m1_bf_cache.py
        |
18_audit_m2_targets.py
        |
19_prepare_m2_targets.py
```

The resulting brightfield cache contains 16-slice, 64 × 64 per-nucleus brightfield crops used by the deep-learning models.

## Longitudinal dataset

The longitudinal sensitivity analysis uses additional Bright2Nuc Zenodo resources.

The root directory is configured through the environment variable:

```bash
export BRIGHT2NUC_DATA_ROOT="/path/to/Bright2Nuc_Zenodo_full/extracted"
```

If this variable is not defined, the scripts use:

```text
data/raw/
```

as their default external-data root.

The longitudinal scripts expect resources including the brightfield 3D time-series, tracking tables, and corresponding In-silico 3D data inside that extracted dataset.

## Generated model artefacts

Training and analysis write intermediate files to:

```text
outputs/
```

This directory is intentionally ignored by Git. It may contain:

- fold-specific checkpoints;
- out-of-fold predictions;
- DINOv2 embeddings;
- ridge-regression probes;
- deep-ensemble member predictions;
- longitudinal intermediate tables;
- generated figures.

Small frozen summary tables used to document the final results are instead stored in:

```text
results/
```

Final publication figures are stored in:

```text
figures/
```

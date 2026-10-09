# Bright2Nuc M3 deep-ensemble weights

The final evaluated system uses five M3 members per outer culture-wise fold,
with training seeds 42, 43, 44, 45 and 46.

The released weights are distributed as five archives:

- `bright2nuc-m3-ensemble-fold0.tar.gz`
- `bright2nuc-m3-ensemble-fold1.tar.gz`
- `bright2nuc-m3-ensemble-fold2.tar.gz`
- `bright2nuc-m3-ensemble-fold3.tar.gz`
- `bright2nuc-m3-ensemble-fold4.tar.gz`

Each archive contains:

```text
foldX/
├── seed42.pt
├── seed43.pt
├── seed44.pt
├── seed45.pt
├── seed46.pt
└── SHA256SUMS.txt
```

Each checkpoint contains the trained model parameters together with the
fold-specific brightfield normalization statistics used during training.

## Inference

After extracting the archive corresponding to the desired outer fold:

```bash
python scripts/demo/predict_m3_ensemble.py \
    --input demo/bright2nuc_fold0_test.npy \
    --weights /path/to/extracted/weights \
    --fold 0 \
    --output predictions.csv
```

The input must contain either:

```text
(16, 64, 64)
```

for one per-nucleus brightfield crop, or:

```text
(N, 16, 64, 64)
```

for a batch of crops, stored as `uint8`.

The output CSV contains the five member predictions together with:

- `ensemble_mean`: mean differentiation prediction;
- `ensemble_std`: sample standard deviation across members;
- `ensemble_min`;
- `ensemble_max`.

The released inference code was numerically checked against the frozen
historical fold-0 predictions on 64 nuclei. The maximum absolute difference
for the ensemble mean was approximately `2.45e-05`.

These checkpoints reproduce the evaluated five-fold out-of-fold system.
They should not be interpreted as a separately trained prospective deployment
model.


## Dataset attribution

The checkpoints were trained using the Bright2Nuc dataset:

- Original study: *Bright2Nuc*
- Dataset DOI: `10.5281/zenodo.7014598`
- Dataset license: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Original code repository: https://github.com/marrlab/Bright2Nuc

The original dataset is not redistributed with these checkpoint archives.
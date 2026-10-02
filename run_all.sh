#!/usr/bin/env bash

set -euo pipefail


# ---------------------------------------------------------------------------
# Bright2Nuc — full reproducibility launcher
#
# Usage:
#   ./run_all.sh
#   ./run_all.sh --static-only
#   ./run_all.sh --dry-run
#   ./run_all.sh --static-only --dry-run
#
# Full mode reproduces:
#   - data validation / preprocessing
#   - M0–M4 static models
#   - M3 deep ensemble
#   - final static results
#   - longitudinal sensitivity analysis
#   - publication figures
#
# All generated artefacts are written under outputs/.
# Frozen tracked release artefacts under results/ and figures/ are not
# automatically overwritten.
# ---------------------------------------------------------------------------


SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
    pwd
)"

ROOT="$SCRIPT_DIR"
cd "$ROOT"


DRY_RUN=0
STATIC_ONLY=0


usage() {
    cat <<'USAGE'
Usage:
  ./run_all.sh [OPTIONS]

Options:
  --dry-run       Print the commands without executing them.
  --static-only   Reproduce only the static M0–M4 + ensemble pipeline.
  -h, --help      Show this help message.

Examples:
  ./run_all.sh
  ./run_all.sh --dry-run
  ./run_all.sh --static-only
  ./run_all.sh --static-only --dry-run
USAGE
}


while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --static-only)
            STATIC_ONLY=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done


run() {
    printf '\n>>>'
    printf ' %q' "$@"
    printf '\n'

    if [[ "$DRY_RUN" -eq 0 ]]; then
        "$@"
    fi
}


section() {
    echo
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
}


echo "Bright2Nuc reproducibility launcher"
echo "Repository : $ROOT"
echo "Dry run    : $DRY_RUN"
echo "Static only: $STATIC_ONLY"
echo

python --version


if [[ "$DRY_RUN" -eq 0 ]]; then
    if [[ ! -d "$ROOT/data/raw/TF_prediction" ]]; then
        echo
        echo "ERROR: Bright2Nuc static data were not found at:"
        echo "  $ROOT/data/raw/TF_prediction"
        echo
        echo "See data/README.md for the expected dataset layout."
        exit 2
    fi
fi


# ===========================================================================
# 1. Data validation and preprocessing
# ===========================================================================

section "1/10 — DATA VALIDATION AND PREPROCESSING"

run python scripts/data/11_validate_target_and_features.py
run python scripts/models/13_freeze_folds_and_run_m0.py
run python scripts/data/15_build_m1_bf_cache.py
run python scripts/data/18_audit_m2_targets.py
run python scripts/data/19_prepare_m2_targets.py


# ===========================================================================
# 2. M1
# ===========================================================================

section "2/10 — M1 RESNET18"

for fold in 0 1 2 3 4; do
    run env M1_FOLD="$fold" \
        python scripts/models/16_train_m1_fold0.py
done

run python scripts/evaluation/17_aggregate_m1_oof.py


# ===========================================================================
# 3. M2
# ===========================================================================

section "3/10 — M2 MULTITASK RESNET18"

for fold in 0 1 2 3 4; do
    run env M2_FOLD="$fold" \
        python scripts/models/20_train_m2_fold0.py
done

run python scripts/evaluation/21_aggregate_m2_oof.py


# ===========================================================================
# 4. M3 — frozen seed 42
# ===========================================================================

section "4/10 — M3 MULTITASK CONVNEXT-TINY — SEED 42"

for fold in 0 1 2 3 4; do
    run env M3_FOLD="$fold" \
        python scripts/models/21_train_m3_frozen.py
done

run python scripts/evaluation/26_aggregate_m3_oof.py


# ===========================================================================
# 5. M4
# ===========================================================================

section "5/10 — M4 DINOV2 + RIDGE"

run python scripts/models/42_extract_m4_dinov2_embeddings.py
run python scripts/models/43_fit_m4_ridge_oof.py


# ===========================================================================
# 6. Static comparison
# ===========================================================================

section "6/10 — FINAL M0–M4 STATIC COMPARISON"

run python scripts/evaluation/64_build_m0_m4_final_table.py


# ===========================================================================
# 7. M3 deep ensemble
# ===========================================================================

section "7/10 — M3 DEEP ENSEMBLE"

run bash scripts/models/66_run_m3_ensemble_remaining.sh
run python scripts/evaluation/67_aggregate_m3_deep_ensemble.py


# ===========================================================================
# 8. Final static release snapshot
# ===========================================================================

section "8/10 — FINAL RELEASE SNAPSHOT"

run python scripts/evaluation/72_build_final_results_snapshot.py


# ===========================================================================
# 9. Longitudinal sensitivity analysis
# ===========================================================================

if [[ "$STATIC_ONLY" -eq 0 ]]; then

    section "9/10 — LONGITUDINAL SENSITIVITY ANALYSIS"

    LONG_ROOT="${BRIGHT2NUC_DATA_ROOT:-$ROOT/data/raw}"

    echo "Longitudinal data root: $LONG_ROOT"

    run python scripts/longitudinal/27_extract_m3_oof_embeddings.py
    run python scripts/longitudinal/28_build_m3_crossfold_pc1.py
    run python scripts/longitudinal/29_map_longitudinal_frames.py
    run python scripts/longitudinal/30_validate_longitudinal_crops.py
    run python scripts/longitudinal/31_audit_longitudinal_domain.py
    run python scripts/longitudinal/32_run_m3_longitudinal.py
    run python scripts/longitudinal/34_common_support_sensitivity.py
    run python scripts/longitudinal/35_exact_longitudinal_statistics.py
    run python scripts/longitudinal/36_audit_image_quality_density.py
    run python scripts/longitudinal/37_run_m2_longitudinal.py
    run python scripts/longitudinal/38_run_m1_longitudinal.py
    run python scripts/longitudinal/44_run_m4_longitudinal.py
    run python scripts/longitudinal/69_longitudinal_confounder_correlations.py

else

    section "9/10 — LONGITUDINAL ANALYSIS SKIPPED (--static-only)"

fi


# ===========================================================================
# 10. Figures
# ===========================================================================

section "10/10 — FIGURE GENERATION"

run python scripts/figures/68_make_uncertainty_figures.py
run python scripts/figures/76_make_uncertainty_publication_figure.py
run python scripts/figures/77_make_final_pipeline_figure.py
run python scripts/figures/74_make_final_model_comparison_publication.py

if [[ "$STATIC_ONLY" -eq 0 ]]; then
    run python scripts/figures/70_make_longitudinal_final_figures.py
    run python scripts/figures/79_make_longitudinal_delta_pretty.py
fi


echo
echo "========================================================================"
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY RUN COMPLETED — no experiment was executed."
else
    echo "BRIGHT2NUC REPRODUCTION PIPELINE COMPLETED."
fi
echo "========================================================================"

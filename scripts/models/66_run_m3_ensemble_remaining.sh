#!/usr/bin/env bash

set -u

# ----------------------------------------------------------------------
# Repository root
#
# This script lives in:
#   scripts/models/66_run_m3_ensemble_remaining.sh
#
# Derive the repository root from the script location so the launcher
# does not depend on the user's home directory or on a fixed clone path.
# ----------------------------------------------------------------------

SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
    pwd
)"

ROOT="$(
    cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1
    pwd
)"

cd "$ROOT" || exit 1


# ----------------------------------------------------------------------
# Configuration
#
# Seed 42 corresponds to the frozen reference M3 model.
# This launcher generates the additional ensemble members using
# seeds 43--46 for each of the five frozen outer folds.
# ----------------------------------------------------------------------

SEEDS=(43 44 45 46)
FOLDS=(0 1 2 3 4)

LOGDIR="outputs/m3_ensemble/logs"
mkdir -p "$LOGDIR"


echo "============================================================"
echo "M3 DEEP ENSEMBLE — REMAINING RUNS"
echo "============================================================"
echo
echo "Repository root : $ROOT"
echo "Seeds           : ${SEEDS[*]}"
echo "Folds           : ${FOLDS[*]}"
echo


TOTAL=0
SKIPPED=0
DONE=0
FAILED=0


for fold in "${FOLDS[@]}"; do
    for seed in "${SEEDS[@]}"; do

        TOTAL=$((TOTAL + 1))

        OUTDIR="outputs/m3_ensemble/fold${fold}/seed${seed}"
        SUMMARY="${OUTDIR}/summary.txt"
        PRED="${OUTDIR}/test_predictions.csv"

        LOG="${LOGDIR}/fold${fold}_seed${seed}.log"

        echo
        echo "============================================================"
        echo "FOLD ${fold} — SEED ${seed}"
        echo "============================================================"

        # Do not overwrite a run that already completed successfully.
        if [[ -s "$SUMMARY" && -s "$PRED" ]]; then
            echo "Already completed — skip"
            echo "$SUMMARY"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        echo "Starting..."
        echo "Log: $LOG"
        echo

        M3_FOLD="$fold" \
        M3_SEED="$seed" \
        python scripts/models/65_train_m3_ensemble.py \
            2>&1 | tee "$LOG"

        status=${PIPESTATUS[0]}

        if [[ $status -eq 0 && -s "$SUMMARY" && -s "$PRED" ]]; then
            echo
            echo "FOLD ${fold} / SEED ${seed} completed"
            DONE=$((DONE + 1))
        else
            echo
            echo "FAILED — FOLD ${fold} / SEED ${seed}"
            echo "Return code: $status"
            FAILED=$((FAILED + 1))

            # Stop immediately rather than continuing through all runs
            # after a potentially systematic failure.
            break 2
        fi

    done
done


echo
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo "Combinations inspected : $TOTAL"
echo "Already present        : $SKIPPED"
echo "Newly completed        : $DONE"
echo "Failures               : $FAILED"

if [[ $FAILED -eq 0 ]]; then
    echo
    echo "ALL REQUESTED RUNS ARE AVAILABLE"
    exit 0
else
    echo
    echo "Stopped after an error — inspect the last log."
    exit 1
fi

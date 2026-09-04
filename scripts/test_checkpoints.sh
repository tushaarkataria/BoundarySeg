#!/bin/bash
# =============================================================================
# Score the released checkpoints, without retraining anything.
#
#   bash scripts/test_checkpoints.sh                    # all four splits
#   bash scripts/test_checkpoints.sh LA_4lbl_4unl       # just one
#
# Evaluates all three seeds of each split and prints mean +/- std, which should
# match the paper table in README.md to two decimals. Every checkpoint is the
# LAST training checkpoint (`last_model.pth`) -- no validation set was used to
# select it, and no checkpoint was cherry-picked.
#
# Roughly 2 min per case on a modern GPU; LA has 20 test cases and VerSe 52, so
# a full sweep of all four splits takes a few hours.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
GPU="${GPU:-0}"
SPLITS="${@:-LA_4lbl_4unl LA_8lbl_4unl VERSE_5lbl_5unl VERSE_10lbl_5unl}"

for SPLIT in $SPLITS; do
  DS="${SPLIT%%_*}"                          # LA | VERSE
  LBL="${SPLIT#*_}"; LBL="${LBL%%lbl*}"      # labeled count
  UNL="${SPLIT##*_}"; UNL="${UNL%%unl*}"     # unlabeled count
  DIR="checkpoints/$SPLIT"
  [ -d "$DIR" ] || { echo "No such split: $DIR" >&2; exit 1; }

  for CKPT in "$DIR"/seed*.pth; do
    echo "=== $SPLIT  $(basename "$CKPT") ==="
    python test.py --gpu "$GPU" --dataset "$DS" \
      --labelnum "$LBL" --unlabelnum "$UNL" --checkpoint "$CKPT"
  done

  echo
  python tools/summarize.py --label "$SPLIT" "$DIR"/eval_seed*/eval_results.csv
  echo
done

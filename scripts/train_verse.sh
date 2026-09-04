#!/bin/bash
# =============================================================================
# BoundarySEG on VerSe -- the exact configuration behind the paper's VerSe rows.
#
#   bash scripts/train_verse.sh 5    #  5 labeled / 5 unlabeled  -> 83.73 Dice
#   bash scripts/train_verse.sh 10   # 10 labeled / 5 unlabeled  -> 88.98 Dice
#
# Trains seeds 1337, 42 and 123 in sequence, then prints mean +/- std over the
# three, read from each run's last checkpoint (`eval_last`).
#
# GPU MEMORY: VerSe trains on 128^3 patches (vs LA's 112x112x80), and the flip
# branch holds a second forward pass. This needs a 24 GB card -- it does not fit
# in 12 GB. Everything else matches scripts/train_la.sh exactly; the recipe is
# unchanged across datasets.
# =============================================================================
set -euo pipefail

LABELNUM="${1:-10}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEEDS="${SEEDS:-1337 42 123}"
GPU="${GPU:-0}"

case "$LABELNUM" in
  5)  MAX_SAMPLES=10 ;;   #  5 labeled + 5 unlabeled
  10) MAX_SAMPLES=15 ;;   # 10 labeled + 5 unlabeled
  *) echo "Unsupported split '$LABELNUM'. The paper reports VerSe at 5 and 10 labeled." >&2
     exit 1 ;;
esac
UNL=$(( MAX_SAMPLES - LABELNUM ))
EXP="VERSE_champion_${LABELNUM}lbl_${UNL}unl"

cd "$REPO"
for SEED in $SEEDS; do
  echo "=== VerSe ${LABELNUM}lbl/${UNL}unl  seed=$SEED ==="
  python train.py \
    --gpu "$GPU" \
    --dataset VERSE \
    --seed "$SEED" \
    --labelnum "$LABELNUM" \
    --max_samples "$MAX_SAMPLES" \
    --max_iterations 20000 \
    --lr_schedule step \
    --lr_step_size 3333 \
    --grad_clip 0 \
    --method surface_ema \
    --kernel_size 3 \
    --lambda_surface 1.0 \
    --ema_decay 0.99 \
    --consistency 0.01 \
    --consistency_rampup 40.0 \
    --consistency_type mse \
    --normalization instancenorm \
    --uncertainty_thresh 0.1 \
    --copy_paste \
    --label_smoothing 0.1 \
    --flip_axes 2 3 4 \
    --flip_mode labeled \
    --flip_sup_weight 0.5 \
    --exp "${EXP}_seed${SEED}"
done

echo
echo "=== VerSe ${LABELNUM}:${UNL} -- last checkpoint, mean +/- std over seeds ==="
python tools/summarize.py --label "VerSe ${LABELNUM}:${UNL}" \
  results/VERSE/surface_ema/${EXP}_seed*/seed*/label${LABELNUM}/kernel3/lam1.0/eval_last/eval_results.csv

#!/bin/bash
# =============================================================================
# BoundarySEG on LA -- the exact configuration behind the paper's LA rows.
#
#   bash scripts/train_la.sh 4      # 4 labeled / 4 unlabeled   -> 86.95 Dice
#   bash scripts/train_la.sh 8      # 8 labeled / 4 unlabeled   -> 89.29 Dice
#
# Trains seeds 1337, 42 and 123 in sequence, then prints mean +/- std over the
# three, read from each run's last checkpoint (`eval_last`). One seed is about
# 5 h on a 12 GB GPU, so the full sweep is roughly 15 h; run the seeds on
# separate GPUs if you have them (SEEDS=1337 bash scripts/train_la.sh 4).
#
# Every flag below is a paper setting. The champion recipe is:
#   surface_ema + dual-head VNet + instancenorm + uncertainty_thresh 0.1
#   + copy_paste + label_smoothing 0.1 + flip_axes 2 3 4 (labeled, w=0.5),
#   erosion kernel 3, lambda_surface 1.0, 20k iterations, step LR.
# =============================================================================
set -euo pipefail

LABELNUM="${1:-4}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEEDS="${SEEDS:-1337 42 123}"
GPU="${GPU:-0}"

case "$LABELNUM" in
  4) MAX_SAMPLES=8  ;;   # 4 labeled + 4 unlabeled
  8) MAX_SAMPLES=12 ;;   # 8 labeled + 4 unlabeled
  *) echo "Unsupported split '$LABELNUM'. The paper reports LA at 4 and 8 labeled." >&2
     exit 1 ;;
esac
UNL=$(( MAX_SAMPLES - LABELNUM ))
EXP="LA_champion_${LABELNUM}lbl_${UNL}unl"

cd "$REPO"
for SEED in $SEEDS; do
  echo "=== LA ${LABELNUM}lbl/${UNL}unl  seed=$SEED ==="
  python train.py \
    --gpu "$GPU" \
    --dataset LA \
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
echo "=== LA ${LABELNUM}:${UNL} -- last checkpoint, mean +/- std over seeds ==="
python tools/summarize.py --label "LA ${LABELNUM}:${UNL}" \
  results/LA/surface_ema/${EXP}_seed*/seed*/label${LABELNUM}/kernel3/lam1.0/eval_last/eval_results.csv

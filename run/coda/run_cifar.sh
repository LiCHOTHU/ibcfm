#!/usr/bin/env bash
# ------------------------------------------------------------------
#  run_cifar_ib_sweep.sh
#
#  Sweeps over:
#       • model = otcfm | icfm | fm | si
#       • IB switch = on | off       (--use_ib)
#
#  Calls cifar_flow_matching.py with its native flags only.
# ------------------------------------------------------------------

set -e  # abort on error

PROJECT="FlowMatchExp"           # W&B project name
# MODELS=("otcfm" "icfm" "fm" "si")
MODELS=("otcfm")
IB_FLAGS=("on" "off")            # 'on' -> pass --use_ib

# timestamp for unique run names
timestamp=$(date +%Y%m%d-%H%M%S)

# root directory for checkpoints (internal code uses its own checkpoint logic)
CKPT_ROOT="checkpoints"
mkdir -p "$CKPT_ROOT"

for model in "${MODELS[@]}"; do
  for ib in "${IB_FLAGS[@]}"; do

    ib_tag=$([[ "$ib" == "on" ]] && echo "ib" || echo "noib")
    run_name="${model}_${ib_tag}_${timestamp}"

    python scripts/train_cifar10.py
      --model "$model"
      --sigma 0.0
      --num_channel 128
      --lr 1e-4
      --grad_clip 1.0
      --total_steps 70000
      --warmup 5000
      --batch_size 128
      --num_workers 4
      --ema_decay 0.9999
      --save_step 5000
      --device "\$( [ -z "$CUDA_VISIBLE_DEVICES" ] && echo "cpu" || echo "cuda" )"
      --wandb_project "$PROJECT"
      --wandb_run_name "$run_name"
  done
done

#!/usr/bin/env bash
# ------------------------------------------------------------------
#  run_mnist_matcher_ib_sweep.sh
#
#  Sweeps over:
#       • dataset      = mnist | fashion-mnist
#       • matcher      = cfm | ot | sb | target
#       • IB switch    = on | off  (--use_ib)
#
#  Calls scripts/train_cond_mnist.py with its native flags only.
# ------------------------------------------------------------------

set -e  # abort on error

PROJECT="FlowMatchExp"                   # W&B project name
DATASETS=("mnist" "fashion-mnist")
MATCHERS=("cfm" "ot" "sb" "target")
IB_FLAGS=("on" "off")                    # 'on' -> pass --use_ib

timestamp=$(date +%Y%m%d-%H%M%S)
CKPT_ROOT="checkpoints"
mkdir -p "$CKPT_ROOT"

for ds in "${DATASETS[@]}"; do
  for matcher in "${MATCHERS[@]}"; do
    for ib in "${IB_FLAGS[@]}"; do

      ib_tag=$([[ "$ib" == "on" ]] && echo "ib" || echo "noib")
      run_name="${ds}_${matcher}_${ib_tag}_${timestamp}"

      # export a folder for saving checkpoints (if your code uses it)
      export CHECKPOINT_DIR="${CKPT_ROOT}/${run_name}"
      mkdir -p "$CHECKPOINT_DIR"

      # build the command
      cmd=(python -m scripts.train_cond_mnist
           --dataset "$ds"
           --matcher "$matcher"
           --sigma 0.0
           --wandb_project "$PROJECT"
           --wandb_run_name "$run_name"
      )

      [[ "$ib" == "on" ]] && cmd+=(--use_ib)

      echo -e "\nLaunching: ${cmd[*]}"
      "${cmd[@]}"
    done
  done
done

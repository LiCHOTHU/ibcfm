#!/usr/bin/env bash
# ------------------------------------------------------------------
#  run_mnist_ib_sweep.sh
#
#  Sweeps over:
#       • dataset = mnist            (extend list if desired)
#       • IB switch = on | off       (--use_ib)
#
#  Calls scripts/train_cond_mnist.py with its *native* flags only.
# ------------------------------------------------------------------

set -e  # stop if any command fails

PROJECT="FlowMatchExp"               # W&B project
DATASETS=("mnist")                   # add "fashion-mnist" if you like
IB_FLAGS=("on" "off")                # 'on' -> pass --use_ib

timestamp=$(date +%Y%m%d-%H%M%S)

# Root for optional checkpoints; export as env var if your training code uses it.
CKPT_ROOT="checkpoints"
mkdir -p "$CKPT_ROOT"

for ds in "${DATASETS[@]}"; do
  for ib in "${IB_FLAGS[@]}"; do

    ib_tag=$( [[ "$ib" == "on" ]] && echo "ib" || echo "noib" )
    run_name="default_${ds}_${ib_tag}_${timestamp}"

    # Optional: expose a folder for saving checkpoints
    export CHECKPOINT_DIR="${CKPT_ROOT}/${run_name}"
    mkdir -p "$CHECKPOINT_DIR"

    cmd="python -m scripts.train_cond_mnist \
            --dataset $ds \
            --wandb_project $PROJECT \
            --wandb_run_name $run_name"

    [[ "$ib" == "on" ]] && cmd="$cmd --use_ib"

    echo -e "\nLaunching: $cmd"
    eval $cmd
  done
done

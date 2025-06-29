#!/usr/bin/env bash
# submit_imagenet_jobs.sh
# ------------------------------------------------------------------
# Submits sbatch jobs for 64×64 ImageNet flow-matching over model and IB sweeps.
# ------------------------------------------------------------------

set -e

PROJECT="FlowMatchImageNet"
MODELS=(otcfm icfm fm si)
IB_FLAGS=(on off)
timestamp=$(date +%Y%m%d-%H%M%S)

for model in "${MODELS[@]}"; do
  for ib in "${IB_FLAGS[@]}"; do

    ib_tag=$([[ "$ib" == "on" ]] && echo "ib" || echo "noib")
    RUN_NAME="${model}_${ib_tag}_${timestamp}"

    echo "Submitting ImageNet job: model=$model IB=$ib_tag run=$RUN_NAME"

    sbatch \
      --export=MODEL="$model",IB_FLAG="$ib",PROJECT="$PROJECT",RUN_NAME="$RUN_NAME" \
      imagenet.sbatch

  done
done

#!/usr/bin/env bash
# ------------------------------------------------------------------
# submit_cifar_jobs.sh
#
# Submits sbatch jobs for CIFAR-10 flow-matching over model and IB sweeps.
# ------------------------------------------------------------------

set -e

PROJECT="FlowMatchCIFAR"
MODELS=("otcfm" "icfm" "fm" "si")
IB_FLAGS=("on" "off")

# Loop and submit
for model in "${MODELS[@]}"; do
  for ib in "${IB_FLAGS[@]}"; do
    ib_tag=$([[ "$ib" == "on" ]] && echo "ib" || echo "noib")
    RUN_NAME="${model}_${ib_tag}_$(date +%Y%m%d-%H%M%S)"
    echo "Submitting job for model=$model IB=$ib_tag"
    sbatch --export=MODEL=$model,IB_FLAG=$ib_tag,PROJECT=$PROJECT,RUN_NAME=$RUN_NAME \
           cifar.sbatch
  done
done

#!/usr/bin/env bash
# ------------------------------------------------------------------
# submit_mnist_jobs.sh
#
# Submits sbatch jobs for MNIST and Fashion-MNIST flow-matching sweeps.
# Variations:
#   • dataset  = mnist | fashion-mnist
#   • matcher  = cfm | ot | sb | target
#   • IB switch = on | off (--use_ib)
# ------------------------------------------------------------------

set -e  # abort on error

PROJECT="FlowMatchMNIST"                # W&B project name
DATASETS=("mnist" "fashion-mnist")
MATCHERS=("cfm" "ot" "sb" "target")
IB_FLAGS=("on" "off")                 # 'on' → pass --use_ib

timestamp=$(date +%Y%m%d-%H%M%S)

for ds in "${DATASETS[@]}"; do
  for matcher in "${MATCHERS[@]}"; do
    for ib in "${IB_FLAGS[@]}"; do

      ib_tag=$([[ "$ib" == "on" ]] && echo "ib" || echo "noib")
      RUN_NAME="${ds}_${matcher}_${ib_tag}_${timestamp}"

      echo "Submitting MNIST job: dataset=$ds matcher=$matcher ib=$ib_tag run=$RUN_NAME"

      sbatch \
        --export=DATASET=$ds,MATCHER=$matcher,IB_FLAG=$ib_tag,PROJECT=$PROJECT,RUN_NAME=$RUN_NAME \
        train_mnist.sbatch

    done
  done
done

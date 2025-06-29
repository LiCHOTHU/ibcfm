#!/usr/bin/env bash
# submit_mnist_jobs.sh
# Submits sbatch jobs for MNIST and Fashion-MNIST flow-matching sweeps,
# now passing a boolean --use_ib flag (True/False).

set -e

PROJECT="FlowMatchMNIST"
DATASETS=(mnist fashion-mnist)
MATCHERS=(cfm ot sb target)
IB_OPTIONS=(True)
timestamp=$(date +%Y%m%d-%H%M%S)

for ds in "${DATASETS[@]}"; do
  for matcher in "${MATCHERS[@]}"; do
    for use_ib in "${IB_OPTIONS[@]}"; do

      ib_tag=$([[ "$use_ib" == "True" ]] && echo "ib" || echo "noib")
      RUN_NAME="${ds}_${matcher}_${ib_tag}_negative_${timestamp}"

      echo "Submitting job: dataset=$ds matcher=$matcher use_ib=$use_ib run=$RUN_NAME"

      sbatch \
        --export=DATASET="$ds",MATCHER="$matcher",USE_IB="$use_ib",PROJECT="$PROJECT",RUN_NAME="$RUN_NAME" \
        mnist.sbatch

    done
  done
done

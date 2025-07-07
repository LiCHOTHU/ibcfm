# submit_cifar_jobs.sh
#!/usr/bin/env bash
# ------------------------------------------------------------------
# submit_cifar_jobs.sh
#
# Submits sbatch jobs for CIFAR-10 flow-matching over model and IB sweeps,
# now passing a boolean --use_ib flag (True/False).
# ------------------------------------------------------------------

set -e

PROJECT="CondFlowMatchCIFAR"
MODELS=(cfm ot sb target)
IB_OPTIONS=(True False)
timestamp=$(date +%Y%m%d-%H%M%S)

for model in "${MODELS[@]}"; do
  for use_ib in "${IB_OPTIONS[@]}"; do
e
    ib_tag=$([[ "$use_ib" == "True" ]] && echo "ib" || echo "noib")
    RUN_NAME="${model}_${ib_tag}_${timestamp}"

    echo "Submitting job for model=$model use_ib=$use_ib run=$RUN_NAME"

    sbatch \
      --export=MODEL="$model",USE_IB="$use_ib",PROJECT="$PROJECT",RUN_NAME="$RUN_NAME" \
      cond_cifar.sbatch

  done
done
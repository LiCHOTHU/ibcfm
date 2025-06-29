#!/usr/bin/env bash
#SBATCH -N 1                                 # Number of nodes
#SBATCH --gres=gpu:1                         # Number of GPUs
#SBATCH -p gpu-a100,gpu-h100,gpu-h200       # Partition(s)
#SBATCH --cpus-per-task=2                    # CPUs per task
#SBATCH --ntasks=1                           # Tasks
#SBATCH --time=8:00:00                       # Time limit
#SBATCH --mem-per-gpu=48G                    # Memory per GPU
#SBATCH --job-name=cifar_flow                # Job name
#SBATCH --error=/storage/home/hcoda1/8/lwang831/p-agarg35-0/logs/cifar_%j.err
#SBATCH --output=/storage/home/hcoda1/8/lwang831/p-agarg35-0/logs/cifar_%j.out
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35                # Account

# Temporary directory for this job
export TMPDIR=/storage/home/hcoda1/8/lwang831/p-agarg35-0/tmp/$SLURM_JOB_ID
mkdir -p $TMPDIR

echo "Job $SLURM_JOB_ID running on $(hostname), TMPDIR=$TMPDIR"

# Load environment
source ~/.bashrc
conda activate torchcfm

# Parameters passed via --export
# MODEL           : flow matcher ('otcfm','icfm','fm','si')
# IB_FLAG         : 'on' or 'off'
# PROJECT         : W&B project name
# RUN_NAME        : W&B run name

echo "Running MODEL=$MODEL, IB_FLAG=$IB_FLAG, RUN_NAME=$RUN_NAME"

# Prepare checkpoint directory
CKPT_DIR=/storage/home/hcoda1/8/lwang831/p-agarg35-0/workspace/ibcfm/checkpoints/${MODEL}_imagenet64_${IB_FLAG}_${SLURM_JOB_ID}
mkdir -p $CKPT_DIR

# W&B credentials
export WANDB_API_KEY="fe43daee24fa5113fd91167314d9448971f808ef"
export WANDB_PROJECT="$PROJECT"
export WANDB_RUN_NAME="$RUN_NAME"

# Decide IB flags
if [ "$IB_FLAG" = "on" ]; then
  IB_OPTS="--use_ib --ib_lambda 0.03 --ib_beta 5e-5"
else
  IB_OPTS=""
fi

# Navigate to repo root
cd /storage/home/hcoda1/8/lwang831/p-agarg35-0/workspace/ibcfm

# Invoke training
python scripts/train_imagenet64_flow.py \
  --model          "$MODEL" \
  --device         "$( [ -z "$CUDA_VISIBLE_DEVICES" ] && echo cpu || echo cuda )" \
  --wandb_project  "$PROJECT" \
  --wandb_run_name "$RUN_NAME" \
  $IB_OPTS

# Cleanup
rm -rf $TMPDIR
echo "Finished job $SLURM_JOB_ID"

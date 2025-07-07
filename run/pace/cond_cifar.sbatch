#!/usr/bin/env bash
#SBATCH -N 1                                 # Number of nodes
#SBATCH --gres=gpu:1                         # Number of GPUs
#SBATCH -p gpu-a100,gpu-h100,gpu-h200       # Partition(s)
#SBATCH --cpus-per-task=2                    # CPUs per task
#SBATCH --ntasks=1                           # Tasks
#SBATCH --time=8:00:00                       # Time limit
#SBATCH --mem-per-gpu=48G                    # Memory per GPU
#SBATCH --job-name=cifar_flow                # Job name
#SBATCH --error=/storage/home/hcoda1/8/lwang831/p-agarg35-0/logs/cond_cifar_%j.err
#SBATCH --output=/storage/home/hcoda1/8/lwang831/p-agarg35-0/logs/cond_cifar_%j.out
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35                # Account

# Set up TMPDIR
export TMPDIR=/storage/home/hcoda1/8/lwang831/p-agarg35-0/tmp/$SLURM_JOB_ID
mkdir -p $TMPDIR

echo "Job $SLURM_JOB_ID running on $(hostname), TMPDIR=$TMPDIR"

# Environment
source ~/.bashrc
conda activate torchcfm

# Params from sbatch --export
# MODEL, IB_FLAG (on/off), PROJECT, RUN_NAME

echo "Running model=$MODEL, IB=$IB_FLAG, run=$RUN_NAME"

# Prepare checkpoint directory
CKPT_DIR=/storage/home/hcoda1/8/lwang831/p-agarg35-0/workspace/ibcfm/checkpoints/${MODEL}_cifar10_${IB_FLAG}_${SLURM_JOB_ID}
mkdir -p $CKPT_DIR

# Export your W&B key if needed
export WANDB_API_KEY="fe43daee24fa5113fd91167314d9448971f808ef"

if [ "$USE_IB" = "True" ]; then
  IB_FLAG="--use_ib"
else
  IB_FLAG=""
fi

# W&B
echo "Using W&B project $PROJECT, run $RUN_NAME"
export WANDB_PROJECT=$PROJECT
export WANDB_RUN_NAME=$RUN_NAME

# Navigate to repo
dd=/storage/home/hcoda1/8/lwang831/p-agarg35-0/workspace/ibcfm
cd $dd

# Invoke training script
python scripts/train_cond_cifar10.py \
  --matcher "$MODEL" \
  --device "$( [ -z "$CUDA_VISIBLE_DEVICES" ] && echo "cpu" || echo "cuda" )" \
  --wandb_project "$PROJECT" \
  --wandb_run_name "$RUN_NAME" \
  $IB_FLAG

# Clean up
rm -rf $TMPDIR
echo "Finished job $SLURM_JOB_ID"

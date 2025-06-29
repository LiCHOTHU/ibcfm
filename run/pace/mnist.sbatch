#!/usr/bin/env bash
#SBATCH -N 1                                                  # Number of nodes
#SBATCH --gres=gpu:1                                          # Number of GPUs
#SBATCH -p gpu-a100,gpu-h100,gpu-h200                        # Partition(s)
#SBATCH --cpus-per-task=2                                     # CPUs per task
#SBATCH --ntasks=1                                            # Tasks
#SBATCH --time=8:00:00                                        # Time limit
#SBATCH --mem-per-gpu=48G                                     # Memory per GPU
#SBATCH --job-name=mnist_flow                                 # Job name
#SBATCH --error=/storage/home/hcoda1/8/lwang831/p-agarg35-0/logs/mnist_%j.err
#SBATCH --output=/storage/home/hcoda1/8/lwang831/p-agarg35-0/logs/mnist_%j.out
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35                                  # Account

# Temporary directory for this job
env TMPDIR=/storage/home/hcoda1/8/lwang831/p-agarg35-0/tmp/$SLURM_JOB_ID
mkdir -p $TMPDIR

echo "Running on $(hostname), TMPDIR=$TMPDIR"

# Load conda environment
source ~/.bashrc
conda activate torchcfm

# Navigate to project directory
cd /storage/home/hcoda1/8/lwang831/p-agarg35-0/workspace/ibcfm

# Export your W&B key if needed
export WANDB_API_KEY="fe43daee24fa5113fd91167314d9448971f808ef"

if [ "$USE_IB" = "True" ]; then
  IB_FLAG="--use_ib"
else
  IB_FLAG=""
fi

echo "Dataset=$DATASET, Matcher=$MATCHER, IB=$IB_FLAG, Run=$RUN_NAME"
python scripts/train_cond_mnist.py \
    --dataset "$DATASET" \
    --matcher "$MATCHER" \
    --device "$( [ -z "$CUDA_VISIBLE_DEVICES" ] && echo "cpu" || echo "cuda" )" \
    --wandb_project "$PROJECT" \
    --wandb_run_name "$RUN_NAME" \
    $IB_FLAG

# Cleanup temporary files
echo "Cleaning up TMPDIR $TMPDIR"
rm -rf $TMPDIR

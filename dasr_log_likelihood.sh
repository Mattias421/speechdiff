#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=slurm.out
#SBATCH -t 0-90:00:00

module load Anaconda3/5.3.0
# module load cuDNN/7.6.4.38-gcccuda-2019b

source activate speech-diff
python dasr_log_likelihood.py --config-name=config --config-dir=/fastdata/speech-diff-dys/speech-diff/beta_10/UAS_all/.hydra/ +eval.checkpoint=/fastdata/speech-diff-dys/speech-diff/beta_10/UAS_all/checkpoints/grad_500.pt
echo "done"

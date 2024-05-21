#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=1-00:00
#SBATCH --output=%N-%j.out
#SBATCH --account=def-ehyangit
module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

python train_nasty.py --trial 3 --SimulationFolder experiments/cifar100/SelfUnderminingKD/resnet50/O0004
python train_nasty.py --trial 3 --SimulationFolder experiments/cifar100/SelfUnderminingKD/resnet50/O0005
python train_nasty.py --trial 3 --SimulationFolder experiments/cifar100/SelfUnderminingKD/resnet50/O001

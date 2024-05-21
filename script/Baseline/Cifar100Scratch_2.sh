#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=2-00:00
#SBATCH --output=%N-%j.out
#SBATCH --account=def-ehyangit
module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

python train_scratch.py --trial 2 --SimulationFolder experiments/cifar100/cross-entropy/vgg19_bn
python train_scratch.py --trial 3 --SimulationFolder experiments/cifar100/label-smoothing/ls05/vgg11_bn
python train_scratch.py --trial 1 --SimulationFolder experiments/cifar100/cross-entropy/vgg19_bn
python train_scratch.py --trial 2 --SimulationFolder experiments/cifar100/label-smoothing/ls02/resnet18
python train_scratch.py --trial 3 --SimulationFolder experiments/cifar100/cross-entropy/mobilenet
python train_scratch.py --trial 1 --SimulationFolder experiments/cifar100/label-smoothing/ls03/mobilenet
python train_scratch.py --trial 2 --SimulationFolder experiments/cifar100/label-smoothing/ls03/shufflenetv2
python train_scratch.py --trial 1 --SimulationFolder experiments/cifar100/cross-entropy/wrn40_2
python train_scratch.py --trial 2 --SimulationFolder experiments/cifar100/label-smoothing/ls01/shufflenetv2
python train_scratch.py --trial 3 --SimulationFolder experiments/cifar100/label-smoothing/ls04/wrn40_1

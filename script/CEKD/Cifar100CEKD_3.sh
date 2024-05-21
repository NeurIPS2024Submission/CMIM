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

python train_kd.py --trial 2 --SimulationFolder experiments/cifar100/CEKD/resnet50/resnet18
python train_kd.py --trial 2 --SimulationFolder experiments/cifar100/CEKD/resnet50/vgg11_bn
python train_kd.py --trial 2 --SimulationFolder experiments/cifar100/CEKD/vgg19_bn/mobilenet
python train_kd.py --trial 2 --SimulationFolder experiments/cifar100/CEKD/vgg19_bn/vgg11_bn
python train_kd.py --trial 2 --SimulationFolder experiments/cifar100/CEKD/wrn40_2/shufflenetv2
python train_kd.py --trial 2 --SimulationFolder experiments/cifar100/CEKD/wrn40_2/wrn40-1

#!/bin/bash 
#SBATCH -J MGC_gx_gru_2
#SBATCH -t 5760:00
#SBATCH --gres=gpu:V100:1
#SBATCH -p dell
#SBATCH -c 10
#SBATCH -o log/new-experiments/pretrain-gx/MyGAN-cycle_gru_2.out
#SBATCH -e log/FinEvent_train-gx_gru_2.err

source /home/LAB/luogx/anaconda3/etc/profile.d/conda.sh
conda activate lgx
#python /home/LAB/luogx/lcy/viewModel/myGAN-cycle.py --dataset ppmi_622 --load True --generator origin --n_epochs 20 --to1_epochs 20
#python /home/LAB/luogx/lcy/viewModel/myGAN-cycle.py --dataset ppmi_622 --load True --generator cat --n_epochs 20 --to1_epochs 20
python /home/LAB/luogx/lcy/viewModel/myGAN-cycle.py --dataset ppmi_622 --load True --generator gru_2 --n_epochs 20 --to1_epochs 20
#python /home/LAB/luogx/lcy/viewModel/myGAN-cycle.py --dataset ppmi_622 --load True --generator mean --n_epochs 20 --to1_epochs 20
#python /home/LAB/luogx/lcy/viewModel/myGAN-cycle.py --dataset ppmi_622 --load True --generator gru --n_epochs 20 --to1_epochs 20

# python /home/LAB/luogx/lcy/viewModel/myGAN-cycle.py --dataset ppmi_622 --load True --generator cat --n_epochs 11 --to1_epochs 11
# python /home/LAB/luogx/lcy/viewModel/myGAN-cycle.py --dataset ppmi_622 --load True --generator cat --n_epochs 11 --to1_epochs 11
# python /home/LAB/luogx/lcy/viewModel/myGAN-cycle.py --dataset ppmi_622 --load True --generator cat --n_epochs 11 --to1_epochs 11
# python /home/LAB/luogx/lcy/viewModel/myGAN-cycle.py --dataset ppmi_622 --load True --generator cat --n_epochs 11 --to1_epochs 11
# python /home/LAB/luogx/lcy/viewModel/myGAN-cycle.py --dataset ppmi_622 --load True --generator cat --n_epochs 11 --to1_epochs 11
# python /home/LAB/luogx/lcy/viewModel/myGAN-cycle.py --dataset ppmi_622 --load True --generator cat --n_epochs 11 --to1_epochs 11
# python /home/LAB/luogx/lcy/viewModel/myGAN-cycle.py --dataset ppmi_622 --load True --generator cat --n_epochs 11 --to1_epochs 11
# python /home/LAB/luogx/lcy/viewModel/myGAN-cycle.py --dataset ppmi_622 --load True --generator cat --n_epochs 11 --to1_epochs 11
# python /home/LAB/luogx/lcy/viewModel/myGAN-cycle.py --dataset ppmi_622 --load True --generator cat --n_epochs 11 --to1_epochs 11
# python /home/LAB/luogx/lcy/viewModel/myGAN-cycle.py --dataset ppmi_622 --load True --generator cat --n_epochs 11 --to1_epochs 11
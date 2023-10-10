#!/bin/bash
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID
ml purge 
python main.py -i "/scratch/winston1214/talmo/train_img/5750_A2LEBJJDE000589_1604808871900_2_TH.jpg"  --output_path "/home/winston1214/" -tg "/scratch/winston1214/talmo/train_img/0617_A2LEBJJDE000815_1603859729654_3_TH.jpg" --diff_iter 100 --timestep_respacing 200 --skip_timesteps 80 --model_output_size 512 --init_mask /scratch/winston1214/talmo/reverse_ensemble_train/5750_A2LEBJJDE000589_1604808871900_2_TH.jpg --use_range_restart True --use_colormatch True --use_noise_aug_all True --iterations_num 1 --output_file Sebum_DIFFIT_M.png
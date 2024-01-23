#!/bin/bash
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID
ml purge 
python main.py -i $source  --output_path $output_path -tg $target --diff_iter 100 --timestep_respacing 200 --skip_timesteps 80 --model_output_size 256 --init_mask $mask --use_range_restart True --use_colormatch True --use_noise_aug_all True --iterations_num 1 --output_file $output_file_name

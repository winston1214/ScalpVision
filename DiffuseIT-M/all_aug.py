from optimization.image_editor import ImageEditor
from optimization.arguments import get_arguments

import os
import json
import glob
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse
import sys
def main(start_idx, end_idx):
    print(start_idx, end_idx)
    
    
    np.random.seed(42)


    img_src_path = './train_img/'
    img_dst_path = './ensemble_diffuse_output/'
    seg_src_path = './reverse_ensemble_train/'
    csv_path = './comb_csv'

    diffuse_csv_file = pd.read_csv('./diffuse_aug_list.csv').sort_values('src_img_name')

    src_label = []
    target_label = []
    for i,j in zip(diffuse_csv_file['src_label'], diffuse_csv_file['target_label']):
        src_string = str(i)
        if len(src_string) != 3:
            src_label.append(src_string.zfill(3))
        else:
            src_label.append(src_string)
        trg_string = str(j)
        if len(trg_string) != 3:
            target_label.append(trg_string.zfill(3))
        else:
            target_label.append(trg_string)
    diffuse_csv_file['src_label'] == src_label
    diffuse_csv_file['target_label'] == target_label

    diffuse_csv_file = diffuse_csv_file[start_idx:end_idx]

    result_df = pd.DataFrame(columns = ['src_img_name','target_img_name','src_label','target_label'])
    result_df['src_img_name'] = diffuse_csv_file['src_img_name']
    target_img_name_list = []
    src_label_list = []
    target_label_list = []
    for src_img_name, target_img_name,src_label, trg_label in tqdm(zip(diffuse_csv_file['src_img_name'], diffuse_csv_file['target_img_name'],diffuse_csv_file['src_label'], diffuse_csv_file['target_label'])):
        full_src_img_name = os.path.join(img_src_path, src_img_name)
        full_trg_img_name = os.path.join(img_src_path, target_img_name)
        single_img_name = full_src_img_name.split('.')[0]
        if src_label == trg_label:
            pass
        else:
            args.init_image = full_src_img_name
            args.target_image = full_trg_img_name
            args.init_mask = os.path.join(seg_src_path, src_img_name)
            args.model_output_size = 512
            args.diff_iter = 100
            args.timestep_respacing = "200"
            args.skip_timesteps = 80
            args.output_path = img_dst_path
            args.iterations_num = 1
            args.output_file = single_img_name.split('.')[0] +'_' + f'{src_label}_{trg_label}' + '.png'
            image_editor = ImageEditor(args)
            image_editor.edit_image_by_prompt()
        target_img_name_list.append(target_img_name)
        src_label_list.append(src_label)
        target_label_list.append(trg_label)
    result_df['src_label'] = src_label_list
    result_df['target_label'] = target_label_list
    result_df['target_img_name'] = target_img_name_list
    result_df = result_df.reset_index(drop=True)
    result_df.to_csv(f'ensemble_aug_result_{start_idx}_{end_idx}.csv',index=False)


if __name__ == '__main__':   
    args = get_arguments()
    main(args.start_idx, args.end_idx)

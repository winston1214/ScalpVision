import pandas as pd
import argparse
import numpy as np
import os

def merge_label(args):
    data = pd.read_csv(os.path.join(args.file_path,args.csv_file))
    micro_dand = np.maximum(data['value_1'].values, data['value_5'].values)

    ery_pus = np.maximum(data['value_3'].values ,data['value_4'].values)

    data = data.drop('value_6',axis=1)

    data.drop(['value_1','value_2','value_4','value_5'],axis=1,inplace = True)
    data['value_1'] = micro_dand
    data['value_2'] = ery_pus

    data.to_csv(os.path.join(args.file_path,'merge_'+args.csv_file))
    # data.to_csv(f'../scalp_aihub/merge_{args.csv_file}.csv',index=False)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file',type=str,required = True)    
    parser.add_argument('--file_path',type=str,required=True)
    args = parser.parse_args()

    merge_label(args)


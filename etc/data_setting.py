import argparse
import os
import glob
from tqdm import tqdm
def img_label_set(args):

    data_dir = os.path.join(args.data_dir,args.mode)
    label = sorted(glob.glob(os.path.join(data_dir,'label*')))
    img = sorted(glob.glob(os.path.join(data_dir,'img*')))

    if not len(label):
        print('Plz change the name')
        raise ValueError 
    if not len(img):
        print('Plz change the name')
        raise ValueError 

    os.mkdir(os.path.join(data_dir,'images'))
    os.mkdir(os.path.join(data_dir,'labels'))

    for im, lab in zip(img,label):
        os.system(f'unzip -qq {im} -d {os.path.join(data_dir,'images')}')
        os.system(f'unzip -qq {lab} -d {os.path.join(data_dir,'labels')}')
    
    if args.clean:
        for i in tqdm(sorted(glob.glob(data_dir,'*.zip'))):
            os.system(f'rm -rf {i}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,help='data directory')
    parser.add_argument('--mode',type=str,help='train or val')
    parser.add_argument('--clean',store_true = True,help='Remove zip file')
    args = parser.parse_args()
    
    img_label_set(args)
    print('Done!')
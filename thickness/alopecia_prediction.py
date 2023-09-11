import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import shutil
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import f1_score, accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# mode = 'train'
# category = 'sam'
CSV_PATH = '../scalp_aihub/hair_loss_pred_csv/'

def thickness_value(mode, category):
    train_csv = 'hair_loss_train.csv'
    test_csv = 'hair_loss_test.csv'
    train = sorted(glob.glob(f'../scalp_aihub/thickness/{category}/thickness_{mode}/*.npy'))
    df = pd.DataFrame(columns = ['img_name', 'iqr_median','iqr_mean','mean','median','area'])
    image_path = f'../scalp_aihub/{category}_result/{category}_{mode}'
    iqr_median = []
    iqr_mean = []
    median = []
    mean = []
    img_name_ls = []
    length = []
    area_ls = []
    for i in tqdm(train):
        thicknesses_sort = np.load(i)
        img_name = i.split('/')[-1].replace('npy','jpg')
        img = cv2.imread(os.path.join(image_path, img_name),0)
        area = np.sum(img==255)/(640*480)
        area_ls.append(area)
        length.append(len(thicknesses_sort))
        if len(thicknesses_sort) == 0:
            img_name_ls.append(img_name)
            median.append(0)
            mean.append(0)
            iqr_median.append(0)
            iqr_mean.append(0)
        else:
            q1 = np.quantile(thicknesses_sort, 0.25)
            q3 = np.quantile(thicknesses_sort, 0.75)
            iqr = q3-q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_array_median = thicknesses_sort[(thicknesses_sort >= lower_bound) & (thicknesses_sort <= upper_bound)]    

            img_name_ls.append(img_name)
            iqr_median.append(np.median(outlier_array_median))  
            iqr_mean.append(np.mean(outlier_array_median))
            mean.append(np.mean(thicknesses_sort))
            median.append(np.median(thicknesses_sort))

    df['img_name'] = img_name_ls
    df['median'] = median
    df['mean'] = mean
    df['iqr_mean'] = iqr_mean
    df['iqr_median'] = iqr_median
    df['length'] = length
    df['area'] = area_ls


    tr = pd.read_csv(f'../scalp_aihub/{mode}_label.csv')
    tmp = pd.merge(tr,df,on='img_name')[['img_name','iqr_median','iqr_mean','mean','median','length','value_6','area']]
    mer = pd.read_csv(f'../scalp_aihub/merge_{mode}_label.csv')
    final = pd.merge(tmp, mer, on='img_name')
    if mode == 'train':
        final.to_csv(os.path.join(CSV_PATH, category, train_csv), index=False)
    else:
        final.to_csv(os.path.join(CSV_PATH, category, test_csv), index=False)
    print(f'{category}-{mode} Done!')

def alopeica_predict(train_csv, test_csv):
    train = os.path.join(CSV_PATH, category, train_csv)
    test = os.path.join(CSV_PATH, category, test_csv)

    train_x = train.drop(['value_6','img_name'], axis = 1)
    train_y = train['value_6'].values.ravel()

    test_x = test.drop(['value_6','img_name'], axis = 1)
    test_y = test['value_6'].values.ravel()

    rf = RandomForestClassifier(random_state=42)
    rf.fit(train_x, train_y)
    pred = rf.predict(test_x)

    print(f1_score(pred, test_y, average='weighted'))
    print(accuracy_score(pred, test_y))


if __name__ == '__main__':

    category = 'ensemble'
    mode = 'train'

    train_csv = 'hair_loss_train.csv'
    test_csv = 'hair_loss_test.csv'

    thickness_value(mode,category)
    alopeica_predict(train_csv, test_csv)
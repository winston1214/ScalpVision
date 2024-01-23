import torch
import torchvision
import torch.nn as nn
import torch.utils.data as D
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
import numpy as np

class ScalpDataset2(D.Dataset):
    def __init__(self, path, data, mode,transform=None):
        self.path = path
        self.data = data
        self.transform = transform
        self.mode = mode
        
        if self.mode == 'train' or self.mode == 'val' or self.mode == 'test':
            df = pd.read_csv(os.path.join(self.path,f'merge_{self.mode}_label.csv'))
            self.img_path = os.path.join(self.path,self.mode + '_img')
        elif self.mode == 'diffuse_new':
            df = pd.read_csv(os.path.join(self.path,'new_merge_train_label.csv'))
            self.img_path = os.path.join(self.path,'new_aug_train_img')
        elif self.mode == 'diffit_new':
            df = pd.read_csv(os.path.join(self.path,'new_diffit_merge_train_label.csv'))
            self.img_path = os.path.join(self.path,'new_diffit_train_img')
        elif self.mode == 'agg_new':
            df = pd.read_csv(os.path.join(self.path,'new_agg_merge_train_label.csv'))
            self.img_path = os.path.join(self.path,'new_agg_train_img')
        
        df = df.sort_values('img_name').reset_index(drop=True) # re-sort
        col = [f'class_{i}' for i in range(1,4)]
        self.label = np.where(df[col].values>0,1,0) # disease label

        ohe = OneHotEncoder()
        ohe_array = ohe.fit_transform(df[col].astype(int)).toarray()
        self.dand_label = ohe_array[:,:4]
        self.sebum_label = ohe_array[:,4:8]
        self.ery_label = ohe_array[:,8:]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_path,self.data[idx])).convert('RGB')
        label = self.label[idx] 
        dand_label = self.dand_label[idx]
        seb_label = self.sebum_label[idx]
        ery_label = self.ery_label[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, dand_label, seb_label, ery_label
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as D
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class ScalpDataset(D.Dataset):
    def __init__(self, path, data, mode,transform=None):
        self.path = path
        self.data = data
        self.transform = transform
        self.mode = mode
        df = pd.read_csv(os.path.join(self.path,f'{self.mode}_label.csv'))
        df = df.sort_values('img_name').reset_index(drop=True) # re-sort
        col = [f'value_{i}' for i in range(1,7)]
        self.label = np.where(df[col].values>0,1,0)
        # self.label = torch.concat([F.one_hot(torch.tensor(df[c].values)) for c in col],1)
        self.img_path = os.path.join(self.path,self.mode,'images')
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_path,self.data[idx]))
        label = self.label[idx] 
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
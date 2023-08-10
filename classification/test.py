import os
import cv2
from PIL import Image
from tqdm import tqdm
import timm
import glob
import argparse
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as D
import torch.optim as optim
import torchvision.transforms as T

from dataset import ScalpDataset
from rand_seed import random_seed
import warnings

from sklearn.metrics import multilabel_confusion_matrix
warnings.filterwarnings('ignore')
random_seed(42)

def test(args):
    device = 'cuda:0'
    DATA_DIR = args.data_dir
    model = timm.create_model(args.model, pretrained=True, num_classes= 12)
    model.load_state_dict(torch.load(args.ckpt))
    model.to(device)

    img_path = os.path.join(DATA_DIR,'inpaint_test')
    img_data = sorted(os.listdir(img_path))

    transformation = T.Compose([
        T.ToTensor(),
        T.Resize((224,224)),
        T.Normalize([49.307, 51.166, 48.442],[74.02 , 76.705, 73.786])
    ])

    test_dataset = ScalpDataset(DATA_DIR, img_data, 'test', transformation)
    test_dataloader = D.DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, drop_last=False)

    prediction_list = np.array([])
    model.eval()
    test_acc_list = []
    label_list = np.array([])
    with torch.no_grad():
        for img, label in tqdm(test_dataloader):
            images = img.type(torch.FloatTensor).to(device)
            labels = label.type(torch.FloatTensor).to(device)
            prob = model(images)
            probs = torch.sigmoid(prob)
            preds = probs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            label_list = np.append(label_list, labels)
            preds = preds >= 0.5
            prediction_list = np.append(prediction_list,preds)
            batch_acc = f1_score(labels,preds,average='macro')
            test_acc_list.append(batch_acc)
        test_acc = np.mean(test_acc_list)

    pred = prediction_list.reshape(-1,12)
    label = label_list.reshape(-1,12)
    label_df = pd.DataFrame(label)
    label_df.to_csv('label.csv',index=False)
    
    # conf_matrix = multilabel_confusion_matrix(pred,gt)
    # print(conf_matrix)
    print(f"F1 score : {test_acc:.3f}")
    df = pd.DataFrame(pred)
    df.to_csv('results.csv',index=False)
    
    print(multilabel_confusion_matrix(df.values, label_df.values))
    





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',type=str,default = 'ckpt/best_29.pth')
    parser.add_argument('--data_dir',type=str,default = '../scalp_aihub')
    parser.add_argument('--batch_size',type=int,default = 32)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()
    test(args)


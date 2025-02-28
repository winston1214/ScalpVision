import os
import cv2
from PIL import Image
from tqdm import tqdm
import argparse
import numpy as np
from sklearn.metrics import f1_score

import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as D
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F

from dataset import ScalpDataset, ScalpDataset2
from rand_seed import random_seed
from scalp_model import ScalpModel, ScalpModel_4head
import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings('ignore')
random_seed(42)
def transform_to_one_hot(arr):
    result_size = arr.shape[1] * 4

    result = np.zeros((arr.shape[0], result_size), dtype=int)
    
    for i in range(arr.shape[0]):
        for j, val in enumerate(arr[i]):
            result[i, j * 4 + val] = 1
    
    return result


def test(args):
    device = 'cuda'
    DATA_DIR = args.data_dir
    model = ScalpModel(args.model).to(device)
    model.load_state_dict(torch.load(args.ckpt))
    model.to(device)

    img_path = os.path.join(DATA_DIR,'test_img')
    img_data = sorted(os.listdir(img_path))
    print(len(img_data))
    sz = 224

    transformation = T.Compose([
        T.ToTensor(),      
        T.Resize((sz,sz)),
        T.Normalize([0.582448, 0.6022764, 0.57366776],[0.14222942, 0.15106438, 0.16288713]),
    ])

    test_dataset = ScalpDataset(DATA_DIR, img_data, 'test', transformation)
    test_dataloader = D.DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, drop_last=False)

    test_dis_label = np.array([])
    test_sev_label = np.array([])
    test_dis_preds = np.array([])
    test_sev_preds = np.array([])
    test_final_label = np.array([])
    test_final_preds = np.array([])

    check_preds = np.array([])
    model.eval()

    with torch.no_grad():
        for img,label, label2 in tqdm(test_dataloader):
            images = img.type(torch.FloatTensor).to(device)
            labels = label.type(torch.FloatTensor).to(device)
            labels2 = label2.type(torch.FloatTensor).to(device)
            
            disease_output, severity_output = model(images)
            # loss, presence, intensity = joint_loss_function(probs, labels,pos_weight)
            disease_preds = disease_output >= 0.5
            severity_preds = severity_output >= 0.5
            
            disease_preds  = disease_preds.cpu().detach().numpy()
            severity_preds = severity_preds.cpu().detach().numpy()
            severity_output = severity_output.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            labels2 = labels2.cpu().detach().numpy()

            sev = severity_preds * severity_output

            max_indices = np.concatenate([np.argmax(sev[:,0:3],axis=1).reshape(-1,1), np.argmax(sev[:,3:6],axis=1).reshape(-1,1),np.argmax(sev[:,6:9],axis=1).reshape(-1,1)],axis=1)
            final_preds = disease_preds * (max_indices+1)
            ohe_final_preds = transform_to_one_hot(final_preds)

            final_label = np.insert(labels2, [0,3,6], 0,  axis=1)

            row = np.where(labels == 0)[0]
            col = np.where(labels == 0)[1]
            for r,c in zip(row,col):
                final_label[r][c*4] = 1

            test_dis_label = np.append(test_dis_label, labels)
            test_sev_label = np.append(test_sev_label, labels2)
            test_dis_preds = np.append(test_dis_preds, disease_preds)
            test_sev_preds = np.append(test_sev_preds, severity_preds)
            test_final_label = np.append(test_final_label, final_label)
            test_final_preds = np.append(test_final_preds, ohe_final_preds)

    test_dis_f1 = f1_score(test_dis_preds, test_dis_label, average='macro')
    test_sev_f1 = f1_score(test_sev_preds, test_sev_label, average='macro')
    test_final_f1 = f1_score(test_final_preds, test_final_label, average='macro')

    print(test_final_f1)
    final_pred = test_final_preds.reshape(-1,12)
    final_label = test_final_label.reshape(-1,12)


    test_dis_preds = test_dis_preds.reshape(-1,3)
    test_sev_preds = test_sev_preds.reshape(-1,9)
    check = check_preds.reshape(-1,3)
    

    test_acc = f1_score(final_pred,final_label, average='macro')
    
    print(f"F1 score : {test_acc:.3f}")
    df = pd.DataFrame(final_pred)
    df.to_csv(args.save_csv,index=False)


def test_4head(args):
    device = 'cuda'
    DATA_DIR = args.data_dir
    model = ScalpModel_4head(args.model).to(device)
    model.load_state_dict(torch.load(args.ckpt))
    model.to(device)

    img_path = os.path.join(DATA_DIR,'test_img')
    img_data = sorted(os.listdir(img_path))
    print(len(img_data))
    sz = 224

    transformation = T.Compose([
        T.ToTensor(),      
        T.Resize((sz,sz)),
        T.Normalize([0.582448, 0.6022764, 0.57366776],[0.14222942, 0.15106438, 0.16288713]),
    ])

    test_dataset = ScalpDataset2(DATA_DIR, img_data, 'test', transformation)
    test_dataloader = D.DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, drop_last=False)



    test_final_label = np.array([])
    test_final_preds = np.array([])

    model.eval()

    with torch.no_grad():
        for img,label, dand, seb, ery in tqdm(test_dataloader):
            images = img.type(torch.FloatTensor).to(device)
            labels = label.type(torch.FloatTensor).to(device)
            dand = dand.type(torch.FloatTensor).to(device)
            seb = seb.type(torch.FloatTensor).to(device)
            ery = ery.type(torch.FloatTensor).to(device)
            
            disease_output, dand_output, seb_output, ery_output = model(images)

            disease_preds = disease_output >= 0.5

            disease_preds  = disease_preds.cpu().detach().numpy()
            dand_output = dand_output.cpu().detach().numpy()
            seb_output = seb_output.cpu().detach().numpy()
            ery_output = ery_output.cpu().detach().numpy()

            labels = labels.cpu().detach().numpy()
            dand = dand.cpu().detach().numpy()
            seb = seb.cpu().detach().numpy()
            ery = ery.cpu().detach().numpy()



            dand_pred = np.argmax(dand_output,axis=1)
            seb_pred = np.argmax(seb_output,axis=1)
            ery_pred = np.argmax(ery_output,axis=1)

            dand_ohe_pred,seb_ohe_pred,ery_ohe_pred = np.zeros(dand_output.shape), np.zeros(seb_output.shape), np.zeros(ery_output.shape)

            dand_ohe_pred[np.arange(len(dand_pred)), dand_pred] = 1
            seb_ohe_pred[np.arange(len(dand_pred)), seb_pred] = 1
            ery_ohe_pred[np.arange(len(dand_pred)), ery_pred] = 1

            
            ohe_final_preds = np.concatenate([dand_ohe_pred,seb_ohe_pred,ery_ohe_pred],axis=1)    

            final_label = np.concatenate([dand,seb,ery],axis=1)    
            

            test_final_label = np.append(test_final_label, final_label)
            test_final_preds = np.append(test_final_preds, ohe_final_preds)


    final_preds = test_final_preds.reshape(-1,12)
    final_label = test_final_label.reshape(-1,12)
    
    test_acc = f1_score(final_preds,final_label, average='macro')
    
    print(f"F1 score : {test_acc:.3f}")
    df = pd.DataFrame(final_preds)
    df.to_csv(args.save_csv,index=False)

def class_predict(args):
    pred = pd.read_csv(args.save_csv).values
    test = pd.read_csv(os.path.join(args.data_dir, 'merge_test_label.csv'))
    test_label = test[['class_1','class_2','class_3']].values
    test_label = F.one_hot(torch.tensor(test_label)).reshape(-1,12)
    test_label = test_label.numpy()
    f1 = f1_score(pred, test_label, average=None)
    print(np.round(f1,4))
    print(np.mean(f1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',type=str,default = 'scalp_classification/xception_baseline/best_2.pth')
    parser.add_argument('--data_dir',type=str)
    parser.add_argument('--batch_size',type=int,default = 128)
    parser.add_argument('--model', type=str)
    parser.add_argument('--save_csv', type=str)
    parser.add_argument('--head',type=int,default = 2)
    args = parser.parse_args()
    print(args)
    if args.head == 2:
        test(args)
    elif args.head == 4:
        test_4head(args)
    class_predict(args)



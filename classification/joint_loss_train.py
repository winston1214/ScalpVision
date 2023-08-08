import os
import cv2
from PIL import Image
from tqdm import tqdm
import timm
import glob
import argparse
import numpy as np
import warnings
from sklearn.metrics import f1_score
import shutil

import torch
import torch.nn as nn
import torch.utils.data as D
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
from dataset import ScalpDataset
from model import ViT_MCMC
# from loss import FocalLoss
from rand_seed import random_seed


warnings.filterwarnings('ignore')
random_seed(42)

weight=1

def gauss_noise_tensor(img):
    assert isinstance(img, torch.Tensor)
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)
    
    sigma = 25.0
    
    out = img + sigma * torch.randn_like(img)
    
    if out.dtype != dtype:
        out = out.to(dtype)
        
    return out

def joint_loss_function(preds, labels,pos_weight):
    global weight
    # Separate the presence and intensity labels from the combined one-hot label
    presence_labels = labels[:, ::4]
    intensity_labels = labels[:,1::4]

    # Split the predictions into presence and intensity predictions
    presence_preds = preds[:, ::4]
    intensity_preds = preds[:,1::4]

    # Compute the binary cross-entropy loss for presence prediction
    presence_loss = F.binary_cross_entropy_with_logits(presence_preds, presence_labels,pos_weight=pos_weight)

    # Compute the mean squared error loss for intensity prediction
    intensity_loss = F.mse_loss(torch.sigmoid(intensity_preds), intensity_labels)

    # Combine the losses with appropriate weights (you can experiment with these weights)
    total_loss = presence_loss + weight * intensity_loss

    return total_loss, presence_loss, intensity_loss

def training(args):
    device = args.device
    DATA_DIR = args.data_dir
    LABEL_DIR = '/home/jerry0110/scalp_diagnosis-2/data/'
    
    if os.path.exists(args.save_dir):
        # remove_cmd = input("remove? y or n : ")
        remove_cmd ='y'
        if remove_cmd == 'y' or remove_cmd == 'yes':
            shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir,exist_ok = True)
    
    train_img_path = os.path.join(DATA_DIR,'train_img') #'Training',
    val_img_path = os.path.join(DATA_DIR,'val_img') #'Validation',

    train_img_data = sorted(os.listdir(train_img_path))
    valid_img_data = sorted(os.listdir(val_img_path))

    sz=224
    if '448' in args.model:
        sz=448
    elif '384' in args.model:
        sz=384

    transformation = T.Compose([
        # T.AugMix(),
        T.ToTensor(),
        T.Resize((sz,sz)),
        T.Normalize([49.307, 51.166, 48.442],[74.02 , 76.705, 73.786]),
        # gauss_noise_tensor,
        
    ])

    train_dataset = ScalpDataset(DATA_DIR, LABEL_DIR,train_img_data,'train',transformation)
    valid_dataset = ScalpDataset(DATA_DIR, LABEL_DIR,valid_img_data,'val',transformation)


    model = timm.create_model(args.model,pretrained=True, num_classes=12).to(device)

    BATCH_SIZE = args.batch_size

    train_dataloader = D.DataLoader(train_dataset,batch_size = BATCH_SIZE, shuffle=True, drop_last = False, num_workers=8) 
    valid_dataloader = D.DataLoader(valid_dataset,batch_size = BATCH_SIZE, shuffle=False,drop_last = False, num_workers=8)#

    learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(),lr = learning_rate, weight_decay=1e-3)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)


    # label smooth
    num_positives = torch.sum(torch.tensor(train_dataset.label[:, ::4]),dim=0)
    num_negatives = len(train_dataset.label) - num_positives
    pos_weight  = (num_negatives / num_positives).to(device)
    print(pos_weight)

    # if args.loss == 'bce':
    #     criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight).to(device)
    
    EPOCH = args.epoch
    total_step = len(train_dataloader)
    best_val_acc = 0.0
    

    print(len(train_dataset.label))


    for e in range(EPOCH):
        running_loss = 0.0
        running_presence = 0.0
        running_intensity = 0.0
        
        train_acc_list = []
        model.train()
        for idx,(img,label) in enumerate(tqdm(train_dataloader)):
            images = img.type(torch.FloatTensor).to(device)
            labels = label.type(torch.FloatTensor).to(device)
            
            probs = model(images)
            loss, presence, intensity = joint_loss_function(probs, labels,pos_weight)
            probs = torch.sigmoid(probs)
            preds = probs >= 0.5
            
            running_presence+= presence.item()
            running_intensity+= intensity.item()

            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            preds  = preds.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()

            batch_acc = f1_score(labels,preds,average='macro')

            train_acc_list.append(batch_acc)
        train_acc = np.mean(train_acc_list)
        epoch_loss = running_loss/total_step
        
        epoch_presence = running_presence/total_step
        epoch_intensity = running_intensity/total_step

        print(f'Epoch : [{e+1}/{EPOCH}], loss : {epoch_loss:.3f}, F1 : {train_acc:.3f}')
        print(f'Epoch : [{e+1}/{EPOCH}], presence : {epoch_presence:.3f}, intensity : {epoch_intensity:.3f}')

        model.eval()
        valid_acc_list = []
        scheduler.step()

        with torch.no_grad():
            for img,label in tqdm(valid_dataloader):
                images = img.type(torch.FloatTensor).to(device)
                labels = label.type(torch.FloatTensor).to(device)
                probs = model(images)
                probs = torch.sigmoid(probs)
                preds = probs >= 0.5
                preds  = preds.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()

                batch_acc = f1_score(labels,preds,average='macro')
                valid_acc_list.append(batch_acc)

        valid_acc = np.mean(valid_acc_list)
        print(f'Valid F1 : {valid_acc:.3f}')

        if valid_acc > best_val_acc :
            try:
                best_val_acc = valid_acc
                before_best = glob.glob(f'{args.save_dir}/best_*')[0]
                os.remove(before_best)
            except:
                pass
            torch.save(model.state_dict(),f'{args.save_dir}/best_{e}.pth')
        torch.save(model.state_dict(),f'{args.save_dir}/epoch_{e}.pth')
    print(best_val_acc)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,default = '../scalp_aihub')
    parser.add_argument('--batch_size',type=int,default = 16)
    parser.add_argument('--device',type=str,default='cuda')
    parser.add_argument('--epoch',type=int,default = 50)
    parser.add_argument('--weight',type=float,default = 1)
    parser.add_argument('--save_dir',type=str)
    parser.add_argument('--loss',type=str)
    parser.add_argument('--model',type=str)
    args = parser.parse_args()
    weight=args.weight
    print(f"joint loss ration = presence : intensity = 1 : ",weight)
    training(args)




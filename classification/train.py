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

import torch
import torch.nn as nn
import torch.utils.data as D
import torch.optim as optim
import torchvision.transforms as T

from dataset import ScalpDataset
from model import ViT_MCMC
from loss import FocalLoss
from rand_seed import random_seed


warnings.filterwarnings('ignore')
random_seed(42)

def training(args):
    device = args.device
    DATA_DIR = args.data_dir
    train_img_path = os.path.join(DATA_DIR,'train','images')
    val_img_path = os.path.join(DATA_DIR,'val','images')

    train_img_data = sorted(os.listdir(train_img_path))
    valid_img_data = sorted(os.listdir(val_img_path))

    transformation = T.Compose([
        T.ToTensor(),
        T.Resize((224,224)),
        T.Normalize([49.307, 51.166, 48.442],[74.02 , 76.705, 73.786])
    ])

    train_dataset = ScalpDataset(DATA_DIR,train_img_data,'train',transformation)
    valid_dataset = ScalpDataset(DATA_DIR,valid_img_data,'val',transformation)


    vit_model = timm.create_model('resnet50',pretrained=True)
    num_classes = 6
    model = ViT_MCMC(vit_model,num_classes).to(device)

    BATCH_SIZE = args.batch_size

    train_dataloader = D.DataLoader(train_dataset,batch_size = BATCH_SIZE, shuffle=True,drop_last = True,num_workers=8)
    valid_dataloader = D.DataLoader(valid_dataset,batch_size = BATCH_SIZE, shuffle=False,drop_last = False,num_workers = 8)

    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(),lr = learning_rate, weight_decay=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)


    # label smooth
    num_positives = torch.sum(torch.tensor(train_dataset.label),dim=0)
    num_negatives = len(train_dataset.label) - num_positives
    pos_weight  = num_negatives / num_positives
    print(pos_weight)

    criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight).to(device)
    # criterion = FocalLoss().to(device)
    # criterion = nn.MSELoss().to(device)


    EPOCH = args.epoch
    total_step = len(train_dataloader)
    best_val_acc = 0.0
    for e in range(EPOCH):
        running_loss = 0.0
        train_acc_list = []
        model.train()
        for idx,(img,label) in enumerate(tqdm(train_dataloader)):
            images = img.type(torch.FloatTensor).to(device)
            labels = label.type(torch.FloatTensor).to(device)
            optimizer.zero_grad()

            probs = model(images)
            loss = criterion(probs, labels)
            probs = torch.sigmoid(probs)
            preds = probs >= 0.5

            running_loss += loss.item()
            loss.backward()
            optimizer.step()

            preds  = preds.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()

            batch_acc = f1_score(labels,preds,average='macro')

            train_acc_list.append(batch_acc)
        train_acc = np.mean(train_acc_list)
        epoch_loss = running_loss/total_step

        print(f'Epoch : [{e+1}/{EPOCH}], loss : {epoch_loss:.3f}, F1 : {train_acc:.3f}')

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
                before_best = glob.glob(os.path.join(args.ckpt_dir,'best*'))[0]
                # before_best = glob.glob(f'{args.ckpt_dir}/best_*')[0]
                os.remove(before_best)
            except:
                pass
            torch.save(model.state_dict(),os.path.join(args.ckpt_dir,f'best_{e}.pth'))
            # torch.save(model.state_dict(),f'{args.ckpt_dir}/best_{e}.pth')
        torch.save(model.state_dict(),os.path.join(args.ckpt_dir,f'epoch_{e}.pth'))
        torch.save(model.state_dict(),f'{args.ckpt_dir}/epoch_{e}.pth')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,default = '../scalp_aihub')
    parser.add_argument('--batch_size',type=int,default = 16)
    parser.add_argument('--device',type=str,default='cuda:0')
    parser.add_argument('--epoch',type=int,default = 50)
    parser.add_argument('--save_dir',type=str,default = 'ckpt/')
    args = parser.parse_args()
    training(args)





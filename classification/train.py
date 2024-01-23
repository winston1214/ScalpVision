import os
from tqdm import tqdm

import glob
import argparse
import numpy as np
import warnings
from sklearn.metrics import f1_score, accuracy_score
import shutil

import torch
import torch.nn as nn
import torch.utils.data as D
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
from dataset import ScalpDataset2

from scalp_model import ScalpModel_4head
from rand_seed import random_seed


warnings.filterwarnings('ignore')
random_seed(42)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def transform_to_one_hot(arr):
    result_size = arr.shape[1] * 4

    result = np.zeros((arr.shape[0], result_size), dtype=int)
    
    for i in range(arr.shape[0]):
        for j, val in enumerate(arr[i]):
            result[i, j * 4 + val] = 1
    
    return result

def transform_forhead_to_one_hot(dis, onehot):
    assert len(dis[0]) == 3 and len(onehot[0]) == 12
    for i in range(dis.shape[0]):
        for j, val in enumerate(dis[i]):
            if val==0:
                onehot[i,j*4:j*4+4] = [1,0,0,0]
    return onehot



def training_4head(args):
    device = args.device
    DATA_DIR = args.data_dir
    
    if os.path.exists(args.save_dir):
        # remove_cmd = input("remove? y or n : ")
        remove_cmd ='y'
        if remove_cmd == 'y' or remove_cmd == 'yes':
            shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir,exist_ok = True)
    
    if args.mode == 'diffuse':
        train_img_path = os.path.join(DATA_DIR,"aug_train_img")
    elif args.mode == 'diffit':
        train_img_path = os.path.join(DATA_DIR,"diffit_aug_train_img") 
    elif args.mode == 'agg':
        train_img_path = os.path.join(DATA_DIR,"agg_aug_train_img")
    elif args.mode == 'diffit_new':
        train_img_path = os.path.join(DATA_DIR,"new_diffit_train_img") 
    elif args.mode == 'agg_new':
        train_img_path = os.path.join(DATA_DIR,"new_agg_train_img") 
    else: # baseline
        train_img_path = os.path.join(DATA_DIR,"train_img")
    val_img_path = os.path.join(DATA_DIR,"val_img")

    train_img_data = sorted(os.listdir(train_img_path))
    valid_img_data = sorted(os.listdir(val_img_path))

    sz=224
    
    transformation = T.Compose([
        T.ToTensor(),
        T.Resize((sz,sz)),
        T.Normalize([0.5722814, 0.5845881, 0.5582471], [0.14020121, 0.14801143, 0.15981224]),    
    ])

    
    if 'diff' in args.mode or 'agg' in args.mode:
        train_dataset = ScalpDataset2(DATA_DIR, train_img_data, f'{args.mode}', transformation)
    else:
        train_dataset = ScalpDataset2(DATA_DIR, train_img_data, 'train', transformation)
    valid_dataset = ScalpDataset2(DATA_DIR, valid_img_data, 'val', transformation)



    model = ScalpModel_4head(args.model).to(device)

    BATCH_SIZE = args.batch_size

    train_dataloader = D.DataLoader(train_dataset,batch_size = BATCH_SIZE, shuffle=True, drop_last = False, num_workers=8) 
    valid_dataloader = D.DataLoader(valid_dataset,batch_size = BATCH_SIZE, shuffle=False,drop_last = False, num_workers=8)

    learning_rate = 1e-4 # 1e-4
    # optimizer = optim.Adam(model.parameters(),lr = learning_rate, weight_decay=1e-3)
    optimizer = optim.AdamW(model.parameters(),lr = learning_rate)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=len(train_dataloader)//2, T_mult=2, eta_min=1e-5)


    disease_criterion = nn.BCEWithLogitsLoss().to(device)
    
    dand_criterion = nn.CrossEntropyLoss().to(device)
    sebum_criterion = nn.CrossEntropyLoss().to(device)
    ery_criterion = nn.CrossEntropyLoss().to(device)

    # criterion = nn.BCEWithLogitsLoss().to(device)
    
    EPOCH = args.epoch
    total_step = len(train_dataloader)
    best_val_final_acc = 0.0
    # wandb.config = {
    #     "model": args.model,
    #     "learning_rate": 1e-4,
    #     "epochs": args.epoch,
    #     "batch_size": args.batch_size
    # }

    for e in range(EPOCH):
        running_loss = 0.0
        
        model.train()
        
        train_dis_label = np.array([])
        train_dis_preds = np.array([])
        train_dand_preds = np.array([])
        train_seb_preds = np.array([])
        train_ery_preds = np.array([])
        
        train_dand_label = np.array([])
        train_seb_label = np.array([])
        train_ery_label = np.array([])

        train_final_label = np.array([])
        train_final_preds = np.array([])
        
        for img,label, dand, seb, ery in tqdm(train_dataloader):
            images = img.type(torch.FloatTensor).to(device)
            labels = label.type(torch.FloatTensor).to(device)
            dand = dand.type(torch.FloatTensor).to(device)
            seb = seb.type(torch.FloatTensor).to(device)
            ery = ery.type(torch.FloatTensor).to(device)
            
            disease_output, dand_output, seb_output, ery_output = model(images)

            loss1 = disease_criterion(disease_output, labels)
            dand_loss = dand_criterion(dand_output, dand)
            seb_loss = sebum_criterion(seb_output, seb)
            ery_loss = ery_criterion(ery_output, ery)
            
            loss = loss1 + dand_loss + seb_loss + ery_loss
            
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

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

            ohe_preds = np.concatenate([dand_ohe_pred,seb_ohe_pred,ery_ohe_pred],axis=1)
            ohe_final_preds = transform_forhead_to_one_hot(disease_preds, ohe_preds)
            # ohe_final_preds = np.concatenate([dand_ohe_pred,seb_ohe_pred,ery_ohe_pred],axis=1)    

            final_label = np.concatenate([dand,seb,ery],axis=1)        
            
            


            train_dis_label = np.append(train_dis_label, labels)
            train_dis_preds = np.append(train_dis_preds, disease_preds)

            train_dand_label = np.append(train_dand_label, dand)
            train_dand_preds = np.append(train_dand_preds, dand_ohe_pred)
            train_seb_label = np.append(train_seb_label, seb)
            train_seb_preds = np.append(train_seb_preds, seb_ohe_pred)
            train_ery_label = np.append(train_ery_label, ery)
            train_ery_preds = np.append(train_ery_preds, ery_ohe_pred)


            train_final_label = np.append(train_final_label, final_label)
            train_final_preds = np.append(train_final_preds, ohe_final_preds)

        train_dis_preds = train_dis_preds.reshape(-1,3)
        train_dis_label = train_dis_label.reshape(-1,3)
        
        train_dand_preds = train_dand_preds.reshape(-1,4)
        train_dand_label = train_dand_label.reshape(-1,4)
        train_seb_preds = train_seb_preds.reshape(-1,4)
        train_seb_label = train_seb_label.reshape(-1,4)
        train_ery_preds = train_ery_preds.reshape(-1,4)
        train_ery_label = train_ery_label.reshape(-1,4)

        train_final_preds = train_final_preds.reshape(-1,12)
        train_final_label = train_final_label.reshape(-1,12)

            
        train_dis_f1 = f1_score(train_dis_preds, train_dis_label, average='macro')
        train_dand_acc = accuracy_score(train_dand_preds, train_dand_label)
        train_seb_acc = accuracy_score(train_seb_preds, train_seb_label)
        train_ery_acc = accuracy_score(train_ery_preds, train_ery_label)
        train_final_f1 = f1_score(train_final_preds, train_final_label, average='macro')
        epoch_loss = running_loss/total_step
        # wandb.log({"training f1-score" : train_acc, "training epoch_loss":epoch_loss}, step=e)

        print(f'Epoch : [{e+1}/{EPOCH}], loss : {epoch_loss:.3f}, \ndis_F1 : {train_dis_f1:.3f}, dand_acc : {train_dand_acc:.3f}, seb_acc : {train_seb_acc:.3f}, ery_acc : {train_ery_acc:.3f}\n final_F1 : {train_final_f1:.3f}')
        # print(f'Epoch : [{e+1}/{EPOCH}], presence : {epoch_presence:.3f}, intensity : {epoch_intensity:.3f}')
        
        scheduler.step()

        model.eval()

        valid_dis_label = np.array([])
        valid_dis_preds = np.array([])
        valid_dand_preds = np.array([])
        valid_seb_preds = np.array([])
        valid_ery_preds = np.array([])
        
        valid_dand_label = np.array([])
        valid_seb_label = np.array([])
        valid_ery_label = np.array([])

        valid_final_label = np.array([])
        valid_final_preds = np.array([])

        with torch.no_grad():
            for img,label, dand, seb, ery in tqdm(valid_dataloader):
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

                ohe_preds = np.concatenate([dand_ohe_pred,seb_ohe_pred,ery_ohe_pred],axis=1)
                ohe_final_preds = transform_forhead_to_one_hot(disease_preds, ohe_preds)  

                final_label = np.concatenate([dand,seb,ery],axis=1)    
                
                
                
                valid_dis_label = np.append(valid_dis_label, labels)
                valid_dis_preds = np.append(valid_dis_preds, disease_preds)

                valid_dand_label = np.append(valid_dand_label, dand)
                valid_dand_preds = np.append(valid_dand_preds, dand_ohe_pred)
                valid_seb_label = np.append(valid_seb_label, seb)
                valid_seb_preds = np.append(valid_seb_preds, seb_ohe_pred)
                valid_ery_label = np.append(valid_ery_label, ery)
                valid_ery_preds = np.append(valid_ery_preds, ery_ohe_pred)


                valid_final_label = np.append(valid_final_label, final_label)
                valid_final_preds = np.append(valid_final_preds, ohe_final_preds)
        # reshape
        valid_dis_preds = valid_dis_preds.reshape(-1,3)
        valid_dis_label = valid_dis_label.reshape(-1,3)
        
        valid_dand_preds = valid_dand_preds.reshape(-1,4)
        valid_dand_label = valid_dand_label.reshape(-1,4)
        valid_seb_preds = valid_seb_preds.reshape(-1,4)
        valid_seb_label = valid_seb_label.reshape(-1,4)
        valid_ery_preds = valid_ery_preds.reshape(-1,4)
        valid_ery_label = valid_ery_label.reshape(-1,4)

        valid_final_preds = valid_final_preds.reshape(-1,12)
        valid_final_label = valid_final_label.reshape(-1,12)

        valid_dis_f1 = f1_score(valid_dis_preds, valid_dis_label, average='macro')
        valid_dand_acc = accuracy_score(valid_dand_preds, valid_dand_label)
        valid_seb_acc = accuracy_score(valid_seb_preds, valid_seb_label)
        valid_ery_acc = accuracy_score(valid_ery_preds, valid_ery_label)
        valid_final_f1 = f1_score(valid_final_preds, valid_final_label, average='macro')

        print(f'dis_F1 : {valid_dis_f1:.3f}, dand_acc : {valid_dand_acc:.3f}, seb_acc : {valid_seb_acc:.3f}, ery_acc : {valid_ery_acc:.3f}\n final_F1 : {valid_final_f1:.3f}')


        if valid_final_f1 > best_val_final_acc : # final f1 score
            try:
                best_val_final_acc = valid_final_f1
                before_best = glob.glob(f'{args.save_dir}/best_final_*')[0]
                os.remove(before_best)
            except:
                pass
            torch.save(model.state_dict(),f'{args.save_dir}/best_final_{e}.pth')
        
    print(best_val_final_acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,default = 'datasets/talmo/')
    parser.add_argument('--batch_size',type=int,default = 128)
    parser.add_argument('--device',type=str,default='cuda')
    parser.add_argument('--epoch',type=int,default = 50)
    parser.add_argument('--save_dir',type=str)
    parser.add_argument('--mode',type=str, default='baseline')
    parser.add_argument('--model',type=str)
    args = parser.parse_args()
    print(args)
    training_4head(args)
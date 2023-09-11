import numpy as np
import cv2
import glob
from tqdm import tqdm
from sklearn.metrics import f1_score

def iou(predicted_mask, true_mask):
    intersection = np.logical_and(predicted_mask, true_mask)
    union = np.logical_or(predicted_mask, true_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def ours_seg():
    gt = sorted(glob.glob('../scalp_aihub/seg_gt/*.png'))
    # pred = list(map(lambda x: x.replace('seg_gt','seg_test'), gt))
    pred = sorted(glob.glob('add_img/*.jpg'))
    smooth = 1

    pixel_f1 = []
    pixel_iou = []
    pixel_acc = []
    dice_ls = []
    for g,p in tqdm(zip(gt, pred)):
        gt_label = cv2.imread(g,0)
        pred_label = cv2.imread(p,0)
        
        pred_label = np.where(pred_label>125, 1, 0)
        gt_label = np.where(gt_label>0, 1, 0)
        
        intersection = (pred_label * gt_label).sum()
        dice = (2.*intersection + smooth) / (pred_label.sum() + gt_label.sum() + smooth) 
        
        dice_ls.append(dice)
        
        pixel_iou.append(iou(pred_label, gt_label))
        gt_label, pred_label = gt_label.reshape(-1), pred_label.reshape(-1)

        f1 = f1_score(gt_label, pred_label, average='macro')
        pixel_f1.append(f1)
        acc = np.sum(gt_label == pred_label) / (480*640)
        pixel_acc.append(acc)
        
    print("Pixel Acc : ", np.mean(pixel_acc))
    print("f1_score : ",np.mean(pixel_f1))
    print("mIOU : ", np.mean(pixel_iou))
    print("mDice : ", np.mean(dice_ls))

ours_seg()
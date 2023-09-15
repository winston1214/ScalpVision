import cv2
import glob
from tqdm import tqdm
import numpy as np
import os
def make_final_mask(seg_path, sam_path, result_path):

    seg_full_path = sorted(glob.glob(os.path.join(seg_path, '*.png')))
    sam_full_path = sorted(glob.glob(os.path.join(sam_path, '*.jpg')))

    for seg, sam in tqdm(zip(seg_full_path, sam_full_path)):
        seg_img = cv2.imread(seg)
        sam_img = cv2.imread(sam)
        img_name = sam.split('/')[-1]
        added_img = cv2.bitwise_and(seg_img, sam_img)
        binary_map = cv2.cvtColor(added_img, cv2.COLOR_BGR2GRAY)
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)
        # Get CC_STAT_AREA component as stats[label, COLUMN]
        areas = stats[1:, cv2.CC_STAT_AREA]
        result = np.zeros((labels.shape), np.uint8)
        for i in range(0, nlabels - 1):
            if areas[i] >= 400:   # Keep
                result[labels == i + 1] = 255
        
        cv2.imwrite(os.path.join(result_path, img_name), result)

if __name__ == '__main__':
    seg_path = '../scalp_aihub/seg_result/seg_test'
    sam_path = '../scalp_aihub/sam_result/sam_test'
    result_path = '../scalp_aihub/ensemble_result/ensemble_test'
    make_final_mask(seg_path, sam_path, result_path)
import numpy as np
import cv2
import argparse
from tqdm import tqdm
import glob

def threshold_filter(args):
    img_path = args.img_path
    img = cv2.imread(img_path)
    imgray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply threshold
    ret, binary_map = cv2.threshold(imgray, 127, 255, 0)
    
    # Get rid of noises (ex. big white spots that are not hair)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)
    # Get CC_STAT_AREA component as stats[label, COLUMN]
    areas = stats[1:, cv2.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if areas[i] >= args.thr:   # Keep
            result[labels == i + 1] = 255
    cv2.imwrite(args.save_path, result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--thr', type=int, default = 400)
    args = parser.parse_args()
    args.thr = 400
    for im in tqdm(sorted(glob.glob('../scalp_aihub/seg_train/*.png'))):
        args.img_path = im
        args.save_path = im
        threshold_filter(args)

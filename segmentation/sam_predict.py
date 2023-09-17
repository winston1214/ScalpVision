# cluster point
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.pyplot as plt

import numpy as np
import cv2
from tqdm import tqdm
import json
import os

with open('../val_add_points.json','r') as f:
    points = json.load(f)

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

for full_name in tqdm(points.keys()):
    name = full_name.split('.')[0]
    sample_points = points[f'{name}.png']
    if len(sample_points) == 0:
        image = cv2.imread(f'../../scalp_aihub/seg_result/seg_val/{name}.png')
        cv2.imwrite(f'../../scalp_aihub/sam_val/{name}.jpg', image)
        print(name)
    else:
        tmp = np.array(sample_points)
        # tmp = np.array(sum(sample_points,[]))
        tmp = tmp[tmp.min(axis=1) > 0]

        rand_idx = np.random.choice(len(tmp), len(tmp)//2, replace = False)

        image = cv2.imread(f'../../scalp_aihub/val/images/{name}.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
        neg_list = []
        while len(neg_list) < 10:
            xy = [np.random.randint(640), np.random.randint(480)]
            if xy not in tmp:
                neg_list.append(xy)
        neg_arr = np.array(neg_list) # 두피

        input_point = tmp[rand_idx] # 머리카락
        neg_point = tmp[list(set([i for i in range(len(tmp))]) - set(rand_idx))] # 머리카락
        # input_point_tmp = np.append(input_point, neg_point)
        # final_point = np.append(input_point_tmp, neg_arr).reshape(-1,2)

        final_point = np.append(input_point, neg_arr).reshape(-1,2)

        # input_label = np.array([0] * len(input_point) + [1]*len(neg_point) + [1]*len(neg_arr))
        input_label = np.array([0] * len(input_point)  + [1]*len(neg_arr))

        masks, scores, logits = predictor.predict(
            point_coords=final_point,
            point_labels=input_label,
            multimask_output=True,
        )
        sam_mask = masks[np.argmax(scores)].reshape(480,640)
        binary_map = np.where(sam_mask>0,0,255).astype(np.uint8)
        # ret, binary_map = cv2.threshold(sam_mask, 127, 255, 0)

        # Get rid of noises (ex. big white spots that are not hair)
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)
        # Get CC_STAT_AREA component as stats[label, COLUMN]
        areas = stats[1:, cv2.CC_STAT_AREA]
        result = np.zeros((labels.shape), np.uint8)
        for i in range(0, nlabels - 1):
            if areas[i] >= 400:   # Keep
                result[labels == i + 1] = 255
        save_path = os.path.join('../../scalp_aihub/sam_result/sam_val',name+'.jpg')
        cv2.imwrite(save_path, result)

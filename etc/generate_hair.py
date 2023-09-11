import cv2
import random
import numpy as np
import os
from tqdm import tqdm
  
BLUE_LIST = [
    [61, 55, 20],
    [93, 92, 31],
    [151, 139, 67],
    [175, 169, 104],
    [193, 189, 124],
    [186, 182, 99],
    [133, 134, 84],
    [106, 105, 37],
    [131, 165, 141],
    [179, 161, 90],
    [171, 180, 93],
    [193, 166, 110],
    [195, 169, 132],
    [170, 148, 113],
    [134, 114, 29],
    [255, 255, 110],
    [255, 255, 180],
    [255, 255, 130],
    [255, 255, 126],
    [113, 116, 80],
    [217, 215, 118],
]
BROWN_LIST = [
    [16, 37, 52],
    [32, 53, 61],
    [63, 78, 94],

    [14, 37, 56],
    [13, 33, 47],
    [78, 86, 103],
    [80, 133, 153],
    [92, 126, 142],
    [84, 134, 133],
    [138, 170, 169],
]
BLACK_LIST = [
    [66, 66, 61],
    [89, 85, 79],
    [48, 41, 33],
    [16, 16, 16],
    [27, 27, 27],
    [36, 36, 36],
    [44, 44, 44],
    [51, 51, 51],
    [58, 58, 58],
    [65, 65, 65],
]
WHITE_LIST = [
    [255, 255, 255],
    [243, 243, 243],
    [232, 232, 232],
    [221, 221, 221],
    [210, 210, 210],
    [200, 200, 200],
]

# YELLOW_LIST = [
#   [217, 190, 121],
#   [184, 152, 109],
#   [180, 161, 124],
#   [177, 163, 88]
# ]


def draw_hair(img):
    mask = np.zeros_like(img).copy()
    image_height, image_width, channel = img.shape
    count = random.randint(1, 10)
    for i in range(count):
        width_size = random.randint(0, image_width)
        height_size = random.randint(0, image_height)
        color_list = random.choice([BROWN_LIST, BLUE_LIST, BLACK_LIST, WHITE_LIST])
        color_label = random.randrange(0, len(color_list))     
        form = random.choice("cl")
        size = random.randint(1, 7)
        
        if form == "c":
            radian = random.randint(0, 550)
            direction = random.choice("rl")
            if direction == "r":
                cv2.circle(img, (0, 0), radian, color_list[color_label], size)
                cv2.circle(mask, (0, 0), radian, (255,255,255), size)
            else:
                cv2.circle(
                    img,
                    (image_width, image_height),
                    radian,
                    color_list[color_label],
                    size,
                )
                cv2.circle(
                    mask,
                    (image_width, image_height),
                    radian,
                    (255,255,255),
                    size,
                )
        else:
            start_point = (random.randint(0,height_size),random.randint(0,width_size))
            end_point = (random.randint(0,height_size),random.randint(0,width_size))
            cv2.line(img,start_point,end_point,color_list[color_label],size)
            cv2.line(mask,start_point,end_point,(255,255,255),size)

    return img, mask

def draw_dot(img,color):
    image_height, image_width, channel = img.shape
    count = random.randint(100, 200)
    for i in range(count):
        width = random.randint(0, image_width)
        height = random.randint(0, image_height)

        color_list = (
            BLUE_LIST
            if color == "blue"
            else BROWN_LIST
            if color == "brown"
            else BLACK_LIST
            if color == "black"
            else WHITE_LIST
        )
        color_label = random.randrange(0, len(color_list))

        form = random.choice("rc")

        if form == "r":
            width_size = random.randint(0, 10)
            height_size = random.randint(0, 10)
            cv2.rectangle(
                img,
                (width, height),
                (width + width_size, height + height_size),
                color_list[color_label],
                -1,
            )
        else:
            size = random.randint(1, 5)
            cv2.circle(img, (width, height), size, color_list[color_label], -1)
    return img

all_patch = sorted(glob.glob('background_img_patch/*.png'))
val1_patch = list(filter(lambda x: 'val1' in x,all_patch))
val2_patch = list(filter(lambda x: 'val2' in x,all_patch))
val3_patch = list(filter(lambda x: 'val3' in x,all_patch))

for i in tqdm(range(3000)):
    patch_choice = random.choice('abc')
    if patch_choice == 'a':
        patch = val1_patch
    elif patch_choice == 'b':
        patch = val2_patch
    else:
        patch = val3_patch
    
    rand = np.random.choice(patch)
    img = cv2.imread(rand)
    res = cv2.resize(img,(640,480),interpolation=cv2.INTER_CUBIC)
    
    result,mask = draw_hair(res)
    
    name = rand.split('/')[-1]
    save_name = name.replace('.png',f'_{i}.png')
    cv2.imwrite(f'seg/org/{save_name}',result)
    cv2.imwrite(f'seg/mask/{save_name}',mask)

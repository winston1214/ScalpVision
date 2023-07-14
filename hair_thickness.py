import cv2 as cv
import numpy as np
import os

def nms(boxes, thresh):
    if len(boxes) == 0:
        return []

    pick = []
    
    #coordinate
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > thresh)[0])))

    return boxes[pick]
    
#pre-process
img_path = "results/0335_A2LEBJJDE00013X_1602916465540_6_BH.png"

# 여러이미지 테스트 코드 ..ing
# results_folder = "results"
# for images in os.listdir(results_folder) in range(10):
#     im
    

img = cv.imread(img_path)

imgray = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

#apply threshold
ret, binary_map = cv.threshold(imgray,127,255,0)

nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(binary_map, None, None, None, 8, cv.CV_32S)

#get CC_STAT_AREA component as stats[label, COLUMN] 
areas = stats[1:,cv.CC_STAT_AREA]

result = np.zeros((labels.shape), np.uint8)

#get rid of noises(ex. big white spots that are not hair)
for i in range(0, nlabels - 1): 
    if areas[i] >= 300:   #keep
        result[labels == i + 1] = 255
        

filter_size = (10,10) 

white_pixels = np.where(result == 255)

x_coords = white_pixels[1]
y_coords = white_pixels[0]

#another version can be boxes goes to middle of the detection
x1 = np.maximum((x_coords - filter_size[0]),0)
y1 = np.maximum((y_coords - filter_size[1] ),0)
x2 =(x_coords + filter_size[0] )
y2 = (y_coords + filter_size[1] )

white_regions = np.column_stack((x1, y1, x2, y2))  # (N, 4)

white_regions = nms(white_regions, thresh=0.01)

thickness_image = result.copy()

sum_values = []
# Draw bounding boxes on the thickness image
for coor in white_regions:
    x1, y1, x2, y2 = coor
    # print(coor)
    thickness_image = cv.rectangle(thickness_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
    
    # Compute sum of white pixels within each bounding box
    region_pixels = result[y1:y2, x1:x2]
    sum_value = np.sum(region_pixels)
    sum_values.append(sum_value)
    

    #the width of the bounding box = thickness???
    # cv.putText(result, str(thickness), (region[0], region[1] - 5),
    #             cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)
    
# print("thickness for white region1: ", thickness)
# result = cv.rectangle(img, (292,0), (293,0), (255, 0, 0), 1)
cv.imshow("Result", thickness_image)

# cv.imshow('testrec', result)
cv.waitKey(0)
cv.destroyAllWindows()




    
    


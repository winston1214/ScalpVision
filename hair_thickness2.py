import cv2
import numpy as np
from math import atan2, cos, sin, sqrt, pi

def nms(boxes, thresh):
    if len(boxes) == 0:
        return []

    pick = []

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

#function that find point on line
def find_pts_on_line(og, slope, d):
    cx, cy = og
    x1 = float(cx -  d/((1+slope**2)**0.5))
    y1 = float(cy-slope*cx + x1 * slope)
    return (x1,y1)

# Pre-process
img_path = "/home/jerry0110/scalp_diagnosis/results2/1407_A2LEBJJDE001258_1606104267519_4_LH.png"
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
    if areas[i] >= 250:   # Keep
        result[labels == i + 1] = 255
re_copy=result.copy()

# Edge detected (contour) image using Canny edge detection
edgeimg = cv2.Canny(result, 10, 150)
cv2.imwrite('edgeimg.png', edgeimg)

contours, _ = cv2.findContours(edgeimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#skeleton image using morphology
# Step 1: Create an empty skeleton
size = np.size(result)
skel = np.zeros(result.shape, np.uint8)

# Get a Cross Shaped Kernel
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

# Repeat steps 2-4
while True:
    # Step 2: Open the image
    open = cv2.morphologyEx(result, cv2.MORPH_OPEN, element)
    # Step 3: Subtract open from the original image
    temp = cv2.subtract(result, open)
    # Step 4: Erode the original image and refine the skeleton
    eroded = cv2.erode(result, element)
    skel = cv2.bitwise_or(skel, temp)
    result = eroded.copy()
    # Step 5: If there are no white pixels left, i.e., the image has been completely eroded, quit the loop
    if cv2.countNonZero(result) == 0:
        break


# Get rid of noises of skeleton(optional)
nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(skel, None, None, None, 8, cv2.CV_32S)
# Get CC_STAT_AREA component as stats[label, COLUMN]
areas = stats[1:, cv2.CC_STAT_AREA]
skel = np.zeros((labels.shape), np.uint8) 
# Get rid of noises of skeleton(optional)
for i in range(0, nlabels - 1):
    if areas[i] >= 2:   # Keep
        skel[labels == i + 1] = 255



# Displaying the final skeleton
cv2.imwrite("Skeleton.png",skel)

combined_img = cv2.addWeighted(skel, 1, edgeimg, 1, 0)
cv2.imwrite('skeleton+contour.png', combined_img)


filter_size = (20, 20)

white_pixels = np.where(skel == 255)

x_coords = white_pixels[1]
y_coords = white_pixels[0]

x1 = x_coords - filter_size[0] // 2
y1 = y_coords - filter_size[1] // 2
x2 = x_coords + filter_size[0] // 2
y2 = y_coords + filter_size[1] // 2

# Getting center point of boxes
center_x = (x1 + x2) // 2
center_y = (y1 + y2) // 2


white_regions = np.column_stack((x1, y1, x2, y2))  # (N, 4)
white_regions = nms(white_regions, thresh=0.1)

skeleton_image = cv2.cvtColor(skel.copy(), cv2.COLOR_BGR2RGB)

center_points = []

directions =[]

    
# Drawing bounding boxes and center points
def get_direction2(bbox_pixels):
    # Get the coordinates of non-zero (white) pixels within the bounding box
    nonzero_indices = np.column_stack(np.nonzero(bbox_pixels))
    nonzero_indices = np.float32(nonzero_indices)
    # Perform PCA analysis
    if len(nonzero_indices) >= 2:
        mean, eigenvectors = cv2.PCACompute(nonzero_indices, mean=None)
        cntr = ((mean[0, 1]), (mean[0, 0]))
        return eigenvectors[0], cntr
    else:
        return (0,0), (0,0)
    
intersection_points = []
# find intersection btw perpendicular line and contour
def find_intersection_points2(center, slope, img,threshold):
    p2=p1=(-1,-1)
    w,h=img.shape
    
    # you can change step and searching_len for faster search
    # too high step may result in skipping pixel
    # too low searching len may limit searching space
    step=100
    searching_len=50
    for d in range(1,step*searching_len):
        px, py = find_pts_on_line(center,slope,d/step)
        if (0<int(px)<h) and (0<int(py)<w) and img[int(py)][int(px)]>threshold:
            p1=(px,py)
        else:
            break

    for d in range(1,step*searching_len):
        px, py = find_pts_on_line(center,slope,-d/step)
        if (0<int(px)<h) and (0<int(py)<w) and img[int(py)][int(px)]>threshold:
            p2=(px,py)
        else:
            break
    return [p1,p2],np.linalg.norm(np.asarray(p1)-np.asarray(p2))

for coor in white_regions:
    x1, y1, x2, y2 = coor
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    bbox_pixels = skel[y1:y2, x1:x2]

    # Calculate the direction using PCA for the white pixels within the bounding box
    direction, mean = get_direction2(bbox_pixels)
    directions.append(direction)
    center_points.append((mean[0]+x1,mean[1]+y1))
    skeleton_image = cv2.rectangle(skeleton_image, (x1, y1), (x2, y2), (255, 255, 255), 1)
    skeleton_image = cv2.circle(img, (int(mean[0]+x1),int(mean[1]+y1)), 1, (255, 0, 0), 1)

thicknesses = []
perpendicular_slope = []

for direction, ctpt in zip(directions, center_points):
    cx, cy = ctpt
    # Calculate the slope of the direction vector
    if direction[1] != 0:
        perpendicular_slope.append(-1/(direction[0] / direction[1]))        
    else:
        perpendicular_slope.append(0)  # Perpendicular slope is 0 for vertical direction


for center_point, perp_slope in zip(center_points, perpendicular_slope):
    x1, y1= find_pts_on_line(center_point,perp_slope,10)
    x2, y2= find_pts_on_line(center_point,perp_slope,-10)
    
    # Draw the short line on the image
    intersection, dst=find_intersection_points2(center_point,perp_slope, re_copy,200)
    intersection_points+=intersection
    if dst!=0:
        thicknesses.append(dst)
    cv2.line(skeleton_image, (int(x1),int(y1)), (int(x2),int(y2)), (0, 255, 0), 1)
    
for pts in intersection_points:
    x, y = pts
    cv2.circle(skeleton_image, (int(x), int(y)), 1, (255, 128, 0), -1)

avg_thickness = np.mean(thicknesses)

print('avg_thickenss', avg_thickness) 
cv2.imwrite('re.png', re_copy)
cv2.imwrite('bbox+centerpt_image.png', skeleton_image)
cv2.destroyAllWindows()

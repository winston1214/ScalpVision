import cv2
import numpy as np
from sklearn.cluster import KMeans


#TODO
 #i have bbox list and intersection list. i want to map intersection points with each bbox when points are within the box. 
 # Then I want to calculate the L2 distance of two farthest points in a bounding box, save the distance in the array.
 # The average of those distances of all selected points=thickness    
 
 #find intersection points that align with perp_line
 
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

# Pre-process
img_path = "results/0013_A2LEBJJDE00060O_1606550825417_3_TH.png"
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
    if areas[i] >= 400:   # Keep
        result[labels == i + 1] = 255

# Edge detected (contour) image using Canny edge detection
edgeimg = cv2.Canny(result, 100, 200)
cv2.imshow('edgeimg.png', edgeimg)

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
    if areas[i] >= 100:   # Keep
        skel[labels == i + 1] = 255



# Displaying the final skeleton
cv2.imshow("Skeleton.png",skel)


filter_size = (16,16)

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
bboxes = []
    
# Drawing bounding boxes and center points
def get_direction2(bbox_pixels):
    # Get the coordinates of non-zero (white) pixels within the bounding box
    nonzero_indices = np.column_stack(np.nonzero(bbox_pixels))
    nonzero_indices = np.float32(nonzero_indices)
    # Perform PCA analysis
    if len(nonzero_indices) >= 2:
        mean, eigenvectors = cv2.PCACompute(nonzero_indices, mean=None)
        cntr = (int(mean[0, 0]), int(mean[0, 1]))
        return eigenvectors[0], cntr
    else:
        return (0,0), (0,0)
    
def calculate_intersection(start_point, end_point, edge, bbox):
    x1, y1 = start_point
    x2, y2 = end_point
    x, y = edge

    # Check if the edge point is on the line segment and within the bounding box
    if bbox[0][0] <= x <= bbox[1][0] and bbox[0][1] <= y <= bbox[1][1] and min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
        return True
    return False
    
intersection_points = []
def find_intersection_points(start_point, end_point, bounding_boxes):
    for bbox in bounding_boxes:
        bbox_x1, bbox_y1 = bbox[0]
        bbox_x2, bbox_y2 = bbox[1]

        for bbox in bounding_boxes:
            bbox_x1, bbox_y1 = np.clip(bbox[0], 0, edgeimg.shape[1] - 1)
            bbox_x2, bbox_y2 = np.clip(bbox[1], 0, edgeimg.shape[0] - 1)

            for y in range(bbox_y1, bbox_y2 + 1):
                for x in range(bbox_x1, bbox_x2 + 1):
                    if 0 <= y < edgeimg.shape[0] and 0 <= x < edgeimg.shape[1] and edgeimg[y, x] == 255:
                        if calculate_intersection(start_point, end_point, (x, y), bbox):
                            intersection_points.append((x, y)) #(x,y) location of pixels that are white.
                
                    
for coor in white_regions:
    x1, y1, x2, y2 = coor
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    bbox_pixels = skel[y1:y2, x1:x2]

    skeleton_image = cv2.rectangle(skeleton_image, (x1, y1), (x2, y2), (255, 255, 255), 1)
    skeleton_image = cv2.circle(img, (center_x, center_y), 1, (255, 0, 0), 1)

    # Calculate the direction using PCA for the white pixels within the bounding box
    direction, mean = get_direction2(bbox_pixels)
    directions.append(direction)
    center_points.append((mean[0]+x1,mean[1]+y1))
    bboxes.append(((x1,y1),(x2,y2)))

thicknesses = []
perpendicular_slope = []

for direction, ctpt in zip(directions, center_points):
    cx, cy = ctpt
    # Calculate the slope of the direction vector
    # print(direction)
    if direction[1] != 0:
        perpendicular_slope.append(-1/(direction[0] / direction[1]))        
    else:
        perpendicular_slope.append(0)  # Perpendicular slope is 0 for vertical direction


for center_point, perp_slope in zip(center_points, perpendicular_slope):
    cx, cy = center_point
    line_length = 10
    x1 = float(cx - line_length/((1+perp_slope**2)**0.5))
    y1 = float(cy-perp_slope*cx + x1 * perp_slope)
    x2 = float(cx +  line_length/((1+perp_slope**2)**0.5))
    y2 = float(cy-perp_slope*cx + x2 * perp_slope)
    
    # Draw the short line on the image
    # cv2.line(skeleton_image, (int(x1),int(y1)), (int(x2),int(y2)), (0, 255, 0), 1)
    find_intersection_points((int(x1),int(y1)),(int(x2),int(y2)),bboxes)
    

# for i in intersection_points:
#     (x,y) = i
#     cv2.circle(skeleton_image,(x,y),1,(255,255,0),1)


# avg_thickness = np.mean(thicknesses)
# print(intersection_points)
# print('avg_thickenss', avg_thickness) 

print(np.size(intersection_points))

# cv2.imshow('bbox+centerpt_image.png', skeleton_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

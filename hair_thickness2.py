import cv2
import numpy as np


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
    if areas[i] >= 250:   # Keep
        result[labels == i + 1] = 255

# Edge detected (contour) image using Canny edge detection
edgeimg = cv2.Canny(result, 10, 150)
cv2.imshow('edgeimg', edgeimg)


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
cv2.imshow("Skeleton",skel)

combined_img = cv2.addWeighted(skel, 1, edgeimg, 1, 0)
cv2.imshow('skeleton+contour', combined_img)


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
def get_direction(bbox_pixels):
    # Get the coordinates of non-zero (white) pixels within the bounding box
    nonzero_indices = np.column_stack(np.nonzero(bbox_pixels))
    if len(nonzero_indices) >= 2:
        _, _, vt = np.linalg.svd(nonzero_indices)
        direction = vt[0]  # First eigenvector (largest eigenvalue)
        return direction
    else:
        return (0, 0)  # No direction
    
for coor in white_regions:
    x1, y1, x2, y2 = coor
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    bbox_pixels = skel[y1:y2, x1:x2]

    skeleton_image = cv2.rectangle(skeleton_image, (x1, y1), (x2, y2), (255, 255, 255), 1)
    skeleton_image = cv2.circle(img, (center_x, center_y), 1, (255, 0, 0), 1)

    # Calculate the direction using PCA for the white pixels within the bounding box
    direction = get_direction(bbox_pixels)
    directions.append(direction)
print(np.size(directions))     

thicknesses = []
perpendicular_slope = []

for direction, ctpt in zip(directions, center_points):
    cx, cy = ctpt
    # Calculate the slope of the direction vector
    if direction[1] != 0:
        perpendicular_slope.append(-1/(direction[0] / direction[1]))
    else:
        perpendicular_slope.append(0)  # Perpendicular slope is 0 for vertical direction
        
print(np.size(perpendicular_slope))   
# Calculate intersection points and thickness for each 'center point'
for center_point, perp_slope in zip(center_points, perpendicular_slope):
    cx, cy = center_point
    line_length = 10
    x1 = int(cx - line_length / 2)
    y1 = int(cy - (line_length / 2) * perp_slope)
    x2 = int(cx + line_length / 2)
    y2 = int(cy + (line_length / 2) * perp_slope)

    # Draw the short line on the image
    cv2.line(skeleton_image, (x1,y1), (x2,y2), (0, 255, 0), 1)
    
    intersection_points = []
    for x in range(edgeimg.shape[1]):
        # Calculate the y-coordinate of the perpendicular line at the given x-coordinate
        perp_y = int(perp_slope * (x - cx) + cy)

        # Check if the point is within the image boundaries
        if 0 <= perp_y < edgeimg.shape[0] and 0 <= x < edgeimg.shape[1]:
            # Check if the pixel at the intersection point is an edge pixel
            if edgeimg[perp_y, x] == 255:
                intersection_points.append((x, perp_y))
        
# avg_thickness = np.mean(thicknesses)

# print('avg_thickenss', avg_thickness)
# print(directions)

cv2.imshow('bbox,centerpt_image', skeleton_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

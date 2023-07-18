import cv2
import numpy as np

#hyperparameter: threshold, filter size, area to cover

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
img_path = "results/0616_A2LEBJJDE00004W_1606017745245_2_TH.png"

img = cv2.imread(img_path)
imgray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Apply threshold
ret, binary_map = cv2.threshold(imgray, 127, 255, 0)

nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)

# Get CC_STAT_AREA component as stats[label, COLUMN]
areas = stats[1:, cv2.CC_STAT_AREA]

result = np.zeros((labels.shape), np.uint8)

# Get rid of noises (ex. big white spots that are not hair)
for i in range(0, nlabels - 1):
    if areas[i] >= 250:   # Keep
        result[labels == i + 1] = 255

# Edge detected (contour) image using Canny edge detection
edgeimg = cv2.Canny(result, 10, 150)
cv2.imshow('edgeimg', edgeimg)

filter_size = (18, 18)

white_pixels = np.where(result == 255)

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
white_regions = nms(white_regions, thresh=0.01)

skeleton_image = img.copy()

center_points = []
# Drawing bounding boxes and center points
for coor in white_regions:
    x1, y1, x2, y2 = coor
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    skeleton_image = cv2.rectangle(skeleton_image, (x1, y1), (x2, y2), (255, 255, 255), 1)
    skeleton_image = cv2.circle(skeleton_image, (center_x, center_y ), 1, (0, 255, 0), 1)
    center_points.append((center_x, center_y))

# Connect center points if they are adjacent horizontally or vertically
for i in range(len(center_points) - 1):
    if np.abs(x1 - x2) <= filter_size[0] or y1 == y2:  # Horizontally adjacent
        skeleton_image = cv2.line(skeleton_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

    if x1 == x2 or np.abs(y1 - y2) <= filter_size[1]:  # Vertically adjacent
        skeleton_image = cv2.line(skeleton_image, (x1, y1), (x2, y2), (0, 255, 0), 1)


print(np.size(center_points))
cv2.imshow('skeleton_image', skeleton_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

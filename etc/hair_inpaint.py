import cv2
import numpy as np
import matplotlib.pyplot as plt

def group_consecutive_numbers(nums):
    # 결과를 저장할 리스트
    result = []
    # 현재 그룹을 저장할 리스트
    current_group = []

    for i, num in enumerate(nums):
        # 현재 숫자가 리스트의 첫 번째 요소이거나, 이전 숫자와 1만큼 차이가 날 경우
        if i == 0 or num == nums[i - 1] + 1:
            current_group.append(num)
        else:
            # 그룹이 끝났을 경우 결과 리스트에 추가하고, 새로운 그룹 시작
            result.append(current_group)
            current_group = [num]

    # 마지막 그룹을 결과 리스트에 추가
    result.append(current_group)
    return result

img = cv2.imread('../scalp_aihub/train/images/3365_A2LEBJJDE00106D_1605944166106_5_RH.jpg')
mask = cv2.imread('../scalp_aihub/seg_train/3365_A2LEBJJDE00106D_1605944166106_5_RH.png')

result = cv2.bitwise_and(img, 255 - mask)
plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
plt.show()
plt.close()
all_zeros_pixels = np.argwhere(np.all(result == [0, 0, 0], axis=-1))
all_nonzeros_pixels = np.argwhere(np.all(result != [0, 0, 0], axis=-1))

continuous_x = group_consecutive_numbers(all_zeros_pixels[:,1])
y_pixel = all_zeros_pixels[:,0]
hap = 0

y_ls = []
for idx, i in enumerate(continuous_x): # y 좌표 정상
    before_pixel_x = i[0]-1
    last_pixel_x = i[-1]+1
    
    if last_pixel_x >= 640:
        last_pixel_x = 639
        
    y = y_pixel[hap]
    
    
    hap += len(continuous_x[idx])
    y_ls.append(y)

    if idx != 0:
        if y_ls[idx-1] == y_ls[idx]: # 부분 (머리카락과 머리카락 사이)
            before_color = result[y, continuous_x[idx-1][0] : i[0]] # 부분
            # last_color = result[y, last_pixel_x : continuous_x[idx-1][-1]]
    
        else:
            before_color = result[y, : i[0]] # 처음부터
            # last_color = result[y, last_pixel_x : continuous_x[idx+1][0]]
    else: # 첫번째면
        before_color = result[y, : i[0]] # 처음부터
        last_color = result[y, i[-1] : continuous_x[idx+1][0]] # 1번째 까지
    
    
    
    result[y,i] = np.mean(before_color,axis=0).astype(int)

plt.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))
plt.show()
plt.close()

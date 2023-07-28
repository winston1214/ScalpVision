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

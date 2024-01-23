import cv2, os, argparse,glob, csv
from tqdm import tqdm
import numpy as np
import pandas as pd

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

def create_color_pallete(n):
    #create n number of rgb color tuple (r,g,b)
    ret=[]
    
    #select random list of rgb
    for i in range(n):
        r=np.random.randint(0,255)
        g=np.random.randint(0,255)
        b=np.random.randint(0,255)
        ret.append((r,g,b))
    
    return ret


#calcualte min_dst
def min_dst(l1,l2):
    p11=np.array([l1[0][0],l1[0][1]])
    p12=np.array([l1[0][2],l1[0][3]])
    
    p21=np.array([l2[0][0],l2[0][1]])
    p22=np.array([l2[0][2],l2[0][3]])
    
    d1=np.abs(np.linalg.norm(p11-p21))
    d2=np.abs(np.linalg.norm(p11-p22))
    d3=np.abs(np.linalg.norm(p12-p21))
    d4=np.abs(np.linalg.norm(p12-p22))
    
    return min(d1,d2,d3,d4)
    
    
#measure similarity between vectors
#input: list of vecotr consist of start and endpoint
#output: list of similarity score
def measure_similarity(lines,shape, thres_l=40, thres_c=0.9):
    vectors=[]
    tmp=[]
    for idx,l in enumerate(lines):
        v=[l[0][2]-l[0][0],l[0][3]-l[0][1]]
        if (v[0]==0 and not 0+3<l[0][0]<shape[0]-3) or (v[1]==0 and not 0+3<l[0][1]<shape[1]-3):
            tmp.append(idx)
        
        # pts=[int(np.clip(l[0][0],0,skel.shape[0]-1)),int(np.clip(l[0][1],0,skel.shape[1]-1)),int(np.clip(l[0][2],0,skel.shape[0]-1)),int(np.clip(l[0][3],0,skel.shape[1]-1))]
        # # print(skel.shape)
        # # print(pts)
        # if not filtered_image[pts[0]][pts[1]].any() or not filtered_image[pts[2]][pts[3]].any():
        #     tmp.append(idx)
        vectors.append(np.array(v))
    
    vectors=np.array(vectors)
    # print(tmp)
    assert len(lines)==len(vectors)
    
    ret=[]
    #get cosine similary between vectors
    for idx,v in enumerate(vectors):
        # print(f'\nvector {v[0]}:{v[1]}')
        for jdx,v2 in enumerate(vectors[idx:]):
            # print(f'vector2 {v2[0]}:{v2[1]}')
            j=jdx+idx
            if jdx==0 or (j in tmp) or (idx in tmp):
                continue
            
            cosSim=np.abs(np.dot(v,v2)/(np.linalg.norm(v)*np.linalg.norm(v2)))
            dst=min_dst(lines[idx], lines[j])

            if dst<thres_l and cosSim>thres_c:
                
                len1=np.abs(np.linalg.norm(lines[idx][0][:2]-lines[idx][0][2:4]))
                len2=np.abs(np.linalg.norm(lines[j][0][:2]-lines[j][0][2:4]))
                
                # print(f'\nidx:{idx} vs jdx:{j}\nline {lines[idx][0][:4]} || line2 {lines[j][0][:4]}\nvector {v[0]}:{v[1]} || vector2 {v2[0]}:{v2[1]}')
                # print(len1,len2)
                if len1<len2:
                    ret.append(idx)
                    # print(f'putting idx:{idx}')
                else:
                    ret.append(j)
                    # print(f'putting jdx:{j}')
    ret+=tmp
    s= set(ret)
    # print(s)
    return s
    

def cluster(imgray,min_area=250, threshold=90,  minLlength=5,  maxLgap=10 ,skel_thr=10):

    # Apply threshold
    ret, binary_map = cv2.threshold(imgray, 127, 255, 0)

    # Get rid of noises (ex. big white spots that are not hair)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)
    # Get CC_STAT_AREA component as stats[label, COLUMN]
    areas = stats[1:, cv2.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if areas[i] >= min_area:   # Keep
            result[labels == i + 1] = 255
    re_copy=result.copy()

    #skeleton image using morphology
    # Step 1: Create an empty skeleton
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
        if areas[i] >= skel_thr:   # Keep
            skel[labels == i + 1] = 255

    # cv2.imwrite(os.path.join(img_path,'skel_'+im), skel)
    filtered_image = cv2.cvtColor(re_copy.copy(), cv2.COLOR_BGR2RGB)

    lines = cv2.HoughLinesP(skel.astype(np.uint8), 5, np.pi / 180, threshold, minLineLength = minLlength, maxLineGap = maxLgap)    
    
    return lines, filtered_image, skel

def main(args,im):
    img_path=args.img_folder
    # img_path ='/home/jerry0110/scalp_diagnosis/sample_mask/'
    # im='ensemble_2.jpg'
    
    # img = cv2.imread(os.path.join(img_path,im))
    # # img=cv2.resize(img, (0, 0), fx=1/factor, fy=1/factor, interpolation=cv2.INTER_LINEAR)
    # print(os.path.join(img_path,im))
    if os.path.isfile(os.path.join(img_path,im)):
        imgray = cv2.imread(os.path.join(img_path,im), cv2.IMREAD_GRAYSCALE)
    elif os.path.isfile(os.path.join(img_path.replace('train','test'),im)):
        imgray = cv2.imread(os.path.join(img_path.replace('train','test'),im), cv2.IMREAD_GRAYSCALE)
        # print('test')
    elif os.path.isfile(os.path.join(img_path.replace('test','train'),im)):
        imgray = cv2.imread(os.path.join(img_path.replace('test','train'),im), cv2.IMREAD_GRAYSCALE)
        # print('train')
    
    ###
    factor =0.5
    thr=100
    mll=60
    mlg=100
    min_area=300
    skel_thr=25

    imgray1=imgray.copy()
    imgray1=cv2.resize(imgray, (0, 0), fx=1/factor, fy=1/factor, interpolation=cv2.INTER_LINEAR)
    lines1, _, _ = cluster(imgray1,min_area=min_area, threshold=thr,  minLlength=mll, maxLgap=mlg,skel_thr=skel_thr)
    if lines1 is None or len(lines1)==0:
        lines1=[]
    else:
        lines1=np.array(lines1)*factor*0.5

    ###
    factor = 1
    thr = 100
    mll = 30
    mlg = 60
    min_area = 150
    skel_thr = 10
    
    imgray2=imgray.copy()
    imgray2=cv2.resize(imgray, (0, 0), fx=1/factor, fy=1/factor, interpolation=cv2.INTER_LINEAR)
    lines2,filtered_image, skel=cluster(imgray2,min_area=min_area, threshold=thr,  minLlength=mll, maxLgap=mlg,skel_thr=skel_thr)
    if lines2 is None or len(lines2)==0:
        lines2=[]
    else:
        lines2=np.array(lines2)*factor*0.5

    ###
    factor =2
    thr=70
    mll=10
    mlg=20
    min_area=30
    skel_thr=5
    
    imgray3=imgray.copy()
    imgray3=cv2.resize(imgray, (0, 0), fx=1/factor, fy=1/factor, interpolation=cv2.INTER_LINEAR)
    lines3, _, _ = cluster(imgray3,min_area=min_area, threshold=thr,  minLlength=mll, maxLgap=mlg,skel_thr=skel_thr)
    if lines3 is None or len(lines3)==0:
        lines3=[]
    else:
        lines3=np.array(lines3)*factor*0.5
    
    lines=[]
    if len(lines1)==0 and len(lines2)==0 and len(lines3)==0:
        return 0
    elif len(lines1)==0 and len(lines2)==0:
        lines=lines3
    elif len(lines1)==0 and len(lines3)==0:
        lines=lines2
    elif len(lines2)==0 and len(lines3)==0:
        lines=lines1
    elif len(lines1)==0:
        lines=np.concatenate((lines3,lines2),axis=0)
    elif len(lines2)==0:
        lines=np.concatenate((lines1,lines3),axis=0)
    elif len(lines3)==0:
        lines=np.concatenate((lines1,lines2),axis=0)
    else:
        lines=np.concatenate((lines1,lines2,lines3),axis=0)
    
    
    # lines= np.concatenate((lines1,lines2),axis=0)
    # lines= np.concatenate((lines,lines3),axis=0)
    if len(lines)!=0:
        s=measure_similarity(lines, thres_l=15, thres_c=0.9,shape=skel.shape)
    else:
        return 0
    
    # print(f'filtered_imge: {filtered_image.shape}, imgray: {imgray.shape}, skel: {skel.shape}')
    if args.draw_lines ==1: 
        new_filtered_img=filtered_image.copy()
        img1=filtered_image.copy()
        img2=filtered_image.copy()
        img3=filtered_image.copy()
        
        rgb3=create_color_pallete(1500)
        for idx, i in enumerate(lines):
            if idx not in s:
                cv2.line(filtered_image, (int(i[0][0])*factor, int(i[0][1])*factor), (int(i[0][2])*factor, int(i[0][3])*factor), rgb3[idx], 2)
                cv2.putText(filtered_image, str(idx), (int(i[0][0])*factor, int(i[0][1])*factor),fontFace=2,fontScale=1,color=rgb3[idx],thickness =2 )
            # cv2.line(new_skel, (int(i[0][0])*factor, int(i[0][1])*factor), (int(i[0][2])*factor, int(i[0][3])*factor), rgb3[idx], 2)

        for idx, i in enumerate(lines):
            cv2.line(new_filtered_img, (int(i[0][0])*factor, int(i[0][1])*factor), (int(i[0][2])*factor, int(i[0][3])*factor), rgb3[idx], 2)
            cv2.putText(new_filtered_img, str(idx), (int(i[0][0])*factor, int(i[0][1])*factor),fontFace=2,fontScale=1,color=rgb3[idx],thickness =2 )
        
        for idx, i in enumerate(lines1):
            cv2.line(img1, (int(i[0][0])*factor, int(i[0][1])*factor), (int(i[0][2])*factor, int(i[0][3])*factor), rgb3[idx], 2)
            cv2.putText(img1, str(idx), (int(i[0][0])*factor, int(i[0][1])*factor),fontFace=2,fontScale=1,color=rgb3[idx],thickness =2 )
        
        for idx, i in enumerate(lines2):
            cv2.line(img2, (int(i[0][0])*factor, int(i[0][1])*factor), (int(i[0][2])*factor, int(i[0][3])*factor), rgb3[idx], 2)
            cv2.putText(img2, str(idx), (int(i[0][0])*factor, int(i[0][1])*factor),fontFace=2,fontScale=1,color=rgb3[idx],thickness =2 )
            
        for idx, i in enumerate(lines3):
            cv2.line(img3, (int(i[0][0])*factor, int(i[0][1])*factor), (int(i[0][2])*factor, int(i[0][3])*factor), rgb3[idx], 2)
            cv2.putText(img3, str(idx), (int(i[0][0])*factor, int(i[0][1])*factor),fontFace=2,fontScale=1,color=rgb3[idx],thickness =2 )
    
        im_h= cv2.hconcat([filtered_image, cv2.cvtColor(skel.copy(), cv2.COLOR_BGR2RGB),new_filtered_img])
        im_h2 =cv2.hconcat([img1, img2,img3])
        final=cv2.vconcat([im_h,im_h2])
        cv2.imwrite(os.path.join(img_path,'houghline_'+im), final)
        # cv2.imwrite(os.path.join(img_path,'houghline_skel_'+im), skel)
    # print(len(lines), len(s), len(lines)-len(s))
    return len(lines)-len(s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder',type=str, default="ensemble_train")
    parser.add_argument('--label_csv',type=str,default="alopecia.csv")
    parser.add_argument('--save_path',type=str,default="hairline_count/")
    parser.add_argument('--start',type=int,default=0)
    parser.add_argument('--end',type=int,default=20000)
    parser.add_argument('--draw_lines',type=int, default=0)
    args = parser.parse_args()
    
    print(args)
    
    img_lst=pd.read_csv(args.label_csv)
    out_d={}
    for im in tqdm(img_lst["img_name"].to_numpy()[args.start:args.end]):
        out_d[im]=main(args,im)
    
    with open(os.path.join(args.save_path, f'{args.start}_{args.end}.csv'),'w') as f:
        w = csv.writer(f)
        w.writerow(out_d.keys())
        w.writerow(out_d.values())

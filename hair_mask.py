import cv2
import numpy as np
from math import atan2, cos, sin, sqrt, pi
import argparse 
import multiprocessing
import time,os
import asyncio
from tqdm import tqdm
from glob import glob
import warnings
warnings.filterwarnings("ignore")

# create gaussian kernel with k_size(filter size) and sigma
def gaussian_kernel(k_size, sigma):
    size = k_size//2
    y, x = np.ogrid[-size:size+1, -size:size+1]

    filter = 1/(2*np.pi * (sigma**2)) * np.exp(-1 *(x**2 + y**2) /(2*(sigma**2)))
    sum = filter.sum()
    filter /= sum
    return filter

# create copy of img with padding
def padding(img, k_size,is_mask=0):
    pad_size = k_size//2
    if is_mask==0:
        rows, cols, ch = img.shape
        res = np.zeros((rows + (2*pad_size), cols+(2*pad_size), ch), dtype=float)
    else:
        rows, cols = img.shape
        res = np.full((rows + (2*pad_size), cols+(2*pad_size)),255 ,dtype=float)
    
    if pad_size == 0:
        res = img.copy()
    else:
        res[pad_size:-pad_size, pad_size:-pad_size] = img.copy()
    return res

#static gaussian 'conditional' filtering
def gaussian_filtering(img, mask, k_size=3,sigma=1):
    """
    param
    img : input img
    k_size : kernel size
    sigma : standard deviation
    
    return
    filtered_img : gaussian filtered image returned
    """
    rows, cols, channels = img.shape
    m_rows, m_cols = mask.shape
    assert rows==m_rows and cols == m_cols
    
    filter = gaussian_kernel(k_size, sigma)
    pad_img = padding(img,k_size)
    pad_mask= padding(mask,k_size,1)
    
    filtered_img = np.zeros((rows, cols, channels), dtype=np.float32)
    
    #do "conditional" filtering
    #only account pixel outside seg mask(which correspond to scalp)
    for ch in range(0, channels):
        for i in range(rows):
            for j in range(cols):
                if pad_mask[i][j]!=0:
                    sub_matrix=pad_img[i:i+k_size, j:j+k_size, ch]
                    sub_mask_matrix=pad_mask[i:i+k_size, j:j+k_size]
                    avg=np.mean(sub_matrix[:,np.any(sub_mask_matrix==0,axis=0)])
                    sub_matrix=np.where(sub_mask_matrix==0,sub_matrix,avg)
                    product=filter * sub_matrix
                    product=np.sum(product)
                else:
                    product=pad_img[i][j][ch]
                filtered_img[i, j, ch] = product
    return filtered_img.astype(np.uint8)


def per_channel_cal(pad_img, pad_mask, ch, rows, cols, k_size,kerenl, ret_dict,error_set):
    out = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            if pad_mask[i][j]!=0:
                sub_matrix=pad_img[i:i+k_size, j:j+k_size, ch]
                sub_mask_matrix=pad_mask[i:i+k_size, j:j+k_size]
                avg=np.mean(sub_matrix[sub_mask_matrix==0])
                sub_matrix=np.where(sub_mask_matrix==0,sub_matrix,avg)
                pixval=np.sum(kerenl * sub_matrix)
            else:
                pixval=pad_img[i][j][ch]
            out[i][j] = pixval
            
            if pixval<1 or pixval>255 or np.isnan(pixval):
                error_set.append((i,j))
    ret_dict[ch]=out

# def shared_to_numpy(shared_arr, dtype, shape):
#     return np.frombuffer(shared_arr, dtype=dtype).reshape(shape)


# def create_shared_array(dtype, shape):
#     dtype = np.dtype(dtype)
#     cdtype = np.ctypeslib.as_ctypes_type(dtype)
#     shared_arr = multiprocessing.RawArray(cdtype, sum(shape))
#     arr = shared_to_numpy(shared_arr, dtype, shape)
#     return shared_arr, arr



def dynamic_gaussian_filtering_multi(img, mask, k_size=3,sigma=1):
    rows, cols, channels = img.shape
    m_rows, m_cols = mask.shape
    assert rows==m_rows and cols == m_cols
    
    ksz_lst=[i for i in range(k_size,k_size+100,10)]
    kernel_list=[gaussian_kernel(i, sigma) for i in ksz_lst]
    pad_img_lst=[padding(img,i) for i in ksz_lst]
    pad_mask_lst=[padding(mask,i,1) for i in ksz_lst]
   
    filtered_img = np.zeros((rows, cols, channels), dtype=np.float32)
    param=(ksz_lst,pad_img_lst,pad_mask_lst,kernel_list)
    
    filtered_img[:,:,0]=np.where(mask==0,img[:,:,0],-1)
    filtered_img[:,:,1]=np.where(mask==0,img[:,:,1],-1)
    filtered_img[:,:,2]=np.where(mask==0,img[:,:,2],-1)
    
    error_set=set()
    for i in range(rows):
        for j in range(cols):
            if -1 in filtered_img[i][j]:
                error_set.add((i,j))
    asyncio.run(running2(error_set,param,filtered_img))
    
    return filtered_img.astype(np.uint8)

async def running2(error_set,param,filtered_img):
    task_lst=[]
    for pts in error_set:
        task_lst.append(asyncio.create_task(each_pix_cal2(pts,param)))
    
    for task in task_lst:
        await task
    
    ret=[]
    for task in task_lst:
        ret.append( task.result())

    for ele in ret:
        filtered_img[ele[3]][ele[4]]=ele[:3]
    
    
async def each_pix_cal2(pts,rest_of_param):
    ksz_lst,pad_img_lst,pad_mask_lst,kernel_list= rest_of_param
    channels=3
    i,j=pts
    for idx,ksz in enumerate(ksz_lst):#,start=1
        # if idx==0:
        #     continue
        pixval=[0,0,0,i,j]
        for ch in range(0, channels):
            sub_matrix=pad_img_lst[idx][i:i+ksz, j:j+ksz, ch]
            sub_mask_matrix=pad_mask_lst[idx][i:i+ksz, j:j+ksz]
            filtered=sub_matrix[sub_mask_matrix==0]
            if len(filtered)==0:
                avg=0
            else:
                avg=np.mean(filtered)
            sub_matrix=np.where(sub_mask_matrix==0,sub_matrix,avg)
            pixval[ch]=np.sum(kernel_list[idx] * sub_matrix)
            
        if 255>=pixval[0]>=1 and 255>=pixval[1]>=1 and 255>=pixval[2]>=1:
            return pixval
    
    return [0,0,0,i,j]




#dynamic gaussian 'conditional' filtering
#basically increase filter size til caculated pixel value is non-zero
def dynamic_gaussian_filtering(img, mask, k_size=3,sigma=1):
    """
    param
    img : input img
    k_size : starting kernel size(it will grow til pixel value is non-zero)
    sigma : standard deviation
    
    return
    filtered_img : gaussian filtered image returned
    """
    rows, cols, channels = img.shape
    m_rows, m_cols = mask.shape
    assert rows==m_rows and cols == m_cols
    
    ksz_lst=[i for i in range(k_size,k_size+100,10)]
    kernel_list=[gaussian_kernel(i, sigma) for i in ksz_lst]
    pad_img_lst=[padding(img,i) for i in ksz_lst]
    pad_mask_lst=[padding(mask,i,1) for i in ksz_lst]
   
    filtered_img = np.zeros((rows, cols, channels), dtype=np.float32)

    #do "conditional" filtering
    #only account pixel outside seg mask(which correspond to scalp)
    #if pixel value is nan(or zero), added to error set
    
    # error_set=set()       
    # for ch in range(0, channels):
    #     for i in range(rows):
    #         for j in range(cols):       
    #             ksz=ksz_lst[0]
    #             if pad_mask_lst[0][i][j] !=0:
    #                 sub_matrix=pad_img_lst[0][i:i+ksz, j:j+ksz, ch]
    #                 sub_mask_matrix=pad_mask_lst[0][i:i+ksz, j:j+ksz]
    #                 avg=np.mean(sub_matrix[sub_mask_matrix==0])#sub_matrix[:,np.any(sub_mask_matrix==0,axis=0)]
    #                 sub_matrix=np.where(sub_mask_matrix==0,sub_matrix,avg)
    #                 pixval=np.sum(kernel_list[0] * sub_matrix)
    #             else:
    #                 pixval=pad_img_lst[0][i][j][ch]
    #             filtered_img[i, j, ch] = pixval
                
    #             if pixval<1 or pixval>255 or np.isnan(pixval):
    #                 error_set.add((i,j))
    
    # shared_arr, arr= create_shared_array(filtered_img.dtype, filtered_img.shape)
    # shared_arr[:]=filtered_img[:]
    
    #use multiprocessing (per rgb channel)
    with multiprocessing.Manager() as manager:
        shared_lst=manager.list()
        shared_out=manager.dict()
        procs=[]          
        for ch in range(0, channels):
            p=multiprocessing.Process(target= per_channel_cal, args=(pad_img_lst[0],pad_mask_lst[0],ch,rows,cols,ksz_lst[0],kernel_list[0],shared_out,shared_lst))
            p.start()
            procs.append(p)
        
        for p in procs:
            p.join()
        # print(len(shared_lst))
        error_set=set(shared_lst)
        for ch in range(0, channels):
            filtered_img[:,:,ch]=shared_out[ch]
    
    #for pix with nan, increase kernel size til pixel is non-zero
    # for _,pts in enumerate(error_set):
    #     i,j=pts
    #     for idx,ksz in enumerate(ksz_lst):#,start=1
    #         if idx==0:
    #             continue
    #         for ch in range(0, channels):
    #             sub_matrix=pad_img_lst[idx][i:i+ksz, j:j+ksz, ch]
    #             sub_mask_matrix=pad_mask_lst[idx][i:i+ksz, j:j+ksz]
    #             filtered=sub_matrix[sub_mask_matrix==0]
    #             if len(filtered)==0:
    #                 avg=0
    #             else:
    #                 avg=np.mean(filtered)
    #             sub_matrix=np.where(sub_mask_matrix==0,sub_matrix,avg)
    #             pixval=np.sum(kernel_list[idx] * sub_matrix)
    #             filtered_img[i, j, ch] = pixval
                
    #         if 255>=filtered_img[i][j][0]>=1 and 255>=filtered_img[i][j][1]>=1 and 255>=filtered_img[i][j][2]>=1:
    #             break

    param=(ksz_lst,pad_img_lst,pad_mask_lst,kernel_list)
    asyncio.run(running(error_set,param,filtered_img))
    
    return filtered_img.astype(np.uint8)
async def running(error_set,param,filtered_img):
    task_lst=[]
    for pts in error_set:
        task_lst.append(asyncio.create_task(each_pix_cal(pts,param)))
    
    for task in task_lst:
        await task
    
    ret=[]
    for task in task_lst:
        ret.append( task.result())

    for ele in ret:
        filtered_img[ele[3]][ele[4]]=ele[:3]
    
    
async def each_pix_cal(pts,rest_of_param):
    ksz_lst,pad_img_lst,pad_mask_lst,kernel_list = rest_of_param
    channels=3
    i,j=pts
    for idx,ksz in enumerate(ksz_lst):#,start=1
        # if idx==0:
        #     continue
        pixval=[0,0,0,i,j]
        for ch in range(0, channels):
            sub_matrix=pad_img_lst[idx][i:i+ksz, j:j+ksz, ch]
            sub_mask_matrix=pad_mask_lst[idx][i:i+ksz, j:j+ksz]
            filtered=sub_matrix[sub_mask_matrix==0]
            if len(filtered)==0:
                avg=0
            else:
                avg=np.mean(filtered)
            sub_matrix=np.where(sub_mask_matrix==0,sub_matrix,avg)
            pixval[ch]=np.sum(kernel_list[idx] * sub_matrix)
            
        if 255>=pixval[0]>=1 and 255>=pixval[1]>=1 and 255>=pixval[2]>=1:
            return pixval
    
    return [0,0,0,i,j]

def main(args):
    # Pre-process
    img_path = args.img_file
    img = cv2.imread(img_path)
    mask_path= args.mask_file
    mask=cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)

    # Apply threshold
    _, binary_map = cv2.threshold(mask, 1, 255, 0)
            
    # Get rid of noises (ex. big white spots that are not hair)
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)
    # Get CC_STAT_AREA component as stats[label, COLUMN]
    areas = stats[1:, cv2.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if areas[i] >= args.thr:   # Keep
            result[labels == i + 1] = 255
    mask=result

    # s=time.time()
    blur=dynamic_gaussian_filtering(img,mask,args.kernel_size,args.sigma)
    # print(time.time()-s)
    cv2.imwrite(args.save_file,blur)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_file', type=str, default= "/home/jerry0110/talmo/train_img/1407_A2LEBJJDE001258_1606104267519_4_LH.jpg")
    parser.add_argument('--mask_file', type=str, default= "/home/jerry0110/scalp_diagnosis/results2/1407_A2LEBJJDE001258_1606104267519_4_LH.png")
    parser.add_argument('--save_file', type=str, default= "/home/jerry0110/scalp_diagnosis/hair_masking/blur.png")
    parser.add_argument('--img_path', type=str,default= "/scratch/winston1214/talmo/train_img")
    parser.add_argument('--mask_path', type=str, default= '/home/jerry0110/scalp_diagnosis/results')
    parser.add_argument('--save_path', type=str,default='/home/jerry0110/scalp_diagnosis/hair_masking/re')
    parser.add_argument('--thr', type=int, default = 400)
    parser.add_argument('--kernel_size', type=int, default = 17)
    parser.add_argument('--sigma', type=int, default = 1)
    args = parser.parse_args()
    
    for im in tqdm(sorted(glob(os.path.join(args.mask_path,'*.png')))):
        im=im.split('/')[-1]
        args.img_file=os.path.join(args.img_path,im).replace(".png", '.jpg')
        args.save_file=os.path.join(args.save_path,im)
        args.mask_file=os.path.join(args.mask_path,im)
        main(args)

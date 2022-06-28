'''
Author      : poirotsy
Description : 
    1. 大图填充，每一张大图读取长宽，生成一个mask图片，mask记录id值:get-mask(img_path,label_csv)
    2. 大图滑窗
    3. 存储小图和小图的label到新地址
'''
from email.mime import base
from msilib import sequence
import random
import numpy as np
from collections import Counter
import pandas as pd
import cv2 as cv
from PIL import Image
import PIL.ImageOps 
from tqdm import tqdm
import os
import glob
from argparse import ArgumentParser
Image.MAX_IMAGE_PIXELS = 2800000000



def number_of_certain_probability(sequence, probability):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(sequence, probability):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item

def gen_box_id_mask(h,w,label_path):
    '''
    读取label_path的csv文件，并生成mask
    返回mask和csv得到的dataFrame数据
    '''
    label_data=pd.read_csv(label_path,encoding="ansi")
    label_data = pd.DataFrame(label_data)
    label_mask=np.zeros((h,w,3),dtype=np.uint8)
    poly_mask=np.zeros((h,w,3),dtype=np.uint8)

    for index,row in label_data.iterrows():
        xmin,ymin,width,height,id,coordstr=row['Xmin'],row['Ymin'],row['Width'],row['Height'],row['id'],row['Coords']
        line = coordstr.split()
        pt_num=len(line)
        pts = np.array(line, dtype=np.int32).reshape(pt_num//2,2)
        # print(pts.shape)
        cv.fillPoly(poly_mask, [pts], color=[1,1,1])
        for x in range(xmin,xmin+width+1):
            for y in range(ymin,ymin+height+1):
                if(label_mask[y,x,0]==0):
                    label_mask[y,x,0]=id
                elif(label_mask[y,x,1]==0):
                    label_mask[y,x,1]=id
                else:
                    label_mask[y,x,2]=id
    # cv.imwrite("D:\\0404detect-data\\test_mask.jpg",poly_mask)
    return label_mask,poly_mask,label_data

def get_mask_iou(crop_label):
    '''
    返回小图中背景比例
    '''
    h,w=crop_label.shape[0],crop_label.shape[1]
    bg_cnt=0
    for x in range(w):
        for y in range(h):
            if(crop_label[y,x].sum()==0):
                bg_cnt+=1
    return bg_cnt/(h*w)

def crop(cnt,crop_image,crop_label,label_csv,img_x1,img_x2,img_y1,img_y2,is_empty=False):
        '''
        存储小图、小图mask、小图label的csv文件
        cnt:小图个数,
        crop_image:小图数据,
        crop_label:小图mask数据,
        id_list:小图中存在的id,
        label_csv:csv读取后的dataframe
        img_x1,img_x2,img_y1,img_y2:小图在大图中的四个定点坐标
        '''
        id_list=list(Counter(crop_label.flatten()))
        # print(id_list)
        # 存储小图+mask
        image_dir=save_image_dir
        label_dir=save_label_dir
        if is_empty:
            image_dir=empty_image_dir
            label_dir=empty_label_dir
        image_name = os.path.join(image_dir,basename+"_"+str(cnt)+".jpg")
        cv.imwrite(image_name,crop_image)
        if crop_label is not None:
            label_name = os.path.join(label_dir,basename+"_"+str(cnt)+".jpg")
            crop_label=3*crop_label
            cv.imwrite(label_name,crop_label)
        # 计算小图中的label
        n_csv=label_csv
        n_csv=n_csv[n_csv.id.isin(id_list)]

        # n_csv.drop(n_csv["id"].isin(id_list)==False)
        for index,row in n_csv.iterrows():
            xmin,ymin,width,height=row["Xmin"],row["Ymin"],row["Width"],row["Height"]
            xmin+=stride//2
            ymin+=stride//2
            xmax=xmin+width
            ymax=ymin+height
            resx=[xmin,xmax,img_x1,img_x2]
            resy=[ymin,ymax,img_y1,img_y2]
            resx.sort()
            resy.sort()
            bx1,bx2=resx[1],resx[2]
            bx1=min(bx1,bx2)
            bx2=max(bx1,bx2)
            by1,by2=resy[1],resy[2]
            by1=min(by1,by2)
            by2=max(by1,by2)
            n_csv.loc[n_csv['Xmin']==row['Xmin'],'Xmin']=bx1-img_x1 # DEBUG:iterrow不改变原来的数据
            n_csv.loc[n_csv['Ymin']==row['Ymin'],'Ymin']=by1-img_y1
            n_csv.loc[n_csv['Width']==row['Width'],'Width']=bx2-bx1
            n_csv.loc[n_csv['Height']==row['Height'],'Height']=by2-by1
        # 存储csv
        csv_name = os.path.join(label_dir,basename+"_"+str(cnt)+".csv")
        n_csv.to_csv(csv_name,index =False,encoding="ansi")
        # print("END")
        

if __name__ == "__main__":
    parser = ArgumentParser(description="")
    parser.add_argument("-image_path",type=str)
    parser.add_argument("-label_path",type=str)
    parser.add_argument("-save_dir",type=str,default=r"./clip/")
    arg = parser.parse_args()
    image_path = arg.image_path
    label_path = arg.label_path
    save_image_dir = os.path.join(arg.save_dir,"image")
    save_label_dir = os.path.join(arg.save_dir,"label")
    empty_image_dir=os.path.join(arg.save_dir,"empty","image")
    empty_label_dir=os.path.join(arg.save_dir,"empty","label")
    stride = 600 
    target_size=(1200,1200)
    if not os.path.isdir(save_image_dir): os.makedirs(save_image_dir)
    if not os.path.isdir(save_label_dir): os.makedirs(save_label_dir)
    if not os.path.isdir(empty_image_dir): os.makedirs(empty_image_dir)
    if not os.path.isdir(empty_label_dir): os.makedirs(empty_label_dir)
    root_dir,filename = os.path.split(image_path)
    basename,filetype = os.path.splitext(filename)
    
    csv_list = glob.glob(label_path+'\\*.csv') 
    ALL_WHITE=255*1200*1200*3
    all_empty_samples=0
    all_pos_samples=0
    for item in tqdm(csv_list):
        root_dir,filename = os.path.split(item)
        basename,filetype = os.path.splitext(filename)
        image_path2=os.path.join(image_path,basename+".jpg")
        label_path2=item
        image = Image.open(image_path2)
        # image = PIL.ImageOps.invert(image)
        image = np.asarray(image)
        # 生成mask
        label,poly,csv_data = gen_box_id_mask(image.shape[0],image.shape[1],label_path2)
        
        cnt = 0
        target_w,target_h = target_size
        h,w = image.shape[0],image.shape[1]
        # 填充至整数
        new_w = (w//target_w)*target_w if (w//target_w == 0) else (w//target_w+1)*target_w
        new_h = (h//target_h)*target_h if (h//target_h == 0) else (h//target_h+1)*target_h
        image = cv.copyMakeBorder(image,0,new_h-h,0,new_w-w,cv.BORDER_CONSTANT,value=(255,255,255))
        label = cv.copyMakeBorder(label,0,new_h-h,0,new_w-w,cv.BORDER_CONSTANT,0)
        poly = cv.copyMakeBorder(poly,0,new_h-h,0,new_w-w,cv.BORDER_CONSTANT,0)

        # 填充1/2 stride长度的外边框
        h,w = image.shape[0],image.shape[1]
        new_w,new_h = w + stride,h + stride
        image = cv.copyMakeBorder(image,stride//2,stride//2,stride//2,stride//2,cv.BORDER_CONSTANT,value=(255,255,255))
        label = cv.copyMakeBorder(label,stride//2,stride//2,stride//2,stride//2,cv.BORDER_CONSTANT,0)
        poly = cv.copyMakeBorder(poly,stride//2,stride//2,stride//2,stride//2,cv.BORDER_CONSTANT,0)

        # crop
        h,w = image.shape[0],image.shape[1]#新的长宽
        
        empty_samples=0
        positive_samples=0
        print(w//stride-1)
        for i in range(w//stride-1):
            for j in range(h//stride-1):
                topleft_x = i*stride
                topleft_y = j*stride
                crop_image = image[topleft_y:topleft_y+target_h,topleft_x:topleft_x+target_w]
                crop_label = label[topleft_y:topleft_y+target_h,topleft_x:topleft_x+target_w]
                crop_poly = poly[topleft_y:topleft_y+target_h,topleft_x:topleft_x+target_w]

                if crop_image.shape[:2]!=(target_h,target_h):
                    print(topleft_x,topleft_y,crop_image.shape)

                if np.sum(crop_image)>=ALL_WHITE:
                    continue
                if np.sum(crop_poly)<=180:
                    empty_samples+=1
                    pass
                else:
                    crop(cnt,crop_image,crop_label,csv_data,topleft_x,topleft_x+target_w,topleft_y,topleft_y+target_h)
                    cnt+=1
                    positive_samples+=1
        
        if empty_samples/positive_samples>5:
            # empty太多
            sample_rate=3*positive_samples/empty_samples
            new_empty_samples=0
            ransequence=[0,1]
            probability=[1-sample_rate,sample_rate]
            for i in range(w//stride-1):
                for j in range(h//stride-1):
                    topleft_x = i*stride
                    topleft_y = j*stride
                    crop_image = image[topleft_y:topleft_y+target_h,topleft_x:topleft_x+target_w]
                    crop_label = label[topleft_y:topleft_y+target_h,topleft_x:topleft_x+target_w]
                    crop_poly = poly[topleft_y:topleft_y+target_h,topleft_x:topleft_x+target_w]

                    if crop_image.shape[:2]!=(target_h,target_h):
                        print(topleft_x,topleft_y,crop_image.shape)

                    if np.sum(crop_image)>=ALL_WHITE:
                        continue
                    if np.sum(crop_poly)==0:
                        flag=number_of_certain_probability(ransequence,probability)
                        if(flag==1):
                            crop(cnt,crop_image,crop_label,csv_data,topleft_x,topleft_x+target_w,topleft_y,topleft_y+target_h,True)
                            cnt+=1
                            new_empty_samples+=1
            empty_samples=new_empty_samples
        else:
            for i in range(w//stride-1):
                for j in range(h//stride-1):
                    topleft_x = i*stride
                    topleft_y = j*stride
                    crop_image = image[topleft_y:topleft_y+target_h,topleft_x:topleft_x+target_w]
                    crop_label = label[topleft_y:topleft_y+target_h,topleft_x:topleft_x+target_w]
                    crop_poly = poly[topleft_y:topleft_y+target_h,topleft_x:topleft_x+target_w]

                    if crop_image.shape[:2]!=(target_h,target_h):
                        print(topleft_x,topleft_y,crop_image.shape)

                    if np.sum(crop_image)>=ALL_WHITE:
                        continue
                    if np.sum(crop_poly)==0:
                        crop(cnt,crop_image,crop_label,csv_data,topleft_x,topleft_x+target_w,topleft_y,topleft_y+target_h,True)
                        cnt+=1
        all_empty_samples+=empty_samples
        all_pos_samples+=positive_samples
        
    print("pos:empty={}".format(all_pos_samples/all_empty_samples))



    
 # python getclip.py -image_path d:/1data/public -label_path d:/1Label/public

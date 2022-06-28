import os
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
import cv2 as cv
import random
from transform import PolyRR
Image.MAX_IMAGE_PIXELS = 3000000000
MYCLASSES = ('Flush', 'Urinal', 'SquattingPan', 'basin', 'Sink', 'MopPool',
             'WM', 'DL', 'ShowerRoom', 'Shower-cham', 'Bathtub', 'Shower')
angle_list=(9,18,27,36,45,54,63,72,81,90,-9,-18,-27,-36,-45,-54,-63,-72,-81) #[0,18]

def getRandomAngle(num):
    res=random.sample(range(0, 18), num)
    return res

def writeTXT(ptlist,catid,xmin,ymin,path,flag=False):
        txtstr=""
        if flag:
            xmin=2
            ymin=2
        x1=ptlist[1][0]-ptlist[0][0]  # 1:i-i-1
        y1=ptlist[1][1]-ptlist[0][1]
        x2=ptlist[2][0]-ptlist[1][0]
        y2=ptlist[2][1]-ptlist[1][1]
        draw_order=x1*y2-x2*y1#组成向量（pi-pi-1）和（pi+1-pi），判断顺时针
        if draw_order>0:
            for i in range(4):
                txtstr+="{} {} ".format(ptlist[i][0]-xmin+2,ptlist[i][1]-ymin+2)
        else:
            for i in range(4):
                txtstr+="{} {} ".format(ptlist[3-i][0]-xmin+2,ptlist[3-i][1]-ymin+2)
        txtstr+="{} 0\n".format(MYCLASSES[catid])
       
        f=open(path,'w')
        f.write(txtstr)
        f.close()
# rotatePoly(crop,ptlist,(xmin-2,ymin-2),catid,angle[i],basename+"_"+str(id)+"_"+str(i))
def rotatePoly(img,coords,origin,catid,angle,filename):
    txtPath='D:\\0404detect-data\\0404detection-data\\dota\\train\\labelTxt\\'+filename+'.txt'
    imgPath="D:\\0404detect-data\\0404detection-data\\dota\\train\\images\\"+filename+".png"
    checkPath='D:\\0404detect-data\\0404detection-data\\dota\\check\\'+filename+".png"
    xmin,ymin=origin
    angle=angle_list[angle]
    pcoords=np.zeros((4,2),dtype=float)
    for i in range(4):
        pcoords[i,0]=coords[i][0]-xmin
        pcoords[i,1]=coords[i][1]-ymin
    rr=PolyRR(pcoords)
    results={}
    results['img']=img
    results['img_shape']=img.shape
    result=rr(results,angle)

    cv.imwrite(imgPath,result['img'])
    crop=result['img']
    
    pts=result['coords']
    cv.polylines(crop,[pts.astype(np.int)],isClosed=True,color=(0,0,255))
    cv.imwrite(checkPath,crop)
    ptlist=[]
    for i in range(4):
        ptlist.append([pts[i,0],pts[i,1]])
    writeTXT(ptlist,catid,0,0,txtPath,True)

def gen_txt(label_path,basename):
    '''
    读取label_path的csv文件，生成dota格式
    '''
    img_path = "D:\\0404detect-data\\0404detection-data\\image\\"+basename+".jpg"
    save_dir = "D:\\0404detect-data\\0404detection-data\\dota\\train\\images\\"+basename

    img = Image.open(img_path)
    image = np.asarray(img).copy()
    # print(image.shape)
    HH = image.shape[0]
    WW = image.shape[1]

    label_data=pd.read_csv(label_path,encoding="ansi")
    label_data = pd.DataFrame(label_data)
    
    
    for index,row in label_data.iterrows():

        xmin,ymin,width,height,catid,coordstr,id=row['Xmin'],row['Ymin'],row['Width'],row['Height'],row['Category'],row['Coords'],row['id']

        # 顶点判断，删除多余顶点
        line = coordstr.split()
        pt_num=len(line)//2
        pts = np.array(line, dtype=np.int32).reshape(pt_num,2)
        npts=0
        ptlist=[]
        for i in range(pt_num):
            front = (i+pt_num-1)%pt_num
            back = (i+pt_num+1)%pt_num
            res=abs((pts[front,1]-pts[i,1])*(pts[i,0]-pts[back,0])-(pts[front,0]-pts[i,0])*(pts[i,1]-pts[back,1]))
            if res>1e-4:
                ptlist.append([pts[i,0],pts[i,1]])
                # txtstr+="\n{} {} ".format(pts[i,0],pts[i,1])
                npts+=1
        assert npts==4,basename+"_"+str(id)

        txtPath='D:\\0404detect-data\\0404detection-data\\dota\\train\\labelTxt\\'+basename+"_"+str(id)+'.txt'
        writeTXT(ptlist,catid,xmin,ymin,txtPath)

        save_path = save_dir+"_"+str(id)+".png"
        y1 = ymin-2 if ymin-2 >= 0 else 0
        y2 = ymin+height+3 if ymin+height+3 <= HH else HH
        x1 = xmin-2 if xmin-2 >= 0 else 0
        x2 = xmin+width+3 if xmin+width+3 <= WW else WW
        crop = image[y1:y2,x1:x2]
        if crop.size == 0:
            print(save_path)
            continue
        cv.imwrite(save_path,crop)
       

        if catid==0 or catid==3 or catid==7:
            angle=random.randint(0,18)
            rotatePoly(crop,ptlist,(xmin-2,ymin-2),catid,angle,basename+"_"+str(id)+"_1")
        elif catid==4 or catid==6 or catid==11:
            num=5
            angle=getRandomAngle(num)
            
            for i in range(num):
                rotatePoly(crop,ptlist,(xmin-2,ymin-2),catid,angle[i],basename+"_"+str(id)+"_"+str(i))
        else:
            num=10
            angle=getRandomAngle(num)
            
            for i in range(num):
                rotatePoly(crop,ptlist,(xmin-2,ymin-2),catid,angle[i],basename+"_"+str(id)+"_"+str(i))


if __name__ == "__main__":
    csv_list = glob.glob('D:\\0404detect-data\\0404detection-data\\label\\*.csv')
    k=0
    for item in tqdm(csv_list):
        k+=1
        if k<91:
            continue
        root_dir,filename = os.path.split(item)
        basename,filetype = os.path.splitext(filename)
        gen_txt(item,basename)

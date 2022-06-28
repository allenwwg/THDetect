import os
import pandas as pd                         #导入pandas包
import glob
import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
Image.MAX_IMAGE_PIXELS = 2000000000

rootstr="D:\\THdetection\\label\\"
outpath="D:\\THdetection\\output2.csv"
class_list=[]

def merge_csv_into_one():
    """合并文件
    """
    csv_list = glob.glob('D:\\THdetection\\label\\*.csv')
    print(u'共发现%s个CSV文件'% len(csv_list))
    print(u'正在处理............')
    for i in csv_list:
        fr = open(i,'r').read()
        with open(outpath,'a') as f:
            f.write(fr)
    print(u'合并完毕!')

def gen_dict():
    """总文件output.csv生成字典
    """

    f = open(outpath, encoding="ansi")
    data = pd.read_csv(f)
    a=data["Label"].drop_duplicates().values.tolist()
    a.sort()

    # print(a)
    b=[0,3,3,3,0,1,3,3,10,8,3,3,3,3,3,3,0,1,1,0,5,3,4,4,4,4,3,6,8,10,10,11,8,8,8,9,9,3,3,2,3,3] # 根据a打印结果自定义b
    dict_all = dict(zip(a,b)) # 得到字典
    print(len(a))

    print(dict_all)
    return dict_all

def replace_label(dict_all):
    """
    每一个文件添加列替换数据
    """
    csv_list = glob.glob('D:\\THdetection\\label\\*.csv')
    print(u'共发现%s个CSV文件'% len(csv_list))
    print(u'正在处理............')
    for i in csv_list:
        data_i=pd.read_csv(i,encoding="ansi")
        y=1
        def add(x=1):
            try:
                add.sum += x
            except AttributeError:
                add.sum = x
            return add.sum

        data_i["Category"] = data_i["Label"].apply(lambda t: dict_all[t])
        data_i["id"] = data_i["Label"].apply(lambda t: add(y))
        data_i.to_csv(i,index =False,encoding="ansi")
    
def get_max_box():
    """ 
    获取最大box宽度
    """
    csv_list = glob.glob('D:\\THdetection\\label\\*.csv')
    print(u'共发现%s个CSV文件'% len(csv_list))
    print(u'正在处理............')
    max_w=0
    max_h=0
    for i in csv_list:
        data_i=pd.read_csv(i,encoding="ansi")
        w=data_i["Width"].max()
        h=data_i["Height"].max()
        max_w=max(max_w,w)
        max_h=max(max_h,h)

    print('{} and {}'.format(max_w,max_h))

def extract_clip(num,my_counter):
    """
    将每张图中的box图像部分提取存储，统计各类别个数
    """
    print(num)
    label_path="D:\\THdetection\\label\\"+str(num)+".csv"
    img_path="D:\\THdetection\\image\\"+str(num)+".jpg"
    save_dir="D:\\THdetection\\check\\"+str(num)
    is_exist=os.path.exists(save_dir)
    if not is_exist:
        os.makedirs(save_dir)
    img=Image.open(img_path)
    image = np.asarray(img)
    # print(image.shape)

    label_data=pd.read_csv(label_path,encoding="ansi")
    label_data = pd.DataFrame(label_data)

    for index,row in label_data.iterrows():
        cate,xmin,ymin,width,height,id=row['Category'],row['Xmin'],row['Ymin'],row['Width'],row['Height'],row['id']
        my_counter[cate]+=1
        save_path=save_dir+"\\"+str(cate)+"_"+str(id)+".jpg"
        
        crop=image[ymin-1:ymin+height+1,xmin-1:xmin+width+1]
        if crop.size==0:
            print(save_path)
            continue
        cv.imwrite(save_path,crop)
    return my_counter



def showFigure():
    """
    绘图展示数据，每类数据点以不同的颜色显示
    """
    csv_list = glob.glob('D:\\THdetection\\label\\*.csv')
    box_list={}
    for i in range(12):
        box_list[i]=[]
    print(u'共发现%s个CSV文件'% len(csv_list))
    print(u'正在处理............')

    for i in csv_list:
        label_data=pd.read_csv(i,encoding="ansi")
        label_data = pd.DataFrame(label_data)

        for index,row in label_data.iterrows():
            cate,width,height=row['Category'],row['Width'],row['Height']
            box_list[cate].append([width,height])
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)   #界面只需显示一个视图
    ax.set_title('box size')  #视图名称，这里简单统一定这个名称吧
    plt.xlabel('Width')    #坐标轴名称
    plt.ylabel('Height')

    colors = ['r','g','b','c','m','y','k','0.5','#FF00FF','#945a5a','#6a7ee3','#947847']   #定义显示的颜色b为blue，k为black
    labels=['FlushToilet','Urinal','SquattingPan','Washbasin','Sink','MopPool','Tap','WashMachine','ShowerRoom','ShowerRoom_chamber','Bathtub','Shower']
    for i in range(12):
        temp=np.zeros((len(box_list[i]),2),np.int32)
        idx=0
        for item in box_list[i]:
            temp[idx,0]=item[0]
            temp[idx,1]=item[1]
            idx+=1
        ax.scatter(temp[:,0],temp[:,1], marker='o', color=colors[i], label=labels[i], s=3)  #在视图中的显示方式

    plt.legend(loc = 'upper right')   #图例显示位置
    plt.show()

if __name__ == "__main__":
    # # 合并
    # merge_csv_into_one()
    # # 生成字典 
    # dict_all=gen_dict()
    # replace_label(dict_all) # 替换
    
    # get_max_box()
    """
    # 统计类别个数；
    my_counter=[0,0,0,0,0,0,0,0,0,0,0,0]
    csv_list = glob.glob('D:\\THdetection\\label\\*.csv')
    for i in csv_list:
        root_dir,filename = os.path.split(i)
        basename,filetype = os.path.splitext(filename)
        mycounyter=extract_clip(basename,my_counter)

    print(my_counter)
    # gt分布：[1424, 219, 155, 1310, 443, 211, 388, 0, 554, 180, 131, 6]

    """
    showFigure()  # 显示box大小分布范围

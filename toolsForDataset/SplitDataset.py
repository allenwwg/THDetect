# coding=utf-8
import os, random, shutil


if __name__ == '__main__':
    train_path = './dota/train/images'  # 最开始train的文件夹路径
    train_label='./dota/train/labelTxt'
    test_path = './dota/trainval/images'  # 移动到新的文件夹路径
    test_label='./dota/trainval/labelTxt'
    ratio = 0.111111  # 抽取比例  ******自己改*******
    
    pathDir = os.listdir(train_path)  # 取图片的原始路径
    filenumber = len(pathDir)
    picknumber = int(filenumber * ratio)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    for name in sample:
        basename,filetype = os.path.splitext(name)
        shutil.move(os.path.join(train_path, name), os.path.join(test_path, name))
        # shutil.move(os.path.join(train_label, name), os.path.join(test_label, name))
        shutil.move(os.path.join(train_label, basename+".txt"), os.path.join(test_label, basename+".txt"))
    print("Successfully split the train and val!")

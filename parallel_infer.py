import logging
import logging.handlers
# from msilib.schema import Patch
import time
import os
import random
import math
import glob
from tkinter import DoubleVar
from tokenize import Double
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm
from collections import Counter
import pandas as pd
import cv2 as cv
import numpy as np
import cupy as cp
import pickle
import torch
from argparse import ArgumentParser
from mmdet.apis import init_detector, inference_detector
import mmrotate
from mmdet.core.visualization import imshow_det_bboxes
import mmcv
import acceleration
"""
把merge时间缩小、genbox时间缩小
也许可以写c++扩展
"""
Image.MAX_IMAGE_PIXELS = 3000000000
MYCLASSES = ('Flush', 'Urinal', 'SquattingPan', 'basin', 'Sink', 'MopPool',
             'WM', 'DL', 'ShowerRoom', 'Shower-cham', 'Bathtub', 'Shower')
# MYCLASSES =  ('坐便器','小便器','蹲便器','洗脸盆','洗涤槽','拖把池','洗衣机','地漏','淋浴房','转角房','浴缸','花洒')


def get_clips(image, img_scale, step):
    """
    大图从左到右、从上到下滑窗切成块
    """
    time_0 = time.time()

    clip_list = []
    target_size = (img_scale, img_scale)  # 滑窗大小
    center_size = (step, step)
    cnt = 0
    target_w, target_h = target_size
    center_w, center_h = center_size
    h, w = image.shape[0], image.shape[1]
    # 填充至整数
    new_w = (w // center_w + 1) * center_w
    new_h = (h // center_h + 1) * center_h
    image = cv.copyMakeBorder(
        image,
        0,
        new_h - h,
        0,
        new_w - w,
        cv.BORDER_CONSTANT,
        value=(255, 255, 255))

    # 填充1/2 stride长度的外边框
    stride = img_scale - step
    h, w = image.shape[0], image.shape[1]
    new_w, new_h = w + stride, h + stride
    image = cv.copyMakeBorder(
        image,
        stride // 2,
        stride // 2,
        stride // 2,
        stride // 2,
        cv.BORDER_CONSTANT,
        value=(255, 255, 255))
    # crop
    h, w = image.shape[0], image.shape[1]  # 新的长宽
    ALL_WHITE=img_scale*img_scale*3*255
    INFO_LIST=[]
    for j in range(h // step):
        for i in range(w // step):
            topleft_x = i * step
            topleft_y = j * step
            crop_image = image[topleft_y:topleft_y + target_h,
                               topleft_x:topleft_x + target_w].copy()

            if crop_image.shape[:2] != (target_h, target_h):
                logger.warning(topleft_x, topleft_y, crop_image.shape)

            else:
                if np.sum(crop_image) < ALL_WHITE:
                    clip_list.append(crop_image)
                    INFO_LIST.append([j,i,cnt])
                # cv.imwrite("d:\\test\\"+str(cnt)+".jpg",crop_image)
                cnt += 1
    logger.info("patch number:{}".format(cnt))
    time_1 = time.time()
    logger.info('clips cost: {}'.format(time_1 - time_0))
    return clip_list,INFO_LIST


def create_zeros_png(image_w, image_h, img_scale, step):
    '''Description:
        0. 先创造一个空白图像，将滑窗预测结果逐步填充至空白图像中；
        1. 填充右下边界，使原图大小可以杯滑动窗口整除；
        2. 膨胀预测:预测时，对每个(1200,1200)窗口，每次只保留中心(1020,1020)区域预测结果，每次滑窗步长为1020，使预测结果不交叠；
    '''
    margin = (img_scale - step) // 2
    new_h, new_w = (image_h // step + 1) * step, (image_w // step + 1) * step  #填充右边界
    zeros = (new_h, new_w, 3)
    # print(zeros)
    zeros = np.zeros(zeros, np.uint8)
    return zeros

def clip_for_inference(model,
                       clip_list,
                       info_list,
                       jpg_shape,
                       img_scale,
                       step,
                       img_name,
                       score_thres=0.9,
                       device=None,
                       paint_flag=True,
                       result_path=None,
                       scale_para=1):
    """
    将clip_list中的clip逐个推理，并合并结果
    """
    image_h, image_w = jpg_shape
    predict_png = create_zeros_png(image_w, image_h, img_scale, step)
    h = predict_png.shape[0]  # 补整后
    w = predict_png.shape[1]
    mask_category = np.zeros((h, w), np.uint8)  # 类别
    mask_score = np.zeros((h, w), np.uint16)  # 分数
    mask_id = np.zeros((h, w), np.uint16)  # box_id
    dict_for_clips = {}

    clip_cnt = 0
    box_cnt = 0
    margin = (img_scale - step) // 2

    infer_cnt = 0
    infer_time_sum = 0
    time_start_inf = time.time()
    iters=math.ceil(len(info_list)/8.0)
    SAMPLES=8
    result=[]
    for i in range(iters):
        left=i*SAMPLES
        right=i*SAMPLES+SAMPLES if i*SAMPLES+SAMPLES<=len(info_list) else len(info_list)
        result_tmp=inference_detector(model,clip_list[left:right])
        result.extend(result_tmp)
    time_end_inf = time.time()
    infer_time_sum += (time_end_inf - time_start_inf)
    logger.info("inference time [for clips]:{}".format(infer_time_sum))
    
    # result = inference_detector(model,clip_list)
    infer_cnt=len(info_list)
    pm=acceleration.PatchManager(img_scale,step,h,w)
    
    for info_index in range(infer_cnt):
        j,i,clip_index_in_origin = info_list[info_index]
        topleft_x = i * step
        topleft_y = j * step
        bboxes, labels, scores = analyse_result(result[info_index], score_thres)
        pm.add_patch(bboxes,labels,j,i)
        predict_png[topleft_y:topleft_y + step, topleft_x:topleft_x +
                        step] = clip_list[info_index][margin:margin + step,
                                                    margin:margin +
                                                    step]  # 原图拼接:用来直接看检测结果对不对，可以不要的
        # temp_cat, temp_score, temp_id, box_cnt, box_dict = get_inference_block_mask(
        #             bboxes, labels, scores, img_scale, box_cnt,topleft_y,topleft_x)  # 每一个图块内处理
        # mask_category[topleft_y:topleft_y + step, topleft_x:topleft_x + step] = temp_cat[margin:margin + step, margin:margin + step].copy()  # 赋值中心区域
        # mask_score[topleft_y:topleft_y + step, topleft_x:topleft_x + step] = temp_score[margin:margin + step, margin:margin + step].copy()
        # mask_id[topleft_y:topleft_y + step, topleft_x:topleft_x + step] = temp_id[margin:margin + step, margin:margin + step].copy()
        # dict_for_clips.update(box_dict)

        # mask_category, mask_id, box_dict = merge_clip_top_and_left(
        #             (j, i), mask_category, mask_id, step, 0.1,
        #             dict_for_clips)  # 合并当前图块与左&上的边缘部分
        # dict_for_clips.update(box_dict)
    dict_for_clips=pm.getInfo()
    print(len(dict_for_clips))
    logger.info("inference patch number:{}".format(infer_cnt))
    time_end_inf = time.time()
    infer_time_sum += (time_end_inf - time_start_inf)
    logger.info("inference time all:{}".format(time_end_inf - time_start_inf))
    fps_patch.append(infer_cnt)
    fps.append(infer_time_sum)
    # FIXME:导出box结果(只是对每一个box求覆盖点最小和最大坐标)
    # [1,box_cnt] box-id

    time_start_inf = time.time()
    dict_for_box = {}
    cp_mask_id=cp.asarray(mask_id).flatten()
    id_list = list(cp.bincount(cp_mask_id).get())

    time_end_inf = time.time()
    logger.info("flatten time:{}".format(time_end_inf - time_start_inf))
    
    time_start_inf = time.time()
    for i in range(len(id_list)): # 更新box的信息
        id = i
        if id == 0 or id_list[i]==0:
            continue
        if dict_for_clips.get(id) is not None:
            dict_for_box[id] = dict_for_clips[id]

    dict_for_box=dict_for_clips
    id_list_len=pm.getBoxCnt()
    for i in range(id_list_len):
        id_list.append(i)
    time_end_inf = time.time()
    logger.info("gen box time:{}".format(time_end_inf - time_start_inf))

    predict_png, final_box_list = process_hbb(predict_png, id_list, dict_for_box,
                                            paint_flag, scale_para,margin)
    predict_png = predict_png[:image_h, :image_w].copy()  # 去除右下边界


    cv.imwrite(result_path + "compound_img_{}.jpg".format(img_name), predict_png)
    return final_box_list



def analyse_result(result, score_thr=0.8):
    """
    将model推理结果转化为可用数据，并根据score_thr筛选box
    """
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)

    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # draw segmentation masks
    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)

    assert bboxes.shape[1] == 5 or bboxes.shape[1] == 6
    scores = bboxes[:, -1]
    inds = scores > score_thr
    bboxes = bboxes[inds, :]
    labels = labels[inds]
    scores = bboxes[:, -1]

    return bboxes, labels, scores


def paint_chinese_opencv(im, textstr, position, fontsize, color):
    #opencv输出中文
    img_PIL = Image.fromarray(cv.cvtColor(
        im, cv.COLOR_BGR2RGB))  # 图像从OpenCV格式转换成PIL格式
    font = ImageFont.truetype('simhei.ttf', fontsize, encoding="utf-8")
    #color = (255,0,0) # 字体颜色
    #position = (100,100)# 文字输出位置
    draw = ImageDraw.Draw(img_PIL)
    draw.text(
        position, textstr, font=font,
        fill=color)  # PIL图片上打印汉字 # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
    img = cv.cvtColor(np.asarray(img_PIL), cv.COLOR_RGB2BGR)  # PIL图片转cv2 图片
    return img


def process_hbb(img, id_cnt_list, dict_box, flag=True, scale_para=1,margin=90):
    """
    visualize labels and boxxes
    """
    time_vis_start = time.time()
    bbox_color = [(128,30, 225),(229,27,152),
                (230,26,60), (231,157,25), (206,230,26), 
                (80,229,27), (109,147,116), (0,144,191), 
                (29,80,163), (92,171,250), (119,119,223), 
                (0,0,192)
                ]
    text_color = (255, 0, 0)
    width, height = img.shape[1], img.shape[0]
    font = cv.FONT_HERSHEY_SIMPLEX  # 使用默认字体
    cnt = len(id_cnt_list)
    box_list = []
    for i in range(len(id_cnt_list)):
        id = i
        if id == 0 or id_cnt_list[i]==0:
            continue

        if dict_box.get(id) is None:
            continue
        x1, y1, x2, y2, label, score = dict_box[id]
        if x2 - x1 <= 10 / scale_para or y2 - y1 <= 10 / scale_para:  # 过于细长的box跳过
            continue
        # if label==7:
        #     if x2-x1<=10/scale_para or y2-y1<=10/scale_para:  # 过于细长的box跳过
        #         continue
        if label == 8 or label == 9:    # 筛选一下淋浴房的尺寸
            if x2 - x1 <= 160 / scale_para or y2 - y1 <= 160 / scale_para:
                continue
            if x2 - x1 >= 580 / scale_para or y2 - y1 >= 580 / scale_para:
                continue
        # if label==10:
        #     if x2-x1<=80/scale_para or y2-y1<=80/scale_para:
        #         continue
        #     if x2-x1>=520/scale_para or y2-y1>=520/scale_para:
        #         continue
        # if label==3:
        #     if x2-x1<=30/scale_para or y2-y1<=30/scale_para:
        #         continue
        # if label==4:
        #     if x2-x1<=100/scale_para or y2-y1<=100/scale_para:
        #         continue
        # if label==5:
        #     if x2-x1<=70/scale_para or y2-y1<=70/scale_para:
        #         continue
        # if label==6:
        #     if x2-x1<=100/scale_para or y2-y1<=100/scale_para:
        #         continue

        box_list.append([
            (x1-margin) * scale_para, (y1-margin) * scale_para, (x2 - x1) * scale_para,
            (y2 - y1) * scale_para, label, score, id
        ])  # 可能使用缩放，因此还原到原图中的scale [x1,y1,width,height,label_id,score,id]
        # 绘制AABB
        y3 = y1 + y2
        x3=x1+x2
        cv.rectangle(img, (x1-margin, y1-margin), (x2-margin, y2-margin), bbox_color[label], thickness=2)
        # img = paint_chinese_opencv(img,MYCLASSES[label],(x3//2-margin,y3//2-margin),12,bbox_color)    # 绘制中文
        cv.putText(img, MYCLASSES[label]+ ":{:.2f}".format(score)+" :{}".format(id),
                   (x3//2-margin, y3 // 2-margin), font, 0.7, text_color, 1)# 绘制英文

    time_vis_end = time.time()
    logger.info("label_visualize time:{}".format(time_vis_end - time_vis_start))
    return img, box_list

def predict_obb(r_model,img,box_data,scale_para):
    # box_data: list of [x1,y1,width,height,label_id,score,id]
    new_box_data=[]
    WIDTH=img.shape[1]
    HEIGHT=img.shape[0]
    logger.info("hbb number:{}".format(len(box_data)))
    for hbb in box_data:
        x0,y0,w,h,label_id,score,id = hbb
        x1=x0//scale_para-5 if x0//scale_para-5>=0 else x0//scale_para
        y1=y0//scale_para-5 if y0//scale_para-5>=0 else y0//scale_para
        x2=(x0+w)//scale_para+5 if (x0+w)//scale_para+5<WIDTH else (x0+w)//scale_para
        y2=(y0+h)//scale_para+5 if (y0+h)//scale_para+5<HEIGHT else (y0+h)//scale_para
        crop_img=img[y1:y2,x1:x2]
        result = inference_detector(r_model, crop_img)
        bboxes, labels, scores=analyse_result(result, score_thr=0.8)
        test_flag=True

        if bboxes.shape[0]==0 or label_id==7:
            p1 = [x0//scale_para,y0//scale_para]
            p2 = [(x0+w)//scale_para,y0//scale_para]
            p3 = [(x0+w)//scale_para,(y0+h)//scale_para]
            p4 = [x0//scale_para,(y0+h)//scale_para]
            new_box_data.append([p1,p2,p3,p4,label_id,score,id])
            continue
        # get obb
        max_index = np.where(bboxes==np.max(bboxes[:, -1]))
        bbox=bboxes[max_index[0],:]
        xc, yc, w2, h2, ag, s = bbox[0]
        wx, wy = w2 / 2 * np.cos(ag), w2 / 2 * np.sin(ag)
        hx, hy = -h2 / 2 * np.sin(ag), h2 / 2 * np.cos(ag)
        p1 = [xc - wx - hx +x1, yc - wy - hy +y1]
        p2 = [xc + wx - hx +x1, yc + wy - hy +y1]
        p3 = [xc + wx + hx +x1, yc + wy + hy +y1]
        p4 = [xc - wx + hx +x1, yc - wy + hy +y1]
        new_box_data.append([p1,p2,p3,p4,label_id,score,id])
    return new_box_data

def outputjson(data, img_name, json_path,scale_para):
    """
        output json
        """
    # out_file="d:\\img_label\\mjy\\{}.json".format(img_name)
    out_file = json_path + "{}.json".format(img_name)
    output = []
    
    for box_index, box_data in enumerate(data):
        p1,p2,p3,p4, category_id, score, box_id = box_data
        bbox_coords = [p1[0]*scale_para,p1[1]*scale_para,p2[0]*scale_para,p2[1]*scale_para,
                        p3[0]*scale_para,p3[1]*scale_para,p4[0]*scale_para,p4[1]*scale_para]
        output.append({
            'box_id': box_id,
            'class_index': category_id,
            'class_name': MYCLASSES[category_id],
            'bbox': bbox_coords,
            'score': score
        })
    coco_format_json = dict(annotations=output)
    mmcv.dump(output, out_file)

if __name__ == "__main__":
    parser = ArgumentParser(description="")
    parser.add_argument("--image_name", type=str)  # 图片名称（不含后缀）：可以用*来检测目录下所有图片
    parser.add_argument("--data_dir", type=str, default=None)  # 图片存放目录（末尾带\）
    parser.add_argument(
        "--res_dir", type=str, default=None)  # 检测后合成的图片存放目录（末尾带\）
    parser.add_argument(
        "--json_dir", type=str, default=None)  # 检测结果json文件存放目录（末尾带\）
    parser.add_argument("--score_thres", type=float, default=0.1)  # 阈值筛选
    parser.add_argument("--stride", type=int, default=1020)  # 裁切步长
    parser.add_argument("--scale", type=int, default=1)  # 缩小倍数
    arg = parser.parse_args()
    th_img_path = "d:\\THdetection\\image\\" + arg.image_name + ".jpg" if arg.data_dir == None else arg.data_dir + arg.image_name + ".jpg"
    th_json_path = "d:\\THdetection\\label\\" if arg.json_dir == None else arg.json_dir
    th_res_path = "d:\\THdetection\\resimg\\" if arg.res_dir == None else arg.res_dir
    """
    1. cmd环境激活：
        cd ../..
        d:
        cd D:\ProgramData\Anaconda3\mmdetection\myutils
        activate openmmlab
    2. 单例：
        python parallel_infer.py --image_name 1212 --score_thres 0.5
    3. 多例：
        python parallel_infer.py -image_name '*'  -data_dir D:\\THdetection\\test\\ -res_dir d:\img_result\small\    -json_dir d:\img_label\small\ -score_thres 0.4
        python parallel_infer.py -image_name '*' -data_dir /home/mjy/mmdetection/myutils/image/ -res_dir /home/mjy/mmdetection/myutils/result/    -json_dir /home/mjy/mmdetection/myutils/label/ -score_thres 0.6
        python parallel_infer.py -image_name '*' 
    """
    # logger
    print(acceleration.foo()) # 测试扩展能不能正确导入
    # logger初始化设置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s|%(name)-12s: %(levelname)-8s %(message)s')
    logger = logging.getLogger("inference")
    logger.setLevel(logging.INFO)

    handler1 = logging.FileHandler("base-log.log")
    handler1.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s|%(name)-12s+ %(levelname)-8s++%(message)s')
    handler1.setFormatter(formatter)

    handler2 = logging.StreamHandler()
    handler2.setLevel(logging.ERROR)

    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.info("img_path:{} json_dir:{} stride:{} score_thres:{}".format(
        th_img_path, th_json_path, arg.stride, arg.score_thres))

    # FIXME: Specify the path to model config and checkpoint file
    config_file = "D:\ProgramData\Anaconda3\mmdetection\myutils\swin\mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py"
    checkpoint_file = "D:\ProgramData\Anaconda3\mmdetection\myutils\swin\epoch_36.pth" 
    config_rot_file = "D:\ProgramData\Anaconda3\mmrotate\work0501\mycfg.py"
    checkpoint_rot_file="D:\ProgramData\Anaconda3\mmrotate\work0501\epoch_12.pth"
    MY_DEVICE='cpu'

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device=MY_DEVICE)
    rotated_model=init_detector(config_rot_file, checkpoint_rot_file, device=MY_DEVICE)

    window_size = 1200 // arg.scale # 滑窗大小
    center_size = int(arg.stride / arg.scale // 2 * 2)
    logger.info("window_size:{} center_size:{}".format(window_size, center_size))

    if next(model.parameters()).device == "cpu":
        logger.warning("no gpu")

    scale_param = arg.scale
    infer_img_list = glob.glob(th_img_path) # 获取需要推理的图片文件夹下所有图片路径

    item = "d:\\img_data\\test\\1.jpg"  # 这个不用改
    root_dir, filename = os.path.split(item)
    basename, filetype = os.path.splitext(filename)
    fps = []
    fps_patch = []


    img_cnt = 0
    for item in tqdm(infer_img_list):
        time_start = time.time()
        root_dir, filename = os.path.split(item)# 分解图片的名字basename
        basename, filetype = os.path.splitext(filename)
        image_path = item
        print(item)
        image = Image.open(image_path)

        if scale_param != 1:    # 缩小图片
            image = image.resize(
                (image.width // scale_param, image.height // scale_param),
                Image.ANTIALIAS)
        
        image = np.asarray(image)

        paint_flag = True

        img_size = (image.shape[0], image.shape[1])

        clip_list,info_list = get_clips(image, window_size, center_size)  # 切分大图为小图片，返回一个list
        logger.info("get clips done-----")
        box_data = clip_for_inference(
            model,
            clip_list,
            info_list,
            img_size,
            window_size,
            center_size,
            basename,
            arg.score_thres,
            paint_flag=paint_flag,
            result_path=th_res_path,
            scale_para=scale_param)
        logger.info("inference done")
        obb_start=time.time()
        box_data=predict_obb(rotated_model,image,box_data,scale_para=scale_param)
        obb_end=time.time()
        logger.info('obb cost: {}'.format(obb_end - obb_start))
        outputjson(box_data, basename, json_path=th_json_path,scale_para=scale_param)  # 输出所有box信息到$basename$.json文件中
        img_cnt += 1
        time_end = time.time()
        logger.info('totally cost: {}'.format(time_end - time_start))
        # print("-------------------------------------------------------")
        logger.info(
            "infer rate:{:.4f}s/patch | infer fps:{:.2f}patch/s "
            .format(
                sum(fps) / sum(fps_patch),
                sum(fps_patch) / sum(fps)))



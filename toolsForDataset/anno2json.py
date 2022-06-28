import os
import os.path as osp
import pandas as pd
import numpy as np
import mmcv
from tqdm import tqdm
from PIL import Image
"""
image = {
    "id": int,
    "width": int,
    "height": int,
    "file_name": str,
}

annotation = {
    "id": int,
    "image_id": int,
    "category_id": int,
    "segmentation": RLE or [[polygon]],
    "area": float,
    "bbox": [x,y,width,height],
    "iscrowd": 0 or 1,
}

categories = [{
    "id": int,
    "name": str,
    "supercategory": str,
}]

"""


def convert_img_to_coco(img_id, box_num, label_path, img_size, images,
                        annotations):
    label_data = pd.read_csv(label_path, encoding="ansi")
    label_data = pd.DataFrame(label_data)
    root_dir, filename = osp.split(label_path)
    basename, filetype = osp.splitext(filename)

    # for image
    ih = iw = img_size

    images.append(
        dict(id=img_id, file_name=basename + ".jpg", height=ih, width=iw))

    # for labels
    obj_count = 0
    for idx, row in label_data.iterrows():
        x_min, y_min, w, h = row['Xmin'], row['Ymin'], row['Width'], row[
            'Height']
        cid = row['Category']
        x_max = x_min + w
        y_max = y_min + h
        data_anno = dict(image_id=img_id,
                         id=box_num + idx,
                         category_id=cid,
                         bbox=[x_min, y_min, w, h],
                         segmentation=[[
                             x_min, y_min, x_min, y_max, x_max, y_max, x_max,
                             y_min
                         ]],
                         area=w * h,
                         iscrowd=0)
        annotations.append(data_anno)
        obj_count += 1
    box_num += obj_count
    return images, annotations, box_num


if __name__ == '__main__':
    # './train/anno_train.json' "./val/anno_val.json"
    train_image_dir = './train/image'  # "d:\\img_data\\public"
    train_label_dir = './train/label'
    val_image_dir = './val/image'
    val_label_dir = './val/label'
    train_output_path = './train/instances_train2017.json'
    val_output_path = './val/instances_val2017.json'

    mode = "val"  # "val"
    img_dir = train_image_dir  # "d:\\img_data\\public"
    label_dir = train_label_dir  # "d:\\img_gt\\public"
    out_file = train_output_path
    if mode == "val":
        img_dir = val_image_dir  # "d:\\img_data\\public"
        label_dir = val_label_dir  # "d:\\img_gt\\public"
        out_file = val_output_path

    annotations = []
    images = []
    pathDir = os.listdir(img_dir)  # './train/image'
    img_id =17589  #28345 # train:0
    box_num = 19126  #19794 # train:0
    for img_path in tqdm(pathDir):
        root_dir, name = os.path.split(img_path)
        basename, filetype = os.path.splitext(name)
        label_path = os.path.join(label_dir,
                                  basename + ".csv")  # './train/label'
        images, annotaions, box_num = convert_img_to_coco(
            img_id, box_num, label_path, 1200, images, annotations)
        img_id += 1

    coco_format_json = dict(images=images,
                            annotations=annotations,
                            categories=[{
                                'id': 0,
                                'name': 'FlushToilet'
                            }, {
                                'id': 1,
                                'name': 'Urinal'
                            }, {
                                'id': 2,
                                'name': 'SquattingPan'
                            }, {
                                'id': 3,
                                'name': 'Washbasin'
                            }, {
                                'id': 4,
                                'name': 'Sink'
                            }, {
                                'id': 5,
                                'name': 'MopPool'
                            }, {
                                'id': 6,
                                'name': 'WashMachine'
                            }, {
                                'id': 7,
                                'name': 'DL'
                            }, {
                                'id': 8,
                                'name': 'ShowerRoom'
                            }, {
                                'id': 9,
                                'name': 'ShowerRoom_chamber'
                            }, {
                                'id': 10,
                                'name': 'Bathtub'
                            }, {
                                'id': 11,
                                'name': 'Shower'
                            }])
    mmcv.dump(coco_format_json, out_file)
    print("img id: {}  box num:{}".format(img_id, box_num))

# 洁具识别

- configs
  - obb.py 旋转框网络的配置
  - maskrcnn.py 水平框检测网络的配置
- ToolsForDataset
  - dataclean.py c#框架下得到的数据集文件需要进行清洗
  - getclip.py 将超大图片滑窗裁切成小图片以及对应的csv文件
  - splitdataset.py 划分为训练集train和验证集val
  - anno2json.py 划分后将训练集所有标注文件转为一个json文件，验证集相同（coco数据集标注格式）
  - csv2dota.py 将每个物件图像提取出来转为dota数据集的标注格式，并进行离线旋转增强
- inference.py 用于识别
- test.cs c#创建数据集、交互

## 数据集处理

## C#框架

命令[THPRINT],选取左上角、右下角两个对角点，输入一个数字（对应了打印出的文件名），当图片打印出后自动调用本地的python环境（见Ex）

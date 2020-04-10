# YOLOv1

**For better performance, this not the same as the original [paper](https://arxiv.org/pdf/1506.02640.pdf). 
I achieved 0.684 mAP on VOC07test, 76fps on RTX2080Ti**

**My backbone network is resnet50, add a few 1x1 and 3x3 conv to fine-tune the resnet. I also change the 7x7 feature maps to 14x14 feature maps and use the fully convolutional instead of convolution & fully connected which has been Implemented in the original paper.**

![avatar](https://github.com/kevin655/YOLOv1/blob/master/bike.jpg)

![avatar](https://github.com/kevin655/YOLOv1/blob/master/000319.jpg)

## Trained on VOC2007+VOC2012
| model                | backbone | map on VOC07test  | FPS  |
| -------------------- | -------------- | ---------- | -------   |
| YOLO Resnet50   |   ResNet50        | 68.4%      |  76   |
| YOLO original |   VGG-16          | 63.4%      |  45   |


### Prerequisites
- pytorch 1.2.0
- cuda 10.0.1
- pillow 6.2.1
- numpy

### Quick Start
Download the file 
```shell
git clone https://github.com/uoip/KCFpy.git
python demo.py
```
And You will find the demo results picture in 
```shell
./demo/demo_results
```

### Training
1. Download VOC2012train and VOC2007 dataset
2. Download VOC2007test dataset
3. Put them in the dir 
```shell
../Image/
```

To train from scratch, run
```shell
python main.py -s 0 
```
To get parameters help, run
```shell
python main.py -h
```
This will automatically start train on VOC07+12, and test on VOC07 every epochs.


### Details
Some parameters Setting are very Important. For convenience, I list them below.

| YOLO Resnet50   |   ResNet50        | 68.4%      |  76   |
| YOLO original |   VGG-16          | 63.4%      |  45   |








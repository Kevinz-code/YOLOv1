# YOLOv1

**This not the same as the original [paper](https://arxiv.org/pdf/1506.02640.pdf). My Results on voc07test is 0.684 map, 76fps on RTX2080Ti**

My backbone network is resnet50, add a few 1x1 and 3x3 conv to fine-tune the resnet. I also change the 7x7 feature maps to 14x14 feature maps and use the fully convolutional instead of convolution & fully connected which has been Implemented in the original paper.

![avatar](https://github.com/kevin655/YOLOv1/blob/master/bike.jpg)

![avatar](https://github.com/kevin655/YOLOv1/blob/master/000319.jpg)

## Train on voc2012+2007
| model                | backbone | map on voc07test  | FPS  |
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
And You will find the demo results picture in ./demo/demo_results

### Training
1. Download voc2012train adn voc2007 dataset

2. Download voc2007test dataset

to train from scratch, run
```shell
python main.py -s 0 
```

### Others








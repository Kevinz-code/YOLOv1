# YOLOv1 based on Pytorch

**For better performance, this not the same as the original [paper](https://arxiv.org/pdf/1506.02640.pdf).**

**I achieved 0.684 mAP on VOC07test, 76fps on RTX2080Ti**

My backbone network is resnet50, add a few 1x1 and 3x3 conv to fine-tune the resnet. I also change the 7x7 feature maps to 14x14 feature maps and use the fully convolutional instead of convolution & fully connected which has been Implemented in the original paper.

![avatar](http://chuantu.xyz/t6/728/1586530590x992239408.jpg)

![avatar](http://chuantu.xyz/t6/728/1586530639x992239408.jpg)

## Trained on VOC2007+VOC2012
| model                |  map on VOC07test  | FPS  |
| -------------------- |  ---------- | -------   |
| YOLO Resnet50   |   68.4%      |  76   |
| YOLO original |  63.4%      |  45   |


### Prerequisites
- pytorch 1.2.0
- cuda 10.0.1
- pillow 6.2.1
- numpy

### Quick Start
Download the file 
```shell
git clone https://github.com/kevin655/YOLOv1.git
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

**For Training**
| learning rate               |  3e-3, 1e-3  | 
| -------------------- |  ---------- | 
| weight_decay  |   0.0005      |  
| miniBatch |  16      |  
| epoch |  30      |  
| momentum |  0.9      |  
| nms_thresh |  0.26      | 

**For Testing**
| c_thresh              |  0.3  | 
| -------------------- |  ---------- | 
| nms_thresh  |   0.26      |  






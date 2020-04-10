# -*- coding: utf-8 -*-
import os
import numpy as np

from voc_eval import voc_eval  # 注意将voc_eval.py和compute_mAP.py放在同一级目录下

def compute_main():
    detpath = '../Image/VOC2007/My_Results/'  # 各类txt文件路径
    detfiles = os.listdir(detpath)

    classes = ("aeroplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor")

    aps = []  # 保存各类ap
    recs = []  # 保存recall
    precs = []  # 保存精度

    annopath = '../Image/VOC2007/Annotations' + '{:s}.xml'  # annotations的路径，{:s}.xml方便后面根据图像名字读取对应的xml文件
    imagesetfile = '../Image/VOC2007/ImageSets/Main/test.txt'  # 读取图像名字列表文件
    cachedir = '../Image/VOC2007/cache_for_annotations'
    for i, cls in enumerate(classes):
        if cls == '__background__':
            continue
        for f in detfiles:  # 读取cls类对应的txt文件
            if f.find(cls) != -1:
                filename = detpath + f

        rec, prec, ap = voc_eval(  # 调用voc_eval.py计算cls类的recall precision ap
            filename, annopath, imagesetfile, cls, cachedir, ovthresh=0,
            use_07_metric=False)

        aps += [ap]

        print('AP for {} = {:.4f}'.format(cls, ap))
        print('recall for {} = {:.4f}'.format(cls, rec[-1]))
        print('precision for {} = {:.4f}'.format(cls, prec[-1]))

    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')

    print('Results:')
    for i, ap in enumerate(aps, 0):
        print('{:s}:{:.3f}'.format(classes[i], ap))
    print('mAP:{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    return np.mean(aps)


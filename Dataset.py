import os
import xml.dom.minidom as XML
from PIL import Image
import numpy as np
import torch
import lib.transform as T1
import torchvision.transforms as T2
import cv2
import random


All_classes={"aeroplane":1,"bicycle":2,"bird":3,"boat":4,"bottle":5,
         "bus":6,"car":7,"cat":8,"chair":9,"cow":10,"diningtable":11,
         "dog":12,"horse":13,"motorbike":14,"person":15,"pottedplant":16,
         "sheep":17,"sofa":18,"train":19,"tvmonitor":20,}


def get_transform(train=False):
    transforms=[]
    transforms.append(T1.ToTensor())
    if train:
        transforms.append(T1.RandomHorizontalFlip(0.5))
    return T1.Compose(transforms)

def img_Normalize(img):
    trans=T2.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    img=trans(img)
    return img

class VOC_Data(object):
    def __init__(self,root,transforms,input_size=448):
        self.input_size=input_size
        self.root=root
        self.transforms=transforms
        self.imgs=list(sorted(os.listdir(os.path.join(root,"JPEGImages"))))
        self.xml=list(sorted((os.listdir(os.path.join(root,"Annotations")))))
        self.txt_path=os.path.join(root,"ImageSets","Main")

    def train_test_idx(self):
        train_file_path=os.path.join(self.txt_path,"trainval.txt")
        test_file_path=os.path.join(self.txt_path,"test.txt")

        train_file=open(train_file_path,"r")
        test_file=open(test_file_path,"r")

        train_idx=[]
        test_idx=[]
        for line in train_file.readlines():
            train_idx.append(int(line.strip())-1)
        for line in test_file.readlines():
            test_idx.append(int(line.strip())-1)

        return train_idx, test_idx

    def get_bbox_classes(self,idx):
        xml_path=os.path.join(self.root,"Annotations",self.xml[idx])
        DomTree=XML.parse(xml_path)
        Root=DomTree.documentElement

        obj_all=Root.getElementsByTagName("object")
        leng=len(obj_all)
        boxes=[]
        classes=[]
        for obj in obj_all:
            # get the classes
            obj_name=obj.getElementsByTagName('name')[0]
            one_class=obj_name.childNodes[0].data
            classes.append(one_class)

            # get the box
            one_box=[]
            obj_bbox=[]
            for child in obj.childNodes:
                if child.nodeName=='bndbox':
                    obj_bbox=child
                    break
            for i in range(1,8,2):
                dot=obj_bbox.childNodes[i].childNodes[0].data
                one_box.append(float(dot))
            boxes.append(one_box)

        return leng,classes,boxes

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def randomBlur(self, bgr):
        if random.random() < 0.5:
            bgr = cv2.blur(bgr, (5, 5))
        return bgr

    def RandomBrightness(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomSaturation(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            s = s * adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomHue(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            h = h * adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def resize(self, boxes, w, h, leng):
        # boxes are Tensor
        boxes[:, 0] *= (self.input_size / w)
        boxes[:, 2] *= (self.input_size / w)

        boxes[:, 1] *= (self.input_size / h)
        boxes[:, 3] *= (self.input_size / h)

        return boxes

    def __getitem__(self, idx):
        # get important information
        leng,classes,bbox=self.get_bbox_classes(idx)
        labels=[All_classes[i] for i in classes]

        img_path=os.path.join(self.root, "JPEGImages", self.imgs[idx])
        img=cv2.imread(img_path)

        ### for CV2 Image its size=tuple(H x W x C)
        h, w = img.shape[0], img.shape[1]
        h, w = float(h), float(w)
        ###

        boxes=torch.as_tensor(bbox, dtype=torch.float64)
        boxes=self.resize(boxes, w, h, leng)

        labels=torch.as_tensor(labels,dtype=torch.int64)
        image_id=torch.tensor([idx])
        iscrowd=torch.zeros((leng,),dtype=torch.int64)
        area=(boxes[:,3]-boxes[:,1])*(boxes[:,2]-boxes[:,0])

        h, w = torch.tensor(h, dtype=torch.float64), torch.tensor(w, dtype=torch.float64)
        target={}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd
        target['height'] = h
        target['width'] = w

        if self.transforms==True:
            # Data Augmentation
            img = self.randomBlur(img)
            img = self.RandomBrightness(img)
            img = self.RandomHue(img)
            img = self.RandomSaturation(img)

            # Convert to RGB
            img = self.BGR2RGB(img)
            # Special operation for Images and Boxes
            # Default Nearest method for img
            img = cv2.resize(img, (self.input_size, self.input_size))

            img,target=get_transform(train=True)(img,target)

            img=img_Normalize(img)

        else:
            # Convert to RGB
            img = self.BGR2RGB(img)
            # Special operation for Images and Boxes
            # Default Nearest method for img
            img = cv2.resize(img, (self.input_size, self.input_size))

            img, target = get_transform(train=False)(img,target)

            img=img_Normalize(img)

        return img,target


    def __len__(self):
        return len(self.imgs)


class VOC_2012(object):
    def __init__(self, root, transforms, input_size=448):
        self.input_size = input_size
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
        self.xml = list(sorted(os.listdir(os.path.join(root, "Annotations"))))
        self.txt_path = os.path.join(root, "ImageSets", "Main")


    def train_test_idx(self):
        train_file_path = os.path.join(self.txt_path, "trainval.txt")
        train_file = open(train_file_path, "r")
        train_idx = []

        for line in train_file.readlines():
            line = line.strip()
            line = line + ".jpg"
            train_idx.append(self.imgs.index(line))

        return train_idx

    def get_bbox_classes(self,idx):
        xml_path=os.path.join(self.root,"Annotations",self.xml[idx])
        #print(idx, xml_path)
        DomTree=XML.parse(xml_path)
        Root=DomTree.documentElement

        obj_all=Root.getElementsByTagName("object")
        leng=len(obj_all)
        boxes=[]
        classes=[]
        for obj in obj_all:
            # get the classes
            obj_name=obj.getElementsByTagName('name')[0]
            one_class=obj_name.childNodes[0].data
            classes.append(one_class)

            # get the box
            one_box=[]
            obj_bbox=[]
            for child in obj.childNodes:
                if child.nodeName=='bndbox':
                    obj_bbox=child
                    break
            '''
            for i in range(1,8,2):
                dot=obj_bbox.childNodes[i].childNodes[0].data
                one_box.append(float(dot))
            x1 = one_box[1]
            y1 = one_box[3]
            x2 = one_box[0]
            y2 = one_box[2]
            '''
            xmin = obj_bbox.getElementsByTagName('xmin')[0]
            x1 = xmin.childNodes[0].data
            ymin = obj_bbox.getElementsByTagName('ymin')[0]
            y1 = ymin.childNodes[0].data
            xmax = obj_bbox.getElementsByTagName('xmax')[0]
            x2 = xmax.childNodes[0].data
            ymax = obj_bbox.getElementsByTagName('ymax')[0]
            y2 = ymax.childNodes[0].data
            one_box = list(map(int,[x1, y1, x2, y2]))
            boxes.append(one_box)

        return leng,classes,boxes

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def BGR2HSV(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def HSV2BGR(self, img):
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    def randomBlur(self, bgr):
        if random.random() < 0.5:
            bgr = cv2.blur(bgr, (5, 5))
        return bgr

    def RandomBrightness(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            v = v * adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomSaturation(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            s = s * adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def RandomHue(self, bgr):
        if random.random() < 0.5:
            hsv = self.BGR2HSV(bgr)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            h = h * adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = self.HSV2BGR(hsv)
        return bgr

    def resize(self, boxes, w, h, leng):
        # boxes are Tensor
        boxes[:, 0] *= (self.input_size / w)
        boxes[:, 2] *= (self.input_size / w)

        boxes[:, 1] *= (self.input_size / h)
        boxes[:, 3] *= (self.input_size / h)

        return boxes

    def __getitem__(self, idx):
        # get important information
        leng,classes,bbox=self.get_bbox_classes(idx)
        labels=[All_classes[i] for i in classes]

        img_path=os.path.join(self.root,"JPEGImages",self.imgs[idx])
        img=cv2.imread(img_path)

        ### for CV2 Image its size=tuple(H x W x C)
        h, w = img.shape[0], img.shape[1]
        h, w = float(h), float(w)
        #print(h, w)

        boxes=torch.as_tensor(bbox,dtype=torch.float64)
        #print(boxes)
        boxes=self.resize(boxes, w, h, leng)
        #print(boxes)

        labels=torch.as_tensor(labels,dtype=torch.int64)
        image_id=torch.tensor([idx])
        iscrowd=torch.zeros((leng,),dtype=torch.int64)
        area=(boxes[:,3]-boxes[:,1])*(boxes[:,2]-boxes[:,0])

        target={}
        target['boxes']=boxes
        target['labels']=labels
        target['image_id']=image_id
        target['area']=area
        target['iscrowd']=iscrowd

        if self.transforms==True:
            # Data Augmentation
            img = self.randomBlur(img)
            img = self.RandomBrightness(img)
            img = self.RandomHue(img)
            img = self.RandomSaturation(img)

            # Convert to RGB
            img = self.BGR2RGB(img)
            # Special operation for Images and Boxes
            # Default Nearest method for img
            img = cv2.resize(img, (self.input_size, self.input_size))

            img,target=get_transform(train=True)(img,target)

            img=img_Normalize(img)

        else:
            # Convert to RGB
            img = self.BGR2RGB(img)
            # Special operation for Images and Boxes
            # Default Nearest method for img
            img = cv2.resize(img, (self.input_size, self.input_size))

            img, target = get_transform(train=False)(img,target)

            img=img_Normalize(img)

        return img,target


    def __len__(self):
        return len(self.imgs)



import cv2
import torchvision.transforms as T
import os
import torch
from yolonet import YoloNet
import time
from test import from_output_to_right_form
from test import NMS
from PIL import Image, ImageDraw, ImageFont
from test import My_classes

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def BGR2RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def trans(img):
    trans1 = T.ToTensor()
    trans2 = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    img = trans1(img)
    img = trans2(img)
    return img


def get_demo_image(path):
    img = cv2.imread(path)
    ### for CV2 Image its size=tuple(H x W x C)
    h, w = img.shape[0], img.shape[1]
    h = torch.tensor(h, dtype=torch.float, device=device)
    w = torch.tensor(w, dtype=torch.float, device=device)
    # Convert to RGB
    img = BGR2RGB(img)
    # Special operation for Images resize
    # Default Nearest method for img
    img = cv2.resize(img, (448, 448))
    img = trans(img)

    return img, h, w


def set_bbox(bbox, scores, labels, img_path):
    a = Image.open(img_path)
    rear = img_path.split('/')[-1]
    
    # define ImageDraw
    b = ImageDraw.Draw(a)
    setfont = ImageFont.truetype(font="./lib/font/a.TTF", size=20)

    for i in range(labels.shape[0]):
        # get the class(type str) first
        cls_id = labels[i].item()
        cls = My_classes["%s" % (cls_id + 1)]

        # Set the rectangle
        bbox_i = bbox[i].tolist()
        b.rectangle(bbox_i, outline='green', width=2)
        log = "%s:%.2f%%" % (cls, scores[i].item() * 100)
        b.text([bbox_i[0], bbox_i[1]], log, font=setfont)

    a.save("./demo/demo_results/" + rear)


def demo():
    # pre operation
    img_path = list(sorted(os.listdir("./demo/demo_img/")))
    img_path = ["demo/demo_img/" + x for x in img_path]
    model = YoloNet()
    model.load_state_dict(torch.load("./models/Yolo_v1.pth0"))
    model.to(device=device)
    model.eval()
    print("loading models...\n...\nStart Detection")
    time.sleep(1.5)

    with torch.no_grad():
        for i in range(len(img_path)):
            img, height, width = get_demo_image(img_path[i])
            img = torch.unsqueeze(img, dim=0).to(device=device)

            t1_model = float(time.time())
            ori_output = model(img)[0].cpu()
            t2_model = float(time.time())
            print("Speed:%.3fs" % (t2_model - t1_model), end=" ")

            output = from_output_to_right_form(ori_output, C_thresh=0.8, height=height, width=width)
            labels = output['labels']  # Shape torch.Size([nums])
            scores = output['scores']  # Shape torch.Size([nums])
            boxes = output['boxes']  # Shape torch.Size([ nums , 4 ])
            detect_object = scores.shape[0]
            print("Total Detect {} obj".format(detect_object), end=" ")

            # Dealing the NMS
            nms_keep_id = NMS(boxes, labels, scores, nms_thresh=0.25)
            nms_del_object = detect_object - nms_keep_id.shape[0]
            print("NMS Delete {} obj".format(nms_del_object))

            # Redifining ouput
            labels = labels[nms_keep_id]
            scores = scores[nms_keep_id]
            boxes = boxes[nms_keep_id]

            set_bbox(bbox=boxes, scores=scores, labels=labels, img_path=img_path[i])

    print()
    print("Detection Finished!!")
    print("Find your results in demo/demo_results")

demo()

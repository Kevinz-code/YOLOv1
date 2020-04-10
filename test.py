from cal_loss import calculate_loss
import torch
import time
from PIL import Image, ImageDraw, ImageFont
from cal_mAP import *

My_classes={"1":"aeroplane","2":"bicycle","3":"bird","4":"boat","5":"bottle",
            "6":"bus","7":"car","8":"cat","9":"chair","10":"cow","11":"diningtable",
            "12":"dog","13":"horse","14":"motorbike","15":"person","16":"pottedplant",
            "17":"sheep","18":"sofa","19":"train","20":"tvmonitor"}


def from_to_the_ori_bbox(x, y, w, h, x_pos, y_pos, height, width):
    # pre operation
    x_pos = torch.tensor(x_pos, dtype=torch.float)
    y_pos = torch.tensor(y_pos, dtype=torch.float)
    ll = len(x)

    x_coord = (x + x_pos) / 14.0 * width
    y_coord = (y + y_pos) / 14.0 * height

    h = h * height
    w = w * width
    x1 = torch.max(x_coord - w/2, torch.tensor([0.]*ll))
    y1 = torch.max(y_coord - h/2, torch.tensor([0.]*ll))
    x2 = torch.min(x_coord + w/2, torch.tensor([width.item()]*ll))
    y2 = torch.min(y_coord + h/2, torch.tensor([height.item()]*ll))

    return x1, y1, x2, y2


def from_output_to_right_form(out, C_thresh, height, width):
    assert isinstance(C_thresh, float)

    # shape 30 x 14 x 14
    y_pos, x_pos = torch.where(out[20:22, :, :].max(dim=0)[0] > C_thresh)
    while len(y_pos) == 0:
        C_thresh = C_thresh - 0.05
        y_pos, x_pos = torch.where(out[20:22, :, :].max(dim=0)[0] > C_thresh)

    # 0 means the value, 1 means the index
    pr_max_value, pr_max_id = torch.max(out[:20, y_pos, x_pos], dim=0)
    labels = pr_max_id
    #print(pr_max_value, pr_max_id)
    C_max_value, C_max_id = torch.max(out[20:22, y_pos, x_pos], dim=0)
    #print("\n\n",C_max_value)
    #exit()
    scores = pr_max_value
    # shape of them Torch.Size([nums])
    x_ratio = out[22+C_max_id*4, y_pos, x_pos]
    y_ratio = out[23+C_max_id*4, y_pos, x_pos]
    w_ratio = out[24+C_max_id*4, y_pos, x_pos]
    h_ratio = out[25+C_max_id*4, y_pos, x_pos]
    x1, y1, x2, y2 = from_to_the_ori_bbox\
        (x_ratio, y_ratio, w_ratio, h_ratio, x_pos, y_pos, height, width)

    boxes = [[x1[i], y1[i], x2[i], y2[i]] for i in range(len(x1))]
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    # for Easy Know
    output = {}
    output['labels'] = labels # Shape torch.Size([nums])
    output['scores'] = scores # Shape torch.Size([nums])
    output['boxes'] = boxes   # Shape torch.Size([ nums , 4 ])

    return output


def NMS(boxes, labels, scores, nms_thresh):

    # Dealing the NMS Method
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2-x1+1) * (y2-y1+1)
    res = []
    index = torch.argsort(scores, descending=True)
    #print(index)
    while len(index) != 0:
        res.append(index[0])
        #print("x1",x1[index[0]],x1[index[1:]])
        xx1 = torch.max(x1[index[0]], x1[index[1:]])
        yy1 = torch.max(y1[index[0]], y1[index[1:]])
        xx2 = torch.min(x2[index[0]], x2[index[1:]])
        yy2 = torch.min(y2[index[0]], y2[index[1:]])
        #print("xx1",xx1,)
        #print("yy1",yy1)
        #print("xx2",xx2)
        #print("yy2",yy2)

        #print("xx2-xx1",xx2-xx1+1)
        #print("yy2-yy1",yy2-yy1+1)
        ww = torch.max(xx2-xx1+1, torch.tensor([0.]*len(xx1)))
        hh = torch.max(yy2-yy1+1, torch.tensor([0.]*len(xx2)))

        inter = ww * hh
        iou = inter / (area[index[0]] + area[index[1:]] - inter)
        # if iou < thresh we keep it
        # elif the iou> thresh's objects belong to other classes we keep it
        keep_iou_id = torch.where(iou < nms_thresh)[0] + 1
        keep_label_id = torch.where(labels[index[1:]] != labels[index[0]])[0] + 1
        final_keep = torch.unique(torch.cat((keep_iou_id, keep_label_id), dim=0), sorted=True)
        index = index[final_keep]
    return torch.as_tensor(res)


def set_bbox(bbox, scores, labels, idx):
    pth = "../Image/VOC2007/JPEGImages/" + "00%04d.jpg" % (idx)
    a = Image.open(pth)

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

    a.save("demo_results/00%04d.jpg" % (idx))


def clear():
    for i in range(1, 20):
        l = My_classes["%s" % i]
        l = "../Image/VOC2007/My_Results/" + l
        f = open(l, "w")
        f.close()
        os.remove(l)



def test(model, test_loader, device, epoch, C_thresh, nms_thresh, test_idx):
    clear()
    # Setting test mode
    model = model.to(device)
    model.eval()
    print("Testing beginning!\n")
    print("Using {}\nYolo_v1.pth{}".format(device, epoch-1))

    more_obj = 0
    less_obj = 0
    detect_total = 0
    real_total = 0
    # Set no grad function
    with torch.no_grad():
        for i, (image, target) in enumerate(test_loader, 0):
            b = 0
            # image is a tuple
            image = image[0].to(device).unsqueeze(0)
            target = target[0]
            height = target['height']
            width = target['width']

            # out shape is 1 x 14 x 30 x 30
            t1_model = float(time.time())
            ori_output = model(image)[0].cpu()
            t2_model = float(time.time())
            # print("%d Image last%.4fps"%(i+1, 1.0/(t2_model-t1_model)),end=" ")

            output = from_output_to_right_form(ori_output, C_thresh, height, width)
            labels = output['labels']  # Shape torch.Size([nums])
            scores = output['scores']  # Shape torch.Size([nums])
            boxes = output['boxes']   # Shape torch.Size([ nums , 4 ])
            detect_object = scores.shape[0]
            print("Total Detect {} obj".format(detect_object), end=" ")

            # Dealing the NMS
            nms_keep_id = NMS(boxes, labels, scores, nms_thresh)
            nms_del_object = detect_object - nms_keep_id.shape[0]
            print("NMS Delete {} obj".format(nms_del_object), end=" ")

            # Redifining ouput
            labels = labels[nms_keep_id]
            scores = scores[nms_keep_id]
            boxes = boxes[nms_keep_id]

            # Save Results in Class-Files
            for k in range(labels.shape[0]):
                if scores[k].item() < 0.1:
                   b += 1
                   continue
                save_str = "00%04d %.4f %.1f %.1f %.1f %.1f\n"%\
                           (test_idx[i]+1, scores[k].item(),
                            boxes[k][0].item(), boxes[k][1].item(),
                            boxes[k][2].item(), boxes[k][3].item())
                cls = My_classes["%s" % (labels[k].item()+1)]  # +1 important
                with open("../Image/VOC2007/My_Results/" + cls, "a") as f:
                    f.writelines(save_str)
                with open("pr.txt", "a") as f:
                    f.writelines("%.3f\n" % (scores[k].item()))

            # Calculating the more or less object
            error_obj = labels.shape[0] - target["labels"].shape[0] - b
            if error_obj > 0:
                more_obj += error_obj
            else:
                less_obj -= error_obj

            # Sum all the Object
            detect_total += labels.shape[0] - b
            real_total += target["labels"].shape[0]
            print("{}/{}".format(detect_total, real_total))

    # Calculate mAP and Save this result
    mAP = compute_main()
    result = "epoch:%d  mAP:%.4f  %d/%d  more:%d  less:%d c_thresh:%.3f  nms:%.3f\n" % (
        epoch, mAP, detect_total, real_total, more_obj, less_obj, C_thresh, nms_thresh)
    with open("result.txt", "a") as f:
        f.writelines(result)


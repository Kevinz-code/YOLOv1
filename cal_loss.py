import torch
import os

'''Important'''
# Remember to Set every Tensor device=torch.device("cuda")
# Also Remember to every Tensor has grad_function
# Also Remember Not to Use inplace operation for Variable
# which has grad function


def from_ratio_to_bbox(x, y, w, h):
    # Scale
    w = w*14 # CUDA tensor
    h= h*14 # CUDA tensor

    #Transformation
    x1= x - w/2.0  # CUDA tensor
    y1= y - h/2.0  # CUDA tensor
    x2= x + w/2.0  # CUDA tensor
    y2= y + h/2.0  # CUDA tensor

    return x1,y1,x2,y2


def from_bbox_to_ratio(x1 ,y1, x2, y2):
    # Calculate x,y
    center_x = (x1 + x2) / 2.0  # CUDA tensor
    center_y = (y1 + y2) / 2.0 # CUDA tensor

    center_resize_x = center_x * (14.0 / 448.0)  # CUDA tensor
    center_resize_y = center_y * (14.0 / 448.0)  # CUDA tensor

    cur_x=int(center_resize_x)
    cur_y=int(center_resize_y)
    x_ratio = center_resize_x - cur_x  # CUDA tensor
    y_ratio = center_resize_y - cur_y  # CUDA tensor

    # Calculate w,h
    w, h = (x2 - x1), (y2 - y1)  # CUDA tensor
    w_ratio, h_ratio = w / 448.0, h / 448.0  # CUDA tensor

    # from graph we can know
    # cur_x , cur_y points to the right grid with their values
    return cur_x, cur_y, x_ratio, y_ratio, w_ratio, h_ratio


def caculate_IOU(x_ori, y_ori, w_ori, h_ori, other,device,sigma=0.0001):
    # x, y, w, h are special parameters
    # Image Sizes are 448 * 448
    # Extracting
    x_other=other[22:27:4]
    y_other=other[23:28:4]
    w_other=other[24:29:4]
    h_other=other[25:30:4]

    # Transformation
    x1_ori,y1_ori,x2_ori,y2_ori=\
        from_ratio_to_bbox(x_ori,y_ori,w_ori,h_ori)
    x1_other,y1_other,x2_other,y2_other=\
        from_ratio_to_bbox(x_other,y_other,w_other,h_other)

    # Area
    area_ori = (x2_ori - x1_ori ) * (y2_ori - y1_ori )
    area_other =(x2_other - x1_other) * (y2_other - y1_other)

    # max coords for IOU
    xx1=torch.max(x1_ori,x1_other)
    yy1=torch.max(y1_ori,y1_other)
    xx2=torch.min(x2_ori,x2_other)
    yy2=torch.min(y2_ori,y2_other)


    zero=torch.zeros(1,dtype=torch.float64, device=device)
    ww=torch.max(zero, xx2-xx1)
    hh=torch.max(zero, yy2-yy1)
    inter = ww * hh
    IOU= inter / ( area_ori + area_other - inter + sigma)

    idx = IOU.argmax().item()
    max_iou = IOU.max().item()

    # return id and max_value
    return IOU.argmax(), IOU.max()


def calculate_loss(target,output,device):
    # target['boxes']=boxes
    # target['labels']=labels
    # target['image_id']=image_id
    # target['area']=area
    # target['iscrowd']=iscrowd
    # ouput.shape B x C x H x W ==(Batch x 30 x 7 x 7 )
    batch=len(target)

    # Init loss at zeros
    x_y_loss = torch.zeros(batch, dtype=torch.float64,device=device)
    w_h_loss = torch.zeros(batch, dtype=torch.float64,device=device)
    C_loss = torch.zeros(batch, dtype=torch.float64,device=device)
    no_obj_loss = torch.zeros(batch, dtype=torch.float64,device=device)
    pr_loss = torch.zeros(batch, dtype=torch.float64,device=device)

    # Utra parameters
    lamda_coord = 5.0
    lamda_no_obj = 0.08

    # cal all num
    num_all = torch.zeros(1,dtype=torch.float64, device=device)

    for bt, one_target in enumerate(target, 0):

        # object numbers in one target
        obj_num = len(one_target['labels'])

        # Define error been extraly calculated for one Image
        extra_loss=torch.zeros(1,dtype=torch.float64,device=device)

        for i in range(obj_num):

            # MAKING TMP TARGET
            # STARTING
            x1, y1, x2, y2 = one_target['boxes'][i]  # CUDA tensor
            label = one_target['labels'][i]  # CUDA tensor
            # Get the tmp target Tensor
            tmp_target=torch.zeros(30, dtype=torch.float64, device=device)
            tmp_target[(label-1)] = 1.0

            # trans from bbox to raio and add_to  xy_wh Tensor
            cur_x, cur_y, x_ratio, y_ratio, w_ratio, h_ratio=from_bbox_to_ratio(x1,y1,x2,y2) # CUDA tensor
            xy_tensor=torch.as_tensor((x_ratio, y_ratio), device=device,dtype=torch.float64)
            wh_tensor=torch.as_tensor((w_ratio, h_ratio), device=device, dtype=torch.float64)

            # IOU calculating
            idx, max_iou = caculate_IOU\
                (x_ratio, y_ratio, w_ratio, h_ratio, output[bt, :, cur_y, cur_x], device, sigma=0.0)

            tmp_target[20+idx] = max_iou

            # MAKING  TMP TARGET
            # FINISHED!

            # Crop area for easier calculation
            xy_crop = output[bt, 22+4*idx:24+4*idx, cur_y, cur_x,]
            wh_crop = output[bt, 24+4*idx:26+4*idx, cur_y, cur_x, ]
            C_crop = output[bt,  20+idx, cur_y, cur_x,]
            pr_crop = output[bt, 0:20, cur_y, cur_x,]
            
            '''
            final_iou = 0.0
            for a in range(3):
                for b in range(3):
                    y = max(cur_y-1+a, 0)
                    y = min(y, 13)
                    x = max(cur_x-1+b, 0)
                    x = min(x, 13)
                    idx, max_iou = caculate_IOU\
                        (x_ratio, y_ratio, w_ratio, h_ratio, output[bt, :, y, x], device,
                         sigma=0.00)
                    if max_iou > final_iou:
                        final_iou = max_iou
            

            # print("output_C:{}, cal_iou{}".format(C_crop, max_iou))
            with open("results/C_crop.txt", "a") as f1:
                line = "{:.3f}\n".format(C_crop)
                f1.writelines(line)

            with open("results/iou.txt", "a") as f2:
                line = "{:.3f}\n".format(max_iou)
                f2.writelines(line)

            with open("results/pr.txt", "a") as f3:
                line = "{},{}\n".format(label-1, pr_crop.argmax())
                f3.writelines(line)
            '''
            
            # Loss beggining
            x_y_loss[bt] = x_y_loss[bt] + torch.sum(torch.pow(xy_tensor - xy_crop, 2))
            w_h_loss[bt] = w_h_loss[bt] + torch.sum(torch.pow(torch.sqrt(wh_tensor) - torch.sqrt(wh_crop), 2))
            C_loss[bt] = C_loss[bt] + torch.pow(max_iou - C_crop, 2)
            pr_loss[bt] = pr_loss[bt] + torch.sum(torch.pow(tmp_target[:20] - pr_crop, 2))
            extra_loss += torch.pow(output[bt, 20:22, cur_y, cur_x], 2).sum()

        #Sum no_obj_loss
        no_obj_loss[bt] = no_obj_loss[bt] + torch.sum( torch.pow(output[bt,20:22,:,:],2) )
        #Subtract more loss
        no_obj_loss[bt] = no_obj_loss[bt] - extra_loss.sum()

        num_all += obj_num

    Utra_params = torch.as_tensor((
        lamda_coord,
        lamda_coord,
        2.5,
        lamda_no_obj,
        1.5
    ), dtype=torch.float64, device=device)

    Loss_tensor = torch.as_tensor((
        x_y_loss.sum(),
        w_h_loss.sum(),
        C_loss.sum(),
        no_obj_loss.sum(),
        pr_loss.sum()
    ), dtype=torch.float64, device=device)
    Total_loss=\
        x_y_loss.sum() * Utra_params[0]+\
        w_h_loss.sum() * Utra_params[1]+\
        C_loss.sum() * Utra_params[2]+\
        no_obj_loss.sum() * Utra_params[3]+\
        pr_loss.sum() * Utra_params[4]

    return Total_loss, Loss_tensor, num_all



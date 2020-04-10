taset import VOC_Data, VOC_2012
from yolonet import YoloNet
import torch
from torch.utils.data import DataLoader, Subset
from train import train
from test import test
from lib.utils import collate_fn
import argparse
import time

p = argparse.ArgumentParser(description="Adjust Your Own Ultra Parameters")


#Set Ultra Params Fisrt
p.add_argument(
    "-s"
    "--start_epoch",
    type = int,
    default = 0,
    help = "from which loaded_epoch model to continue training"
)
p.add_argument(
    "-l"
    "--learning_rate",
    type = float,
    default = 0.003,
    required = False,
    help = "The learing rate"
)
p.add_argument(
    "-m",
    "--momentum",
    type = float,
    default = 0.9,
    required = False,
    help = "The momentum Method"
)
p.add_argument(
    "-w",
    "--weight_decay",
    type = float,
    default = 0.0001,
    required = False,
    help = "Your Own weight_decay"
)
p.add_argument(
    "--c_thresh",
    type = float,
    default = 0.35,
    required = False,
    help = "The prediction confidence threshold of the Boxes"
)
p.add_argument(
    "--nms_thresh",
    type = float,
    required = False,
    default = 0.3,
    help = "The NMS method during Test Time"
)
p.add_argument(
    "-p",
    "--print_freq",
    type = int,
    required = False,
    default = 10,
    help = "The batch freqency to show results"
)
p.add_argument(
    "--testprint",
    type=int,
    required=False,
    default= 100
)
p.add_argument(
    "-b",
    "--batch_size",
    type = int,
    required = False,
    default = 16,
)


def main(lr, momentum, weight_decay, C_thresh, nms_thresh ,print_freq, start_epoch, test_print, batch_size):
    # Params
    print("learning_rate:{} momentum:{} weight_decay:{}".format(lr, momentum, weight_decay))
    print("C_thresh(during Test):{} NMS_thresh(during test):{}".format(C_thresh, nms_thresh))
    print("print_frequency:{}".format(print_freq))
    print("Test Print:{}".format(test_print))
    print("Batch_size:{}\n".format(batch_size))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("You are using {}".format(device))

    # Loading data
    dataset_07 = VOC_Data("../Image/VOC2007/", transforms=True)
    dataset_12 = VOC_2012("../Image/VOC2012/", transforms=True)
    dataset_test = VOC_Data("../Image/VOC2007/",transforms=False)
    print("Loading Data from VOC2007 and VOC2012...")

    # get the idx of train test
    trainval_idx_07, test_idx = dataset_07.train_test_idx()
    trainval_idx_12 = dataset_12.train_test_idx()
    print("{} Images for train_07".format(len(trainval_idx_07)))
    print("{} Images for train_12".format(len(trainval_idx_12)))
    print("{} Images for test\n".format(len(test_idx)))

    # Seperate data
    trainval07_data = Subset(dataset_07, trainval_idx_07)
    trainval12_data = Subset(dataset_12, trainval_idx_12)
    test_data=Subset(dataset_test, test_idx)
    #print(test_idx)

    # DataLoader
    trainval07_loader = DataLoader(trainval07_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    trainval12_loader = DataLoader(trainval12_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader=DataLoader(test_data,batch_size=1,shuffle=False,num_workers=4,collate_fn=collate_fn)

    # Model Init
    model = YoloNet()
    if start_epoch > 0:   # Train from existing epoch models
        model.load_state_dict(torch.load("save_models/Yolo_v1.pth{}".format(start_epoch-1)))
        print("\nUsing Yolo_v1.pth{}..".format(start_epoch-1))
    else:  # Train from scratch
        pass
        print("Train from scratch...")
    time.sleep(1.5)

    # Params lr
    weight_p, bias_p = [], []
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p.append(p)
        else:
            weight_p.append(p)
    optimizer=torch.optim.SGD(
        [
            {'params': weight_p, 'weight_decay': weight_decay},
            {'params': bias_p, 'weight_decay': 0}
        ]
    , lr=lr, momentum=momentum)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    for epoch_id in range(start_epoch, 100):
        print("lr_scheduler", lr_scheduler.state_dict())
        print("optimizer", optimizer.state_dict()['param_groups'][0]['lr'], "\n")

        train(model, trainval07_loader, test_loader, optimizer, device, epoch_id, print_freq, test_print, batch_size)
        train(model, trainval12_loader, test_loader, optimizer, device, epoch_id, print_freq, test_print, batch_size)
        torch.save(model.state_dict(), "save_models/Yolo_v1.pth{}".format(epoch_id))
        lr_scheduler.step()

        test(model, test_loader, device, start_epoch, C_thresh, nms_thresh, test_idx)

# Decompose the argv and to dict
argv = vars(p.parse_args())
lr = argv["l__learning_rate"]
momentum = argv["momentum"]
weight_decay = argv["weight_decay"]
C_thresh = argv["c_thresh"]
nms_thresh = argv["nms_thresh"]
print_freq = argv["print_freq"]
start_epoch = argv["s__start_epoch"]
batch_size = argv["batch_size"]
test_print = argv["testprint"]

if __name__ =="__main__":
    main(lr=lr,
         momentum=momentum,
         weight_decay=weight_decay,
         C_thresh=C_thresh,
         nms_thresh=nms_thresh,
         print_freq=print_freq,
         start_epoch=start_epoch,
         batch_size=batch_size,
         test_print=test_print
    )





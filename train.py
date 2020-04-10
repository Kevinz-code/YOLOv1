from cal_loss import calculate_loss
import torch
import time
import os

def train(model, trainval_loader, test_loader, optimizer, device, epoch, print_freq, test_print, batch_size):
    print("\nStart training!")
    # training begining
    model.to(device)
    model.train()
    one_epoch_loss=torch.zeros(1,dtype=torch.float64,device=device)
    # Set print_freq loss to zero
    print_freq_loss=torch.zeros(1,dtype=torch.float64,device=device)
    print_freq_loss_tesor=torch.zeros(5,dtype=torch.float64,device=device)
    print_freq_loss.requires_grad=False
    print_freq_loss_tesor.requires_grad=False

    for i,(img,target) in enumerate(trainval_loader,0):
        # First zero grad
        optimizer.zero_grad()

        # Setting transformation for Image and target
        # And to CUDA
        target=[{k:v.to(device) for k,v in t.items()} for t in target]
        img=torch.stack(tuple(item for item in img),dim=0).to(device)
        img.to(torch.float64)

        # forward functions img.to(device)
        t1_model=time.time()
        output=model(img).to(torch.float64)
        t2_model=time.time()

        # Loss Tensor Structure:
        # xy_loss,
        # wh_loss,
        # C_loss,
        # no_obj_loss,
        # pr_loss
        t1_loss=time.time()
        Total_loss_batch, Loss_tensor_batch, num_per_batch=\
            calculate_loss(target,output,device=device)
        t2_loss=time.time()

        print_freq_loss += Total_loss_batch/num_per_batch
        print_freq_loss_tesor += Loss_tensor_batch/num_per_batch
        one_epoch_loss += Total_loss_batch /num_per_batch

        # backward
        (Total_loss_batch/ batch_size).backward()
        optimizer.step()

        if i%print_freq==0:
            print("[{}] ({}/{})".format(epoch,i,len(trainval_loader)),end=" ")
            print("Average loss per Object:{:.5f}".format(print_freq_loss[0]/print_freq))
            print("bbox_loss {:.4f} C_loss:{:.4f} No_Obj_loss:{:.8f} Pr_loss:{:.4f} ".format(
                (print_freq_loss_tesor[0]+print_freq_loss_tesor[1])/print_freq,
                print_freq_loss_tesor[2]/print_freq,
                print_freq_loss_tesor[3]/print_freq,
                print_freq_loss_tesor[4]/print_freq), end="  ")
            print("Model time: {:.4f} Loss time {:.4f}".format
                  (float(t2_model-t1_model), float(t2_loss-t1_loss)))

            print_freq_loss.zero_()
            print_freq_loss_tesor.zero_()

    print("One epoch Loss",one_epoch_loss/(len(trainval_loader)+1))
    with open("results/trainloss.txt","a") as f:
        f.writelines("[{}] loss: {}\n".format(epoch,one_epoch_loss/(len(trainval_loader)+1) ))
    print("")
    #End one epoch Training

    # Validation
    print("Start Validation")
    time.sleep(2)
    optimizer.zero_grad()

    print_freq_loss.zero_()
    print_freq_loss_tesor.zero_()
    one_epoch_loss.zero_()

    model.eval()
    with torch.no_grad():
        for i,(img,target) in enumerate(test_loader, 1):
            target = [{k:v.to(device) for k,v in t.items()} for t in target]
            img = img[0].unsqueeze(0).to(device).to(torch.float32)

            t1=time.time()
            output=model(img).to(torch.float64)
            t2=time.time()

            t3=time.time()
            Total_loss_batch, Loss_tensor_batch, num_per_batch=\
                calculate_loss(target, output, device)
            t4=time.time()

            print_freq_loss += Total_loss_batch/num_per_batch
            print_freq_loss_tesor += Loss_tensor_batch/num_per_batch
            one_epoch_loss += Total_loss_batch/ 1.0

            if i % test_print == 0:
                print("[T_{}] ({}/{})".format(epoch, i, len(test_loader)), end=" ")
                print("Average loss per Object:{:.5f}".format(print_freq_loss[0] / test_print))
                print("bbox_loss {:.4f} C_loss:{:.4f} No_Obj_loss:{:.8f} Pr_loss:{:.4f} ".format(
                    (print_freq_loss_tesor[0] + print_freq_loss_tesor[1]) / test_print,
                    print_freq_loss_tesor[2] / test_print,
                    print_freq_loss_tesor[3] / test_print,
                    print_freq_loss_tesor[4] / test_print), end="  ")
                print("Model time: {:.4f} Loss time {:.4f}".format
                      (float(t2 - t1), float(t4 - t3)))

                print_freq_loss.zero_()
                print_freq_loss_tesor.zero_()

    print("One epoch Loss",one_epoch_loss/(len(test_loader)))
    with open("results/testloss.txt","a") as f:
        f.writelines("[{}] loss: {}\n".format(epoch,one_epoch_loss/(len(test_loader)) ))
    print("")

    # End one epoch Testing






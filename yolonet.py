from backbone.resnet import *
import torch.nn as nn
import torch

device = torch.device("cuda:0")


class my_conv2d(nn.Module):
    def __init__(self, in_channel,out_channel,kernel,pad,stride=1,dp=0):
        #Necessary Init
        super(my_conv2d,self).__init__()

        self.dp_tag = dp

        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel,stride=stride,padding=pad,groups=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.dp = nn.Dropout(p=0.5)
        self.l_relu = nn.LeakyReLU(negative_slope=0.1,inplace=True)

        # weight init
        assert isinstance(self.conv1, nn.Conv2d)
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity='leaky_relu')
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)
        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        if self.dp_tag == 1:
            x = self.dp(x)

        x = self.l_relu(x)

        return x


def minmaxscalr_forone(tensor):
    Max=torch.max(tensor)
    Min=torch.min(tensor)

    if Max<1.0 and Min>0.0:
        return tensor
    else :
        return (tensor-Min)/(Max-Min +0.005)

def optional_squeeze(tensor, batch):
    back=torch.zeros((batch,10,7,7),dtype=torch.float32,device=device)
    for k in range(0,10):
        back[:,k,:,:]=minmaxscalr_forone(tensor[:,k,:,:])

    return back

class YoloNet(nn.Module):
    def __init__(self):
        super(YoloNet,self).__init__()

        # download resnet18
        self.backbone = resnet50(pretrained=True, progress=True).to(device=device)
        # 1-3
        self.add_conv1=my_conv2d(2048,512,kernel=1,pad=0,dp=0)
        self.add_conv2=my_conv2d(512,512,kernel=3,pad=1,dp=0)
        self.add_conv3=my_conv2d(512,30,kernel=1,pad=0,dp=0)
        self.add_conv4=my_conv2d(30,30,kernel=3,pad=1,dp=0)

        self.sm = nn.Softmax(dim=1)
        self.sig = nn.Sigmoid()

        # Weight Init for the last Layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.backbone(x)
        x = self.add_conv1(x)
        x = self.add_conv2(x)
        x = self.add_conv3(x)
        x = self.add_conv4(x)


        #x = self.conv1(x)
        #x = self.conv2(x)
        #x = self.conv3(x)
        #x = self.conv4(x)

        # x=self.conv5(x)
        # x=self.conv6(x)
        # x=self.conv7(x)

        # fc
        #x = x.view(-1, 7 * 7 * 1024)
        #x = self.fc1(x)
        #x = self.dp(x)
        #x = self.leaky_relu(x)
        #x = self.fc2(x)

        # Reshape to 30*7*7
        #x = x.reshape(-1, 30, 7, 7)

        # tmp_front=self.sm(x[:, :20, :, :])
        # tmp_back=self.sig(x[:, 20:, :, :]) +0.001
        # tmp_back=optional_squeeze(x[:,20:,:,:], batch=x.size()[0])
        # tmp_back=torch.clamp(x[:,20:,:,:],0,1)
        # x=torch.cat((tmp_front,tmp_back),dim=1)

        tmp_front = self.sm(x[:, :20, :, :])
        tmp_back = self.sig(x[:, 20:, :, :])+0.00001

        #x=self.sigmoid(x)+0.00001
        x=torch.cat((tmp_front,tmp_back),dim=1)

        return x



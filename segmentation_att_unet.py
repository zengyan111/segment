import torch
from torch import nn
from segmentation_att import CBAtten_res, chan_att2
import torchvision
import torchvision.models as models
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self,input):
        return self.conv(input)
    
class UnetCbam(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UnetCbam,self).__init__()

        # through 2 layer decoder
        self.conv1 = DoubleConv(in_ch,64)
        # extract main features
        self.pool1 = nn.MaxPool2d(2)
 
        # upcoder
        self.conv2 = DoubleConv(64,128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512,1024)
        # 反卷积:隔一次采样，做一次注意力控制
        self.up6 = nn.ConvTranspose2d(1024,512,2,stride=2)
        self.c1 = CBAtten_res(1024,1024,reduction=16,stride=1)
        self.conv6 = DoubleConv(1024,512)

        self.up7 = nn.ConvTranspose2d(512,256,2,stride=2)
        self.conv7 = DoubleConv(512,256)

        self.up8 = nn.ConvTranspose2d(256,128,2,stride=2)
        self.c2 = CBAtten_res(256,256,reduction=8,stride=1)
        self.conv8 = DoubleConv(256,128)

        self.up9 = nn.ConvTranspose2d(128,64,2,stride=2)
        self.conv9 = DoubleConv(128,64)
        self.conv10 = nn.Conv2d(64,out_ch,1)
    
    def forward(self,x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6,c4],dim=1)
        merge6 = self.c1(merge6)
        c6 = self.conv6(merge6)

        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7,c3],dim=1)
        c7 = self.conv7(merge7)

        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8,c2],dim=1)
        merge8 = self.c2(merge8)
        c8 = self.conv8(merge8)
        # print(f"Shape of up_8: {up_8.shape}")
        up_9 = self.up9(c8)
        # print(f"Shape of up_9: {up_9.shape}")
        merge9 = torch.cat([up_9,c1],dim=1)
        c9 = self.conv9(merge9)

        c10 = self.conv10(c9)
        if c10.shape[1] == 1 or c10.shape[1] == 2:
            out = nn.Sigmoid()(c10)
        else:
            out = torch.softmax(c10,dim=1)
        return out
    

class UnetCabm_drop(nn.Module):
    def __init__(self, in_ch, out_ch, drop_rate = 0.3):
        super(UnetCabm_drop, self).__init__()
        # gain features:downsampling image depth and WH reduce half
        self.conv1 = DoubleConvD2(in_ch, 64, drop_rate)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = DoubleConvD2(64, 128, drop_rate)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConvD2(128, 256, drop_rate)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConvD2(256, 512, drop_rate)
        self.pool4 = nn.MaxPool2d(2)
        # finish downsampling
        self.conv5 = DoubleConvD2(512, 1024, drop_rate)

        # strat upsampling:image enhance WH, the depth of image becomes shallower
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.c1 = CBAtten_res(1024, 1024, reduction=16, stride=1)
        self.conv6 = DoubleConvD2(1024, 512, drop_rate)

        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = DoubleConvD2(512, 256, drop_rate)

        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.c2 = CBAtten_res(256, 256, reduction=16, stride=1)
        self.conv8 = DoubleConvD2(256, 128, drop_rate)

        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = DoubleConvD2(128, 64, drop_rate)

        self.conv10 = nn.Conv2d(64, out_ch, kernel_size=1)
    
    
    def forward(self,x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        c5 = self.conv5(p4)
        # start upsampling
        up6 = self.up6(c5)
        merge6 = torch.cat([up6, c4], dim=1)
        merge6 = self.c1(merge6)
        c6 = self.conv6(merge6)

        up7 = self.up7(c6)
        merge7 = torch.cat([up7, c3], dim=1)
        c7 = self.conv7(merge7)

        up8 = self.up8(c7)
        merge8 = torch.cat([up8, c2], dim=1)
        merge8 = self.c2(merge8)
        c8 = self.conv8(merge8)

        up9 = self.up9(c8)
        merge9 = torch.cat([up9, c1], dim=1)
        c9 = self.conv9(merge9)

        c10 = self.conv10(c9)
        if c10.shape[1] <= 2:
            out = nn.Sigmoid()(c10)
        else:
            out = torch.softmax(c10, dim = 1)
        return out


# add dropout to reduce overfiftting
class DoubleConvD2(nn.Module):
    def __init__(self, in_ch, out_ch, drop_rate = 0.3):
        super(DoubleConvD2, self).__init__()
        self.conv = nn.Sequential(
            # extract features
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            # prevent oscillation and overfitting
            nn.BatchNorm2d(out_ch),
            # rectified linear to fitting complex function
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # froce the network not to rely on a single feature, prevent overfitting 
            nn.Dropout2d(p=drop_rate)
        )
    
    def forward(self, input):
        return self.conv(input)
    

class ResUnet(nn.Module):
    def __init__(self, in_ch, out_ch, feature = 16, kernel = 3):
        super(ResUnet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding = 1),
            nn.InstanceNorm2d(16),
            nn.SiLU(inplace = True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding = 1),
            nn.InstanceNorm2d(16),
            nn.SiLU(inplace = True)
        )
        # c16 128, c32 256, c64 512
        #ResidualConvTrans
        self.resnet = ResNet(16)
        self.up1 = ResConvTrans(512, 256)
        self.up2 = nn.Sequential(
            chan_att2(512, 128),
            ResConvTrans(128, 128),
        )
        self.up3 = nn.Sequential(
            chan_att2(256, 128),
            ResConvTrans(128, 128),
            ResConvTrans(128, 128),
            ResConvTrans(128, 32),
        )
        self.up4 = nn.Sequential(
            chan_att2(48, 16),
            ResConvTrans(16, 16, stride = 1),
        )

        self.fusion = nn.Sequential(
            chan_att2(32, 16),
            ResConv(16),
            ResConv(16),
            ResConv(16),
        )
        self.out = nn.Sequential(
            nn.Conv2d(16, out_ch, 3, 1, 1, bias=False),
        )
    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        feat8x, feat16x, feat32x = self.resnet(c2)
        feat16x = torch.cat([self.up1(feat32x), feat16x], dim=1)
        feat8x  = torch.cat([self.up2(feat16x), feat8x],  dim=1)
        c2 = torch.cat([self.up3(feat8x), c2], dim=1)
        c3 = self.up4(c2)
        if c3.shape[2] != c1.shape[2] or c3.shape[3] != c1.shape[3] :
            c3 = F.interpolate(c3, size=(c1.size(2), c1.size(3)), mode='bilinear', align_corners=False)
        c1 = torch.cat([c3, c1], dim=1)
        c1 = self.fusion(c1)
        return self.out(c1)
        
  
class ResConvTrans(nn.Module):
    def __init__(self, in_ch, out_ch, stride = 2):
        super(ResConvTrans, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_ch ,out_ch, 2, stride = stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
    def forward(self, x):
        x = self.conv1(x)
        x2 = self.conv2(x)
        return x + x2

class ResConv(nn.Module):
    def __init__(self, in_ch):
        super(ResConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace = True),
        )
    def forward(self, x):
        c1 = self.conv1(x)
        return c1 + x

class ResNet(nn.Module):
    def __init__(self, in_ch, model_fn = models.resnet18):
        super(ResNet, self).__init__()
        backbone = model_fn(pretrained = False) 
        self.conv1 = backbone.conv1
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False) if in_ch != 3 else backbone.conv1,
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2  #128
        self.layer3 = backbone.layer3  #256
        self.layer4 = backbone.layer4  #512

    def forward(self, x):
        feat4x      = self.in_conv(x)
        feat4x      = self.layer1(feat4x)
        feat8x      = self.layer2(feat4x)
        feat16x     = self.layer3(feat8x)
        feat32x     = self.layer4(feat16x)
        return feat8x, feat16x, feat32x


if __name__ == '__main__':
    device = torch.device('cuda')
    x = torch.zeros((4, 140, 256,256)).to(device)
    # unet = UnetCbam(140,100).to(device)
    unet = ResUnet(140,100).to(device)
    out = unet(x)
    print(out.shape)


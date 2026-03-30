import torch
import torch.nn as nn

class chan_att2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(chan_att2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace = True),
        )
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_ch, out_ch, 1, 1, 0,bias=True),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = self.conv(x)
        at = self.att(x)
        return x * at


class ChannleAttention(nn.Module):
    def __init__(self, in_channels, reduction = 4, batch_frist = True):
        super(ChannleAttention,self).__init__()
        self.batch_frist = batch_frist
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # get pre channle weights，reduction control channels scaling
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1, bias= True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels,kernel_size=1, bias=True)
        )
        # get important feature
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        # get queeue : batch, channels, weight, height
        if not self.batch_frist:
            x = x.permute(1, 0, 2, 3)
        # get MLP weights score
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        # get weight and attention for channels 
        weights = self.sigmoid(avgout + maxout)
        atten_channels_score = x * weights.expand_as(x)
        if not self.batch_frist:
            x = x.permute(1,0,2,3)

        return atten_channels_score


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size = 3, batch_frist = True):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 5, 7),"kernel size must be 3 or 7"
        padding = kernel_size // 2

        self.batch_frist = batch_frist
        self.conv = nn.Conv2d(2, 1, kernel_size, padding = padding, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if not self.batch_frist:
            x = x.permute(1, 0, 2, 3)

        avgout = torch.mean(x , dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avgout,maxout], dim=1)
        x1 = self.conv(x1)
        weights = self.sigmoid(x1)
        spatial_attention = x *weights
        return spatial_attention

class CBAtten_res(nn.Module):
    """
        CBAM:channel attention,spatial anttention ,resnet
        in_channls: input number, size: (batch,in_channls,weight,height), if frist_bath == False   --> (in_clannls, batch, weight,height)
        kernel:3
        stride: 2
        reduction: 4
        batch_frist: Ture
    """
    def __init__(self, in_clannls, out_channels, kernel_size = 3, stride = 2, reduction = 4,batch_frist = True):
        super(CBAtten_res,self).__init__()
        self.batch_frist = batch_frist
        self.reduction = reduction
        self.padding = kernel_size // 2

        # decoder pooling 
        self.max_pooling = nn.MaxPool2d(3,stride = stride, padding = self.padding)
        # change channle depth
        self.con_res = nn.Conv2d(in_clannls, out_channels, kernel_size = 1, stride = 1, bias = True)
        # get pre part spatial feature
        self.conv1 = nn.Conv2d(in_clannls, out_channels, kernel_size= kernel_size, stride= stride, padding=self.padding, bias= True)
        # stablity training and accelerate train processing
        self.bn1 = nn.BatchNorm2d(out_channels)
        # nonliner-activation: get more feature and avoid gradient missing 
        self.relu = nn.ReLU(inplace=True)
        # add channle feature attention
        self.channels_attention = ChannleAttention(out_channels, reduction = self.reduction, batch_frist = self.batch_frist)
        # add spatial feature attention
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size, batch_frist=self.batch_frist)

    def forward(self,x):
        if not self.batch_frist:
            x = x.permute(1, 0, 2, 3)
        residual = x
        # (batch, in_channels, weight,height)

        out = self.conv1(x)  # (batch, out_channels, weight, height)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.channels_attention(out)
        out = self.spatial_attention(out)

        residual = self.max_pooling(residual)
        residual = self.con_res(residual)

        out += residual
        out = self.relu(out)

        if not self.batch_frist:
            out = out.permute(1, 0, 2, 3)
        return out 
        

if __name__ == '__main__':
    x = torch.randn(size=[4, 8, 20, 20])
    cba = CBAtten_res(8, 16, reduction=2, stride=1)
    y = cba(x)
    print("y.size: ", y.size())

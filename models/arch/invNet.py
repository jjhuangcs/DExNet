import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter
import math

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

    def inverse(self, *inputs):
        for module in reversed(self._modules.values()):
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class ResBlock(nn.Module):
    def __init__(self, in_ch, f_ch, f_sz, dilate=1):
        super(ResBlock, self).__init__()

        if_bias = True
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=f_ch, kernel_size=f_sz,
                               padding=math.floor(f_sz / 2) + dilate - 1, dilation=dilate, bias=if_bias,
                               groups=1)
        # torch.nn.init.kaiming_uniform_(self.conv1.weight, a=0, mode='fan_out', nonlinearity='relu')
        self.conv2 = nn.Conv2d(in_channels=f_ch, out_channels=in_ch, kernel_size=f_sz,
                              padding=math.floor(f_sz / 2) + dilate - 1, dilation=dilate, bias=if_bias,
                               groups=1)
        # torch.nn.init.kaiming_uniform_(self.conv2.weight, a=0, mode='fan_out', nonlinearity='relu')

        self.relu = nn.LeakyReLU() # ReLU

    def forward(self, x):
        return self.relu(x + self.conv2(self.relu(self.conv1(x))))




''' FROM ERRNet 20230902'''

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, f_ch, f_sz, dilate=1):
        super(ResidualBlock, self).__init__()

        if_bias = True
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=f_ch, kernel_size=f_sz,
                               padding=math.floor(f_sz / 2) + dilate - 1, dilation=dilate, bias=if_bias,
                               groups=1)
        self.conv2 = nn.Conv2d(in_channels=f_ch, out_channels=in_ch, kernel_size=f_sz,
                              padding=math.floor(f_sz / 2) + dilate - 1, dilation=dilate, bias=if_bias,
                               groups=1)

        self.relu = nn.ReLU() # ReLU
        self.se_layer = SELayer(in_ch)

    def forward(self, x):
        residual  = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        out = self.se_layer(out)
        out = out + residual
        return out

class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, scales=(4, 8, 16, 32), ct_channels=1):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, scale, ct_channels) for scale in scales])
        self.bottleneck = nn.Conv2d(in_channels + len(scales) * ct_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def _make_stage(self, in_channels, scale, ct_channels):
        prior = nn.AvgPool2d(kernel_size=(scale, scale))
        conv = nn.Conv2d(in_channels, ct_channels, kernel_size=1, bias=False)
        relu = nn.LeakyReLU(0.2, inplace=True)
        return nn.Sequential(prior, conv, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = torch.cat([F.interpolate(input=stage(feats), size=(h, w), mode='nearest') for stage in self.stages] + [feats], dim=1)
        return self.relu(self.bottleneck(priors))

''''''

class PUNet(nn.Module):
    def __init__(self, in_ch, out_ch, f_ch, f_sz, num_layers, dilate):
        super(PUNet, self).__init__()

        if_bias = False

        self.layers = []
        self.layers.append(nn.Conv2d(in_ch, f_ch, f_sz, stride=1, padding=math.floor(f_sz / 2), bias=False))
        self.layers.append(nn.ReLU())
        for i in range(num_layers):
            self.layers.append(ResBlock(in_ch=f_ch, f_ch=f_ch, f_sz=f_sz, dilate=1))
        self.net = nn.Sequential(*self.layers)

        self.convOut = nn.Conv2d(f_ch, out_ch, f_sz, stride=1, padding=math.floor(f_sz / 2) + dilate - 1,
                                 dilation=dilate, bias=if_bias)
        self.convOut.weight.data.fill_(0.)

    def forward(self, x):
        x = self.net(x)
        out = self.convOut(x)
        return out


class LiftingStep(nn.Module):
    def __init__(self, pin_ch, f_ch, uin_ch, f_sz, dilate, num_layers):
        super(LiftingStep, self).__init__()

        self.dilate = dilate

        pf_ch = int(f_ch)
        uf_ch = int(f_ch)
        self.predictor = PUNet(pin_ch, uin_ch, pf_ch, f_sz, num_layers, dilate)
        self.updator = PUNet(uin_ch, pin_ch, uf_ch, f_sz, num_layers, dilate)

    def forward(self, xc, xd):
        Fxc = self.predictor(xc)
        xd = - Fxc + xd
        Fxd = self.updator(xd)
        xc = xc + Fxd

        return xc, xd

    def inverse(self, xc, xd):
        Fxd = self.updator(xd)
        xc = xc - Fxd
        Fxc = self.predictor(xc)
        xd = xd + Fxc

        return xc, xd

class invNet(nn.Module):
    def __init__(self, pin_ch, f_ch, uin_ch, f_sz, dilate, num_step, num_layers):
        super(invNet, self).__init__()
        self.layers = []
        for _ in range(num_step):
            self.layers.append(LiftingStep(pin_ch, f_ch, uin_ch, f_sz, dilate, num_layers))
        self.net = mySequential(*self.layers)

    def forward(self, xc, xd):
        for i in range(len(self.net)):
            xc, xd = self.net[i].forward(xc, xd)
        return xc, xd

    def inverse(self, xc, xd):
        for i in reversed(range(len(self.net))):
            xc, xd = self.net[i].inverse(xc, xd)
        return xc, xd
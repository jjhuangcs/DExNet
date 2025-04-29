import torch.nn as nn
from torch.nn import Parameter
import torch
import math
import numpy as np
import pywt
import torch.nn.functional as F
from models.arch.invNet import *

from collections import OrderedDict

'''---------------------------------------------------------------------'''
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

'''---------------------------------------------------------------------'''
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


'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''


class FUM(nn.Module):
    def __init__(self, fxin, chals, fsz, proxNetlayers=1, DW_Expand=1):
        super().__init__()

        SC = 3#False True
        if SC==1:
            # Depth-wise Separable Conv
            self.convDx = nn.Sequential(nn.Conv2d(chals, fxin, 1, stride=1, padding=0, bias=False),
                nn.Conv2d(fxin, fxin, fsz, stride=1, padding=math.floor(fsz / 2), bias=False, groups=fxin))
            self.convDy = nn.Sequential(nn.Conv2d(chals, fxin, 1, stride=1, padding=0, bias=False),
                nn.Conv2d(fxin, fxin, fsz, stride=1, padding=math.floor(fsz / 2), bias=False, groups=fxin))
            self.convDz = nn.Sequential(nn.Conv2d(chals, fxin, 1, stride=1, padding=0, bias=False),
                nn.Conv2d(fxin, fxin, fsz, stride=1, padding=math.floor(fsz / 2), bias=False, groups=fxin))
            self.convDxT = nn.Sequential(nn.ConvTranspose2d(fxin, fxin, fsz, stride=1, padding=math.floor(fsz / 2), bias=False, groups=fxin),
                nn.ConvTranspose2d(fxin, chals, 1, stride=1, padding=0, bias=False))
            self.convDyT = nn.Sequential(nn.ConvTranspose2d(fxin, fxin, fsz, stride=1, padding=math.floor(fsz / 2), bias=False, groups=fxin),
                nn.ConvTranspose2d(fxin, chals, 1, stride=1, padding=0, bias=False))
            self.convDzT = nn.Sequential(nn.ConvTranspose2d(fxin, fxin, fsz, stride=1, padding=math.floor(fsz / 2), bias=False, groups=fxin),
                nn.ConvTranspose2d(fxin, chals, 1, stride=1, padding=0, bias=False))

            self.convDx_ = nn.Sequential(nn.Conv2d(chals, chals * DW_Expand, 1, stride=1, padding=0, bias=False),
                                         nn.Conv2d(chals * DW_Expand, chals * DW_Expand, fsz, stride=1, padding=math.floor(fsz / 2), bias=False, groups=chals * DW_Expand))
            self.convDx_T = nn.Sequential(nn.ConvTranspose2d(chals * DW_Expand, chals * DW_Expand, fsz, stride=1, padding=math.floor(fsz / 2), bias=False, groups=chals * DW_Expand),
                                          nn.ConvTranspose2d(chals * DW_Expand, chals, 1, stride=1, padding=0, bias=False))
            self.convDy_ = nn.Sequential(nn.Conv2d(chals, chals * DW_Expand, 1, stride=1, padding=0, bias=False),
                                         nn.Conv2d(chals * DW_Expand, chals * DW_Expand, fsz, stride=1, padding=math.floor(fsz / 2), bias=False, groups=chals * DW_Expand))
            self.convDy_T = nn.Sequential(nn.ConvTranspose2d(chals * DW_Expand, chals * DW_Expand, fsz, stride=1, padding=math.floor(fsz / 2), bias=False, groups=chals * DW_Expand),
                                          nn.ConvTranspose2d(chals * DW_Expand, chals, 1, stride=1, padding=0, bias=False))
        elif SC==2:
            # Standard Convolution
            self.convDx = nn.Conv2d(chals, fxin, fsz, stride=1, padding=math.floor(fsz / 2), bias=False)
            self.convDy = nn.Conv2d(chals, fxin, fsz, stride=1, padding=math.floor(fsz / 2), bias=False)
            self.convDz = nn.Conv2d(chals, fxin, fsz, stride=1, padding=math.floor(fsz / 2), bias=False)
            self.convDxT = nn.ConvTranspose2d(fxin, chals, fsz, stride=1, padding=math.floor(fsz / 2), bias=False)
            self.convDyT = nn.ConvTranspose2d(fxin, chals, fsz, stride=1, padding=math.floor(fsz / 2), bias=False)
            self.convDzT = nn.ConvTranspose2d(fxin, chals, fsz, stride=1, padding=math.floor(fsz / 2), bias=False)

            self.convDx_ = nn.Conv2d(chals, chals * DW_Expand, fsz, stride=1, padding=math.floor(fsz / 2), bias=False)
            self.convDx_T = nn.ConvTranspose2d(chals * DW_Expand, chals, fsz, stride=1, padding=math.floor(fsz / 2),
                                               bias=False)
            self.convDy_ = nn.Conv2d(chals, chals * DW_Expand, fsz, stride=1, padding=math.floor(fsz / 2), bias=False)
            self.convDy_T = nn.ConvTranspose2d(chals * DW_Expand, chals, fsz, stride=1, padding=math.floor(fsz / 2),
                                               bias=False)
        else:
            # Default Setting
            self.convDx = nn.Conv2d(chals, fxin, fsz, stride=1, padding=math.floor(fsz / 2), bias=False)
            self.convDy = nn.Conv2d(chals, fxin, fsz, stride=1, padding=math.floor(fsz / 2), bias=False)
            self.convDz = nn.Conv2d(chals, fxin, fsz, stride=1, padding=math.floor(fsz / 2), bias=False)
            self.convDxT = nn.ConvTranspose2d(fxin, chals, fsz, stride=1, padding=math.floor(fsz / 2), bias=False)
            self.convDyT = nn.ConvTranspose2d(fxin, chals, fsz, stride=1, padding=math.floor(fsz / 2), bias=False)
            self.convDzT = nn.ConvTranspose2d(fxin, chals, fsz, stride=1, padding=math.floor(fsz / 2), bias=False)

            self.convDx_ = nn.Sequential(nn.Conv2d(chals, chals * DW_Expand, 1, stride=1, padding=0, bias=False),
                                         nn.Conv2d(chals * DW_Expand, chals * DW_Expand, fsz, stride=1,
                                                   padding=math.floor(fsz / 2), bias=False, groups=chals * DW_Expand))
            self.convDy_ = nn.Sequential(nn.Conv2d(chals, chals * DW_Expand, 1, stride=1, padding=0, bias=False),
                                         nn.Conv2d(chals * DW_Expand, chals * DW_Expand, fsz, stride=1,
                                                   padding=math.floor(fsz / 2), bias=False, groups=chals * DW_Expand))
            self.convDx_T = nn.Sequential(nn.ConvTranspose2d(chals * DW_Expand, chals * DW_Expand, fsz, stride=1, padding=math.floor(fsz / 2),
                                   bias=False, groups=chals * DW_Expand),
                                          nn.ConvTranspose2d(chals * DW_Expand, chals, 1, stride=1, padding=0, bias=False))
            self.convDy_T = nn.Sequential(nn.ConvTranspose2d(chals * DW_Expand, chals * DW_Expand, fsz, stride=1, padding=math.floor(fsz / 2),
                                   bias=False, groups=chals * DW_Expand),
                                          nn.ConvTranspose2d(chals * DW_Expand, chals, 1, stride=1, padding=0, bias=False))


        self.PNetT = proxyNetNAF(nlayer=proxNetlayers, nchal=int(chals), fsz=3)
        self.PNetR = proxyNetNAF(nlayer=proxNetlayers, nchal=int(chals), fsz=3)
        self.PNetN = proxyNetNAF(nlayer=proxNetlayers, nchal=int(chals), fsz=3)
        self.PNetA = proxyNetNAF(nlayer=proxNetlayers, nchal=int(chals) * DW_Expand, fsz=3)

        self.tau = nn.Parameter(1 * torch.ones(1), requires_grad=True)
        self.eta = nn.Parameter(1 * torch.ones(1), requires_grad=True)


        self.criterion = nn.L1Loss()

    def forward(self, I, zT, zR, zN, zA):
        '''TFU'''
        RReszT = self.convDxT(I - self.convDx(zT) - self.convDy(zR) - self.convDz(zN))
        GatezT = self.convDx_T(torch.clamp(self.convDy_(zR) * (torch.clamp(self.convDx_(zT) * self.convDy_(zR), -1, 1) - zA), -1, 1)) * self.tau

        zT = zT + RReszT
        zT = zT + GatezT
        zT = self.PNetT(zT)

        '''RFU'''
        RReszR = self.convDyT(I - self.convDx(zT) - self.convDy(zR) - self.convDz(zN))
        GatezR = self.convDy_T(torch.clamp(self.convDx_(zT) * (torch.clamp(self.convDx_(zT) * self.convDy_(zR), -1, 1) - zA), -1, 1)) * self.tau

        zR = zR + RReszR
        zR = zR + GatezR
        zR = self.PNetR(zR)

        '''NFU'''
        RReszE = self.convDzT(I - self.convDx(zT) - self.convDy(zR) - self.convDz(zN))

        zN = zN + RReszE
        zN = self.PNetN(zN)

        '''AFU'''
        GatezA = self.eta * (zA - torch.clamp(self.convDx_(zT) * self.convDy_(zR), -1, 1))

        zA = zA - GatezA
        zA = self.PNetA(zA)

        return zT, zR, zN, zA




'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''
class NAFProxNet(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)#20240310
        self.norm2 = LayerNorm2d(c)#20240310

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.sg = SimpleGate()

    def forward(self, inp):
        x = inp

        x = self.norm1(x)#20240310

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        # x = self.conv4(y)#20240310
        x = self.conv4(self.norm2(y))#20240310
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class proxyNetNAF(nn.Module):
    def __init__(self, nlayer, nchal, fsz):
        super(proxyNetNAF, self).__init__()
        layers = []
        for ii in range(nlayer):
            layers.append(NAFProxNet(c=nchal))
        self.PNet = nn.Sequential(*layers)
    def forward(self, x):
        out = self.PNet(x)
        return out
'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

class quadFeatInput(nn.Module):
    def __init__(self, n_channels, chals):
        super().__init__()
        self.seqv = nn.Sequential(
            nn.Conv2d(n_channels, chals, 1, 1, padding=math.floor(1 / 2), bias=False),
            ResBlock(in_ch=chals, f_ch=chals, f_sz=3)

        )

        self.seqw = nn.Sequential(
            nn.Conv2d(n_channels, chals, 1, 1, padding=math.floor(1 / 2), bias=False),
            ResBlock(in_ch=chals, f_ch=chals, f_sz=3)
        )

        self.seqx = nn.Sequential(
            nn.Conv2d(n_channels, chals, 1, 1, padding=math.floor(1 / 2), bias=False),
            ResBlock(in_ch=chals, f_ch=chals, f_sz=3)
        )

        self.seqy = nn.Sequential(
            nn.Conv2d(n_channels, chals, 1, 1, padding=math.floor(1 / 2), bias=False),
            ResBlock(in_ch=chals, f_ch=chals, f_sz=3)
        )

    def forward(self, v, w, x, y):

        return self.seqv(v), self.seqw(w), self.seqx(x), self.seqy(y)

class triFeatInput(nn.Module):
    def __init__(self, n_channels, chals):
        super().__init__()
        self.seqv = nn.Sequential(
            nn.Conv2d(n_channels, chals, 1, 1, padding=math.floor(1 / 2), bias=False),
            ResBlock(in_ch=chals, f_ch=chals, f_sz=3)
            # NAFProxNet(c=chals)
        )

        self.seqw = nn.Sequential(
            nn.Conv2d(n_channels, chals, 1, 1, padding=math.floor(1 / 2), bias=False),
            ResBlock(in_ch=chals, f_ch=chals, f_sz=3)
            # NAFProxNet(c=chals)
        )

        self.seqx = nn.Sequential(
            nn.Conv2d(n_channels, chals, 1, 1, padding=math.floor(1 / 2), bias=False),
            ResBlock(in_ch=chals, f_ch=chals, f_sz=3)
            # NAFProxNet(c=chals)
        )

    def forward(self, v, w, x):

        return self.seqv(v), self.seqw(w), self.seqx(x)

class quadUps(nn.Module):
    def __init__(self, n_channels, chals):
        super().__init__()
        self.seqv = nn.Sequential(
            nn.Conv2d(chals, 4 * chals, 1, bias=False),
            nn.PixelShuffle(2)
        )

        self.seqw = nn.Sequential(
            nn.Conv2d(chals, 4 * chals, 1, bias=False),
            nn.PixelShuffle(2)
        )

        self.seqx = nn.Sequential(
            nn.Conv2d(chals, 4 * chals, 1, bias=False),
            nn.PixelShuffle(2)
        )

        self.seqy = nn.Sequential(
            nn.Conv2d(chals, 4 * chals, 1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, v, w, x, y):

        return self.seqv(v), self.seqw(w), self.seqx(x), self.seqy(y)


class triUps(nn.Module):
    def __init__(self, n_channels, chals):
        super().__init__()
        self.seqv = nn.Sequential(
            nn.Conv2d(chals, 4 * chals, 1, bias=False),
            nn.PixelShuffle(2)
        )

        self.seqw = nn.Sequential(
            nn.Conv2d(chals, 4 * chals, 1, bias=False),
            nn.PixelShuffle(2)
        )

        self.seqx = nn.Sequential(
            nn.Conv2d(chals, 4 * chals, 1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, v, w, x):

        return self.seqv(v), self.seqw(w), self.seqx(x)

class quadFeatUp(nn.Module):
    def __init__(self, n_channels, chals):
        super().__init__()
        self.seqv = nn.Sequential(
            nn.Conv2d(2 * chals, chals, 1, stride=1, padding=math.floor(1 / 2), bias=False)
        )

        self.seqw = nn.Sequential(
            nn.Conv2d(2 * chals, chals, 1, stride=1, padding=math.floor(1 / 2), bias=False)
        )

        self.seqx = nn.Sequential(
            nn.Conv2d(2 * chals, chals, 1, stride=1, padding=math.floor(1 / 2), bias=False)
        )

        self.seqy = nn.Sequential(
            nn.Conv2d(2 * chals, chals, 1, stride=1, padding=math.floor(1 / 2), bias=False)
        )

    def forward(self, v, w, x, y):

        return self.seqv(v), self.seqw(w), self.seqx(x), self.seqy(y)

class triFeatUp(nn.Module):
    def __init__(self, n_channels, chals):
        super().__init__()
        self.seqv = nn.Sequential(
            nn.Conv2d(2 * chals, chals, 1, stride=1, padding=math.floor(1 / 2), bias=False)
        )

        self.seqw = nn.Sequential(
            nn.Conv2d(2 * chals, chals, 1, stride=1, padding=math.floor(1 / 2), bias=False)
        )

        self.seqx = nn.Sequential(
            nn.Conv2d(2 * chals, chals, 1, stride=1, padding=math.floor(1 / 2), bias=False)
        )

    def forward(self, v, w, x):

        return self.seqv(v), self.seqw(w), self.seqx(x)

class triRecon(nn.Module):
    def __init__(self, n_channels, chals):
        super().__init__()
        self.seqv = nn.Sequential(
            nn.Conv2d(chals, 3, 3, stride=1, padding=math.floor(3 / 2), bias=False)
        )

        self.seqw = nn.Sequential(
            nn.Conv2d(chals, 3, 3, stride=1, padding=math.floor(3 / 2), bias=False)
        )

        self.seqx = nn.Sequential(
            nn.Conv2d(chals, 3, 3, stride=1, padding=math.floor(3 / 2), bias=False)
        )

    def forward(self, v, w, x):

        return self.seqv(v), self.seqw(w), self.seqx(x)

class DExNet(nn.Module):
    '''DURRNet with Residual Modelling'''
    def __init__(self, n_channels, out_channels):
        super(DExNet, self).__init__()
        fxin = 3

        self.chals = 64
        self.layers = 2
        self.scale = 4
        self.fsz = 3

        DW_Expand = 1

        layers_net = []
        for _ in range(self.layers * self.scale):
            layers_net.append(FUM(fxin=fxin, fsz=self.fsz, chals=self.chals, DW_Expand=DW_Expand))
        self.DUSepNet = mySequential(*layers_net)

        self.featinputs = nn.ModuleList()
        self.featupdates = nn.ModuleList()
        self.ups = nn.ModuleList()
        for _ in range(self.scale):
            self.featinputs.append(
                quadFeatInput(n_channels, self.chals)
            )

        for _ in range(self.scale-1):
            '''learnable upsampling'''
            self.ups.append(
                quadUps(n_channels, self.chals)
            )
            self.featupdates.append(
                quadFeatUp(n_channels, self.chals)
            )
        self.recon = triRecon(n_channels, self.chals)

        self.scale_factor = 2
        self.up = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear')  # scale_factor=2
        self.down = nn.Upsample(scale_factor=1. / self.scale_factor, mode='bilinear')  # scale_factor=0.5

    def forward(self, x, y=None, fn=None):
        out_l, out_r, out_rr = [], [], []
        out_A = []
        if y is None:
            y = x

        yd = []
        xd = x
        yd.append(y)
        for s in range(self.scale - 1):
            xd = self.down(xd)
            yd.append(self.down(yd[s]))

        for s in range(self.scale):
            xd = x
            for _ in range(self.scale - s - 1):
                xd = self.down(xd)

            fzT, fzR, fzN, fzA = self.featinputs[s](yd[-s - 1], yd[-s - 1], yd[-s - 1], yd[-s - 1])#0 * yd[-s - 1], 0 * yd[-s - 1]
            if s == 0:
                zT, zR, zN, zA = fzT, fzR, fzN, fzA
            else:
                zT, zR, zN, zA = \
                    self.featupdates[s-1](torch.cat((zT, fzT), 1), torch.cat((zR, fzR), 1),
                                    torch.cat((zN, fzN), 1), torch.cat((zA, fzA), 1)
                                    )

            for i in range(self.layers * s, self.layers * (s + 1)):
                zT, zR, zN, zA = self.DUSepNet[i](xd, zT, zR, zN, zA)
            if s < self.scale - 1:
                zT, zR, zN, zA = self.ups[s](zT, zR, zN, zA)

        out_A.append(zA)
        zT, zR, zN = self.recon(zT, zR, zN)

        out_l.append(zT)
        out_r.append(zR)
        out_rr.append(zN)

        return out_l, out_r, out_rr, out_A

class DExNet_expand(nn.Module):
    '''DURRNet with Residual Modelling'''
    def __init__(self, n_channels, out_channels):
        super(DExNet_expand, self).__init__()
        fxin = 3

        self.chals = 64
        self.layers = 5
        self.scale = 4
        self.fsz = 3

        DW_Expand = 2

        layers_net = []
        for _ in range(self.layers * self.scale):
            layers_net.append(FUM(fxin=fxin, fsz=self.fsz, chals=self.chals, DW_Expand=DW_Expand))
        self.DUSepNet = mySequential(*layers_net)

        self.featinputs = nn.ModuleList()
        self.featinputzA = nn.ModuleList()
        self.featupdates = nn.ModuleList()
        self.featupdatezA = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.upzA = nn.ModuleList()
        for _ in range(self.scale):
            self.featinputs.append(
                triFeatInput(n_channels, self.chals)
            )
            self.featinputzA.append(
                nn.Sequential(
                    nn.Conv2d(n_channels, self.chals*DW_Expand, 1, 1, padding=math.floor(1 / 2), bias=False),
                    ResBlock(in_ch=self.chals*DW_Expand, f_ch=self.chals*DW_Expand, f_sz=3)                )
            )

        for _ in range(self.scale-1):
            '''learnable upsampling'''
            self.ups.append(
                triUps(n_channels, self.chals)
            )
            self.featupdates.append(
                triFeatUp(n_channels, self.chals)
            )
            self.upzA.append(
                nn.Sequential(nn.Conv2d(self.chals*DW_Expand, 4 * self.chals*DW_Expand, 1, bias=False),
                    nn.PixelShuffle(2))
            )
            self.featupdatezA.append(nn.Conv2d(2 * self.chals * DW_Expand, self.chals * DW_Expand, 1, stride=1, padding=math.floor(1 / 2), bias=False))

        self.recon = triRecon(n_channels, self.chals)
        self.scale_factor = 2
        self.up = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear')  # scale_factor=2
        self.down = nn.Upsample(scale_factor=1. / self.scale_factor, mode='bilinear')  # scale_factor=0.5

    def forward(self, x, y=None, fn=None):
        out_l, out_r, out_rr = [], [], []
        out_A = []
        if y is None:
            y = x

        yd = []
        xd = x
        yd.append(y)
        for s in range(self.scale - 1):
            xd = self.down(xd)
            yd.append(self.down(yd[s]))

        for s in range(self.scale):
            xd = x
            for _ in range(self.scale - s - 1):
                xd = self.down(xd)

            fzT, fzR, fzN = self.featinputs[s](yd[-s - 1], yd[-s - 1], yd[-s - 1])#0 * yd[-s - 1], 0 * yd[-s - 1]
            fzA = self.featinputzA[s](yd[-s - 1])
            if s == 0:
                zT, zR, zN, zA = fzT, fzR, fzN, fzA
            else:
                zT, zR, zN = self.featupdates[s-1](torch.cat((zT, fzT), 1), torch.cat((zR, fzR), 1), torch.cat((zN, fzN), 1))
                zA = self.featupdatezA[s-1](torch.cat((zA, fzA), 1))

            for i in range(self.layers * s, self.layers * (s + 1)):
                zT, zR, zN, zA = self.DUSepNet[i](xd, zT, zR, zN, zA)
            if s < self.scale - 1:
                zT, zR, zN = self.ups[s](zT, zR, zN)
                zA = self.upzA[s](zA)

        out_A.append(zA)
        zT, zR, zN = self.recon(zT, zR, zN)

        out_l.append(zT)
        out_r.append(zR)
        out_rr.append(zN)

        return out_l, out_r, out_rr, out_A

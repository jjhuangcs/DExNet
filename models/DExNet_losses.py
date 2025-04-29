import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vgg import Vgg19


def compute_grad(img):
    gradx = img[..., 1:, :] - img[..., :-1, :]
    grady = img[..., 1:] - img[..., :-1]
    return gradx, grady


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, predict, target):
        predict_gradx, predict_grady = compute_grad(predict)
        target_gradx, target_grady = compute_grad(target)

        return self.loss(predict_gradx, target_gradx) + self.loss(predict_grady, target_grady)


class MultipleLoss(nn.Module):
    def __init__(self, losses, weight=None):
        super(MultipleLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        self.weight = weight or [1 / len(self.losses)] * len(self.losses)

    def forward(self, predict, target):
        total_loss = 0
        for weight, loss in zip(self.weight, self.losses):
            total_loss += loss(predict, target) * weight
        return total_loss


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False


class VGGLoss(nn.Module):
    def __init__(self, vgg=None, weights=None, indices=None, normalize=True):
        super(VGGLoss, self).__init__()
        if vgg is None:
            self.vgg = Vgg19().cuda()
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10 / 1.5]
        self.indices = indices or [2, 7, 12, 21, 30]
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        else:
            self.normalize = None

    def forward(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        return loss



class ReconsLossDURRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()#20230908
        # self.criterion = nn.MSELoss()

    def get_loss(self, out_l, out_r, out_rr, input_i):
        content_diff = self.criterion(out_l[-1] + out_r[-1] + out_rr[-1], input_i)
        return content_diff

class ExcluULossDURRNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.criterion = nn.L1Loss(size_average=False)#20230908
        self.criterion = nn.L1Loss()  # 20230908
        # self.criterion = nn.MSELoss()

    def get_loss(self, out_u):
        content_diff = self.criterion(out_u[-1], torch.zeros_like(out_u[-1]))
        return content_diff

class ContentLoss():
    def initialize(self, loss):
        self.criterion = loss

    def get_loss(self, fakeIm, realIm):
        loss = 0
        for i in range(len(fakeIm)):
            if fakeIm[i].shape != realIm.shape:
                realIm_ = nn.functional.upsample(realIm, size=[fakeIm[i].size(2), fakeIm[i].size(3)], mode='bilinear')
                loss = loss + 1 * self.criterion(fakeIm[i], realIm_)
            else:
                loss = loss + 1 * self.criterion(fakeIm[i], realIm)
        # loss = loss / len(fakeIm)
        return loss

class ContentLoss0():
    def initialize(self, loss):
        self.criterion = loss

    def get_loss(self, fakeIm, realIm):
        loss = 1 * self.criterion(fakeIm, realIm)
        return loss

def init_loss(opt):
    loss_dic = {}
    pixel_loss = ContentLoss()
    pixel_loss.initialize(MultipleLoss([nn.MSELoss(), GradientLoss()], [0.3, 0.6]))
    # pixel_loss.initialize(MultipleLoss([nn.MSELoss()], [1]))
    loss_dic['t_pixel'] = pixel_loss
    loss_dic['r_pixel'] = pixel_loss
    loss_dic['recons'] = ReconsLossDURRNet()
    loss_dic['excluU'] = ExcluULossDURRNet()

    return loss_dic


if __name__ == '__main__':
    x = torch.randn(3, 32, 224, 224).cuda()

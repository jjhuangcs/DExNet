import os
from collections import OrderedDict
from os.path import join

import numpy as np
import torch
from PIL import Image

import models.networks as networks
import util.index as index
import util.util as util
from models import arch
from .base_model import BaseModel

import torch.nn.functional as F
import models.DExNet_losses as losses



def tensor2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    # image_numpy = image_numpy.astype(imtype)
    return image_numpy

def tensor2im_woclip(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor[0].cpu().float().numpy()
    # image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    return image_numpy

class DExNetBase(BaseModel):
    def _init_optimizer(self, optimizers):
        self.optimizers = optimizers
        for optimizer in self.optimizers:
            util.set_opt_param(optimizer, 'initial_lr', self.opt.lr)
            util.set_opt_param(optimizer, 'weight_decay', self.opt.wd)

    def set_input(self, data, mode='train'):
        target_t = None
        target_r = None
        data_name = None
        identity = False
        mode = mode.lower()
        if mode == 'train':
            input, target_t, target_r = data['input'], data['target_t'], data['target_r']
        elif mode == 'eval':
            input, target_t, target_r, data_name = data['input'], data['target_t'], data['target_r'], data['fn']
        elif mode == 'test':
            input, data_name = data['input'], data['fn']
        else:
            raise NotImplementedError('Mode [%s] is not implemented' % mode)

        if len(self.gpu_ids) > 0:  # transfer data into gpu
            input = input.to(device=self.gpu_ids[0])
            if target_t is not None:
                target_t = target_t.to(device=self.gpu_ids[0])
            if target_r is not None:
                target_r = target_r.to(device=self.gpu_ids[0])

        self.input = input
        self.identity = identity
        self.target_t = target_t
        self.target_r = target_r
        self.data_name = data_name

        self.issyn = False if 'real' in data else True
        self.aligned = False if 'unaligned' in data else True

    def eval(self, data, savedir=None, suffix=None, pieapp=None):
        # only the 1st input of the whole minibatch would be evaluated
        self._eval()
        self.set_input(data, 'eval')

        with torch.no_grad():
            self.forward()

            output_t = tensor2im(self.output_t[-1])
            output_r = tensor2im(self.output_r[-1])
            output_rr = tensor2im(torch.clip((self.output_rr[-1] + 1) / 2, 0., 1.))
            target = tensor2im(self.target_t)
            target_r = tensor2im(self.target_r)

            if self.aligned:
                res = index.quality_assess(output_t, target)
            else:
                res = {}

            if savedir is not None:
                if self.data_name is not None:
                    name = os.path.splitext(os.path.basename(self.data_name[0]))[0]
                    savedir = join(savedir, suffix, name)
                    os.makedirs(savedir, exist_ok=True)
                    Image.fromarray(output_t.astype(np.uint8)).save(
                        join(savedir, '{}_t.png'.format(self.opt.name)))
                    Image.fromarray(output_r.astype(np.uint8)).save(
                        join(savedir, '{}_r.png'.format(self.opt.name)))
                    Image.fromarray(output_rr.astype(np.uint8)).save(
                        join(savedir, '{}_rr.png'.format(self.opt.name)))

                    Image.fromarray(target.astype(np.uint8)).save(join(savedir, 't_label.png'))
                    Image.fromarray(target_r.astype(np.uint8)).save(join(savedir, 'r_label.png'))
                    Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(join(savedir, 'm_input.png'))
                else:
                    if not os.path.exists(join(savedir, 'transmission_layer')):
                        os.makedirs(join(savedir, 'transmission_layer'))
                        os.makedirs(join(savedir, 'blended'))
                    Image.fromarray(target.astype(np.uint8)).save(
                        join(savedir, 'transmission_layer', str(self._count) + '.png'))
                    Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(
                        join(savedir, 'blended', str(self._count) + '.png'))
                    self._count += 1

            return res

    def test(self, data, savedir=None):
        # only the 1st input of the whole minibatch would be evaluated
        self._eval()
        self.set_input(data, 'test')

        if self.data_name is not None and savedir is not None:
            name = os.path.splitext(os.path.basename(self.data_name[0]))[0]
            if not os.path.exists(join(savedir, name)):
                os.makedirs(join(savedir, name))

            if os.path.exists(join(savedir, name, '{}.png'.format(self.opt.name))):
                return

        with torch.no_grad():
            output_i, output_j, output_rr = self.forward()
            output_i = tensor2im(output_i)
            output_j = tensor2im(output_j)
            output_rr = tensor2im(output_rr)
            if self.data_name is not None and savedir is not None:
                Image.fromarray(output_i.astype(np.uint8)).save(join(savedir, name, '{}_l.png'.format(self.opt.name)))
                Image.fromarray(output_j.astype(np.uint8)).save(join(savedir, name, '{}_r.png'.format(self.opt.name)))
                Image.fromarray(tensor2im(torch.clip((output_rr + 1) / 2, 0., 1.)).astype(np.uint8)).save(
                    join(savedir, name, '{}_rr.png'.format(self.opt.name)))
                Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(join(savedir, name, 'm_input.png'))


class DExNetModel(DExNetBase):
    def name(self):
        return 'DExNet'

    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.if_hypercolumn = True

    def print_network(self):
        print('--------------------- Model ---------------------')
        print('##################### NetG #####################')
        networks.print_network(self.network)

    def _eval(self):
        self.network.eval()

    def _train(self):
        self.network.train()

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        in_channels = 3
        self.vgg = losses.Vgg19(requires_grad=False).to(self.device)
        if self.if_hypercolumn:
            in_channels += 1472
        self.network = arch.__dict__[self.opt.inet](in_channels, 3).to(self.device)
        networks.init_weights(self.network, init_type=opt.init_type)  # using default initialization as EDSR

        if self.isTrain:
            # define loss functions
            self.loss_dic = losses.init_loss(opt)
            ''''''
            vggloss = losses.ContentLoss0()
            vggloss.initialize(losses.VGGLoss(self.vgg))
            self.loss_dic['vgg'] = vggloss
            ''''''
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.network.parameters(),
                                                lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)

            self._init_optimizer([self.optimizer_G])

        if opt.resume:
            self.load(self)

        if opt.no_verbose is False:
            self.print_network()

    def get_loss(self, out_t, out_r, out_rr):
        loss_t_pixel = self.loss_dic['t_pixel'].get_loss(out_t, self.target_t)
        loss_r_pixel = self.loss_dic['r_pixel'].get_loss(out_r, self.target_r)
        loss_t_vgg = self.loss_dic['vgg'].get_loss(out_t[-1], self.target_t)
        loss_recons = self.loss_dic['recons'].get_loss(out_t, out_r, out_rr, self.input)
        loss_exclu_u = self.loss_dic['excluU'].get_loss(self.output_u[-1])
        return loss_t_pixel, loss_r_pixel, loss_t_vgg, loss_recons, loss_exclu_u

    def backward_G(self):
        self.loss_t_pixel, self.loss_r_pixel, self.loss_t_vgg, self.loss_recons, self.loss_excluU = self.get_loss(self.output_t, self.output_r, self.output_rr)
        self.loss_G = self.loss_t_pixel + self.loss_r_pixel + \
                      self.loss_t_vgg * self.opt.lambda_vgg \
                      + self.loss_excluU * self.opt.lambda_excl \
                      + self.loss_recons * self.opt.lambda_rec
        self.loss_G.backward()

    def hyper_column(self, input_img):
        hypercolumn = self.vgg(input_img)
        _, C, H, W = input_img.shape
        hypercolumn = [F.interpolate(feature.detach(), size=(H, W), mode='bilinear', align_corners=False) for
                       feature in hypercolumn]
        input_i = [input_img]
        input_i.extend(hypercolumn)
        input_i = torch.cat(input_i, dim=1)
        return input_i

    def forward(self):
        # without edge
        input_i = self.input
        # if self.vgg is not None:
        if self.if_hypercolumn:
            input_i = self.hyper_column(self.input)
        output_t, output_r, output_rr, output_u = self.network(self.input,
                                                     input_i,
                                                     fn=self.data_name[0] if self.data_name else None)
        self.output_t = output_t
        self.output_r = output_r
        self.output_rr = output_rr
        self.output_u = output_u
        return output_t, output_r, output_rr, output_u


    def optimize_parameters(self):
        self._train()
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()

        self.optimizer_G.step()

    def get_current_errors(self):
        ret_errors = OrderedDict()
        if self.loss_r_pixel is not None:
            ret_errors['R_P'] = self.loss_r_pixel.item()
        if self.loss_t_pixel is not None:
            ret_errors['I_P'] = self.loss_t_pixel.item()
        if self.loss_t_vgg is not None:
            ret_errors['VGG'] = self.loss_t_vgg.item()
        if self.loss_excluU is not None:
            ret_errors['Ex'] = self.loss_excluU.item()
        if self.loss_recons is not None:
            ret_errors['Re'] = self.loss_recons.item()

        ret_errors['lr'] = self.optimizer_G.param_groups[0]['lr']
        ret_errors['seed'] = self.opt.seed
        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input).astype(np.uint8)
        ret_visuals['output_t'] = tensor2im(self.output_t[-1]).astype(np.uint8)
        ret_visuals['output_r'] = tensor2im(self.output_r[-1]).astype(np.uint8)
        ret_visuals['output_rr'] = tensor2im(torch.clip((self.output_rr[-1] + 1) / 2, 0., 1.)).astype(np.uint8)
        ret_visuals['target'] = tensor2im(self.target_t).astype(np.uint8)
        ret_visuals['reflection'] = tensor2im(self.target_r).astype(np.uint8)

        ret_visuals['output_u'] = tensor2im(self.output_u).astype(np.uint8)

        return ret_visuals

    def load(self, model):
        weight_path = model.opt.weight_path
        state_dict = torch.load(weight_path)
        model.network.load_state_dict(state_dict['weight'])
        model.epoch = state_dict['epoch']
        model.iterations = state_dict['iterations']
        return state_dict

    def state_dict(self, save_extra_state=True):
        state_dict = {
            'weight': self.network.state_dict(),
            'epoch': self.epoch,
            'iterations': self.iterations
        }

        if save_extra_state:
            state_dict.update({
                'opt_g': self.optimizer_G.state_dict()
            })

        return state_dict

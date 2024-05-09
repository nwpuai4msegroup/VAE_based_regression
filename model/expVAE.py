from __future__ import print_function

from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import os


class PropBase(object):

    def __init__(self, model, target_layer, unmask=1, z_dim=128, cuda=True, Anomaly_Detection=False, Disentanglement=False):
        self.model = model
        self.cuda = cuda
        self.z_dim = z_dim
        self.unmask = unmask
        self.Anomaly_Detection = Anomaly_Detection
        self.Disentanglement = Disentanglement
        if self.cuda:
            self.model.cuda()
        self.model.eval()
        self.target_layer = target_layer
        self.outputs_backward = OrderedDict()
        self.outputs_forward = OrderedDict()
        self.set_hook_func()

    def set_hook_func(self):
        raise NotImplementedError

    # set the target class as one others as zero. use this vector for back prop
    # def encode_one_hot(self, idx):
    #     one_hot = torch.FloatTensor(1, self.n_class).zero_()
    #     one_hot[0][idx] = 1.0
    #     return one_hot

    # set the target class as one others as zero. use this vector for back prop added by Lezi
    def encode_one_hot_batch(self, z, mu, logvar, mu_avg, logvar_avg):
        arr = np.ones((5, self.z_dim))
        arr[:, self.unmask] = 0
        masked = torch.ByteTensor(arr)
        if self.Anomaly_Detection == False:
            if self.Disentanglement ==False:
                one_hot = z
            elif self.Disentanglement == True:#如果为True为z添加一个mask
                one_hot = z.masked_fill(masked, 0)
        elif self.Anomaly_Detection == True:
            std = torch.exp(0.5 * logvar)
            std_avg = torch.exp(0.5 * logvar_avg)
            mu_anomaly = mu - mu_avg
            std_anomaly = torch.sqrt(torch.square(std) + torch.square(std_avg))
            eps_anomaly = torch.randn_like(std_anomaly)

            #one_hot = eps_anomaly * std_anomaly + mu_anomaly
            one_hot = eps_anomaly.mul(std_anomaly).add_(mu_anomaly)
            if self.Disentanglement ==False:
                one_hot = one_hot
            elif self.Disentanglement == True:
                one_hot = one_hot.masked_fill(masked, 0)
            #one_hot_batch = torch.FloatTensor(z.size()).zero_()

        return one_hot

    def forward(self, x):
        self.preds = self.model(x)
        self.image_size = x.size(-1)
        recon_batch, self.mu, self.logvar = self.model(x)
        return recon_batch, self.mu, self.logvar

    # back prop the one_hot signal
    def backward(self, mu, logvar, mu_avg, logvar_avg):
        self.model.zero_grad()
        z = self.model.reparameterize(mu, logvar)#.cuda()
        one_hot = self.encode_one_hot_batch(z, mu, logvar, mu_avg, logvar_avg)

        if self.cuda:
            one_hot = one_hot.cuda()
        flag = 2
        if flag == 1:
            self.score_fc = torch.sum(F.relu(one_hot * mu))
        else:
            self.score_fc = torch.sum(one_hot)
        self.score_fc.backward(retain_graph=True)

    def get_conv_outputs(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('invalid layer name: {}'.format(target_layer))

class GradCAM(PropBase):

    def set_hook_func(self):
        def func_b(module, grad_in, grad_out):
            self.outputs_backward[id(module)] = grad_out[0].cpu()

        def func_f(module, input, f_output):
            self.outputs_forward[id(module)] = f_output

        for module in self.model.named_modules():
            module[1].register_backward_hook(func_b)
            module[1].register_forward_hook(func_f)

    def normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm.item()

    def compute_gradient_weights(self):
        self.grads = self.normalize(self.grads.squeeze())
        self.map_size = self.grads.size()[2:]
        self.weights = nn.AvgPool2d(self.map_size)(self.grads)

    def generate(self):
        # get gradient
        self.grads = self.get_conv_outputs(
            self.outputs_backward, self.target_layer)
        # compute weithts based on the gradient
        self.compute_gradient_weights()

        # get activation
        self.activiation = self.get_conv_outputs(
            self.outputs_forward, self.target_layer)

        self.weights.volatile = False
        self.activiation = self.activiation[None, :, :, :, :]
        self.weights = self.weights[:, None, :, :, :]
        gcam = F.conv3d(self.activiation, (self.weights), padding=0, groups=len(self.weights))
        gcam = F.relu(gcam)
        gcam = gcam.squeeze(dim=0)
        gcam = F.upsample(gcam, (self.image_size, self.image_size), mode="bilinear")#上采样
        #gcam = torch.abs(gcam)

        # compute attention map for each convolution
        # gcam = self.activiation * self.weights
        # gcam = F.relu(gcam)

        # # average the attention
        # gcam = torch.mean(gcam, dim = 1)[:,None,:,:]


        # # upsamples through interpolation to increase image size
        # gcam = F.interpolate(gcam, (self.image_size, self.image_size),
        #                         mode="bilinear", align_corners=True)


        return gcam



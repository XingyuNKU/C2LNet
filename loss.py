#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models.inception import inception_v3
from math import exp


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.mseLoss = nn.MSELoss()

    def forward(self, image, isReal=True):
        if isReal:
            self.target = torch.tensor(1.0).cuda().expand_as(image)
        else:
            self.target = torch.tensor(0.0).cuda().expand_as(image)
        loss_dis = self.mseLoss(image, self.target)
        return loss_dis


class InceptionEntropyHingeLoss(nn.Module):
    def __init__(self, m=0.5):
        super(InceptionEntropyHingeLoss, self).__init__()
        self.inception = inception_v3(pretrained=True, transform_input=False, aux_logits=True).eval().cuda()
        self.m = m
        for param in self.inception.parameters():
            param.requires_grad = False

    def forward(self, image_ir, image_y, generate_img):
        image_ir = F.interpolate(image_ir, size=(299, 299), mode='bilinear', align_corners=False)
        image_y = F.interpolate(image_y, size=(299, 299), mode='bilinear', align_corners=False)
        generate_img = F.interpolate(generate_img, size=(299, 299), mode='bilinear', align_corners=False)

        def preprocess(x):
            x = (x - 0.5) * 2  # Normalize to [-1, 1]
            x = x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x
            return x

        ir_pred = self.inception(preprocess(image_ir))
        y_pred = self.inception(preprocess(image_y))
        gen_pred = self.inception(preprocess(generate_img))

        def entropy(p):
            p = F.softmax(p, dim=1)
            log_p = torch.log(p + 1e-8)
            return -torch.sum(p * log_p, dim=1).mean()

        entropy_ir = entropy(ir_pred)
        entropy_y = entropy(y_pred)
        entropy_gen = entropy(gen_pred)

        entropy_max = torch.max(entropy_ir, entropy_y)
        loss = torch.clamp(entropy_gen + self.m - entropy_max, min=0)
        return loss


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.guass = Gradient()
        self.target_real_label = torch.tensor(1.0).cuda()
        self.mseLoss = nn.MSELoss()
        self.Cross_loss = nn.MSELoss()
        self.ssim = SSIM_Loss()
        self.l1Loss = nn.L1Loss()
        self.entropy_loss_fn = InceptionEntropyHingeLoss(m=0.5)

    def forward(self, image_vis, image_ir, labels, generate_img, ir_fake_output, vis_fake_output, att_source,
                logits_list):
        image_y = image_vis[:, :1, :, :]
        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(x_in_max, generate_img)

        y_grad = self.guass(image_y)
        ir_grad = self.guass(image_ir)
        generate_img_grad = self.guass(generate_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)

        target_real = self.target_real_label.expand_as(ir_fake_output)
        gan_loss_ir = self.mseLoss(ir_fake_output, target_real)
        gan_loss_vis = self.mseLoss(vis_fake_output, target_real)
        loss_gen = gan_loss_ir + gan_loss_vis

        loss_ssim = ((1 - self.ssim.msssim(generate_img, image_y)) + (1 - self.ssim.msssim(generate_img, image_ir))) / 2

        loss_att = self.Cross_loss(att_source[2], torch.ones_like(att_source[3]))

        loss_entropy = self.entropy_loss_fn(image_ir, image_y, generate_img)

        loss_total = 20 * loss_in + 10 * loss_grad + 1 * loss_gen + 1 * loss_ssim + 0.5 * loss_att + 0.1 * loss_entropy
        return loss_total, loss_in, loss_grad, loss_gen, loss_att, loss_entropy


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


class Gradient(nn.Module):
    def __init__(self, channels=1):
        super(Gradient, self).__init__()
        self.channels = channels
        kernel = [[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False).cuda()

    def __call__(self, x):
        x = F.conv2d(x, self.weight, padding=1, groups=self.channels)
        return x


class SSIM_Loss(nn.Module):
    def __init__(self):
        super(SSIM_Loss, self).__init__()

    def msssim(self, img1, img2, window_size=11, size_average=True, val_range=None, normalize=True):
        device = img1.device
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
        levels = weights.size()[0]
        mssim = []
        mcs = []
        for _ in range(levels):
            sim, cs = self.ssim(img1, img2, window_size=window_size, size_average=size_average, full=True,
                                val_range=val_range)
            mssim.append(sim)
            mcs.append(cs)

            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))

        mssim = torch.stack(mssim)
        mcs = torch.stack(mcs)

        # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
        if normalize:
            mssim = (mssim + 1) / 2
            mcs = (mcs + 1) / 2

        pow1 = mcs ** weights
        pow2 = mssim ** weights
        # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
        output = torch.prod(pow1[:-1] * pow2[-1])
        return output

    def ssim(self, img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        if val_range is None:
            if torch.max(img1) > 128:
                max_val = 255
            else:
                max_val = 1

            if torch.min(img1) < -0.5:
                min_val = -1
            else:
                min_val = 0
            L = max_val - min_val
        else:
            L = val_range

        padd = 0
        (_, channel, height, width) = img1.size()
        if window is None:
            real_size = min(window_size, height, width)
            window = self.create_window(real_size, channel=channel).to(img1.device)

        mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs
        return ret

    def create_window(self, window_size, channel=1):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()


if __name__ == '__main__':
    pass
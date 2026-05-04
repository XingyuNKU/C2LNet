# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBnLeakyRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class ConvBnTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvBnTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return torch.tanh(self.conv(x)) / 2 + 0.5


class ConvLeakyRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


class Sobelxy(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, groups=channels, bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))

    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x


class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)

    def forward(self, x):
        return self.conv(x)


class DenseBlock(nn.Module):
    def __init__(self, channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2 * channels, channels)

    def forward(self, x):
        x = torch.cat((x, self.conv1(x)), dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        return x


class DRC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DRC, self).__init__()
        self.dense = DenseBlock(in_channels)
        self.convdown = Conv1(3 * in_channels, out_channels)
        self.sobelconv = Sobelxy(in_channels)
        self.convup = Conv1(in_channels, out_channels)

    def forward(self, x):
        x1 = self.dense(x)
        x1 = self.convdown(x1)
        x2 = self.sobelconv(x)
        x2 = self.convup(x2)
        return F.leaky_relu(x1 + x2, negative_slope=0.1)


class SA1Attention(nn.Module):
    def __init__(self, channel, reduction=4, output_maps=2):
        super(SA1Attention, self).__init__()
        self.output_maps = output_maps
        self.body = nn.ModuleList()
        for i in range(output_maps):
            self.body.append(nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 3, padding=1),
                nn.BatchNorm2d(channel // reduction),
                nn.ReLU(True),
                nn.Conv2d(channel // reduction, 1, 3, padding=1),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            ))

    def forward(self, input):
        return [att(input) for att in self.body]

class ScalarPredictor(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ScalarPredictor, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        hidden = max(in_channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, _, _ = x.shape
        y = self.gap(x).view(b, c)       # (B, C)
        y = self.fc(y)                   # (B, 2), 已 Sigmoid
        a1 = y[:, 0].view(b, 1, 1, 1)    # (B, 1, 1, 1)
        a2 = y[:, 1].view(b, 1, 1, 1)    # (B, 1, 1, 1)
        return a1, a2

class FusionNet(nn.Module):
    def __init__(self, output, out_channel=16, ngf=64, use_dropout=False, n_blocks=1, padding_type='reflect'):
        super(FusionNet, self).__init__()
        vis_ch = [48, 32, 48]
        inf_ch = [48, 32, 48]
        output = 1

        # Visible branch
        self.vis_conv = ConvLeakyRelu2d(1, vis_ch[0])       # 1 → 48
        self.vis_drc1 = DRC(vis_ch[0], vis_ch[1])           # 48 → 32

        # Infrared branch
        self.inf_conv = ConvLeakyRelu2d(1, inf_ch[0])       # 1 → 48
        self.inf_drc1 = DRC(inf_ch[0], inf_ch[1])           # 48 → 32

        self.MSAB_att = SA1Attention(channel=vis_ch[1] * 2, output_maps=2)

        self.prob_predictor = ScalarPredictor(in_channels=vis_ch[0] + inf_ch[0])  # 96

        self.channel_adjust_vis = Conv1(vis_ch[0], vis_ch[1])  # 48 → 32
        self.channel_adjust_inf = Conv1(inf_ch[0], inf_ch[1])  # 48 → 32

        self.decode4 = ConvBnLeakyRelu2d(vis_ch[1] + inf_ch[1], vis_ch[0])  # 64 → 48
        self.decode3 = ConvBnLeakyRelu2d(vis_ch[0], vis_ch[1])              # 48 → 32
        self.decode2 = ConvBnLeakyRelu2d(vis_ch[1], vis_ch[1])              # 32 → 32
        self.decode1 = ConvBnTanh2d(vis_ch[1], output)                       # 32 → 1

    def get_drc1_features(self, image_vis, image_ir):
        x_vis_origin = image_vis[:, :1]
        x_inf_origin = image_ir

        x_vis_p = self.vis_conv(x_vis_origin)
        x_inf_p = self.inf_conv(x_inf_origin)

        x_vis_p1 = self.vis_drc1(x_vis_p)
        x_inf_p1 = self.inf_drc1(x_inf_p)

        return x_vis_p1, x_inf_p1

    def forward(self, image_vis, image_ir):
        x_vis_origin = image_vis[:, :1]
        x_inf_origin = image_ir

        x_vis_p = self.vis_conv(x_vis_origin)   # 48ch
        x_inf_p = self.inf_conv(x_inf_origin)   # 48ch

        concat_initial = torch.cat((x_vis_p, x_inf_p), dim=1)  # 96ch
        a1, a2 = self.prob_predictor(concat_initial)

        # ===== RB =====
        x_vis_p1 = self.vis_drc1(x_vis_p)   # 32ch
        x_inf_p1 = self.inf_drc1(x_inf_p)   # 32ch

        x_vis_weighted = a1 * x_vis_p1
        x_inf_weighted = a2 * x_inf_p1

        x_vis_p_adjusted = self.channel_adjust_vis(x_vis_p)  # 48 → 32
        x_inf_p_adjusted = self.channel_adjust_inf(x_inf_p)  # 48 → 32

        x_vis_p1_res = x_vis_weighted + (1 - a1) * x_vis_p_adjusted  # 32ch
        x_inf_p1_res = x_inf_weighted + (1 - a2) * x_inf_p_adjusted  # 32ch

        att_SA3 = torch.cat((x_vis_p1_res, x_inf_p1_res), dim=1)  # 64ch
        SAM3, SAM4 = self.MSAB_att(att_SA3)

        fv3 = SAM3 * x_vis_p1_res   # 32ch
        fi3 = SAM4 * x_inf_p1_res   # 32ch

        x = self.decode4(torch.cat((fv3, fi3), dim=1))  # 64 → 48
        x = self.decode3(x)   # 48 → 32
        x = self.decode2(x)   # 32 → 32
        x = self.decode1(x)   # 32 → 1

        return x, [None, None, SAM3, SAM4, x_vis_p1, x_inf_p1], a1, a2, fv3, fi3


def unit_test():
    import numpy as np
    vis = torch.tensor(np.random.rand(2, 1, 480, 640).astype(np.float32))
    ir = torch.tensor(np.random.rand(2, 1, 480, 640).astype(np.float32))
    model = FusionNet(output=1)

    output, att_list, a1, a2, fv3, fi3 = model(vis, ir)
    print('output shape:', output.shape)
    print('fv3 shape:', fv3.shape)
    print('fi3 shape:', fi3.shape)
    print('a1 shape:', a1.shape, 'values:', a1.flatten().tolist())  # (2,1,1,1)
    print('a2 shape:', a2.shape, 'values:', a2.flatten().tolist())
    print('SAM3 shape:', att_list[2].shape)

    x_vis_p1, x_inf_p1 = model.get_drc1_features(vis, ir)
    print('x_vis_p1 shape:', x_vis_p1.shape)
    print('x_inf_p1 shape:', x_inf_p1.shape)

    assert output.shape == (2, 1, 480, 640), 'output shape (2,1,480,640) is expected!'
    print('test ok!')

if __name__ == '__main__':
    unit_test()
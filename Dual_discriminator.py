import torch
import torch.nn as nn

class DisVIS_net(nn.Module):

    def __init__(self, in_channels=32):
        super(DisVIS_net, self).__init__()
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True))
        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True))
        self.C3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True))
        self.C4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True))
        self.C5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True))
        self.C6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True))
        self.C7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=8,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, True))
        self.Linear1 = nn.Sequential(
            nn.Linear(160, 64, bias=False),
            nn.LeakyReLU(0.2, True)
        )
        self.Linear2 = nn.Sequential(
            nn.Linear(64, 1, bias=False)
        )

    def forward(self, input):
        """Standard forward."""
        out = self.C1(input)
        out = self.C2(out)
        out = self.C3(out)
        out = self.C4(out)
        out = self.C5(out)
        out = self.C6(out)
        out = self.C7(out)
        out = out.reshape(out.shape[0], -1)
        out = self.Linear1(out)
        out = self.Linear2(out)
        return out


class DisIR_net(nn.Module):

    def __init__(self, in_channels=32):
        super(DisIR_net, self).__init__()
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True))
        self.C2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True))
        self.C3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True))
        self.C4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True))
        self.C5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True))
        self.C6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True))
        self.C7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=8,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, True))
        self.Linear1 = nn.Sequential(
            nn.Linear(160, 64, bias=False),
            nn.LeakyReLU(0.2, True))
        self.Linear2 = nn.Sequential(
            nn.Linear(64, 1, bias=False)
        )

    def forward(self, input):
        """Standard forward."""
        out = self.C1(input)
        out = self.C2(out)
        out = self.C3(out)
        out = self.C4(out)
        out = self.C5(out)
        out = self.C6(out)
        out = self.C7(out)
        out = out.view(out.shape[0], -1)
        out = self.Linear1(out)
        out = self.Linear2(out)
        return out

if __name__ == '__main__':
    net = DisIR_net(in_channels=32).cuda()
    a = torch.randn([1, 32, 480, 640]).cuda()
    b = net(a)
    print('DisIR_net output shape:', b.shape)

    net2 = DisVIS_net(in_channels=32).cuda()
    c = torch.randn([1, 32, 480, 640]).cuda()
    d = net2(c)
    print('DisVIS_net output shape:', d.shape)
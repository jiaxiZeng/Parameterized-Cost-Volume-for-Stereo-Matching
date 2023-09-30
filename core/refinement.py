import torch.nn as nn
import torch


class refineNet(nn.Module):
    def __init__(self, args):
        super(refineNet, self).__init__()
        self.args = args
        self.conv0 = nn.Sequential(nn.Conv2d(2 * args.gauss_num + 1, 64, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 64, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.conv_softmask = nn.Sequential(nn.Conv2d(64, 1, 3, 1, 1),
                                           nn.Sigmoid())
        self.conv_disp = nn.Sequential(nn.Conv2d(1, 32, 7, 1, 3),
                                       nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(64 + 64 + 2 * args.gauss_num, 64, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 3, dilation=3),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 7, dilation=7),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Conv2d(64, 1, 3, 1, 1)

    def forward(self, w, sigma, mu, disp, features):
        # uncertainty map
        w_sigma = w * sigma
        uncertainty_feature = self.conv0(torch.cat((w_sigma, mu, disp), dim=1))
        uncertainty_map = self.conv_softmask(uncertainty_feature)
        # propagation
        x = self.conv_disp(disp)
        x1 = self.conv1(torch.cat((x, features, w_sigma, mu, uncertainty_feature), dim=1))
        x = self.conv2(x1)
        x = self.conv3(x)
        x = self.conv4(x)
        disp = disp + x * uncertainty_map
        return disp
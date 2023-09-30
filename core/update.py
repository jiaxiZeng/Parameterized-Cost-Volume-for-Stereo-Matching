import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, h, cz, cr, cq, *x_list):
        x = torch.cat(x_list, dim=1)  # correlation+flow
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx) + cz)
        r = torch.sigmoid(self.convr(hx) + cr)
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)) + cq)

        h = (1 - z) * h + z * q
        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        self.args = args

        cor_planes = args.sample_num * args.corr_levels  # 27
        self.convc1 = nn.Conv2d(cor_planes, 64, 3, padding=1)
        self.convc2 = nn.Conv2d(64, 64, 3, padding=1)
        self.convc3 = nn.Conv2d(64, 48, 3, padding=1)
        self.convf1 = nn.Conv2d(3 * args.gauss_num, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 64 - 3 * args.gauss_num, 3, padding=1)

    def forward(self, disp, corr, w, sigma):
        N, C, H, W = corr.shape
        corr = corr.reshape(N, self.args.corr_levels, self.args.gauss_num, self.args.sample_num, H, W).permute(0, 2, 1,
                                                                                                               3, 4, 5)
        corr = corr.reshape(-1, self.args.corr_levels * self.args.sample_num, H, W)
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        cor = F.relu(self.convc3(cor))
        cor = cor.reshape(N, -1, H, W)
        param = torch.cat((disp, w.detach(), sigma.detach()), dim=1)
        param_f = F.relu(self.convf1(param))
        param_f = F.relu(self.convf2(param_f))
        return torch.cat([cor, param_f, param], dim=1)


def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)


def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)


def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)


class ParametersUpdater(nn.Module):
    def __init__(self, args, input_dim, hidden_dim):
        super(ParametersUpdater, self).__init__()
        self.args = args
        self.head = FlowHead(input_dim, hidden_dim, args.gauss_num)
        self.sigma0 = 0.5
        self.eps = 1e-3
        self.gamma1 = 1
        self.gamma2 = 1
        self.gamma3 = 1

    def forward(self, hidden_state, mu, sigma, w):
        delta = self.head(hidden_state)
        _, M, _, _ = delta.shape

        # feed forward gradients
        delta_sigma = 0.5 * (((1 - M * w) * sigma ** 2 - self.sigma0 ** 2 - delta ** 2) / (M * sigma ** 3) + w * sigma / (self.sigma0 ** 2))
        delta_mu = -0.5 * delta * (1 / (M * sigma ** 2) + w / (self.sigma0 ** 2))
        beta = 0.5 * (-1 / (M * w + self.eps) + torch.log(self.sigma0 * M * w / sigma + self.eps) + (sigma ** 2 + delta ** 2) / (2 * self.sigma0 ** 2) + 0.5)
        delta_w = beta - torch.sum(beta, dim=1, keepdim=True) / M

        # clip the gradients
        delta_sigma = torch.clip(delta_sigma * self.gamma1, min=-3, max=3)
        delta_mu = torch.clip(delta_mu * self.gamma2, min=-128, max=128)
        delta_w = torch.clip(delta_w * self.gamma3, min=-1 / (M * 4), max=1 / (M * 4))

        # update & clip the parameters
        sigma = torch.clip(sigma - delta_sigma, min=0.1, max=16)
        mu = mu - delta_mu
        w = torch.clip(w - delta_w, min=0, max=1)
        # normalize
        w = w / torch.sum(w, dim=1, keepdim=True)
        return mu, w, sigma


class BasicMultiUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dims=None):
        super().__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        encoder_output_dim = 256
        self.gru04 = ConvGRU(hidden_dims[3], encoder_output_dim + hidden_dims[2] * (args.n_gru_layers > 1))
        self.gru08 = ConvGRU(hidden_dims[2], 128 + hidden_dims[1] * (args.n_gru_layers > 2) + hidden_dims[3])
        self.gru16 = ConvGRU(hidden_dims[1], 128 + hidden_dims[0] * (args.n_gru_layers > 3) + hidden_dims[2])
        factor = 2 ** self.args.n_downsample
        self.mask = nn.Sequential(nn.Conv2d(hidden_dims[3], 256, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(256, (factor ** 2) * 9, 1, padding=0))
        self.ParametersUpdater = ParametersUpdater(self.args, input_dim=128, hidden_dim=256)
        self.conv2 = nn.Sequential(nn.Conv2d(256, 128, 3, 2, 1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 3, 2, 1), nn.ReLU())
        self.conv2_out = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU())
        self.conv3_out = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU())

    def forward(self, net, inp, corr=None, mu=None, w=None, sigma=None, iter04=True, iter08=True, iter16=True,
                update=True, test_mode=False, motion_features_list=None):
        if motion_features_list is None:
            if self.args.n_gru_layers >= 1:
                motion_features = self.encoder(mu, corr, w, sigma)
                motion_features_list = [motion_features]
            if self.args.n_gru_layers >= 2:
                motion_features_08_0 = self.conv2(motion_features.detach())
                motion_features_08 = self.conv2_out(motion_features_08_0)
                motion_features_list = [motion_features, motion_features_08]
            if self.args.n_gru_layers >= 3:
                motion_features_16 = self.conv3(motion_features_08_0.detach())
                motion_features_16 = self.conv3_out(motion_features_16)
                motion_features_list = [motion_features, motion_features_08, motion_features_16]

        if iter16:
            net[2] = self.gru16(net[2], *(inp[2]), motion_features_list[2], pool2x(net[1]))
        if iter08:
            if self.args.n_gru_layers > 2:
                net[1] = self.gru08(net[1], *(inp[1]), motion_features_list[1], pool2x(net[0]),
                                    interp(net[2], net[1]))
            else:
                net[1] = self.gru08(net[1], *(inp[1]), motion_features_list[1], pool2x(net[0]))
        if iter04:
            if self.args.n_gru_layers > 1:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features_list[0], interp(net[1], net[0]))
            else:
                net[0] = self.gru04(net[0], *(inp[0]), motion_features_list[0])

        if not update:
            return net, motion_features_list

        mu, w, sigma = self.ParametersUpdater(net[0], mu, sigma, w)

        if not test_mode:
            mask = self.mask(net[0]) * 0.25
        else:
            mask = torch.zeros_like(mu)
        return net, mask, mu, sigma, w

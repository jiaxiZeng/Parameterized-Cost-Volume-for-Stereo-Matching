import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicMultiUpdateBlock
from core.extractor import MultiBasicEncoder, ResidualBlock
from core.corr import CorrBlock1D
from core.utils.utils import coords_grid
from core.refinement import refineNet

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


class PCVNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.test_mode = None
        self.args = args
        context_dims = args.hidden_dims
        # feature extractor
        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn="batch",
                                      down_sample=args.n_downsample, n_gru_layers=args.n_gru_layers)
        # feed-forward differential module
        self.FDM = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList(
            [nn.Conv2d(context_dims[i], args.hidden_dims[i] * 3, 3, padding=3 // 2) for i in
             range(self.args.n_gru_layers)])

        self.conv2 = nn.Sequential(
            ResidualBlock(128, 128, 'instance', stride=1),
            nn.Conv2d(128, 256, 3, padding=1))

        # uncertainty-aware refinement
        self.refineNet = refineNet(args)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_disp(self, img):
        n, _, h, w = img.shape
        gauss_num = self.args.gauss_num
        start_point = np.array(self.args.init_mu) / (2 ** self.args.n_downsample)
        coords0 = coords_grid(n, h, w, gauss_num).to(img.device)
        coords1 = coords_grid(n, h, w, gauss_num, start_point).to(img.device)
        return coords0[:, :, 0], coords1[:, :, 0]  # n,g,h,w

    def convex_upsample(self, x, mask, scale=True):
        n, d, h, w = x.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(n, 1, 9, factor, factor, h, w)
        mask = torch.softmax(mask, dim=2)

        up_x = F.unfold(factor * x, (3, 3), padding=1) if scale else F.unfold(x, (3, 3), padding=1)
        up_x = up_x.view(n, d, 9, 1, 1, h, w)

        up_x = torch.sum(mask * up_x, dim=2)
        up_x = up_x.permute(0, 1, 4, 2, 5, 3)
        return up_x.reshape(n, d, factor * h, factor * w)

    def forward(self, image1, image2, iters=12, test_mode=False, init_param=None, cascade=False):
        self.test_mode = test_mode
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        # feature extraction
        with autocast(enabled=self.args.mixed_precision):
            *cnet_list, x, low_f = self.cnet(torch.cat((image1, image2), dim=0), dual_inp=True,
                                             num_layers=self.args.n_gru_layers)
            fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0] // 2)
            net_list = [torch.tanh(f[0]) for f in cnet_list]
            inp_list = [torch.relu(f[1]) for f in cnet_list]
            inp_list = [list(conv(i).split(split_size=conv.out_channels // 3, dim=1)) for i, conv in
                        zip(inp_list, self.context_zqr_convs)]

        # correlation
        N, C, H, W = net_list[0].shape
        fmap1, fmap2 = fmap1.float(), fmap2.float()
        corr_fn = CorrBlock1D(fmap1, fmap2, sample_num=self.args.sample_num, num_levels=self.args.corr_levels, downsample=self.args.n_downsample)

        # parameters initialization
        if init_param is not None:
            coords0, _ = self.initialize_disp(fmap1)  # n,g(aussians),h,w
            factor = coords0.shape[3] / init_param['mu'].shape[3]
            coords1 = coords0 - factor * F.interpolate(init_param['mu'], size=(coords0.shape[2], coords0.shape[3]),
                                                       mode='bilinear', align_corners=True)
            N, C, H, W = net_list[0].shape
            sigma = torch.ones(N, self.args.gauss_num, H, W).to(
                coords0.device) * self.args.init_sigma / (
                            2 ** self.args.n_downsample)
            w = torch.ones(N, self.args.gauss_num, H, W).to(coords0.device) / self.args.gauss_num
        else:
            coords0, coords1 = self.initialize_disp(fmap1)
            N, C, H, W = net_list[0].shape
            sigma = torch.ones(N, self.args.gauss_num, H, W).to(
                coords0.device) * self.args.init_sigma / (
                            2 ** self.args.n_downsample)
            w = torch.ones(N, self.args.gauss_num, H, W).to(coords0.device) / self.args.gauss_num

        disp_seqs = []
        mu_seqs = []
        w_seqs = []
        sigma_seqs = []

        # iterative update
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1, sigma, self.test_mode)
            mu = (coords0 - coords1).detach()
            motion_features_list = None
            with autocast(enabled=self.args.mixed_precision):
                if self.args.n_gru_layers >= 3 and self.args.slow_fast_gru:
                    net_list, motion_features_list = self.FDM(net_list, inp_list, corr, mu.detach(), w=w.detach(),
                                                              sigma=sigma.detach(),
                                                              iter16=True,
                                                              iter08=False,
                                                              iter04=False,
                                                              update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:
                    net_list, motion_features_list = self.FDM(net_list, inp_list, corr, mu.detach(), w=w.detach(),
                                                              sigma=sigma.detach(),
                                                              iter16=self.args.n_gru_layers >= 3,
                                                              iter08=True,
                                                              iter04=False,
                                                              update=False,
                                                              motion_features_list=motion_features_list)
                net_list, up_mask, mu, sigma, w = self.FDM(net_list, inp_list, corr, mu=mu.detach(), w=w.detach(),
                                                           sigma=sigma.detach(),
                                                           iter16=self.args.n_gru_layers >= 3,
                                                           iter08=self.args.n_gru_layers >= 2,
                                                           iter04=True,
                                                           test_mode=(test_mode and itr < iters - 1),
                                                           motion_features_list=motion_features_list)

            coords1 = coords0 - mu
            if test_mode and itr < iters - 1:
                continue

            # disparity regression
            disp = torch.sum(w * mu, dim=1, keepdim=True)

            # refinement
            if itr == self.args.valid_iters - 1:
                with autocast(enabled=self.args.mixed_precision):
                    refined_disp = self.refineNet(w.detach(), sigma.detach(),
                                                  mu.detach(),
                                                  disp.detach(), low_f)
                refined_disp_up = self.convex_upsample(refined_disp, up_mask.detach())

            # up-sampling
            disp_up = self.convex_upsample(disp, up_mask)

            sigma_up = self.convex_upsample(sigma.reshape(-1, 1, H, W),
                                            up_mask.repeat(1, self.args.gauss_num, 1, 1).reshape(N * self.args.gauss_num, -1, H, W).detach())

            mu_up = self.convex_upsample(mu.reshape(-1, 1, H, W),
                                         up_mask.repeat(1, self.args.gauss_num, 1, 1).reshape(N * self.args.gauss_num, -1, H, W).detach())

            w_up = self.convex_upsample(w.reshape(-1, 1, H, W),
                                        up_mask.repeat(1, self.args.gauss_num, 1, 1).reshape(N * self.args.gauss_num, -1, H, W).detach(), scale=False)

            mu_seqs.append(mu_up)
            disp_seqs.append(disp_up)
            sigma_seqs.append(sigma_up)
            w_seqs.append(w_up)

        if cascade:
            init_params={'disp': disp_up,
                        'sigma': sigma_up.reshape(disp_up.shape[0], -1, disp_up.shape[2], disp_up.shape[3]),
                        'mu': mu_up.reshape(disp_up.shape[0], -1, disp_up.shape[2], disp_up.shape[3]),
                        "w": w_up.reshape(disp_up.shape[0], -1, disp_up.shape[2], disp_up.shape[3])}
            if test_mode:
                return init_params
            else:
                return refined_disp_up, disp_seqs, mu_seqs, w_seqs, sigma_seqs, init_params
        else:
            if test_mode:
                return refined_disp_up
            return refined_disp_up, disp_seqs, mu_seqs, w_seqs, sigma_seqs

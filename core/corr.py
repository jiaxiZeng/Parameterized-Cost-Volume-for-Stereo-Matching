import torch
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler
import math

try:
    import corr_sampler
except:
    pass

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrBlock1D:
    def __init__(self, fmap1, fmap2, sample_num, num_levels=4, downsample=2):
        self.sample_num = sample_num
        self.num_levels = num_levels
        corr = CorrBlock1D.corr(fmap1, fmap2)
        batch, h1, w1, _, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, 1, 1, w2)
        self.index = torch.range(-(sample_num // 2), (sample_num // 2))
        self.corr_pyramid = []
        self.corr_pyramid.append(corr)
        self.compress_factor = 4 if downsample == 2 else 2
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, [1, self.compress_factor], stride=[1, self.compress_factor])
            self.corr_pyramid.append(corr)

    def __call__(self, coords, sigma, test_mode=False):
        batch, gauss_num, h1, w1 = coords.shape
        sigma = sigma.permute(0, 2, 3, 1).contiguous().reshape(batch * h1 * w1, 1, gauss_num, 1)
        coords = coords.permute(0, 2, 3, 1).contiguous().reshape(batch * h1 * w1, 1, gauss_num, 1)
        dx = self.index.to(coords.device).detach().view(1, 1, 1, self.sample_num)
        x = dx * sigma + coords
        out_pyramid = []

        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            x0 = x / self.compress_factor ** i
            x0 = x0.reshape(batch * h1 * w1, 1, gauss_num * self.sample_num, 1)
            y0 = torch.zeros_like(x0)
            coords_lvl = torch.cat([x0, y0], dim=-1)
            corr = bilinear_sampler(corr.contiguous(), coords_lvl.contiguous())
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr / torch.sqrt(torch.tensor(D).float())

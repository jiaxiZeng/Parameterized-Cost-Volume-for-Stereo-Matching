from __future__ import print_function, division
import torch
import wandb
import logging
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import torch.optim as optim
from core.model import PCVNet
import torch.nn.functional as F
from evaluate_stereo import validate
import core.stereo_datasets as datasets
from core.utils.data_parallel import BalancedDataParallel

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sequence_loss(disp_refine, final_disp_preds, mu_preds, w_preds, sigma_preds, disp_gt, valid, gauss_num,
                  max_disp=512):
    """ Loss function defined over sequence of disp predictions """

    n_predictions = len(mu_preds)
    assert n_predictions >= 1
    disp_loss = 0.0

    # exclude extremely large displacements
    valid = (disp_gt < max_disp) & (valid[:, None].bool()) & (disp_gt >= 0)
    # print("mask_rate:", 1 - valid.float().mean())
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()
    N, C, H, W = mu_preds[0].shape
    i_weights = [0.4, 0.6, 0.8, 1, 1.2, 1.4]
    if (valid == False).all():
        disp_loss = 0
        metrics = {
            'epe': 0.,
            '1px': 1.,
            '3px': 1.,
            '5px': 1.,
        }
    else:
        for i in range(n_predictions):
            assert not torch.isnan(mu_preds[i]).any() and not torch.isinf(mu_preds[i]).any()
            assert not torch.isnan(w_preds[i]).any() and not torch.isinf(w_preds[i]).any()
            assert not torch.isnan(sigma_preds[i]).any() and not torch.isinf(sigma_preds[i]).any()
            assert not torch.isnan(final_disp_preds[i]).any() and not torch.isinf(final_disp_preds[i]).any()
            # split the segment
            mu_preds[i] = mu_preds[i].view(N // gauss_num, gauss_num, 1, H, W)

            w_preds[i] = w_preds[i].view(N // gauss_num, gauss_num, 1, H, W)
            sigma_preds[i] = sigma_preds[i].view(N // gauss_num, gauss_num, 1, H, W)
            w = w_preds[i]
            print("mu%d:" % i, torch.mean(mu_preds[i], dim=[0, 2, 3, 4]).detach())
            print("w%d:" % i, torch.mean(w, dim=[0, 2, 3, 4]).detach())
            print("sigma%d:" % i, torch.mean(sigma_preds[i], dim=[0, 2, 3, 4]).detach())

            i_loss1 = (final_disp_preds[i] - disp_gt).abs()
            i_loss2 = torch.mean((mu_preds[i] - disp_gt[:, None]).abs(), dim=1)
            disp_loss += i_weights[i] * (
                    i_loss1.view(-1)[valid.view(-1)].mean() + i_loss2.view(-1)[valid.view(-1)].mean())
        disp_loss += 1.4 * F.smooth_l1_loss(disp_refine[valid], disp_gt[valid], size_average=True)

        epe_final = torch.abs(disp_refine - disp_gt)
        epe_final = epe_final.view(-1)[valid.view(-1)]

        epe = torch.abs(final_disp_preds[3] - disp_gt)
        epe = epe.view(-1)[valid.view(-1)]
        metrics = {
            'epe': epe.mean().item(),
            '1px': (epe < 1).float().mean().item(),
            '3px': (epe < 3).float().mean().item(),
            '5px': (epe < 5).float().mean().item(),
            'bad1': (epe > 1).float().mean().item(),
            'bad2': (epe > 2).float().mean().item(),
            'bad5': (epe > 5).float().mean().item(),

            'epe_final': epe_final.mean().item(),
            '1px_final': (epe_final < 1).float().mean().item(),
            '3px_final': (epe_final < 3).float().mean().item(),
            '5px_final': (epe_final < 5).float().mean().item(),
            'bad1_final': (epe_final > 1).float().mean().item(),
            'bad2_final': (epe_final > 2).float().mean().item(),
            'bad5_final': (epe_final > 5).float().mean().item(),
        }

    return disp_loss, metrics


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
                                              pct_start=args.pct_start, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


class Logger:
    def __init__(self, model, scheduler, sum_freq):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = wandb
        self.sum_freq = sum_freq

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = wandb

        self.writer.log(self.running_loss, commit=False)

    def push(self, metrics):
        """
        this function is used to record the running metrics when training
        :param metrics:
        :return:
        """
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.sum_freq == self.sum_freq - 1:
            for key in metrics:
                self.running_loss[key] /= self.sum_freq
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        """
        this function is used to record the running metrics when testing
        :return:
        """
        if self.writer is None:
            self.writer = wandb

        self.writer.log(results)


def train(args):
    # load model
    if len(args.device) > 1:
        if args.gpu0_bsz >= 0:
            model = BalancedDataParallel(args.gpu0_bsz, PCVNet(args), dim=0)
        else:
            model = nn.DataParallel(PCVNet(args))
    else:
        model = PCVNet(args)

    print("Parameter Count: %d" % count_parameters(model))

    state = np.random.get_state()
    np.random.seed(1000)
    val_idxs = set(np.random.permutation(200)[:40])
    np.random.set_state(state)
    args.valid_set = val_idxs

    train_loader = datasets.fetch_dataloader(args, occ_mask=False)
    optimizer, scheduler = fetch_optimizer(args, model)
    logger = Logger(model, scheduler, args.log_freq)
    scaler = GradScaler(enabled=args.mixed_precision)

    # load checkpoint
    if args.restore_ckpt is not None:
        # assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        new_ckpt = dict()
        if len(args.device) > 1:
            for k, v in checkpoint.items():
                k = k.replace('.update_block.', '.FDM.')
                k = k.replace('.uncertainty_aware_updater.', '.ParametersUpdater.')
                if k[:7] != 'module.':
                    new_ckpt['module.' + k] = v
                else:
                    new_ckpt[k] = v
        else:
            for k, v in checkpoint.items():
                new_ckpt[k.replace('module.', '')] = v
        model.load_state_dict(new_ckpt, strict=True)
        logging.info(f"Done loading checkpoint")

    model.cuda(args.device[0])
    model.train()
    if len(args.device) > 1:
        model.module.freeze_bn()  # We keep BatchNorm frozen
    else:
        model.freeze_bn()  # We keep BatchNorm frozen

    total_steps = 0
    validation_frequency = args.valid_freq
    should_keep_training = True
    gauss_num_train = args.gauss_num
    best_metric = float('inf')
    # training
    while should_keep_training:
        for i_batch, (files_path, *data_blob) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            image1, image2, disp, valid = [x.cuda(args.device[0]) for x in data_blob]

            if args.cascade:
                image1_dw2 = F.interpolate(image1, scale_factor=(0.5, 0.5), mode='bilinear', align_corners=True)
                image2_dw2 = F.interpolate(image2, scale_factor=(0.5, 0.5), mode='bilinear', align_corners=True)
                *output_list_dw2, init_param = model(image1_dw2, image2_dw2, iters=args.valid_iters, cascade=True)
            else:
                init_param = None

            assert model.training
            output_list = model(image1, image2, iters=args.train_iters, init_param=init_param)
            assert model.training

            # calculate loss
            if args.cascade:
                output_list_dw2 = [
                    2 * F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=True) if type(x) is not list
                    else [2 * F.interpolate(y, scale_factor=(2, 2), mode='bilinear', align_corners=True) for y in x]
                    for x in output_list_dw2]
                loss_dw2, metrics_dw2 = sequence_loss(*output_list_dw2, disp, valid, gauss_num_train, args.max_disp)
                loss, metrics = sequence_loss(*output_list, disp, valid, gauss_num_train, args.max_disp)
                loss += 0.5 * loss_dw2
            else:
                loss, metrics = sequence_loss(*output_list, disp, valid, gauss_num_train, args.max_disp)

            # log
            logger.push(metrics)
            logger.writer.log({"live_loss": loss.item(), 'learning_rate': optimizer.param_groups[0]['lr']})

            # gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # backward
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            # validation
            if total_steps % validation_frequency == validation_frequency - 1:
                results, best_metric = validate(model, args.valid_iters, args, step=total_steps,
                                                best_metric=best_metric,
                                                valid_set=args.valid_set, cascade=args.cascade)
                if results:
                    logger.write_dict(results)
                # convert to training
                model.train()
                if len(args.device) > 1:
                    model.module.freeze_bn()
                else:
                    model.freeze_bn()

            total_steps += 1
            if total_steps > args.num_steps:
                should_keep_training = False
                break

        # saving the checkpoint
        if len(train_loader) >= 10000:
            save_path = Path('%s/%d_epoch_%s.pth.gz' % (args.saving_path, total_steps + 1, args.name))
            logging.info(f"Saving file {save_path}")
            torch.save(model.state_dict(), save_path)

    print("FINISHED TRAINING")
    PATH = '%s/%s.pth' % (args.saving_path, args.name)
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='pcvnet', help="name your experiment")

    # Training parameters
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--batch_size', type=int, default=8, help="batch size used during training")
    parser.add_argument('--train_datasets', nargs='+', default=['sceneflow', 'middlebury_F', 'kitti', 'booster', 'crestereo'], help="training dataset(s)")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate")
    parser.add_argument('--pct_start', type=float, default=0.01, help="percentage of total number of epochs when learning rate rises during one cycle")
    parser.add_argument('--num_steps', type=int, default=100000, help="length of training schedule")
    parser.add_argument('--image_size', type=int, nargs='+', default=[320, 720], help="size of the random image crops used during training")
    parser.add_argument('--train_iters', type=int, default=6, help="number of updates to the disparity field in training forward pass")
    parser.add_argument('--valid_iters', type=int, default=4, help='number of disparity updates during validation forward pass')
    parser.add_argument('--valid_freq', type=int, default=10000, help="the frequency of validation while training")
    parser.add_argument('--wdecay', type=float, default=.00001, help="weight decay in optimizer")
    parser.add_argument('--device', nargs='+', type=int, default=[0], help="the index of gpu")
    parser.add_argument('--saving_path', type=str, default='.', help="the path to save the checkpoint.")
    parser.add_argument('--gpu0_bsz', type=int, default=-1, help="the batch size of GPU 0, which is used to balance the load of GPUs")
    parser.add_argument('--log_freq', type=int, default=100, help="the frequency of logging")
    parser.add_argument('--max_disp', type=int, default=512, help="the max disparity when training")
    parser.add_argument('--valid_set', nargs='+', type=int, default=[0], help="the index of validation set for kitti validation")
    parser.add_argument('--cascade', action='store_true', help="cascade inference like crestereo")

    # Distribution related parameters
    parser.add_argument('--gauss_num', type=int, default=4, help="the number of Gaussian distributions")
    parser.add_argument('--sample_num', type=int, default=9, help="the number of sampling points in each Gaussian distribution")
    parser.add_argument('--init_sigma', type=int, default=32, help="the initial sigma of each Gaussian")
    parser.add_argument('--init_mu', type=int, nargs='+', default=[0, 64, 128, 192], help="the initial mu of each Gaussian distribution")

    # Architecture choices
    parser.add_argument('--corr_levels', type=int, default=3, help="number of levels in the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 4, help="hidden state and context dimensions")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=[0, 1.4], help='color saturation')
    parser.add_argument('--do_flip', default=False, nargs='+', choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.2, 0.4], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    args = parser.parse_args()

    # seed
    torch.manual_seed(1234)
    np.random.seed(1234)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    Path(args.saving_path).mkdir(exist_ok=True, parents=True)
    wandb.init(
        job_type="train",
        project=args.name,
        entity="zengjiaxi"
    )
    # add the args to wandb
    wandb.config.update(args)
    train(args)

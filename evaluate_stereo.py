from __future__ import print_function, division

import os.path
import sys
import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from core.model import PCVNet, autocast
import core.stereo_datasets as datasets
from core.utils.utils import InputPadder
from core.utils.visualization import logFeatureMap
from core.utils.frame_utils import writePFM
from pathlib import Path
import wandb
import skimage
import torch.nn.functional as F

sys.path.append('core')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def validate_eth3d(model, iters=32, mixed_prec=False):
    """ Perform validation using the ETH3D (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, disp_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            disp_pr = model(image1, image2, iters=iters, test_mode=True)
        disp_pr = padder.unpad(disp_pr).cpu().squeeze(0)  # s,1,h,w
        assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)
        epe = (disp_pr - disp_gt).abs()

        epe_flattened = epe.flatten()
        val = valid_gt.flatten() >= 0.5
        out = (epe_flattened > 1.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(
            f"ETH3D {val_id + 1} out of {len(val_dataset)}. EPE {round(image_epe, 4)} D1 {round(image_out, 4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation ETH3D: EPE %f, D1 %f" % (epe, d1))
    return {'eth3d-epe': epe, 'eth3d-d1': d1}


@torch.no_grad()
def validate_kitti(model, iters=32, mixed_prec=False, device=[0], val_set=None, submission=False):
    """ Perform validation using the KITTI-2015 (train) split """
    if submission:
        out_path = './kitti_output/'
        Path(out_path).mkdir(exist_ok=True, parents=True)
        model.eval()
        aug_params = {}
        val_dataset = datasets.KITTI(aug_params, image_set='testing', valid=True, val_set=val_set, submission=submission)
        torch.backends.cudnn.benchmark = True
        elapsed_list = []
        for val_id in range(len(val_dataset)):
            image1, image2, disp_name = val_dataset[val_id]
            image1 = image1[None].cuda(device[0])
            image2 = image2[None].cuda(device[0])
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            with autocast(enabled=mixed_prec):
                start = time.time()
                torch.cuda.synchronize()
                disp_pr = model(image1, image2, iters=iters, test_mode=True)
                torch.cuda.synchronize()
                end = time.time()

            if val_id > 50:
                elapsed_list.append(end - start)
            disp_pr = padder.unpad(disp_pr).cpu().squeeze(0).squeeze(0)  # h,w
            disp_pr = np.array(disp_pr, dtype=np.float32)
            disp_pr_uint = np.round(disp_pr * 256).astype(np.uint16)
            pt = os.path.join(out_path, disp_name.split('/')[-1])
            skimage.io.imsave(pt, disp_pr_uint)
        print("time cost: %.4f s" % np.mean(elapsed_list))
        return None
    else:
        model.eval()
        aug_params = {}
        val_dataset = datasets.KITTI(aug_params, image_set='training', valid=True, val_set=val_set)
        torch.backends.cudnn.benchmark = True
        out_list, epe_list, elapsed_list = [], [], []
        for val_id in range(len(val_dataset)):
            _, image1, image2, disp_gt, valid_gt = val_dataset[val_id]
            image1 = image1[None].cuda(device[0])
            image2 = image2[None].cuda(device[0])
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            with autocast(enabled=mixed_prec):
                start = time.time()
                torch.cuda.synchronize()
                disp_pr = model(image1, image2, iters=iters, test_mode=True)
                torch.cuda.synchronize()
                end = time.time()

            if val_id > 50:
                elapsed_list.append(end - start)
            disp_pr = padder.unpad(disp_pr).cpu().squeeze(0)

            assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)
            epe = (disp_pr - disp_gt).abs()

            epe_flattened = epe.flatten()
            val = (valid_gt.flatten() >= 0.5)

            out = (epe_flattened > 3.0) & (epe_flattened > 0.05 * disp_gt.flatten())
            image_out = out[val].float().mean().item()
            image_epe = epe_flattened[val].mean().item()
            if val_id < 9 or (val_id + 1) % 10 == 0:
                logging.info(
                    f"KITTI Iter {val_id + 1} out of {len(val_dataset)}. EPE {round(image_epe, 4)} D1 {round(image_out, 4)}. Runtime: {format(end - start, '.3f')}s ({format(1 / (end - start), '.2f')}-FPS)")
            epe_list.append(epe_flattened[val].mean().item())
            out_list.append(out[val].cpu().numpy())

        epe_list = np.array(epe_list)
        out_list = np.concatenate(out_list)

        epe = np.mean(epe_list)
        d1 = 100 * np.mean(out_list)

        avg_runtime = np.mean(elapsed_list)

        print(
            f"Validation KITTI: EPE {format(epe, '.3f')}, D1 {format(d1, '.3f')}, {format(1 / avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
        return {'kitti-epe': epe, 'kitti-d1': d1}


@torch.no_grad()
def validate_things(model, iters=32, mixed_prec=False, device=[0], version='frames_finalpass'):
    """ Perform validation using the FlyingThings3D (TEST) split """
    model.eval()
    val_dataset = datasets.SceneFlowDatasets(dstype=version, things_test=True)

    out_list, epe_list = [], []
    for val_id in tqdm(range(len(val_dataset))):
        _, image1, image2, disp_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda(device[0])  # 相当于把第一个N维度扩展了
        image2 = image2[None].cuda(device[0])

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            disp_pr = model(image1, image2, iters=iters, test_mode=True)
        disp_pr = padder.unpad(disp_pr).cpu().squeeze(0)  # 1,1,h,w
        assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)
        epe = (disp_pr - disp_gt).abs()

        epe = epe.flatten()
        val = (disp_gt.abs().flatten() < 192)
        if (val == False).all():
            continue
        epe_1 = epe[val].mean()
        assert not torch.isnan(epe_1), (epe, val)
        # out = (epe > 1.0)
        epe_list.append(epe[val].mean().item())
        # out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    # out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    # d1 = 100 * np.mean(out_list)

    print("Validation FlyingThings: %f" % (epe))
    return {'things-epe': epe}


@torch.no_grad()
def validate_middlebury(model, iters=32, split='H', mixed_prec=False, submission=False, cascade=False):
    """ Perform validation using the Middlebury-V3 dataset """
    if submission:
        model.eval()
        aug_params = {}
        val_dataset = datasets.Middlebury(aug_params, root='Middlebury/MiddEval3/testH', split=split, is_test=True, submission=True)
        for val_id in range(len(val_dataset)):
            image1, image2, name = val_dataset[val_id]
            print("Processing image: %s" % (name))
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            with autocast(enabled=mixed_prec):
                s = time.time()
                disp_pr = model(image1, image2, iters=iters, test_mode=True)
                e = time.time()
            disp_pr = np.array(padder.unpad(disp_pr).cpu().squeeze(0).squeeze(0))  # 1,h,w
            time_cost = e - s
            out_name = name.replace('datasets/Middlebury/MiddEval3', './mboutput')
            print(out_name + '/disp0PCVNet.pfm' + "time:" + str(time_cost) + ' s')
            Path(out_name).mkdir(exist_ok=True, parents=True)
            # if not os.path.exists(out_name):
            #     os.mkdir(out_name)
            writePFM(out_name + '/disp0PCVNet.pfm', disp_pr)
            with open(out_name + '/timePCVNet.txt', 'w') as f:
                f.write(str(time_cost))
        return None
    else:
        model.eval()
        aug_params = {}
        val_dataset = datasets.Middlebury(aug_params, root='datasets/mb-ex-training/trainingF', split=split, is_test=True)
        for val_id in range(len(val_dataset)):
            image1, image2, name = val_dataset[val_id]
            print("Processing image: %s" % (name))
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()
            if cascade:
                image1_dw2 = F.interpolate(image1, scale_factor=(0.5, 0.5), mode='bilinear', align_corners=True)
                image2_dw2 = F.interpolate(image2, scale_factor=(0.5, 0.5), mode='bilinear', align_corners=True)
                padder = InputPadder(image1_dw2.shape, divis_by=32)
                image1_dw2, image2_dw2 = padder.pad(image1_dw2, image2_dw2)
                with autocast(enabled=mixed_prec):
                    params = model(image1_dw2, image2_dw2, iters=iters, test_mode=True, cascade=cascade)
                init_params = params
            else:
                init_params = None

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            with autocast(enabled=mixed_prec):
                disp_pr = model(image1, image2, iters=iters, test_mode=True, init_param=init_params)
            disp_pr = padder.unpad(disp_pr).cpu().squeeze(0)  # 1,h,w
            logFeatureMap(disp_pr, name, wandb, vmin=0, cmap='jet')
        return None


@torch.no_grad()
def validate_booster(model, iters=4, mixed_prec=False, submission=False, cascade=False):
    """ Perform validation using the Booster dataset """
    out_path = './booster_output/'
    model.eval()
    aug_params = {}
    val_dataset = datasets.Booster(aug_params, is_test=True)
    for val_id in tqdm(range(len(val_dataset))):
        image1, image2, img_name = val_dataset[val_id]
        print("Processing image: %s" % img_name)
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        if cascade:
            image1_dw2 = F.interpolate(image1, scale_factor=(0.25, 0.25), mode='bilinear', align_corners=True)
            image2_dw2 = F.interpolate(image2, scale_factor=(0.25, 0.25), mode='bilinear', align_corners=True)
            padder = InputPadder(image1_dw2.shape, divis_by=32)
            image1_dw2, image2_dw2 = padder.pad(image1_dw2, image2_dw2)
            with autocast(enabled=mixed_prec):
                # params = model(image1_dw2, image2_dw2, iters=iters, test_mode=True, init_param=params, cascade=cascade)
                params = model(image1_dw2, image2_dw2, iters=iters, test_mode=True, cascade=cascade)
            init_params = params
        else:
            init_params = None

        image1 = F.interpolate(image1, scale_factor=(0.5, 0.5), mode='bilinear', align_corners=True)
        image2 = F.interpolate(image2, scale_factor=(0.5, 0.5), mode='bilinear', align_corners=True)
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            disp_pr = model(image1, image2, iters=iters, test_mode=True, init_param=init_params)
        disp_pr = padder.unpad(disp_pr).cpu().squeeze(0)  # 1,h,w
        if submission:
            disp_pr = disp_pr.squeeze(0)  # h,w
            disp_pr = np.array(disp_pr, dtype=np.float32)
            disp_pr_uint = np.round(disp_pr * 64).astype(np.uint16)
            pt = os.path.join(out_path, img_name.replace('/camera_00', ''))
            Path(os.path.dirname(pt)).mkdir(exist_ok=True, parents=True)
            skimage.io.imsave(pt, disp_pr_uint)
        else:
            logFeatureMap(disp_pr, img_name, wandb, vmin=0, cmap='jet', local_save=True)
    return None


def validate(model, iters, args, step=0, best_metric=1, valid_set=None, submission=False, cascade=False):
    datasets = args.train_datasets
    results = None

    for dataset in datasets:
        if dataset[:10] == 'middlebury':
            save_path = Path('%s/mb_latest_%s.pth' % (args.saving_path, args.name))
            logging.info(f"Saving file {save_path.absolute()}")
            torch.save(model.state_dict(), save_path)
            validate_middlebury(model.module if len(args.device) > 1 else model, iters=iters, split=dataset[-1], mixed_prec=args.mixed_precision,
                                submission=submission)
        elif dataset == 'sceneflow':
            save_path = Path('%s/sceneflow_%d_%s.pth' % (args.saving_path, step + 1, args.name))
            logging.info(f"Saving file {save_path.absolute()}")
            torch.save(model.state_dict(), save_path)
            results = validate_things(model.module if len(args.device) > 1 else model, iters=iters, mixed_prec=args.mixed_precision)
        elif dataset == 'kitti':
            results = validate_kitti(model.module if len(args.device) > 1 else model, iters=iters, mixed_prec=args.mixed_precision, submission=submission,
                                     val_set=valid_set)
            if results and best_metric > results['kitti-d1']:
                best_metric = results['kitti-d1']
                save_path = Path('%s/best_kitti_%d_%s.pth' % (args.saving_path, step + 1, args.name))
                logging.info(f"Saving file {save_path.absolute()}")
                torch.save(model.state_dict(), save_path)
        elif dataset == 'booster':
            save_path = Path('%s/booster_%d_%s.pth' % (args.saving_path, step, args.name))
            logging.info(f"Saving file {save_path.absolute()}")
            torch.save(model.state_dict(), save_path)
            validate_booster(model.module if len(args.device) > 1 else model, iters=iters, mixed_prec=args.mixed_precision, submission=submission,
                             cascade=cascade)
    return results, best_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--dataset', help="dataset for evaluation", required=True,
                        choices=["eth3d", "kitti", "things", "booster"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=4, help='number of flow-field updates during forward pass')
    parser.add_argument('--submission', action='store_true', help="whether to save the benchmark submission files")

    # Distribution related parameters
    parser.add_argument('--gauss_num', type=int, default=4, help="the number of Gaussian distributions")
    parser.add_argument('--sample_num', type=int, default=9, help="the number of sampling points in each Gaussian distribution")
    parser.add_argument('--init_sigma', type=int, default=32, help="the initial sigma of each Gaussian")
    parser.add_argument('--init_mu', type=int, nargs='+', default=[0, 64, 128, 192], help="the initial mu of each Gaussian distribution")

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 4, help="hidden state and context dimensions")
    parser.add_argument('--corr_levels', type=int, default=3, help="number of levels in the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    args = parser.parse_args()

    model = torch.nn.DataParallel(PCVNet(args), device_ids=[0])

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        new_ckpt = dict()
        for k, v in checkpoint.items():
            if k[:7] != 'module.':
                new_ckpt['module.' + k] = v
            else:
                new_ckpt[k] = v
        model.load_state_dict(new_ckpt, strict=True)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model) / 1e6, '.2f')}M learnable parameters.")

    use_mixed_precision = args.mixed_precision

    if args.dataset == 'eth3d':
        validate_eth3d(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)

    elif args.dataset == 'kitti':
        validate_kitti(model, iters=args.valid_iters, mixed_prec=use_mixed_precision, submission=args.submission)

    elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:
        validate_middlebury(model, iters=args.valid_iters, split='H', mixed_prec=use_mixed_precision, submission=args.submission)

    elif args.dataset == 'booster':
        validate_booster(model, iters=args.valid_iters, mixed_prec=use_mixed_precision, submission=args.submission, cascade=True)

    elif args.dataset == 'things':
        validate_things(model, iters=args.valid_iters, mixed_prec=use_mixed_precision, version='frames_cleanpass')

# Data loading based on https://github.com/NVIDIA/flownet2-pytorch
import copy
import logging
import os
import re
import os.path as osp
import random
from glob import glob
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from core.utils import frame_utils
from core.utils.augmentor import DispAugmentor, SparseDispAugmentor


def get_occ(disparity_cuda):
    if len(disparity_cuda.shape) == 2:
        height, width = disparity_cuda.shape
        batch_size = 1
        disp_num = 1
        disparity_cuda = disparity_cuda.reshape((batch_size, disp_num, height, width))
    elif len(disparity_cuda.shape) == 3:
        batch_size, height, width = disparity_cuda.shape
        disp_num = 1
        disparity_cuda = disparity_cuda.reshape((batch_size, disp_num, height, width))
    elif len(disparity_cuda.shape) == 4:
        batch_size, disp_num, height, width = disparity_cuda.shape
    else:
        raise Exception("Only accept disparity with dim 3 or 4, but given {}".format(len(disparity_cuda.shape)))
    if isinstance(disparity_cuda, torch.Tensor):
        # get the normal position
        pos_y, pos_x = torch.meshgrid([torch.arange(0, height, dtype=disparity_cuda.dtype),
                                       torch.arange(0, width, dtype=disparity_cuda.dtype)])  # (H, W)
        pos_x = pos_x.reshape(1, 1, height, width).repeat(batch_size, disp_num, 1, 1)
        pos_y = pos_y.reshape(1, 1, height, width).repeat(batch_size, disp_num, 1, 1)  # (B, S, H, W)

        if disparity_cuda.is_cuda:
            pos_x = pos_x.cuda()
            pos_y = pos_y.cuda()

        # get the warped position
        shift_cuda = pos_x - disparity_cuda
        shift_numpy = shift_cuda.detach().cpu().numpy()

    elif isinstance(disparity_cuda, np.ndarray):
        # get the normal position
        pos_x, pos_y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        pos_x = pos_x.reshape(1, 1, height, width).repeat(batch_size, axis=0).repeat(disp_num, axis=1)
        pos_y = pos_y.reshape(1, 1, height, width).repeat(batch_size, axis=0).repeat(disp_num, axis=1)

        # get the warped position
        shift_cuda = pos_x - disparity_cuda
        shift_numpy = shift_cuda

    # compute the minimum position from rightmost pixel to leftmost pixel
    min_shift = np.zeros_like(shift_numpy)
    min_col = np.ones((batch_size, disp_num, height)) * width
    for col in np.arange(width - 1, -1, -1):
        min_col = (min_col > shift_numpy[..., col]) * shift_numpy[..., col] + (
                min_col <= shift_numpy[..., col]) * min_col
        min_shift[..., col] = min_col
    without_occ = (shift_numpy <= min_shift) & (shift_numpy > 0)
    return without_occ[0, 0]


class StereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None, occ_mask=False, submission=False,
                 load_right_disp=False):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        self.occ_mask = occ_mask
        self.load_right_disp = load_right_disp
        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                self.augmentor = SparseDispAugmentor(**aug_params)
            else:
                self.augmentor = DispAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader

        self.submission = submission
        self.init_seed = False
        self.disp_list = []
        self.disparity_list = []
        self.right_disparity_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):
        # inference
        if self.submission:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]  # 3,H,W
        # set the random seed
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        # read images
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # read disparity
        disp = self.disparity_reader(self.disparity_list[index])
        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 1024
        disp = np.array(disp).astype(np.float32)[:, :, None]

        if self.load_right_disp:
            disp_r = self.disparity_reader(self.right_disparity_list[index])
            if isinstance(disp_r, tuple):
                disp_r, valid_r = disp_r
            else:
                valid_r = disp_r < 1024
            disp_r = np.array(disp_r).astype(np.float32)[:, :, None]

        # for grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        # data augmentation
        if self.augmentor is not None:
            if self.sparse:
                img1, img2, disp, valid = self.augmentor(img1, img2, disp, valid,
                                                         valid_r=valid_r if self.load_right_disp else None,
                                                         disp_r=disp_r if self.load_right_disp else None)
            else:
                img1, img2, disp = self.augmentor(img1, img2, disp, disp_r=disp_r if self.load_right_disp else None)

        # to tensor
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()  # C,H,W
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()  # C,H,W
        disp = torch.from_numpy(disp).permute(2, 0, 1).float()  # 1,H,W

        if self.occ_mask:
            occ_mask = get_occ(disp)

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            if self.occ_mask:
                valid = (disp[0] < 1024) & (disp[0] > 0) & occ_mask  # H,W
            else:
                valid = (disp[0] < 1024) & (disp[0] > 0)

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW] * 2 + [padH] * 2)
            img2 = F.pad(img2, [padW] * 2 + [padH] * 2)

        return self.image_list[index] + [self.disparity_list[index]], img1, img2, disp, valid.float()

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.disp_list = v * copy_of_self.disp_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.right_disparity_list = v * copy_of_self.right_disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self

    def __len__(self):
        return len(self.image_list)


class SceneFlowDatasets(StereoDataset):
    def __init__(self, aug_params=None, root='datasets', dstype='frames_cleanpass', things_test=False, occ_mask=False,
                 load_right_disp=False, only_things=False):
        super(SceneFlowDatasets, self).__init__(aug_params, occ_mask=occ_mask, load_right_disp=load_right_disp)
        self.root = root
        self.dstype = dstype

        if things_test:
            self._add_things("TEST")
        else:
            self._add_things("TRAIN")
            if not only_things:
                self._add_monkaa()
                self._add_driving()

    def _add_things(self, split='TRAIN'):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'FlyingThings3D')
        left_images = sorted(glob(osp.join(root, self.dstype, split, '*/*/left/*.png')))
        right_images = [im.replace('left', 'right') for im in left_images]
        disparity_images = [im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images]
        right_disparity_images = [im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in right_images]

        # Choose a random subset of 400 images for validation
        state = np.random.get_state()
        np.random.seed(1000)
        # val_idxs = set(np.random.permutation(len(left_images))[:400])
        np.random.set_state(state)

        for idx, (img1, img2, disp, disp_r) in enumerate(
                zip(left_images, right_images, disparity_images, right_disparity_images)):
            # if (split == 'TEST' and idx in val_idxs) or split == 'TRAIN':
            if (split == 'TEST') or split == 'TRAIN':
                self.image_list += [[img1, img2]]
                self.disparity_list += [disp]
                self.right_disparity_list += [disp_r]
        logging.info(f"Added {len(self.disparity_list) - original_length} from FlyingThings {self.dstype}")

    def _add_monkaa(self):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'Monkaa')
        left_images = sorted(glob(osp.join(root, self.dstype, '*/left/*.png')))
        right_images = [image_file.replace('left', 'right') for image_file in left_images]
        disparity_images = [im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images]
        right_disparity_images = [im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in right_images]

        for img1, img2, disp, disp_r in zip(left_images, right_images, disparity_images, right_disparity_images):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]
            self.right_disparity_list += [disp_r]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Monkaa {self.dstype}")

    def _add_driving(self):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'Driving')
        left_images = sorted(glob(osp.join(root, self.dstype, '*/*/*/left/*.png')))
        right_images = [image_file.replace('left', 'right') for image_file in left_images]
        disparity_images = [im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images]
        right_disparity_images = [im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in right_images]

        for img1, img2, disp, disp_r in zip(left_images, right_images, disparity_images, right_disparity_images):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]
            self.right_disparity_list += [disp_r]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Driving {self.dstype}")


class ETH3D(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/ETH3D', split='training'):
        super(ETH3D, self).__init__(aug_params, sparse=True)

        image1_list = sorted(glob(osp.join(root, f'two_view_{split}/*/im0.png')))
        image2_list = sorted(glob(osp.join(root, f'two_view_{split}/*/im1.png')))
        disp_list = sorted(glob(osp.join(root, 'two_view_training_gt/*/disp0GT.pfm'))) if split == 'training' else \
            [osp.join(root, 'two_view_training_gt/playground_1l/disp0GT.pfm')] * len(image1_list)

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class KITTI(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/KITTI', image_set='training', valid=False, val_set=None,
                 submission=False):
        super(KITTI, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI, submission=submission)
        assert os.path.exists(root)
        if valid:  # validation
            image1_list = sorted(glob(os.path.join(root, 'Kitti15', image_set, 'image_2/*_10.png')))
            image2_list = sorted(glob(os.path.join(root, 'Kitti15', image_set, 'image_3/*_10.png')))
            disp_list = sorted(glob(
                os.path.join(root, 'Kitti15', 'training', 'disp_occ_0/*_10.png'))) if image_set == 'training' else \
                [osp.join(root, 'training/disp_occ_0/000085_10.png')] * len(image1_list)

            for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
                if val_set is None or idx in val_set:
                    self.image_list += [[img1, img2]]
                    self.disparity_list += [disp]
                    self.extra_info += [img1[-13:]]
        else:  # training
            image1_list = sorted(glob(os.path.join(root, 'Kitti15', image_set, 'image_2/*_10.png')))
            image2_list = sorted(glob(os.path.join(root, 'Kitti15', image_set, 'image_3/*_10.png')))
            disp_list = sorted(glob(
                os.path.join(root, 'Kitti15', 'training', 'disp_occ_0/*_10.png'))) if image_set == 'training' else \
                [osp.join(root, 'training/disp_occ_0/000085_10.png')] * len(image1_list)
            image1_list += sorted(glob(os.path.join(root, 'Kitti12', image_set, 'image_0/*_10.png')))
            image2_list += sorted(glob(os.path.join(root, 'Kitti12', image_set, 'image_1/*_10.png')))
            disp_list += sorted(
                glob(os.path.join(root, 'Kitti12', 'training', 'disp_occ/*_10.png'))) if image_set == 'training' else \
                [osp.join(root, 'training/disp_occ_0/000085_10.png')] * len(image1_list)

            for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
                if val_set is None or (idx not in val_set):
                    self.image_list += [[img1, img2]]
                    self.disparity_list += [disp]


class Middlebury(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/mb-ex-training/trainingF/2014', split='F', is_test=False,
                 submission=False):
        super(Middlebury, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispMiddlebury,
                                         submission=is_test, load_right_disp=True)
        assert os.path.exists(root)
        assert split in ["F", "H", "Q", "2014", "2021"]
        if is_test:
            if submission:
                image1_list = glob(
                    os.path.join(root, "**/im0.png"), recursive=True)
                image2_list = glob(
                    os.path.join(root, "**/im1.png"), recursive=True)
                image1_list += glob(
                    os.path.join(root.replace('/test', '/training'), "**/im0.png"), recursive=True)
                image2_list += glob(
                    os.path.join(root.replace('/test', '/training'), "**/im1.png"), recursive=True)
            else:
                image1_list = glob(os.path.join(root.replace('mb-ex-training/trainingF', 'Middlebury/MiddEval3/testH'), "**/im0.png"), recursive=True)
                image2_list = glob(os.path.join(root.replace('mb-ex-training/trainingF', 'Middlebury/MiddEval3/testH'), "**/im1.png"), recursive=True)
            assert len(image1_list) == len(image2_list), [image1_list, split]
            for img1, img2 in zip(image1_list, image2_list):
                self.image_list += [[img1, img2]]
                self.extra_info += [img1.replace(root + '/', '').replace('/im0.png', '')]
        else:
            if split == "2014":  # datasets/Middlebury/2014/Pipes-perfect/im0.png
                scenes = list((Path(root) / "2014").glob("*"))
                for scene in scenes:
                    for s in ["E", "L", ""]:
                        self.image_list += [[str(scene / "im0.png"), str(scene / f"im1{s}.png")]]
                        self.disparity_list += [str(scene / "disp0.pfm")]
            else:
                image1_list = sorted(glob(os.path.join(root, "**/im0.png"), recursive=True)
                                     # + glob(os.path.join(root, "**/view5.png"), recursive=True)
                                     )
                image2_list = sorted(glob(os.path.join(root, "**/im1.png"), recursive=True)
                                     # + glob(os.path.join(root, "**/view1.png"), recursive=True)
                                     )
                disp_list = sorted(glob(os.path.join(root, "**/disp0GT.pfm"), recursive=True)
                                   # + glob(os.path.join(root, "**/disp0.pfm"), recursive=True)
                                   # + glob(os.path.join(root, "**/disp5.png"), recursive=True)
                                   )
                disp_r_list = [x.replace('disp0GT.pfm', 'disp1GT.pfm') for x in disp_list]

                assert len(image1_list) == len(image2_list) == len(disp_list) == len(disp_r_list) > 0, [len(image1_list), len(image2_list),
                                                                                                        len(disp_list), len(disp_r_list)]
                for img1, img2, disp, disp_r in zip(image1_list, image2_list, disp_list, disp_r_list):
                    self.image_list += [[img1, img2]]
                    self.disparity_list += [disp]
                    self.right_disparity_list += [disp_r]


class Booster(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/booster/train/balanced', is_test=False, load_right_disp=False):
        super(Booster, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispBooster,
                                      submission=is_test, load_right_disp=load_right_disp)
        assert os.path.exists(root)
        if is_test:
            root = root.replace('train', 'test')
            image1_list = sorted(glob(os.path.join(root, "**/camera_00/im*.png"), recursive=True))
            image2_list = sorted(glob(os.path.join(root, "**/camera_02/im*.png"), recursive=True))
            assert len(image1_list) == len(image2_list) > 0, [len(image1_list), len(image2_list)]
            for img1, img2 in zip(image1_list, image2_list):
                self.image_list += [[img1, img2]]
                self.extra_info += [img1.replace(root + '/', '').replace('/im*.png', '')]
        else:
            image1_list = sorted(glob(os.path.join(root, "**/camera_00/im*.png"), recursive=True))
            image2_list = sorted(glob(os.path.join(root, "**/camera_02/im*.png"), recursive=True))

            disp_list = [os.path.join(os.path.split(x)[0], '../disp_00.npy') for x in image1_list]
            right_disp_list = [os.path.join(os.path.split(x)[0], '../disp_02.npy') for x in image1_list]
            assert len(image1_list) == len(image2_list) == len(disp_list) > 0, [len(image1_list), len(image2_list),
                                                                                len(disp_list)]
            for img1, img2, disp, disp_r in zip(image1_list, image2_list, disp_list, right_disp_list):
                self.image_list += [[img1, img2]]
                self.disparity_list += [disp]
                self.right_disparity_list += [disp_r]


class Crestereo(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/crestereo', is_test=False, load_right_disp=False):
        super(Crestereo, self).__init__(aug_params, sparse=False, reader=frame_utils.readDispCrestereo,
                                        submission=is_test, load_right_disp=load_right_disp)
        assert os.path.exists(root)

        image1_list = sorted(glob(os.path.join(root, "**/*left.jpg"), recursive=True))
        image2_list = sorted(glob(os.path.join(root, "**/*right.jpg"), recursive=True))

        disp_list = [x.replace('.jpg', '.disp.png') for x in image1_list]
        right_disp_list = [x.replace('.jpg', '.disp.png') for x in image2_list]
        assert len(image1_list) == len(image2_list) == len(disp_list) == len(right_disp_list) > 0, \
            [len(image1_list), len(image2_list), len(disp_list), len(right_disp_list)]
        for img1, img2, disp, disp_r in zip(image1_list, image2_list, disp_list, right_disp_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]
            self.right_disparity_list += [disp_r]


def fetch_dataloader(args, occ_mask):
    """ Create the data loader for the corresponding trainign set """
    load_right_disp = False
    aug_params = {'crop_size': args.image_size, 'min_scale': args.spatial_scale[0], 'max_scale': args.spatial_scale[1],
                  'do_flip': False, 'yjitter': not args.noyjitter}
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip
        if type(args.do_flip) is list and 'h' in args.do_flip:
            load_right_disp = True

    train_dataset = None
    for dataset_name in args.train_datasets:
        if dataset_name.startswith("middlebury_"):
            new_dataset = Middlebury(aug_params, split=dataset_name.replace('middlebury_', ''))
        elif dataset_name == 'sceneflow':
            clean_dataset = SceneFlowDatasets(aug_params, dstype='frames_cleanpass', occ_mask=occ_mask,
                                              load_right_disp=load_right_disp)
            final_dataset = SceneFlowDatasets(aug_params, dstype='frames_finalpass', occ_mask=occ_mask,
                                              load_right_disp=load_right_disp)
            new_dataset = (clean_dataset * 4) + (final_dataset * 4)
            logging.info(f"Adding {len(new_dataset)} samples from SceneFlow")
        elif dataset_name == 'kitti':
            new_dataset = KITTI(aug_params, val_set=args.valid_set) * 5
            logging.info(f"Adding {len(new_dataset)} samples from KITTI")
        elif dataset_name == 'booster':
            # aug_params['min_scale'] = -2  # 1/4
            # aug_params['max_scale'] = -0.8  # 1/2
            new_dataset = Booster(aug_params, load_right_disp=load_right_disp) * 10 * 10  #
            logging.info(f"Adding {len(new_dataset)} samples from Booster")
        elif dataset_name == 'crestereo':
            new_dataset = Crestereo(aug_params, load_right_disp=load_right_disp) * 2
            logging.info(f"Adding {len(new_dataset)} samples from CreStereo")
        elif dataset_name == 'eth3d':
            new_dataset = ETH3D(aug_params)
            logging.info(f"Adding {len(new_dataset)} samples from ETH3D")

        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,  # 28 * 4
                                   pin_memory=True, shuffle=True, num_workers=24, prefetch_factor=4, drop_last=True)

    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_loader

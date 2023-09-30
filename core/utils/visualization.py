import os.path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import wandb
from pathlib import Path


def pseudoColorMap(arr, vmin=None, vmax=None, cmap=None):
    """
    :param arr:单通道的arrey
    :param vmin:截断的下限
    :param vmax:截断的上限
    :param cmap:colormap类型
    :return: rgb伪彩图像
    """
    sm = cm.ScalarMappable(cmap=cmap)
    sm.set_clim(vmin, vmax)
    rgba = sm.to_rgba(arr, bytes=True)
    return rgba[:, :, :3]


def logFeatureMap(inputs, logname, wandb, vmin=None, vmax=None, cmap=None, local_save=False):
    """
    :param inputs: tensor like [N,C,H,W]
    :param wandb: wandb handle
    :return:
    """
    if len(inputs.shape) == 4:
        N, C, H, W = inputs.shape
        inputs = inputs.detach().cpu().numpy()
        log_dict = dict()
        # 为了不占用太多的空间，只使用第一个instance的各个channel进行可视化
        for j in range(C):
            slices = inputs[0, j, :, :]
            slicesMap = pseudoColorMap(slices, vmin, vmax, cmap)
            log_dict[logname + "_" + str(j)] = wandb.Image(slicesMap)
        wandb.log(log_dict, commit=False)
    elif len(inputs.shape) == 3:
        N, H, W = inputs.shape
        inputs = inputs.detach().cpu().numpy()
        log_dict = dict()
        # only use the first instance
        slices = inputs[0, :, :]
        slicesMap = pseudoColorMap(slices, vmin, vmax, cmap)
        if local_save:
            pt="./localSave/" + logname.replace("/camera_00", "")
            Path(os.path.split(pt)[0]).mkdir(exist_ok=True, parents=True)
            img=Image.fromarray(slicesMap)
            img.save(pt, format='png')
        else:
            log_dict[logname] = wandb.Image(slicesMap)
            wandb.log(log_dict, commit=False)


def gen_error_colormap():
    cols = np.array(
        [[0 / 3.0, 0.1875 / 3.0, 49, 54, 149],
         [0.1875 / 3.0, 0.375 / 3.0, 69, 117, 180],
         [0.375 / 3.0, 0.75 / 3.0, 116, 173, 209],
         [0.75 / 3.0, 1.5 / 3.0, 171, 217, 233],
         [1.5 / 3.0, 3 / 3.0, 224, 243, 248],
         [3 / 3.0, 6 / 3.0, 254, 224, 144],
         [6 / 3.0, 12 / 3.0, 253, 174, 97],
         [12 / 3.0, 24 / 3.0, 244, 109, 67],
         [24 / 3.0, 48 / 3.0, 215, 48, 39],
         [48 / 3.0, np.inf, 165, 0, 38]], dtype=np.float32)
    cols[:, 2: 5] /= 255.
    return cols


def logErrorMap(disp_pr, disp_gt, log_name, wandb, abs_thres=3., rel_thres=0.05, dilate_radius=1):
    D_gt_np = disp_gt.detach().cpu().numpy()[:1, :, :]
    D_est_np = disp_pr.detach().cpu().numpy()[:1, :, :]
    B, H, W = D_gt_np.shape
    # valid mask
    mask = D_gt_np > 0
    # error in percentage. When error <= 1, the pixel is valid since <= 3px & 5%
    error = np.abs(D_gt_np - D_est_np)
    error[np.logical_not(mask)] = 0
    error[mask] = np.minimum(error[mask] / abs_thres, (error[mask] / D_gt_np[mask]) / rel_thres)
    # get colormap
    cols = gen_error_colormap()
    # create error image
    error_image = np.zeros([B, H, W, 3], dtype=np.float32)
    for i in range(cols.shape[0]):
        error_image[np.logical_and(error >= cols[i][0], error < cols[i][1])] = cols[i, 2:]

    # error_image = cv2.imdilate(D_err, strel('disk', dilate_radius));
    error_image[np.logical_not(mask)] = 0.
    # show color tag in the top-left cornor of the image
    for i in range(cols.shape[0]):
        distance = 20
        error_image[:, :10, i * distance:(i + 1) * distance, :] = cols[i, 2:]
    log_dict = dict()
    log_dict[log_name] = wandb.Image(error_image)
    wandb.log(log_dict, commit=False)


if __name__ == "__main__":
    wandb.init(project="ToyExperiment", entity="zengjiaxi")
    img = np.random.random((1, 3, 16, 16))
    logFeatureMap(img, "feature", wandb, cmap="jet")

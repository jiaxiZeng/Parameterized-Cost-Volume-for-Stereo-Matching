# Parameterized Cost Volume for Stereo Matching (ICCV2023)
This repository is the implementation of [Parameterized Cost Volume for Stereo Matching](https://openaccess.thecvf.com/content/ICCV2023/papers/Zeng_Parameterized_Cost_Volume_for_Stereo_Matching_ICCV_2023_paper.pdf).

We thank the authors of [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo), as our code is built on top of their project.
## Data
To evaluate/train this model, you need to download the required datasets,
- [sceneflow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html#:~:text=on%20Academic%20Torrents-,FlyingThings3D,-Driving)
- [middlebury](https://vision.middlebury.edu/stereo/data/)
- [kitti15](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

and organize them as following:

```
├── datasets
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── disparity
    ├── Monkaa
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── disparity
    ├── Driving
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── disparity
    ├── KITTI
        ├──Kitti12
           ├── testing
           ├── training
        ├──Kitti15
           ├── testing
           ├── training
    ├── Middlebury
        ├── trainingF
        ├── trainingH
        ├── trainingQ
        ├── testF
        ├── testH
        ├── testQ
```

The ```datasets``` directory need to be placed on the root of this project.

## Environment
CUDA 11.3 

python 3.8.12

pytorch 1.12.0

```
pip install scipy
pip install tqdm
pip install tensorboard
pip install opt_einsum
pip install imageio
pip install opencv-python
pip install scikit-image
pip install einops
pip install wandb
pip install matplotlib
```

## Train
```
bash ./train_sceneflow.sh
```

## Evaluate
You need to create a directory called ```pcv_ckpts```  in the root directory of this project, then download [chechkpoint](https://www.dropbox.com/scl/fi/za096sxpi7t6d5uk1f1da/pcvnet_sceneflow_sigma32.pth?rlkey=367wody43u6tj4uzx4lzf7lcz&dl=0) to the directory, and run the following code:

```
bash ./test_sceneflow.sh
```

## Citation
```
@inproceedings{zeng2023parameterized,
  title={Parameterized Cost Volume for Stereo Matching},
  author={Zeng, Jiaxi and Yao, Chengtang and Yu, Lidong and Wu, Yuwei and Jia, Yunde},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={18347--18357},
  year={2023}
}
```


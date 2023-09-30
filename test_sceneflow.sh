python evaluate_stereo.py \
--dataset things \
--mixed_precision \
--valid_iters 4 \
--init_sigma 32 \
--slow_fast_gru \
--restore_ckpt ./pcv_ckpts/pcvnet_sceneflow_sigma32.pth

python evaluate_stereo.py \
--dataset booster \
--mixed_precision \
--slow_fast_gru \
--valid_iter 4 \
--init_sigma 8 \
--submission \
--restore_ckpt ./pcv_ckpts/pcvnet_booster_sigma8_final.pth
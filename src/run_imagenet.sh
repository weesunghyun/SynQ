config_path="./config/imagenet_3bit.hocon"

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 main_direct.py --conf_path $config_path --lambda_ce 0.5 --lambda_cam 50 --save_model True
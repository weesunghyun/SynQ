config_path="./config/cifar100_6bit.hocon"

CUDA_VISIBLE_DEVICES=$1 torchrun --nproc_per_node 1 --master_port 29501 main_direct.py --conf_path $config_path --lambda_ce 0.05 --lambda_cam 20 --few_shot True --save_model True
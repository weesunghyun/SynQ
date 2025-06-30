config_path=(
    # "./config/imagenet_3bit.hocon"
    "./config/imagenet_4bit.hocon"
)

for config in "${config_path[@]}"; do
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port=29503 main_direct.py --conf_path $config --lambda_ce 0.5 --lambda_cam 50 --save_model True
done

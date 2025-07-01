gpu_id=0
master_port=8900

config_path=(
    "./config/dermamnist_3bit.hocon"
    "./config/dermamnist_4bit.hocon"
)

for config in "${config_path[@]}"; do
    CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=${master_port}  main_direct.py --conf_path $config --lambda_ce 0.5 --lambda_cam 50 --save_model True
done

config_path="./config/cifar100_6bit.hocon"
# for j in 20 50 100 200
# do
#     for i in 0.005 0.05 0.5
#     do
#     CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 main_direct.py --conf_path $config_path --lambda_ce $i --lambda_cam $j --few_shot True
#     done
# done

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node 1 main_direct.py --conf_path $config_path --lambda_ce 0.005 --lambda_cam 20 --few_shot True

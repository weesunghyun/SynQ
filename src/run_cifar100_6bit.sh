config_path="./config/cifar100_6bit.hocon"
<<<<<<< HEAD
for j in 20 50 100 200
do
    for i in 0.005 0.05 0.5
    do
    CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node 1 main_direct.py --conf_path $config_path --lambda_ce $i --lambda_cam $j --few_shot True
    done
done
=======
# for j in 20 50 100 200
# do
#     for i in 0.005 0.05 0.5
#     do
#     CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 main_direct.py --conf_path $config_path --lambda_ce $i --lambda_cam $j --few_shot True
#     done
# done
>>>>>>> parent of 59f1dbb... 240604 commit (1)

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node 1 main_direct.py --conf_path $config_path --lambda_ce 0.005 --lambda_cam 20 --few_shot True
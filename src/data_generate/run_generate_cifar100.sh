python generate_data.py \
        --model=resnet34 \
        --batch_size=256 \
        --test_batch_size=512 \
        --group=1 \
        --beta=0.1 \
        --gamma=0.5 \
        --save_path_head=../data/cifar100 \
        --lbns True
for g in 1 2 3 4
do
		# --model=resnet34_cifar100 	 \
python generate_data.py 		\
		--model=resnet20_cifar100 	 \
		--batch_size=256 		\
		--test_batch_size=512 \
		--group=$g \
		--beta=0.1 \
		--gamma=0.5 \
		--save_path_head=../data/cifar
done

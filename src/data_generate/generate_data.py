"""
SynQ: Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning (ICLR 2025)

Authors:
- Minjun Kim (minjun.kim@snu.ac.kr), Seoul National University
- Jongjin Kim (j2kim99@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

Version : 1.1

Date : Feb. 7th, 2025

Main Contact: Minjun Kim

This software is free of charge under research purposes.

For other purposes (e.g. commercial), please contact the authors.

generate_data.py
    - codes for generating synthetic data for zero-shot quantization

This code is mainly based on
    - ZeroQ: https://github.com/amirgholami/ZeroQ
    - HAST: https://github.com/lihuantong/HAST
"""
import os
import sys
import argparse

import torch
torch.backends.cudnn.enabled = False

from distill_data import generate_calib_centers, DistillData
from collections import OrderedDict
from pytorchcv.model_provider import get_model as ptcv_get_model

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# model settings
def arg_parse():
    """
    Parses arguments for data generation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        default='resnet18',
                        choices=[
                            'resnet18', 'resnet34_cifar100', 'resnet50', 'mobilenet_w1',
                            'mobilenetv2_w1', 'shufflenet_g1_w1',
                            'resnet20_cifar10', 'resnet20_cifar100', 'regnetx_600m'
                        ],
                        help='model to be quantized')
    parser.add_argument('--dataset',
                        type=str,
                        default=None,
                        choices=[
                            'pathmnist', 'octmnist', 'pneumoniamnist',
                            'breastmnist', 'dermamnist', 'bloodmnist',
                            'tissuemnist', 'organamnist', 'organcmnist',
                            'organsmnist'
                        ],
                        help='dataset to generate calibration data for')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='batch size of distilled data')
    parser.add_argument('--num_data',
                        type=int,
                        default=1280,
                        help='batch size of test data')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=128,
                        help='batch size of test data')
    parser.add_argument('--group',
                        type=int,
                        default=1,
                        help='group of generated data')
    parser.add_argument('--beta',
                        type=float,
                        default=1.0,
                        help='beta')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.0,
                        help='gamma')
    parser.add_argument('--save_path_head',
                        type=str,
                        default='',
                        help='save_path_head')
    parser.add_argument('--radius', type=float, default=0.05, metavar='radius')
    parser.add_argument('--lbns', type=bool, default=False, metavar='lbns')
    parser.add_argument('--fft', type=bool, default=False, metavar='fft')

    arguments = parser.parse_args()

    if arguments.lbns:
        arguments.save_path_head = arguments.save_path_head + "_lbns"

    if arguments.fft:
        arguments.save_path_head = arguments.save_path_head + "_fft"
    return arguments


def convert_state_dict(pretrained_state_dict, new_model):
    """
    Converts a pretrained state_dict to match the key format of a new model.

    Args:
        pretrained_state_dict (OrderedDict): The state_dict object from the pretrained model.
        new_model (torch.nn.Module): An instance of the new model architecture.

    Returns:
        OrderedDict: The converted state_dict.
    """
    # Create a new state_dict with the correct keys
    new_state_dict = OrderedDict()

    # --- Key mapping logic remains the same ---
    key_map = {
        'conv1.weight': 'features.init_block.conv.conv.weight',
        'bn1.weight': 'features.init_block.conv.bn.weight',
        'bn1.bias': 'features.init_block.conv.bn.bias',
        'bn1.running_mean': 'features.init_block.conv.bn.running_mean',
        'bn1.running_var': 'features.init_block.conv.bn.running_var',
        'fc.weight': 'output.weight',
        'fc.bias': 'output.bias',
    }

    # Map the layer weights for a ResNet-18 like structure
    # This part might need adjustment depending on the exact ResNet variant.
    # The error message implies a structure with 4 stages (layers) and 2 blocks (units) per stage.
    for i in range(1, 5):  # For layer 1 to 4
        for j in range(2):  # For each block in the layer (assuming 2 blocks for resnet18)
            # Body convolutions
            key_map[f'layer{i}.{j}.conv1.weight'] = f'features.stage{i}.unit{j+1}.body.conv1.conv.weight'
            key_map[f'layer{i}.{j}.bn1.weight'] = f'features.stage{i}.unit{j+1}.body.conv1.bn.weight'
            key_map[f'layer{i}.{j}.bn1.bias'] = f'features.stage{i}.unit{j+1}.body.conv1.bn.bias'
            key_map[f'layer{i}.{j}.bn1.running_mean'] = f'features.stage{i}.unit{j+1}.body.conv1.bn.running_mean'
            key_map[f'layer{i}.{j}.bn1.running_var'] = f'features.stage{i}.unit{j+1}.body.conv1.bn.running_var'
            key_map[f'layer{i}.{j}.conv2.weight'] = f'features.stage{i}.unit{j+1}.body.conv2.conv.weight'
            key_map[f'layer{i}.{j}.bn2.weight'] = f'features.stage{i}.unit{j+1}.body.conv2.bn.weight'
            key_map[f'layer{i}.{j}.bn2.bias'] = f'features.stage{i}.unit{j+1}.body.conv2.bn.bias'
            key_map[f'layer{i}.{j}.bn2.running_mean'] = f'features.stage{i}.unit{j+1}.body.conv2.bn.running_mean'
            key_map[f'layer{i}.{j}.bn2.running_var'] = f'features.stage{i}.unit{j+1}.body.conv2.bn.running_var'

            # Downsample (identity) convolutions for stages 2, 3, and 4
            # In ResNet, the downsample block is typically only in the first unit of a stage.
            if i > 1 and j == 0:
                key_map[f'layer{i}.{j}.downsample.0.weight'] = f'features.stage{i}.unit{j+1}.identity_conv.conv.weight'
                key_map[f'layer{i}.{j}.downsample.1.weight'] = f'features.stage{i}.unit{j+1}.identity_conv.bn.weight'
                key_map[f'layer{i}.{j}.downsample.1.bias'] = f'features.stage{i}.unit{j+1}.identity_conv.bn.bias'
                key_map[f'layer{i}.{j}.downsample.1.running_mean'] = f'features.stage{i}.unit{j+1}.identity_conv.bn.running_mean'
                key_map[f'layer{i}.{j}.downsample.1.running_var'] = f'features.stage{i}.unit{j+1}.identity_conv.bn.running_var'

    # Populate the new_state_dict
    for old_key, new_key in key_map.items():
        if old_key in pretrained_state_dict:
            new_state_dict[new_key] = pretrained_state_dict[old_key]
        # else:
        #     print(f"Warning: key '{old_key}' not found in pretrained model")

    # Handle unexpected keys that are not mapped (like 'num_batches_tracked')
    # These are often not needed for inference, so we can usually ignore them.
    for key in pretrained_state_dict:
        if 'num_batches_tracked' in key:
            # You might need to map these as well if your new model requires them,
            # but usually they are not critical for loading weights.
            pass

    # Check if all keys in the new model are present in the converted dict
    # new_model_keys = set(new_model.state_dict().keys())
    # converted_keys = set(new_state_dict.keys())
    # missing_keys = new_model_keys - converted_keys
    # if missing_keys:
    #     print("Missing keys in converted state_dict:", missing_keys)

    return new_state_dict


if __name__ == '__main__':
    args = arg_parse()
    print(args)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    model = ptcv_get_model(args.model, pretrained=True)
    print('****** Full precision model loaded ******')

    if args.dataset is not None:
        args.dataset = args.dataset.lower()
        args.model = args.model + '_' + args.dataset
        args.save_path_head = args.save_path_head + '_' + args.dataset
        pretrained_path = f'../checkpoints/{args.model}.pth'
        if not os.path.exists(pretrained_path):
            raise ValueError(f"Pretrained model {pretrained_path} not found")
        
        # Determine the correct number of classes for the dataset
        if args.dataset == 'dermamnist':
            num_classes = 7
        elif args.dataset == 'pathmnist':
            num_classes = 9
        elif args.dataset == 'octmnist':
            num_classes = 4
        elif args.dataset == 'pneumoniamnist':
            num_classes = 2
        elif args.dataset == 'breastmnist':
            num_classes = 2
        elif args.dataset == 'bloodmnist':
            num_classes = 8
        elif args.dataset == 'tissuemnist':
            num_classes = 8
        elif args.dataset == 'organamnist':
            num_classes = 11
        elif args.dataset == 'organcmnist':
            num_classes = 11
        elif args.dataset == 'organsmnist':
            num_classes = 11
        else:
            num_classes = 1000
        
        # Create a new model with the correct number of classes
        model = ptcv_get_model(args.model.split('_')[0], pretrained=False, num_classes=num_classes)
        print(f'****** Model created with {num_classes} classes for {args.dataset} ******')
        
        # Load checkpoint and handle different formats
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'net' in checkpoint:
            converted_state_dict = convert_state_dict(checkpoint['net'], model)
        else:
            converted_state_dict = convert_state_dict(checkpoint, model)

        model.load_state_dict(converted_state_dict)
        print(f'****** Pretrained model {pretrained_path} loaded ******')

    if args.lbns:
        args.calib_centers = generate_calib_centers(args, model.cuda())

    DD = DistillData(args)
    dataloader = DD.get_distil_data(
        model_name=args.model,
        teacher_model=model.cuda(),
        batch_size=args.batch_size,
        group=args.group,
        beta=args.beta,
        gamma=args.gamma,
        save_path_head=args.save_path_head
    )

    print('****** Data Generated ******')

import argparse

import numpy as np

import torch

from pytorchcv.models.resnet_cifar import CIFARResNet
from pytorchcv.model_provider import get_model as ptcv_get_model

from distill_data import *

from ..utils.get_resnet34 import *


# model settings
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        default='resnet34',
                        choices=[
                            'resnet18', 'resnet34_cifar100', 'resnet50', 'mobilenet_w1',
                            'mobilenetv2_w1', 'shufflenet_g1_w1',
                            'resnet20_cifar10', 'resnet20_cifar100', 'regnetx_600m'
                        ],
                        help='model to be quantized')
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
    
    args = parser.parse_args()
    
    if args.lbns:
        args.save_path_head = args.save_path_head + "_lbns"
    
    if args.fft:
        args.save_path_head = args.save_path_head + "_fft"
    return args


if __name__ == '__main__':
    args = arg_parse()
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    if args.model == 'resnet34_cifar100':
        model = resnet34_get_model()
        # model.load_state_dict(torch.load('resnet34_cifar100.pth'))
    else:
        model = ptcv_get_model(args.model, pretrained=True)
        print('****** Full precision model loaded ******')

    if args.lbns:
        args.calib_centers = generate_calib_centers(args, model.cuda())
    
    DD = DistillData(args)
    dataloader = DD.getDistilData(
        model_name=args.model,
        teacher_model=model.cuda(),
        batch_size=args.batch_size,
        group=args.group,
        beta=args.beta,
        gamma=args.gamma,
        save_path_head=args.save_path_head
    )

    print('****** Data Generated ******')





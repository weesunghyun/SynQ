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

main_direct.py
    - codes for main function for zero-shot quantization (SynQ)

This code is mainly based on
    - ZeroQ: https://github.com/amirgholami/ZeroQ
    - HAST: https://github.com/lihuantong/HAST
"""
import os
import copy
import time
import pickle
import shutil
import logging
import argparse
import datetime
import warnings
import traceback

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import Dataset
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision import datasets
from torchvision import transforms

import utils
from options import Option
from dataloader import DataLoader
from trainer_direct import Trainer
from quantization_utils.quant_modules import QuantAct, QuantLinear, QuantConv2d
from quantization_utils import AdaRoundConv2d, AdaRoundLinear
from utils.get_resnet34 import resnet34_get_model
from pytorchcv.model_provider import get_model as ptcv_get_model
from conditional_batchnorm import CategoricalConditionalBatchNorm2d
from tqdm import tqdm
from collections import OrderedDict
# from regularizer import get_reg_criterions

# Classification task datasets from MedMNIST2D
CLASSIFICATION_DATASETS = {
    'pathmnist': 9,      # Colon Pathology - Multi-Class (9)
    'dermamnist': 7,     # Dermatoscope - Multi-Class (7)
    'octmnist': 4,       # Retinal OCT - Multi-Class (4)
    'pneumoniamnist': 2, # Chest X-Ray - Binary-Class (2)
    'retinamnist': 5,    # Fundus Camera - Ordinal Regression (5) - treated as classification
    'breastmnist': 2,    # Breast Ultrasound - Binary-Class (2)
    'bloodmnist': 8,     # Blood Cell Microscope - Multi-Class (8)
    'tissuemnist': 8,    # Kidney Cortex Microscope - Multi-Class (8)
    'organamnist': 11,   # Abdominal CT - Multi-Class (11)
    'organcmnist': 11,   # Abdominal CT - Multi-Class (11)
    'organsmnist': 11,   # Abdominal CT - Multi-Class (11)
}

medmnist_dataset = list(CLASSIFICATION_DATASETS.keys())


class Generator_32(nn.Module):
	def __init__(self, options=None, conf_path=None):
		super(Generator_32, self).__init__()
		self.settings = options or Option(conf_path)
		self.label_emb = nn.Embedding(self.settings.num_classes, self.settings.latent_dim)
		self.init_size = self.settings.img_size // 4
		self.l1 = nn.Sequential(nn.Linear(self.settings.latent_dim, 128 * self.init_size ** 2))

		self.conv_blocks0 = nn.Sequential(
			nn.BatchNorm2d(128),
		)

		self.conv_blocks1 = nn.Sequential(
			nn.Conv2d(128, 128, 3, stride=1, padding=1),
			nn.BatchNorm2d(128, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
		)
		self.conv_blocks2 = nn.Sequential(
			nn.Conv2d(128, 64, 3, stride=1, padding=1),
			nn.BatchNorm2d(64, 0.8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(64, self.settings.channels, 3, stride=1, padding=1),
			nn.Tanh(),
			nn.BatchNorm2d(self.settings.channels, affine=False)
		)

	def forward(self, z, labels):
		gen_input = torch.mul(self.label_emb(labels), z)
		out = self.l1(gen_input)
		out = out.view(out.shape[0], 128, self.init_size, self.init_size)
		img = self.conv_blocks0(out)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks1(img)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks2(img)
		return img

class Generator_224(nn.Module):
	def __init__(self, options=None, conf_path=None):
		self.settings = options or Option(conf_path)

		super(Generator_224, self).__init__()

		self.init_size = self.settings.img_size // 4
		self.l1 = nn.Sequential(nn.Linear(self.settings.latent_dim, 128 * self.init_size ** 2))

		self.conv_blocks0_0 = CategoricalConditionalBatchNorm2d(self.settings.num_classes, 128)

		self.conv_blocks1_0 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
		self.conv_blocks1_1 = CategoricalConditionalBatchNorm2d(self.settings.num_classes, 128, 0.8)
		self.conv_blocks1_2 = nn.LeakyReLU(0.2, inplace=True)

		self.conv_blocks2_0 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
		self.conv_blocks2_1 = CategoricalConditionalBatchNorm2d(self.settings.num_classes, 64, 0.8)
		self.conv_blocks2_2 = nn.LeakyReLU(0.2, inplace=True)
		self.conv_blocks2_3 = nn.Conv2d(64, self.settings.channels, 3, stride=1, padding=1)
		self.conv_blocks2_4 = nn.Tanh()
		self.conv_blocks2_5 = nn.BatchNorm2d(self.settings.channels, affine=False)

	def forward(self, z, labels):
		out = self.l1(z)
		out = out.view(out.shape[0], 128, self.init_size, self.init_size)
		img = self.conv_blocks0_0(out, labels)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks1_0(img)
		img = self.conv_blocks1_1(img, labels)
		img = self.conv_blocks1_2(img)
		img = nn.functional.interpolate(img, scale_factor=2)
		img = self.conv_blocks2_0(img)
		img = self.conv_blocks2_1(img, labels)
		img = self.conv_blocks2_2(img)
		img = self.conv_blocks2_3(img)
		img = self.conv_blocks2_4(img)
		img = self.conv_blocks2_5(img)
		return img


def create_generator(options=None, conf_path=None):
	if options is not None:
		settings = options
	elif conf_path is not None:
		settings = Option(conf_path)
	else:
		raise ValueError("options or conf_path must be provided")

	if settings.img_size == 32:
		return Generator_32(options=options, conf_path=conf_path)
	elif settings.img_size == 224:
		return Generator_224(options=options, conf_path=conf_path)
	# elif image_size == 64:
	#     return Generator_64(options=options, conf_path=conf_path) # 향후 확장 가능
	else:
		raise ValueError(f"Unsupported image size: {settings.img_size}")


class DirectDataset(Dataset):
    """
    Direct dataset class
    Args:
        args: Arguments
        settings: Option class contains varios configuration
        logger: Logger
        dataset: Dataset name
    """
    def __init__(self, args, settings, logger, dataset):
        self.settings = settings
        self.logger = logger
        self.args = args


        # 이미지 크기 기반으로 transform 설정
        if settings.img_size == 32:
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=32, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
            ])
               
        if self.args.few_shot:
            self.fewshot_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=32, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.50705882, 0.48666667, 0.44078431],
                    [0.26745098, 0.25568627, 0.27607843])
            ])

            cifar100_train = datasets.CIFAR100(
                root='./data/cifar100',
                train=True,
                download=True,
                transform=self.fewshot_transform
                )
            real_data, real_label = np.array(cifar100_train.data), np.array(cifar100_train.targets)

            # Sample few-shot data
            sampled_data = []
            sampled_labels = []
            for class_id in range(100):
                class_indices = np.where(real_label == class_id)[0]
                sampled_indices = np.random.choice(class_indices, 10, replace=False)
                sampled_data.append(real_data[sampled_indices])
                sampled_labels.append(real_label[sampled_indices])
            sampled_data = np.concatenate(sampled_data, axis=0)
            sampled_data = sampled_data.transpose(0, 3, 1, 2)
            sampled_labels = np.concatenate(sampled_labels, axis=0)

            self.tmp_data = sampled_data
            self.tmp_label = sampled_labels

        else:
            self.tmp_data = None
            self.tmp_label = None

            for i in range(1,5):
                path = self.settings.generate_data_path +str(i)+".pickle"
                self.logger.info(path)
                with open(path, "rb") as fp:
                    gaussian_data = pickle.load(fp)
                if self.tmp_data is None:
                    self.tmp_data = np.concatenate(gaussian_data, axis=0)
                else:
                    self.tmp_data = np.concatenate(
                        (self.tmp_data, np.concatenate(gaussian_data, axis=0))
                        )

                path = self.settings.genera_label_path + str(i) + ".pickle"
                self.logger.info(path)
                with open(path, "rb") as fp:
                    labels_list = pickle.load(fp)
                if self.tmp_label is None:
                    self.tmp_label = np.concatenate(labels_list, axis=0)
                else:
                    self.tmp_label = np.concatenate(
                        (self.tmp_label, np.concatenate(labels_list, axis=0))
                        )

        if self.args.calib_centers:
            if not self.settings.model_name == 'resnet18':
                temp = "_" + self.settings.model_name
            else:
                temp = ""
            calib_path = f"../new_generate/data/{self.settings.dataset}{temp}_lbns" + \
                         f"/{self.settings.model_name}_calib_centers.pickle"
            with open(calib_path, "rb") as fp:
                gaussian_data = pickle.load(fp)
            labels_list = range(self.settings.num_classes)
            self.tmp_data = np.concatenate(
                (self.tmp_data, np.concatenate(gaussian_data, axis=0)[:len(labels_list)])
                )
            self.tmp_label = np.concatenate((self.tmp_label, np.array(labels_list)))

        print(self.tmp_data.shape, self.tmp_label.shape)
        assert len(self.tmp_label) == len(self.tmp_data)
        print('direct dataset image number', len(self.tmp_label))


    def __getitem__(self, index):
        img = self.tmp_data[index]
        label = self.tmp_label[index]
        if self.args.few_shot:
            img = img.transpose(1, 2, 0)
            img = Image.fromarray(img)
            img = self.fewshot_transform(img)
            # img = self.fewshot_transform(torch.from_numpy(img))
        else:
            img = self.train_transform(torch.from_numpy(img))
        return img, label

    def __len__(self):
        return len(self.tmp_label)



def convert_state_dict(pretrained_state_dict, new_model):
    """
    Converts a pretrained ResNet-18 state_dict to match the key format of the new model,
    including all 'num_batches_tracked' keys for BatchNorm layers.

    Args:
        pretrained_state_dict (OrderedDict): The state_dict object from the pretrained model.
        new_model (torch.nn.Module): An instance of the new model architecture.

    Returns:
        OrderedDict: The converted state_dict.
    """
    # For debugging: Check if the problematic key exists in the source state_dict
    if 'bn1.num_batches_tracked' not in pretrained_state_dict:
        print("Warning: 'bn1.num_batches_tracked' not found in the source checkpoint!")

    new_state_dict = OrderedDict()
    key_map = {}

    # 1. Initial block (conv1 and bn1)
    key_map.update({
        'conv1.weight': 'features.init_block.conv.conv.weight',
        'bn1.weight': 'features.init_block.conv.bn.weight',
        'bn1.bias': 'features.init_block.conv.bn.bias',
        'bn1.running_mean': 'features.init_block.conv.bn.running_mean',
        'bn1.running_var': 'features.init_block.conv.bn.running_var',
        'bn1.num_batches_tracked': 'features.init_block.conv.bn.num_batches_tracked'  # The missing key
    })

    # 2. ResNet stages (layer1 to layer4)
    for i in range(1, 5):  # Stages 1-4
        for j in range(2):  # Units 1-2 (for ResNet-18)
            # Body convolutions and their batchnorms
            for conv_idx in [1, 2]:
                old_prefix = f'layer{i}.{j}.conv{conv_idx}'
                new_prefix = f'features.stage{i}.unit{j+1}.body.conv{conv_idx}'
                key_map[f'{old_prefix}.weight'] = f'{new_prefix}.conv.weight'

                old_bn_prefix = f'layer{i}.{j}.bn{conv_idx}'
                new_bn_prefix = f'features.stage{i}.unit{j+1}.body.conv{conv_idx}'
                key_map[f'{old_bn_prefix}.weight'] = f'{new_bn_prefix}.bn.weight'
                key_map[f'{old_bn_prefix}.bias'] = f'{new_bn_prefix}.bn.bias'
                key_map[f'{old_bn_prefix}.running_mean'] = f'{new_bn_prefix}.bn.running_mean'
                key_map[f'{old_bn_prefix}.running_var'] = f'{new_bn_prefix}.bn.running_var'
                key_map[f'{old_bn_prefix}.num_batches_tracked'] = f'{new_bn_prefix}.bn.num_batches_tracked'

            # Downsample (identity) convolution for stages 2, 3, 4
            if i > 1 and j == 0:
                old_ds_prefix = f'layer{i}.{j}.downsample'
                new_ds_prefix = f'features.stage{i}.unit{j+1}.identity_conv'
                key_map[f'{old_ds_prefix}.0.weight'] = f'{new_ds_prefix}.conv.weight'
                key_map[f'{old_ds_prefix}.1.weight'] = f'{new_ds_prefix}.bn.weight'
                key_map[f'{old_ds_prefix}.1.bias'] = f'{new_ds_prefix}.bn.bias'
                key_map[f'{old_ds_prefix}.1.running_mean'] = f'{new_ds_prefix}.bn.running_mean'
                key_map[f'{old_ds_prefix}.1.running_var'] = f'{new_ds_prefix}.bn.running_var'
                key_map[f'{old_ds_prefix}.1.num_batches_tracked'] = f'{new_ds_prefix}.bn.num_batches_tracked'

    # 3. Final fully-connected layer
    key_map.update({
        'fc.weight': 'output.weight',
        'fc.bias': 'output.bias'
    })

    # Populate the new_state_dict using the generated map
    for old_key, new_key in key_map.items():
        if old_key in pretrained_state_dict:
            new_state_dict[new_key] = pretrained_state_dict[old_key]

    return new_state_dict



class ExperimentDesign:
    """
    Experiment design class
    Args:
        options: Option class contains varios configuration
        args: Arguments
        logger: Logger
    """
    def __init__(self, options=None, args=None, logger=None):
        self.settings = options
        self.args = args
        self.logger = logger

        self.train_loader = None
        self.test_loader = None
        self.model = None
        self.model_teacher = None
        self.optimizer_state = None
        self.trainer = None
        self.start_epoch = 0

        self.prepare()

    def set_logger(self):
        """
        Set logger
        """
        if dist.get_rank()==0:
            file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
            file_handler = logging.FileHandler(
                os.path.join(self.settings.save_path, "train_test.log")
                )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO if self.args.local_rank in [-1, 0] else logging.WARN)
        return self.logger

    def prepare(self):
        """
        Prepare for the experiment
        """
        torch.cuda.set_device(self.args.local_rank)
        dist.init_process_group(backend='nccl')
        if dist.get_rank() == 0:
            self.settings.set_save_path()
            print(self.settings.save_path)
            shutil.copyfile(
                self.args.conf_path,
                os.path.join(self.settings.save_path, os.path.basename(self.args.conf_path))
                )
            shutil.copyfile(
                './main_direct.py',
                os.path.join(self.settings.save_path, 'main_direct.py')
                )
            shutil.copyfile(
                './trainer_direct.py',
                os.path.join(self.settings.save_path, 'trainer_direct.py')
                )
        self.logger = self.set_logger()
        self.settings.paramscheck(self.logger)
        self._set_gpu()
        self._set_dataloader()
        self._set_model()
        self._replace()
        self._set_trainer()

    def _set_gpu(self):
        """
        Set GPU
        """
        torch.manual_seed(self.settings.manual_seed)
        torch.cuda.manual_seed(self.settings.manual_seed)
        cudnn.benchmark = True

    def _set_dataloader(self):
        """
        Set data loader
        """
        data_loader = DataLoader(dataset=self.settings.dataset,
                                 batch_size=self.settings.batch_size,
                                 data_path=self.settings.data_path,
                                 n_threads=self.settings.num_threads,
                                 ten_crop=self.settings.ten_crop,
                                 logger=self.logger)

        self.train_loader, self.test_loader = data_loader.getloader()

    def _set_model(self):
        """
        Set model
        """
        if self.settings.dataset in medmnist_dataset:
            num_classes = self.settings.num_classes
            self.model = ptcv_get_model(self.settings.model_name, pretrained=False, num_classes=num_classes)
            self.model_teacher = ptcv_get_model(self.settings.model_name, pretrained=False, num_classes=num_classes)
            print(f'****** Model created with {num_classes} classes for {self.settings.dataset} ******')

            # Load checkpoint and handle different formats
            checkpoint = torch.load(self.settings.pretrained_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'net' in checkpoint:
                converted_state_dict = convert_state_dict(checkpoint['net'], self.model)
            else:
                converted_state_dict = convert_state_dict(checkpoint, self.model)

            self.model.load_state_dict(converted_state_dict)
            self.model_teacher.load_state_dict(converted_state_dict)
            print(f'****** Pretrained model {self.settings.pretrained_path} loaded ******')

        else:
            self.model = ptcv_get_model(self.settings.model_name, pretrained=True)
            self.model_teacher = ptcv_get_model(self.settings.model_name, pretrained=True)

        self.generator = create_generator(options=self.settings)
        self.model_teacher.eval()

        self.model_teacher = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model_teacher)

        self.model_teacher = DDP(
            self.model_teacher.to(self.args.local_rank),
            device_ids=[self.args.local_rank],
            output_device=self.args.local_rank,
            broadcast_buffers=False
            )
        self.generator = DDP(
            self.generator.to(self.args.local_rank),
            device_ids=[self.args.local_rank],
            output_device=self.args.local_rank,
            broadcast_buffers=False
            )

    def _set_trainer(self):
        """
        Set trainer
        """
        lr_master_s = utils.LRPolicy(self.settings.lr_s,
                                     self.settings.num_epochs,
                                     self.settings.lr_policy_s)
        lr_master_g = utils.LRPolicy(self.settings.lr_g,
                                     self.settings.num_epochs,
                                     self.settings.lr_policy_g)

        params_dict_s = {
            'step': self.settings.step_s,
            'decay_rate': self.settings.decay_rate_s
        }

        params_dict_g = {
            'step': self.settings.step_g,
            'decay_rate': self.settings.decay_rate_g
        }

        lr_master_s.set_params(params_dict=params_dict_s)
        lr_master_g.set_params(params_dict=params_dict_g)

        self.trainer = Trainer(
            model=self.model,
            model_teacher=self.model_teacher,
            generator = self.generator,
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            lr_master_s=lr_master_s,
            lr_master_g=lr_master_g,
            settings=self.settings,
            args = self.args,
            logger=self.logger,
            opt_type=self.settings.opt_type,
            optimizer_state=self.optimizer_state,
            run_count=self.start_epoch)

    def quantize_model(self, model):
        """
        Quantize model
        Args:
            model: Original Pretrained Model (Full-Precision)
        """
        weight_bit = self.settings.qw
        act_bit = self.settings.qa

        use_ptq = getattr(self.settings, 'quant_method', 'qat') == 'ptq'

        if isinstance(model, nn.Conv2d):
            if use_ptq:
                quant_mod = AdaRoundConv2d(weight_bit=weight_bit)
            else:
                quant_mod = QuantConv2d(weight_bit=weight_bit)
            quant_mod.set_param(model)
            return quant_mod
        if isinstance(model, nn.Linear):
            if use_ptq:
                quant_mod = AdaRoundLinear(weight_bit=weight_bit)
            else:
                quant_mod = QuantLinear(weight_bit=weight_bit)
            quant_mod.set_param(model)
            return quant_mod
        if isinstance(model, (nn.ReLU, nn.ReLU6)):
            return nn.Sequential(*[model, QuantAct(activation_bit=act_bit)])
        if isinstance(model, nn.Sequential):
            mods = []
            for _, m in model.named_children():
                mods.append(self.quantize_model(m))
            return nn.Sequential(*mods)

        q_model = copy.deepcopy(model)
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                setattr(q_model, attr, self.quantize_model(mod))
        return q_model

    def _replace(self):
        """
        Replace model with quantized model
        """
        self.model = self.quantize_model(self.model)
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = DDP(
            self.model.to(self.args.local_rank),
            device_ids=[self.args.local_rank],
            output_device=self.args.local_rank,
            broadcast_buffers=False
            )

    def freeze_model(self, model):
        """
        Freeze model
        Args:
            model: Model to freeze
        """
        if isinstance(model, QuantAct):
            model.fix()
        elif isinstance(model, nn.Sequential):
            for _, m in model.named_children():
                self.freeze_model(m)
        else:
            for attr in dir(model):
                mod = getattr(model, attr)
                if isinstance(mod, nn.Module) and 'norm' not in attr:
                    self.freeze_model(mod)
            # return model

    def unfreeze_model(self, model):
        """
        Unfreeze model
        Args:
            model: Model to unfreeze
        """
        if isinstance(model, QuantAct):
            model.unfix()
        elif isinstance(model, nn.Sequential):
            for _, m in model.named_children():
                self.unfreeze_model(m)
        else:
            for attr in dir(model):
                mod = getattr(model, attr)
                if isinstance(mod, nn.Module) and 'norm' not in attr:
                    self.unfreeze_model(mod)
            # return model

    def run(self):
        """
        Run the experiment
        """
        best_top1 = 100
        best_top5 = 100
        start_time = time.time()

        dataset = DirectDataset(self.args, self.settings, self.logger, self.settings.dataset)

        bs = min(self.settings.batch_size, len(dataset))
        direct_dataload = torch.utils.data.DataLoader(dataset,
                                                      batch_size= bs,
                                                      sampler = DistributedSampler(dataset))
        test_error, _, test5_error = self.trainer.test(epoch=-1)

        if self.settings.quant_method == 'ptq':
            self.trainer.ptq_calibrate(direct_dataload)
            test_error, _, test5_error = self.trainer.test(epoch=0)
            return test_error, test5_error

        try:
            for epoch in range(self.start_epoch, self.settings.num_epochs):
                self.epoch = epoch
                self.start_epoch = 0

                if epoch < 4:
                    self.unfreeze_model(self.model)

                # train_error, train_loss, train5_error =
                self.trainer.train(epoch=epoch, direct_dataload=direct_dataload)

                self.freeze_model(self.model)

                if epoch > 4:
                # if (epoch > self.settings.num_epochs // 5) and (epoch % 5 == 0):
                    test_error, _, test5_error = self.trainer.test(epoch=epoch)
                else:
                    print(f"skip eval for epoch {epoch}")
                    self.logger.info(f"skip eval for epoch {epoch}")
                    continue

                if best_top1 >= test_error:
                    best_top1 = test_error
                    best_top5 = test5_error

                    if self.args.save_model:
                        self.logger.info(
                            f"Save model! The path is "
                            f"{os.path.join(self.settings.save_path, 'model.pth')}"
                            )
                        print(
                            f"Save model! The path is "
                            f"{os.path.join(self.settings.save_path, 'model.pth')}"
                            )
                        torch.save(
                            self.model.state_dict(),
                            os.path.join(self.settings.save_path, "model.pth")
                            )

                self.logger.info(
                    f"#==>Best Result is: Top1 Error: {best_top1}, Top5 Error: {best_top5}"
                    )
                self.logger.info(
                    f"#==>Best Result is: Top1 Accuracy: {100 - best_top1}, "
                    f"Top5 Accuracy: {100 - best_top5}"
                    )
                print(
                    f"#==>Best Result is: Top1 Accuracy: {100 - best_top1}, "
                    f"Top5 Accuracy: {100 - best_top5}"
                    )

        except BaseException as e:
            self.logger.error(f"Training is terminating due to exception: {str(e)}")
            traceback.print_exc()

        end_time = time.time()
        time_interval = end_time - start_time
        t_string = "Running Time is: " + str(datetime.timedelta(seconds=time_interval)) + "\n"
        self.logger.info(t_string)

        return best_top1, best_top5


def main():
    """
    Main function
    """
    logger = logging.getLogger()
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument('--conf_path', type=str, metavar='conf_path',
                        help='input the path of config file')
    parser.add_argument('--cam_type', type=str, default='gradcam', metavar='cam_type')
    parser.add_argument('--lambda_cam', type=float, default=200, metavar='lambda_cam')
    parser.add_argument('--lambda_pert', type=float, default=1, metavar='lambda_pert')
    parser.add_argument('--selce', type=bool, default=True, metavar='selce')
    parser.add_argument('--tau_selce', type=float, default=0.5, metavar='tau_selce')
    parser.add_argument('--lambda_ce', type=float, default=5, metavar='lambda_ce')
    parser.add_argument('--d_zero', type=int, default=80, metavar='d_zero')
    parser.add_argument('--calib_centers', type=bool, default=False, metavar='calib_centers')
    parser.add_argument('--save_model', type=bool, default=False, metavar='save_model')
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--few_shot", default=False, type=bool)
    args = parser.parse_args()

    if not args.selce:
        args.tau_selce = 0

    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])

    if args.local_rank != -1:
        print(f"Running in distributed mode. Local rank: {args.local_rank}")
    else:
        print("Running in non-distributed mode.")

    option = Option(args.conf_path)
    option.manual_seed = 1

    print(args)
    logger.info(args)

    experiment = ExperimentDesign(option, args, logger)
    experiment.run()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()

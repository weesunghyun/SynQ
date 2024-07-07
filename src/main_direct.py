"""
    # TODO: add description
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
from quantization_utils.quant_modules import * # import waildcard
from utils.get_resnet34 import resnet34_get_model
from pytorchcv.model_provider import get_model as ptcv_get_model
from conditional_batchnorm import CategoricalConditionalBatchNorm2d


class Generator(nn.Module):
    """
    Generator for CIFAR-10 and CIFAR-100
    Args:
        options: Option class
        conf_path: Path to the configuration file
    """
    def __init__(self, options=None, conf_path=None):
        super(Generator, self).__init__()
        self.settings = options or Option(conf_path)
        self.label_emb = nn.Embedding(self.settings.nClasses, self.settings.latent_dim)
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
        """
        Forward pass of the generator
        Args:
            z: Latent vector
            labels: Labels
        """
        gen_input = torch.mul(self.label_emb(labels), z)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        return img


class Generator_imagenet(nn.Module):
    """
    Generator for ImageNet
    Args:
        options: Option class
        conf_path: Path to the configuration file
    """
    def __init__(self, options=None, conf_path=None):
        self.settings = options or Option(conf_path)

        super(Generator_imagenet, self).__init__()

        self.init_size = self.settings.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(self.settings.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks0_0 = CategoricalConditionalBatchNorm2d(1000, 128)

        self.conv_blocks1_0 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_blocks1_1 = CategoricalConditionalBatchNorm2d(1000, 128, 0.8)
        self.conv_blocks1_2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv_blocks2_0 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.conv_blocks2_1 = CategoricalConditionalBatchNorm2d(1000, 64, 0.8)
        self.conv_blocks2_2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv_blocks2_3 = nn.Conv2d(64, self.settings.channels, 3, stride=1, padding=1)
        self.conv_blocks2_4 = nn.Tanh()
        self.conv_blocks2_5 = nn.BatchNorm2d(self.settings.channels, affine=False)

    def forward(self, z, labels):
        """
        Forward pass of the generator
        Args:
            z: Latent vector
            labels: Labels
        """
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


class direct_dataset(Dataset):
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

        if dataset in ["cifar10", "cifar100"]:
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
                transforms.Normalize([0.50705882, 0.48666667, 0.44078431], [0.26745098, 0.25568627, 0.27607843])
            ])

            cifar100_train = datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=self.fewshot_transform)
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
                path = self.settings.generateDataPath +str(i)+".pickle"
                self.logger.info(path)
                with open(path, "rb") as fp:
                    gaussian_data = pickle.load(fp)
                if self.tmp_data is None:
                    self.tmp_data = np.concatenate(gaussian_data, axis=0)
                else:
                    self.tmp_data = np.concatenate((self.tmp_data, np.concatenate(gaussian_data, axis=0)))

                path = self.settings.generateLabelPath + str(i) + ".pickle"
                self.logger.info(path)
                with open(path, "rb") as fp:
                    labels_list = pickle.load(fp)
                if self.tmp_label is None:
                    self.tmp_label = np.concatenate(labels_list, axis=0)
                else:
                    self.tmp_label = np.concatenate((self.tmp_label, np.concatenate(labels_list, axis=0)))

        if self.args.calib_centers:
            temp = "_" + self.settings.model_name if not self.settings.model_name == 'resnet18' else ""
            calib_path = f'../new_generate/data/{self.settings.dataset}{temp}_lbns/{self.settings.model_name}_calib_centers.pickle'
            with open(calib_path, "rb") as fp:
                gaussian_data = pickle.load(fp)
            labels_list = range(self.settings.nClasses)
            self.tmp_data = np.concatenate((self.tmp_data, np.concatenate(gaussian_data, axis=0)[:len(labels_list)]))
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
            file_handler = logging.FileHandler(os.path.join(self.settings.save_path, "train_test.log"))
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
            shutil.copyfile(self.args.conf_path, os.path.join(self.settings.save_path, os.path.basename(self.args.conf_path)))
            shutil.copyfile('./main_direct.py', os.path.join(self.settings.save_path, 'main_direct.py'))
            shutil.copyfile('./trainer_direct.py', os.path.join(self.settings.save_path, 'trainer_direct.py'))
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
        torch.manual_seed(self.settings.manualSeed)
        torch.cuda.manual_seed(self.settings.manualSeed)
        cudnn.benchmark = True

    def _set_dataloader(self):
        """
        Set data loader
        """
        data_loader = DataLoader(dataset=self.settings.dataset,
                                 batch_size=self.settings.batchSize,
                                 data_path=self.settings.dataPath,
                                 n_threads=self.settings.nThreads,
                                 ten_crop=self.settings.tenCrop,
                                 logger=self.logger)

        self.train_loader, self.test_loader = data_loader.getloader()

    def _set_model(self):
        """
        Set model
        """
        if self.settings.dataset in ["cifar100", "cifar10"]:
            if self.settings.model_name == "resnet34_cifar100":
                self.model = resnet34_get_model()
                self.model_teacher = resnet34_get_model()
            else:
                self.model = ptcv_get_model(self.settings.model_name, pretrained=True)
                self.model_teacher = ptcv_get_model(self.settings.model_name, pretrained=True)
            self.generator = Generator(self.settings)
            self.model_teacher.eval()

        elif self.settings.dataset in ["imagenet"]:
            self.model = ptcv_get_model(self.settings.model_name, pretrained=True)
            self.model_teacher = ptcv_get_model(self.settings.model_name, pretrained=True)
            self.generator = Generator_imagenet(self.settings)
            self.model_teacher.eval()

        else:
            assert False, "unsupport data set: " + self.settings.dataset

        self.model_teacher = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model_teacher)
        self.model_teacher = DDP(self.model_teacher.to(self.args.local_rank), device_ids=[self.args.local_rank], output_device=self.args.local_rank, broadcast_buffers=False)
        self.generator = DDP(self.generator.to(self.args.local_rank), device_ids=[self.args.local_rank], output_device=self.args.local_rank, broadcast_buffers=False)

    def _set_trainer(self):
        """
        Set trainer
        """
        lr_master_S = utils.LRPolicy(self.settings.lr_S,
                                     self.settings.nEpochs,
                                     self.settings.lrPolicy_S)
        lr_master_G = utils.LRPolicy(self.settings.lr_G,
                                     self.settings.nEpochs,
                                     self.settings.lrPolicy_G)

        params_dict_S = {
            'step': self.settings.step_S,
            'decay_rate': self.settings.decayRate_S
        }

        params_dict_G = {
            'step': self.settings.step_G,
            'decay_rate': self.settings.decayRate_G
        }

        lr_master_S.set_params(params_dict=params_dict_S)
        lr_master_G.set_params(params_dict=params_dict_G)

        self.trainer = Trainer(
            model=self.model,
            model_teacher=self.model_teacher,
            generator = self.generator,
            train_loader=self.train_loader,
            test_loader=self.test_loader,
            lr_master_S=lr_master_S,
            lr_master_G=lr_master_G,
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

        if type(model) == nn.Conv2d:
            quant_mod = QuantConv2d(weight_bit=weight_bit)
            quant_mod.set_param(model)
            return quant_mod
        elif type(model) == nn.Linear:
            quant_mod = QuantLinear(weight_bit=weight_bit)
            quant_mod.set_param(model)
            return quant_mod
        elif type(model) == nn.ReLU or type(model) == nn.ReLU6:
            return nn.Sequential(*[model, QuantAct(activation_bit=act_bit)])
        elif type(model) == nn.Sequential:
            mods = []
            for n, m in model.named_children():
                mods.append(self.quantize_model(m))
            return nn.Sequential(*mods)
        else:
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
        self.model = DDP(self.model.to(self.args.local_rank), device_ids=[self.args.local_rank], output_device=self.args.local_rank, broadcast_buffers=False)

    def freeze_model(self, model):
        """
        Freeze model
        Args:
            model: Model to freeze
        """
        if type(model) == QuantAct:
            model.fix()
        elif type(model) == nn.Sequential:
            for n, m in model.named_children():
                self.freeze_model(m)
        else:
            for attr in dir(model):
                mod = getattr(model, attr)
                if isinstance(mod, nn.Module) and 'norm' not in attr:
                    self.freeze_model(mod)
            return model

    def unfreeze_model(self, model):
        """
        Unfreeze model
        Args:
            model: Model to unfreeze
        """
        if type(model) == QuantAct:
            model.unfix()
        elif type(model) == nn.Sequential:
            for n, m in model.named_children():
                self.unfreeze_model(m)
        else:
            for attr in dir(model):
                mod = getattr(model, attr)
                if isinstance(mod, nn.Module) and 'norm' not in attr:
                    self.unfreeze_model(mod)
            return model

    def run(self):
        """
        Run the experiment
        """
        best_top1 = 100
        best_top5 = 100
        start_time = time.time()

        dataset = direct_dataset(self.args, self.settings, self.logger, self.settings.dataset)

        direct_dataload = torch.utils.data.DataLoader(dataset,
                                                      batch_size=min(self.settings.batchSize, len(dataset)),
                                                      sampler = DistributedSampler(dataset))
        test_error, test_loss, test5_error = self.trainer.test(epoch=-1)

        try:
            for epoch in range(self.start_epoch, self.settings.nEpochs):
                self.epoch = epoch
                self.start_epoch = 0

                if epoch < 4:
                    self.unfreeze_model(self.model)

                train_error, train_loss, train5_error = self.trainer.train(epoch=epoch, direct_dataload=direct_dataload)

                self.freeze_model(self.model)

                if epoch > 4:
                # if (epoch > self.settings.nEpochs // 5) and (epoch % 5 == 0):
                        test_error, test_loss, test5_error = self.trainer.test(epoch=epoch)
                else:
                    print(f"skip eval for epoch {epoch}")
                    self.logger.info(f"skip eval for epoch {epoch}")
                    continue

                if best_top1 >= test_error:
                    best_top1 = test_error
                    best_top5 = test5_error

                    if self.args.save_model:
                        self.logger.info('Save model! The path is ' + os.path.join(self.settings.save_path, "model.pth"))
                        print('Save model! The path is ' + os.path.join(self.settings.save_path, "model.pth"))
                        torch.save(self.model.state_dict(), os.path.join(self.settings.save_path, "model.pth"))

                self.logger.info("#==>Best Result is: Top1 Error: {:f}, Top5 Error: {:f}".format(best_top1, best_top5))
                self.logger.info("#==>Best Result is: Top1 Accuracy: {:f}, Top5 Accuracy: {:f}".format(100 - best_top1, 100 - best_top5))
                print("#==>Best Result is: Top1 Accuracy: {:f}, Top5 Accuracy: {:f}".format(100 - best_top1, 100 - best_top5))

        except BaseException as e:
            self.logger.error("Training is terminating due to exception: {}".format(str(e)))
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
    parser.add_argument('--calib_centers', type=bool, default=False, metavar='calib_centers') # False로 바꿈.
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
    option.manualSeed = 1

    print(args)
    logger.info(args)

    experiment = ExperimentDesign(option, args, logger)
    experiment.run()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()

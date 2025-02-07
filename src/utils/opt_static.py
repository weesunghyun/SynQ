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

opt_static.py
    - codes for setting the options for training and testing

This code is mainly based on
    - ZeroQ: https://github.com/amirgholami/ZeroQ
    - HAST: https://github.com/lihuantong/HAST
"""
import torch

__all__ = ["NetOption"]

"""
You can run your script with CUDA_VISIBLE_DEVICES=5,6 python your_script.py
or set the environment variable in the script by os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'
to map GPU 5, 6 to device_ids 0, 1, respectively.
"""

class NetOption:
    """
    Options for training and testing
    """

    def __init__(self):
        #  ------------ General options ----------------------------------------
        self.save_path = ""  # log path
        self.data_path = "/home/dataset/"  # path for loading data set
        self.dataset = "cifar10"  # options: imagenet | cifar10 | cifar100 | imagenet100 | mnist
        self.manual_seed = 1  # manually set RNG seed
        self.num_gpu = 1  # number of GPUs to use by default
        self.gpu = 0  # default gpu to use, options: range(num_gpu)

        # ------------- Data options -------------------------------------------
        self.num_threads = 4  # number of data loader threads

        # ------------- Training options ---------------------------------------
        self.test_only = False  # run on validation set only
        self.ten_crop = False  # Ten-crop testing

        # ---------- Optimization options --------------------------------------
        self.num_epochs = 200  # number of total epochs to train
        self.batch_size = 128  # mini-batch size
        self.momentum = 0.9  # momentum
        self.weight_decay = 1e-4  # weight decay 1e-4
        self.opt_type = "SGD"

        self.lr = 0.1  # initial learning rate
        self.lr_policy = "multi_step"  # options: multi_step | linear | exp | fixed
        self.power = 1  # power for learning rate policy (inv)
        self.step = [0.6, 0.8]  # step for linear or exp learning rate policy
        self.endlr = 0.001  # final learning rate, oly for "linear lrpolicy"
        self.decay_rate = 0.1  # lr decay rate

        # ---------- Model options ---------------------------------------------
        self.net_type = "PreResNet"  # options: ResNet | PreResNet | GreedyNet | NIN | LeNet5
        self.experiment_id = "refator-test-01"
        self.depth = 20  # resnet depth: (n-2)%6==0
        self.num_classes = 10  # number of classes in the dataset
        self.wide_factor = 1  # wide factor for wide-resnet

        # ---------- Resume or Retrain options ---------------------------------------------
        self.retrain = None  # path to model to retrain with, load model state_dict only
        self.resume = None  # path to directory containing checkpoint
                            # load state_dicts of model and optimizer
                            # as well as training epoch

        # ---------- Visualization options -------------------------------------
        self.draw_network = True
        self.draw_interval = 30

        self.torch_version = torch.__version__
        torch_version_split = self.torch_version.split("_")
        self.torch_version = torch_version_split[0]
        # check parameters
        # self.paramscheck()

    def paramscheck(self):
        """
        Check the parameters
        """
        if self.torch_version != "0.2.0":
            self.draw_network = False
            print(
                f"|===>draw_network is supported by PyTorch with version: 0.2.0. "
                f"The used version is {self.torch_version}"
                )

        if self.net_type in ["PreResNet", "ResNet"]:
            self.save_path = (
                f"log {self.net_type}{self.depth}_{self.dataset}_"
                f"bs{self.batch_size}_lr{self.lr:.3f}_{self.experiment_id}/"
            )

        else:
            self.save_path = (
                f"log_{self.net_type}_{self.dataset}_"
                f"bs{self.batch_size}_lr{self.lr:.3f}_{self.experiment_id}/"
            )

        if self.dataset in ["cifar10", "mnist"]:
            self.num_classes = 10
        elif self.dataset == "cifar100":
            self.num_classes = 100
        elif self.dataset in ["imagenet", "thi_imgnet"]:
            self.num_classes = 1000
        elif self.dataset == "imagenet100":
            self.num_classes = 100

        if self.depth >= 100:
            self.draw_network = False
            print("|===>draw network with depth over 100 layers, skip this step")

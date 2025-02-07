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

options.py
    - codes for setting the options (reading the configuration file)

This code is mainly based on
    - ZeroQ: https://github.com/amirgholami/ZeroQ
    - HAST: https://github.com/lihuantong/HAST
"""
import os
from datetime import datetime

import pytz
from pyhocon import ConfigFactory

from utils.opt_static import NetOption


class Option(NetOption):
    """
    Options class for the few- or zero-shot quantization
    Args:
        conf_path: str, the path of the configuration file
    """
    def __init__(self, conf_path):
        super().__init__()
        self.conf = ConfigFactory.parse_file(conf_path)
        #  ------------ General options ----------------------------------------
        self.model_name = self.conf['model_name']
        self.generate_data_path = self.conf['generateDataPath']
        self.genera_label_path = self.conf['generateLabelPath']
        self.data_path = self.conf['dataPath']  # path for loading data set
        self.dataset = self.conf['dataset']  # options: imagenet | cifar100

        # ------------- Data options -------------------------------------------
        self.num_threads = self.conf['nThreads']  # number of data loader threads

        # ---------- Optimization options --------------------------------------
        self.num_epochs = self.conf['nEpochs']  # number of total epochs to train
        self.batch_size = self.conf['batchSize']  # mini-batch size
        self.momentum = self.conf['momentum']  # momentum
        self.weight_decay = float(self.conf['weightDecay'])  # weight decay
        self.opt_type = self.conf['opt_type']

        self.lr_s = self.conf['lr_S']  # initial learning rate
        self.lr_policy_s = self.conf['lrPolicy_S']  # [multi_step, linear, exp, const, step]
        self.step_s = self.conf['step_S']  # step for linear or exp learning rate policy
        self.decay_rate_s = self.conf['decayRate_S']  # lr decay rate

        # ---------- Model options ---------------------------------------------
        self.num_classes = self.conf['nClasses']  # number of classes in the dataset

        # ---------- Quantization options ---------------------------------------
        self.qw = self.conf['qw']
        self.qa = self.conf['qa']

        # ----------KD options ---------------------------------------------
        self.temperature = self.conf['temperature']
        self.alpha = self.conf['alpha']

        # ----------Generator options ---------------------------------------------
        self.latent_dim = self.conf['latent_dim']
        self.img_size = self.conf['img_size']
        self.channels = self.conf['channels']

        self.lr_g = self.conf['lr_G']
        self.lr_policy_g = self.conf['lrPolicy_G']  # [multi_step, linear, exp, const, step]
        self.step_g = self.conf['step_G']  # step for linear or exp learning rate policy
        self.decay_rate_g = self.conf['decayRate_G']  # lr decay rate

        self.b1 = self.conf['b1']
        self.b2 = self.conf['b2']

        # ----------- parameter --------------------------------------
        self.lam = 1000
        self.eps = 0.01

    def set_save_path(self):
        """
        Set the save path for the model
        """
        path='log'
        if not os.path.isdir(path):
            os.mkdir(path)
        path = os.path.join(path, self.model_name+"_"+self.dataset)
        if not os.path.isdir(path):
            os.mkdir(path)
        pathname = 'W' + str(self.qw) + 'A' + str(self.qa)
        current_utc_time = datetime.utcnow()
        kst = pytz.timezone('Asia/Seoul')
        current_kst_time = current_utc_time.replace(tzinfo=pytz.utc).astimezone(kst)
        num = current_kst_time.strftime("%m_%d_%H_%M_%S")
        pathname += '_' + str(num)
        path = os.path.join(path, pathname)
        if not os.path.isdir(path):
            os.mkdir(path)
        self.save_path = path

    def paramscheck(self, logger):
        """
        Check the parameters
        Args:
            logger: logger
        """
        logger.info(f"|===>The used PyTorch version is {self.torch_version}")

        if self.dataset in ["cifar10", "mnist"]:
            self.num_classes = 10
        elif self.dataset == "cifar100":
            self.num_classes = 100
        elif self.dataset in ["imagenet", "thi_imgnet"]:
            self.num_classes = 1000
        elif self.dataset == "imagenet100":
            self.num_classes = 100

import os
import pytz
from datetime import datetime
from pyhocon import ConfigFactory

from utils.opt_static import NetOption


class Option(NetOption):
    def __init__(self, conf_path):
        super(Option, self).__init__()
        self.conf = ConfigFactory.parse_file(conf_path)
        #  ------------ General options ----------------------------------------
        self.model_name = self.conf['model_name']
        self.generateDataPath = self.conf['generateDataPath']
        self.generateLabelPath = self.conf['generateLabelPath']
        self.dataPath = self.conf['dataPath']  # path for loading data set
        self.dataset = self.conf['dataset']  # options: imagenet | cifar100

        # ------------- Data options -------------------------------------------
        self.nThreads = self.conf['nThreads']  # number of data loader threads

        # ---------- Optimization options --------------------------------------
        self.nEpochs = self.conf['nEpochs']  # number of total epochs to train
        self.batchSize = self.conf['batchSize']  # mini-batch size
        self.momentum = self.conf['momentum']  # momentum
        self.weightDecay = float(self.conf['weightDecay'])  # weight decay
        self.opt_type = self.conf['opt_type']

        self.lr_S = self.conf['lr_S']  # initial learning rate
        self.lrPolicy_S = self.conf['lrPolicy_S']  # options: multi_step | linear | exp | const | step
        self.step_S = self.conf['step_S']  # step for linear or exp learning rate policy
        self.decayRate_S = self.conf['decayRate_S']  # lr decay rate

        # ---------- Model options ---------------------------------------------
        self.nClasses = self.conf['nClasses']  # number of classes in the dataset

# ---------- Quantization options ---------------------------------------------
        self.qw = self.conf['qw']
        self.qa = self.conf['qa']

        # ----------KD options ---------------------------------------------
        self.temperature = self.conf['temperature']
        self.alpha = self.conf['alpha']

        # ----------Generator options ---------------------------------------------
        self.latent_dim = self.conf['latent_dim']
        self.img_size = self.conf['img_size']
        self.channels = self.conf['channels']

        self.lr_G = self.conf['lr_G']
        self.lrPolicy_G = self.conf['lrPolicy_G']  # options: multi_step | linear | exp | const | step
        self.step_G = self.conf['step_G']  # step for linear or exp learning rate policy
        self.decayRate_G = self.conf['decayRate_G']  # lr decay rate

        self.b1 = self.conf['b1']
        self.b2 = self.conf['b2']

        # ----------- parameter --------------------------------------
        self.lam = 1000
        self.eps = 0.01

    def set_save_path(self):
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
        logger.info("|===>The used PyTorch version is {}".format(
                self.torch_version))

        if self.dataset in ["cifar10", "mnist"]:
            self.nClasses = 10
        elif self.dataset == "cifar100":
            self.nClasses = 100
        elif self.dataset == "imagenet" or "thi_imgnet":
            self.nClasses = 1000
        elif self.dataset == "imagenet100":
            self.nClasses = 100

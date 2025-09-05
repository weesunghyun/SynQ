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

dataloader.py
    - codes for loading the dataset

This code is mainly based on
    - ZeroQ: https://github.com/amirgholami/ZeroQ
    - HAST: https://github.com/lihuantong/HAST
"""
import os

import torch
from torch.utils import data

from torchvision import transforms
from torchvision import datasets as dsets


# __all__ = ["DataLoader", "PartDataLoader"]
__all__ = ["DataLoader"]

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

class ImageLoader(data.Dataset):
    """Image Loader"""
    def __init__(self, dataset_dir, transform=None, target_transform=None):
        """
        Initialize Image Loader
        Args:
            dataset_dir: Directory containing the dataset
            transform: Transformation to apply to the input
            target_transform: Transformation to apply to the target
        """
        class_list = os.listdir(dataset_dir)
        datasets = []
        for cla in class_list:
            cla_path = os.path.join(dataset_dir, cla)
            files = os.listdir(cla_path)
            for file_name in files:
                file_path = os.path.join(cla_path, file_name)
                if os.path.isfile(file_path):
                    datasets.append((file_path, [float(cla)]))

        self.dataset_dir = dataset_dir
        self.datasets = datasets
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """get the item corresponding to the index"""		
        file_path, label = self.datasets[index]
        noise = torch.load(file_path, map_location=torch.device('cpu'))
        return noise, torch.Tensor(label)

    def __len__(self):
        """the length of the dataset"""
        return len(self.datasets)


class DataLoader:
    """Data Loader"""
    def __init__(self, dataset, batch_size, n_threads=4,
	             ten_crop=False, data_path='/home/dataset/', logger=None):
        """
        Initialize Data Loader
        Args:
            dataset: Dataset to use
            batch_size: Batch size
            n_threads: Number of threads to use
            ten_crop: Use ten crop for testing
            data_path: Path to the dataset
            logger: Logger
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_threads = n_threads
        self.ten_crop = ten_crop
        self.data_path = data_path
        self.logger = logger
        self.dataset_root = data_path

        if self.dataset in ["cifar100", "cifar10"]:
            self.train_loader, self.test_loader = self.cifar(dataset=self.dataset)

        elif self.dataset in ["imagenet"]:
            self.train_loader, self.test_loader = self.imagenet(dataset=self.dataset)

        elif self.dataset in medmnist_dataset:
            import medmnist
            from medmnist import INFO, Evaluator

            info = INFO[self.dataset]
            download = True

            DataClass = getattr(medmnist, info['python_class'])

            # preprocessing
            data_transform = transforms.Compose([
                transforms.ToTensor(),
				transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),  # Convert grayscale to RGB
                transforms.Normalize(mean=[.5], std=[.5])
            ])

            # load the data
            # train_dataset = DataClass(split='train', transform=data_transform, download=download, size=224)
            # test_dataset = DataClass(split='test', transform=data_transform, download=download, size=224)
            test_dataset = DataClass(split='test', transform=data_transform, download=download, size=28)

            # encapsulate data into dataloader form
            # self.train_loader = data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
            self.train_loader = None
            self.test_loader = data.DataLoader(dataset=test_dataset, batch_size=500, shuffle=False)
        else:
            assert False, "invalid data set"

    def getloader(self):
        """Get the data loader"""
        return self.train_loader, self.test_loader

    def imagenet(self, dataset="imagenet"):
        """Get the ImageNet data loader"""
        testdir = os.path.join(self.data_path, "val")
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

        test_loader = torch.utils.data.DataLoader(
            dsets.ImageFolder(testdir, test_transform),
            batch_size=500,
            shuffle=False,
            num_workers=self.n_threads,
            pin_memory=False)

        return None, test_loader

    def cifar(self, dataset="cifar100"):
        """Get the CIFAR data loader"""
        if dataset == "cifar10":
            norm_mean = [0.49139968, 0.48215827, 0.44653124]
            norm_std = [0.24703233, 0.24348505, 0.26158768]
        elif dataset == "cifar100":
            norm_mean = [0.50705882, 0.48666667, 0.44078431]
            norm_std = [0.26745098, 0.25568627, 0.27607843]

        else:
            assert False, "Invalid cifar dataset"

        test_data_root = self.dataset_root
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)])

        if self.dataset == "cifar10":
            test_dataset = dsets.CIFAR10(root=test_data_root,
                                         train=False,
                                         transform=test_transform)

        elif self.dataset == "cifar100":
            test_dataset = dsets.CIFAR100(root=test_data_root,
                                          train=False,
                                          transform=test_transform,
                                          download=True)

        else:
            assert False, "invalid data set"

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.batch_size, # 10000
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=self.n_threads)

        return None, test_loader

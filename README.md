# Accurate Few-shot Quantization by Synthesis-aware Fine-tuning

This project is a PyTorch implementation of **"Accurate Few-shot Quantization by Synthesis-aware Fine-tuning"**.
The paper proposes **SaFT** (Synthesis-aware Fine-tuning), an accurate Few-shot Quantization (FSQ) method.

![Overall Architecture of SaFT](./images/saft.jpg)


## Prerequisites

Our implementation is based on PyTorch, TorchVision, and PyTorchCV libraries.

- Python 3.9.12
- PyTorch 1.13.1
- PyTorchCV 0.0.67

We include `requirements.txt`, which contains all the packages used for the experiment. 
We also contain a working code for PyTorchCV library within `src/pytorchcv` to mitigate the dependency issue.
We checked the dependency using a workstation with Intel Xeon Silver 4214 and RTX 3090, where its CUDA version is 11.6.
Install the required packages with the following code:

```shell
pip install -r requirements.txt
```

### Usage
For usage, first generate the synthetic dataset with the code under `src/data_generate/`.
We include `run_generate_cifar100.sh`, which generates the synthetic dataset using a ResNet-34 model pre-trained on the CIFAR-100 dataset.

```shell
cd src/data_generate
bash run_generate_cifar100.sh
```

Fine-tune the quantized model by executing `src/main_direct.py`.
We include `run_cifar100_6bit.sh`, for the 6bit Few-shot Quantization (FSQ) for ResNet-34 model pre-trained on the CIFAR-100 dataset.

```
cd src/
bash run_cifar100_6bit.sh 0
```

To run with different settings, modify the config files under `src/config/` or the arguments passed into `src/data_generate/generate_data.py` and `src/main_direct.py`.

### Code Description

This repository is written based on the codes from **ZeroQ** (CVPR '20) \[[Github](https://github.com/amirgholami/ZeroQ)\], **HAST** (CVPR '23) \[[Github](https://github.com/lihuantong/HAST)\], and a PyTorch implementation of Grad-CAM and Grad-CAM++ \[[Github](https://github.com/1Konny/gradcam_plus_plus-pytorch)\].
Here is an overview of our codes.

``` Unicode
SaFT/
├── images/
│   └── saft.jpg                     # the overall architecture
├── src/
│   ├── config/                      # configurations for fine-tuning      
│   ├── data_generate/               # synthetic dataset generation
│   ├── pytorchcv/                   # pytorchcv library
│   ├── quantization_utils/          # utils for quantization
│   ├── utils/                       # frequently used code snippets
│   ├── conditional_batchnorm.py     # conditional batch normalization layer
│   ├── dataloader.py                # dataloader for test data
│   ├── gradcam.py                   # gradcam and gradcam++ code
│   ├── main_direct.py               # fine-tuning code
│   ├── options.py                   # storing code for configurations
│   ├── run_cifar_100_6bit.sh        # codes for running SaFT 
│   ├── trainer_direct.py            # trainer code
│   └── unit_test.py                 # unit-test code
├── .gitignore                       # gitignore file
├── LICENSE                          # license for the use of the code
├── README.md                        # readme file
└── requirements.txt                 # required packages
```

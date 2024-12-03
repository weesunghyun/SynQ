"""
[SW Starlab]
Few-shot quantization with SaFT (Synthesis-aware Fine-Tuning)

Author
    - Minjun Kim (minjun.kim@snu.ac.kr), Seoul National University
    - U Kang (ukang@snu.ac.kr), Seoul National University

Version : 1.0

Date : Dec 3rd, 2023

Main Contact: Minjun Kim

This software is free of charge under research purposes.

For commercial purposes, please contact the authors.

model_transform.py
    - codes for model transformation
        e.g., data parallel, model to list, list to sequential, model to state_dict

This code is mainly based on
    - ZeroQ: https://github.com/amirgholami/ZeroQ
    - HAST: https://github.com/lihuantong/HAST
"""
import torch
from torch import nn

__all__ = ["data_parallel", "model2list",
           "list2sequential", "model2state_dict"]


def data_parallel(model, ngpus, gpu0=0):
    """
    assign model to multi-gpu mode
    Args:
        model: target model
        ngpus: number of gpus to use
        gpu0: id of the master gpu
    Returns:
        model: model, type is Module or Sequantial or DataParallel
    """
    if ngpus == 0:
        assert False, "only support gpu mode"
    gpu_list = list(range(gpu0, gpu0 + ngpus))
    assert torch.cuda.device_count() >= gpu0 + ngpus, "Invalid Number of GPUs"
    if isinstance(model, list):
        for i, model_i in enumerate(model):
            if ngpus >= 2:
                if not isinstance(model_i, nn.DataParallel):
                    model[i] = torch.nn.DataParallel(model_i, gpu_list).cuda()
            else:
                model[i] = model_i.cuda()

    else:
        if ngpus >= 2:
            if not isinstance(model, nn.DataParallel):
                model = torch.nn.DataParallel(model, gpu_list).cuda()
        else:
            model = model.cuda()

    return model


def model2list(model):
    """
    convert model to list type
    Args:
        model: should be type of list or nn.DataParallel or nn.Sequential
    Return:
        model: no return params    
    """
    if isinstance(model, nn.DataParallel):
        model = list(model.module)
    elif isinstance(model, nn.Sequential):
        model = list(model)

    return model


def list2sequential(model):
    """
    Convert model to nn.Sequential
    Args:
        model: should be type of list
    Return:
        model: nn.Sequential
    """
    if isinstance(model, list):
        model = nn.Sequential(*model)

    return model


def model2state_dict(file_path):
    """
    Convert model to state_dict
    Args:
        file_path: model file path
    """

    model = torch.load(file_path)
    if model['model'] is not None:
        model_state_dict = model['model'].state_dict()
        torch.save(model_state_dict, file_path.replace(
            '.pth', 'state_dict.pth'))

    else:
        print((type(model)))
        print(model)
        print("skip")

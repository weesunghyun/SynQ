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

distill_data.py
    - codes for generating distilled data

This code is mainly based on
    - ZeroQ: https://github.com/amirgholami/ZeroQ
    - HAST: https://github.com/lihuantong/HAST
"""
import gc
import os
import sys
import time
import pickle
import random

import numpy as np

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from torchvision import transforms


def check_path(model_path):
    """
    Check if the directory exists, if not create it.
    Args:
        model_path: path to the model
    """
    directory = os.path.dirname(model_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_calib_centers(args, teacher_model, beta_ce = 5):
    """
    Generate calibration centers for the teacher model.
    Args:
        args: arguments
        teacher_model: teacher model
        beta_ce: beta for cross entropy loss
    Returns:
        refined_gaussian: refined gaussian data
    """
    calib_path = os.path.join(args.save_path_head, args.model + "_calib_centers" + ".pickle")
    if not os.path.exists(calib_path):
        model_name = args.model

        if model_name in ['resnet20_cifar10','resnet20_cifar100', 'resnet34_cifar100']:
            shape = (args.batch_size, 3, 32, 32)
        else:
            shape = (args.batch_size, 3, 224, 224)

        teacher_model = teacher_model.cuda()
        teacher_model = teacher_model.eval()

        # Determine number of classes from model output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, *shape[1:]).cuda()
            dummy_output = teacher_model(dummy_input)
            num_classes = dummy_output.shape[1]
            print(f"Model output dimension: {num_classes} classes")

        refined_gaussian = []

        ce_loss = nn.CrossEntropyLoss(reduction='none').cuda()
        mse_loss = nn.MSELoss().cuda()

        mean_list = []
        var_list = []
        teacher_running_mean = []
        teacher_running_var = []

        def hook_fn_forward(module, _input, output):
            _input = _input[0]
            mean = _input.mean([0, 2, 3])
            var = _input.var([0, 2, 3], unbiased=False)

            mean_list.append(mean)
            var_list.append(var)
            teacher_running_mean.append(module.running_mean)
            teacher_running_var.append(module.running_var)

        for _, m in teacher_model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.register_forward_hook(hook_fn_forward)

        total_time = time.time()

        for i in range(num_classes//args.batch_size + 1):
            gaussian_data = torch.randn(shape).cuda()
            gaussian_data.requires_grad = True
            # optimizer = optim.Adam([gaussian_data], lr=0.5)
            optimizer = optim.Adam([gaussian_data], lr=0.1)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                # min_lr=0.05,
                                                                min_lr=1e-4,
                                                                verbose=False,
                                                                patience=50)

            if (i + 1) * args.batch_size <= num_classes:
                labels = torch.tensor([i * args.batch_size + j for j in range(args.batch_size)],
                                      dtype=torch.long, device='cuda')
            else:
                labels = torch.tensor([i * args.batch_size + j \
                                       for j in range(num_classes % args.batch_size)],
                                      dtype=torch.long, device='cuda')

            if len(labels) < args.batch_size:
                labels = torch.nn.functional.pad(labels, (0, args.batch_size - len(labels)))

            batch_time = time.time()
            for it in range(1500):
                new_gaussian_data = []
                for j, jth_data in enumerate(gaussian_data):
                    new_gaussian_data.append(jth_data)
                new_gaussian_data = torch.stack(new_gaussian_data).cuda()

                mean_list.clear()
                var_list.clear()
                teacher_running_mean.clear()
                teacher_running_var.clear()

                output = teacher_model(new_gaussian_data)
                loss_target = beta_ce * (ce_loss(output, labels)).mean()

                mean_loss = torch.zeros(1).cuda()
                var_loss = torch.zeros(1).cuda()
                for n, nth_mean in enumerate(mean_list):
                    if n < (len(mean_list)+2) // 2 - 2:
                        mean_loss += 0.2 * mse_loss(nth_mean,
                                                    teacher_running_mean[n].detach())
                        var_loss += 0.2 * mse_loss(var_list[n],
                                                   teacher_running_var[n].detach())
                    else:
                        mean_loss += 1.1 * mse_loss(nth_mean,
                                                    teacher_running_mean[n].detach())
                        var_loss += 1.1 * mse_loss(var_list[n],
                                                    teacher_running_var[n].detach())

                mean_loss = mean_loss / len(mean_list)
                var_loss = var_loss / len(mean_list)

                total_loss = mean_loss + var_loss + loss_target

                print(i, it, 'lr', optimizer.state_dict()['param_groups'][0]['lr'],
                    'mean_loss', mean_loss.item(), 'var_loss',
                    var_loss.item(), 'loss_target', loss_target.item())

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(gaussian_data, max_norm=1.0)
                optimizer.step()
                scheduler.step(total_loss.item())

            with torch.no_grad():
                output = teacher_model(gaussian_data.detach())
                d_acc = np.mean(np.argmax(output.data.cpu().numpy(), axis=1) == labels)
                print('d_acc', d_acc)

            refined_gaussian.append(gaussian_data.detach().cpu().numpy())

            print(f"Time for {i} batch for {it} iters: {time.time()-batch_time:.2f} sec.")

            gaussian_data = gaussian_data.cpu()
            del gaussian_data
            del optimizer
            del scheduler
            del labels
            torch.cuda.empty_cache()

        print(f"Total time for {num_classes//args.batch_size} "
              f"batches: {time.time()-total_time:.2f} sec.")
        check_path(calib_path)
        with open(calib_path, "wb") as fp:
            pickle.dump(refined_gaussian, fp, protocol=pickle.HIGHEST_PROTOCOL)
        del refined_gaussian

    with open(calib_path, 'rb') as f:
        refined_gaussian = pickle.load(f)
    return refined_gaussian

class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        """
        Forward pass for the LabelSmoothing module.
        Args:
            x: input tensor
            target: target tensor
        Returns:
            loss: loss value
        """
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class OutputHook:
    """
        Forward_hook used to get the output of the intermediate layer. 
    """
    def __init__(self):
        self.outputs = None

    def hook(self, m, i, output):
        """
        Output hook function.
        Args:
            m: module
            i: input
            output: output
        """
        self.outputs = output

    def clear(self):
        """
        Clear the output.
        """
        self.outputs = None


class DistillData:
    """
    Construct the distilled data.
    Args:
        args: arguments
    """
    def __init__(self, args):
        self.mean_list = []
        self.var_list = []
        self.teacher_running_mean = []
        self.teacher_running_var = []
        self.args = args
        if args.lbns:
            self.calib_centers = args.calib_centers
            self.calib_running_mean = []
            self.calib_running_var = []
            self.calib_data_means = []
            self.calib_data_vars = []

    def hook_fn_forward(self, module, _input, _):
        """
        Forward hook function for the batch normalization layer.
        Args:
            module: module
            _input: input
            output: output
        """
        _input = _input[0]
        mean = _input.mean([0, 2, 3])
        var = _input.var([0, 2, 3], unbiased=False)

        self.mean_list.append(mean)
        self.var_list.append(var)
        self.teacher_running_mean.append(module.running_mean)
        self.teacher_running_var.append(module.running_var)

    def get_distil_data(self, model_name="resnet18", teacher_model=None, batch_size=256,
                      num_batch=1, group=1, aug_margin=0.4, beta=1.0, gamma=0, save_path_head=""):
        """
        Generate the distilled data.
        Args:
            model_name: model name
            teacher_model: teacher model
            batch_size: batch size
            num_batch: number of batches
            group: group
            aug_margin: augmentation margin
            beta: beta
            gamma: gamma
            save_path_head: save path head
        """

        data_path = os.path.join(save_path_head, model_name+"_refined_gaussian_hardsample_" \
                    + "beta"+ str(beta) +"_gamma" + str(gamma) + "_group" + str(group) + ".pickle")
        label_path = os.path.join(save_path_head, model_name+"_labels_hardsample_" \
                    + "beta"+ str(beta) +"_gamma" + str(gamma) + "_group" + str(group) + ".pickle")

        print(data_path, label_path)

        check_path(data_path)
        check_path(label_path)

        if model_name in ['resnet20_cifar10','resnet20_cifar100', 'resnet34_cifar100']:
            shape = (batch_size, 3, 32, 32)
        else:
            shape = (batch_size, 3, 224, 224)

        teacher_model = teacher_model.cuda()
        teacher_model = teacher_model.eval()

        # Determine number of classes from model output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, *shape[1:]).cuda()
            dummy_output = teacher_model(dummy_input)
            self.num_classes = dummy_output.shape[1]
            print(f"Model output dimension: {self.num_classes} classes")

        refined_gaussian = []
        labels_list = []

        ce_loss = nn.CrossEntropyLoss(reduction='none').cuda()
        mse_loss = nn.MSELoss().cuda()

        for _, m in teacher_model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.register_forward_hook(self.hook_fn_forward)

                total_calib_means = []
        total_calib_vars = []
        sum_means_squared = []
        num_centers = self.num_classes // batch_size + (self.num_classes % batch_size != 0)
        total_samples = 0

        if self.args.lbns and not self.calib_data_means:
            for j in range(num_centers):
                self.mean_list.clear()
                self.var_list.clear()
                with torch.no_grad():
                    calib_center = torch.tensor(self.calib_centers[j]).to('cuda', non_blocking=True)
                    teacher_model(calib_center)
                    num_samples = calib_center.size(0)

                if j == 0:
                    total_calib_means = [x.clone().cpu() * num_samples for x in self.mean_list]
                    total_calib_vars = [x.clone().cpu() for x in self.var_list]
                    sum_means_squared = [x.clone().cpu()**2 * num_samples for x in self.mean_list]
                else:
                    for idx, (mean, var) in enumerate(zip(self.mean_list, self.var_list)):
                        total_calib_means[idx] += mean.cpu() * num_samples
                        total_calib_vars[idx] += var.cpu() * num_samples
                        sum_means_squared[idx] += (mean.cpu()**2) * num_samples

                total_samples += num_samples
                del calib_center
                torch.cuda.empty_cache()
                gc.collect()

            global_means = [total_mean / total_samples for total_mean in total_calib_means]
            global_vars = []

            for idx in range(len(total_calib_means)):
                mean_of_squares = sum_means_squared[idx] / total_samples
                square_of_mean = global_means[idx]**2
                var_between = mean_of_squares - square_of_mean
                var_within = total_calib_vars[idx] / total_samples
                global_vars.append(var_within + var_between)

            self.calib_running_mean = global_means
            self.calib_running_var = global_vars

            print(len(self.calib_running_mean))

        total_time = time.time()

        if self.args.lbns:
            assert self.calib_centers

        assert self.num_classes

        for i in range(self.args.num_data//batch_size):

            if model_name in ['resnet20_cifar10', 'resnet20_cifar100', 'resnet34_cifar100']:
                rrc = transforms.RandomResizedCrop(size=32,scale=(aug_margin, 1.0))
            else:
                rrc = transforms.RandomResizedCrop(size=224,scale=(aug_margin, 1.0))
            rhf = transforms.RandomHorizontalFlip()

            gaussian_data = torch.randn(shape).cuda()
            gaussian_data.requires_grad = True
            # optimizer = optim.Adam([gaussian_data], lr=0.5)
            optimizer = optim.Adam([gaussian_data], lr=0.05)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                # min_lr=0.05,
                                                                min_lr=0.005,
                                                                verbose=False,
                                                                patience=50)

            labels = torch.randint(0, self.num_classes, (len(gaussian_data),)).cuda()
            # labels_mask = F.one_hot(labels, num_classes=self.num_classes).float()
            gt = labels.data.cpu().numpy()

            batch_time = time.time()
            for it in range(500*2):
                if model_name in ['resnet20_cifar10', 'resnet20_cifar100', 'resnet34_cifar100']:
                    new_gaussian_data = []
                    for j, jth_data in enumerate(gaussian_data):
                        new_gaussian_data.append(jth_data)
                    new_gaussian_data = torch.stack(new_gaussian_data).cuda()
                else:
                    if random.random() < 0.5:
                        new_gaussian_data = []
                        for j, jth_data in enumerate(gaussian_data):
                            new_gaussian_data.append(rhf(rrc(jth_data)))
                        new_gaussian_data = torch.stack(new_gaussian_data).cuda()
                    else:
                        new_gaussian_data = []
                        for j, jth_data in enumerate(gaussian_data):
                            new_gaussian_data.append(jth_data)
                        new_gaussian_data = torch.stack(new_gaussian_data).cuda()

                self.mean_list.clear()
                self.var_list.clear()
                self.teacher_running_mean.clear()
                self.teacher_running_var.clear()

                output = teacher_model(new_gaussian_data)
                d_acc = np.mean(np.argmax(output.data.cpu().numpy(), axis=1) == gt)
                a = F.softmax(output, dim=1)
                mask = torch.zeros_like(a)
                b=labels.unsqueeze(1)
                mask=mask.scatter_(1,b,torch.ones_like(b).float())
                p=a[mask.bool()]

                # loss_target = beta * F.kl_div(input=F.log_softmax(output, dim=1),
                #                               target=labels_mask.to(output.device),
                #                               reduction='batchmean')
                loss_target = beta * ((1-p).pow(gamma) * ce_loss(output, labels)).mean()

                mean_loss = torch.zeros(1).cuda()
                var_loss = torch.zeros(1).cuda()
                for n, nth_mean in enumerate(self.mean_list):
                    mean_loss += mse_loss(nth_mean.cpu(),
                                          self.teacher_running_mean[n].detach().cpu())
                    var_loss += mse_loss(self.var_list[n].cpu(),
                                        self.teacher_running_var[n].detach().cpu())

                if self.args.lbns:
                    print(f"Length of mean list: {len(self.mean_list)}")
                    lbns_loss = torch.zeros(1).cuda()
                    for n, nth_mean in enumerate(self.mean_list):
                        if n >= (len(self.mean_list)+2) // 2 - 2 :
                            lmean_loss = mse_loss(nth_mean.cuda(),
                                                  self.calib_running_mean[n].detach().cuda())
                            lvar_loss = mse_loss(self.var_list[n].cuda(),
                                                 self.calib_running_var[n].detach().cuda())
                            lbns_loss += lmean_loss + lvar_loss
                    lbns_loss = lbns_loss / (len(self.mean_list) * len(labels))

                mean_loss = mean_loss / len(self.mean_list)
                var_loss = var_loss / len(self.mean_list)

                if self.args.lbns:
                    total_loss = 0.4 * (mean_loss + var_loss) + loss_target + 0.02 * lbns_loss
                else:
                    total_loss = mean_loss + var_loss + loss_target

                print(f"Batch: {i}, Iter: {it}, LR: {optimizer.state_dict()['param_groups'][0]['lr']:.4f}, "
                      f"Mean Loss: {mean_loss.item():.4f}, Var Loss: {var_loss.item():.4f}, "
                      f"Target Loss: {loss_target.item():.4f}")

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(gaussian_data, max_norm=1.0)
                optimizer.step()
                scheduler.step(total_loss.item())


            with torch.no_grad():
                output = teacher_model(gaussian_data.detach())
                d_acc = np.mean(np.argmax(output.data.cpu().numpy(), axis=1) == gt)
                print('d_acc', d_acc)

            refined_gaussian.append(gaussian_data.detach().cpu().numpy())

            labels_list.append(labels.detach().cpu().numpy())

            print(f"Time for {i} batch for {it} iters: {time.time()-batch_time:.2f} sec.")

            gaussian_data = gaussian_data.cpu()
            del gaussian_data
            del optimizer
            del scheduler
            del labels
            torch.cuda.empty_cache()

        print(f"Total time for {self.args.num_data//batch_size} "
              f"batches: {time.time()-total_time:.2f} sec.")
        with open(data_path, "wb") as fp:  # Pickling
            pickle.dump(refined_gaussian, fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open(label_path, "wb") as fp:  # Pickling
            pickle.dump(labels_list, fp, protocol=pickle.HIGHEST_PROTOCOL)
        sys.exit()

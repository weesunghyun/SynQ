import os
import time
from builtins import isinstance

import numpy as np

import torch
from torch import nn
import torch.autograd
import torch.distributed as dist
from torch.autograd import Variable
from torch.nn import functional as F

import utils

from gradcam import GradCAM, GradCAMpp

from pytorchcv.models.resnet import ResUnit
from pytorchcv.models.mobilenet import DwsConvBlock
from pytorchcv.models.mobilenetv2 import LinearBottleneck

from quantization_utils.quant_modules import *

__all__ = ["Trainer"]

class Trainer(object):
    """
    Trainer class
    """
    def __init__(self, model, model_teacher, generator, lr_master_S, lr_master_G,
                 train_loader, test_loader, settings, args, logger, tensorboard_logger=None,
                 opt_type="SGD", optimizer_state=None, run_count=0):
        """
        Initialize the trainer class
        Args:
            model: student model
            model_teacher: teacher model
            generator: generator model
            lr_master_S: learning rate scheduler for student model
            lr_master_G: learning rate scheduler for generator model
            train_loader: training data loader
            test_loader: testing data loader
            settings: settings
            args: arguments
            logger: logger
            tensorboard_logger: tensorboard logger
            opt_type: optimizer type
            optimizer_state: optimizer state
            run_count: run count
        """

        self.settings = settings
        self.args = args
        self.model = model
        self.model_teacher = model_teacher
        self.generator = generator
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.tensorboard_logger = tensorboard_logger
        self.criterion = nn.CrossEntropyLoss().to(self.args.local_rank)
        self.bce_logits = nn.BCEWithLogitsLoss().to(self.args.local_rank)
        self.MSE_loss = nn.MSELoss().to(self.args.local_rank)
        self.KLloss = nn.KLDivLoss(reduction='batchmean').to(self.args.local_rank)
        self.lr_master_S = lr_master_S
        self.lr_master_G = lr_master_G
        self.opt_type = opt_type

        if opt_type == "SGD":
            self.optimizer_S = torch.optim.SGD(
                params=self.model.parameters(),
                lr=self.lr_master_S.lr,
                momentum=self.settings.momentum,
                weight_decay=self.settings.weightDecay,
                nesterov=True,
            )
        elif opt_type == "RMSProp":
            self.optimizer_S = torch.optim.RMSprop(
                params=self.model.parameters(),
                lr=self.lr_master_S.lr,
                eps=1.0,
                weight_decay=self.settings.weightDecay,
                momentum=self.settings.momentum,
                alpha=self.settings.momentum
            )
        elif opt_type == "Adam":
            self.optimizer_S = torch.optim.Adam(
                params=self.model.parameters(),
                lr=self.lr_master_S.lr,
                eps=1e-5,
                weight_decay=self.settings.weightDecay
            )
        else:
            assert False, "invalid type: %d" % opt_type

        if optimizer_state is not None:
            self.optimizer_S.load_state_dict(optimizer_state)\

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.settings.lr_G,
                                            betas=(self.settings.b1, self.settings.b2))

        self.logger = logger
        self.run_count = run_count
        self.scalar_info = {}
        self.mean_list = []
        self.var_list = []
        self.teacher_running_mean = []
        self.teacher_running_var = []
        self.save_BN_mean = []
        self.save_BN_var = []
        self.activation_teacher = []
        self.activation = []
        self.handle_list = []


        if self.settings.dataset == "imagenet":
            self.teacher_dict = dict(type='resnet',
                                     arch=model_teacher,
                                     layer_name='stage4',
                                     input_size=(224, 224))
        else:
            self.teacher_dict = dict(type='resnet',
                                     arch=model_teacher,
                                     layer_name='stage4',
                                     input_size=(32, 32))

        # if self.args.cam_type == 'cam':
        #     self.cam_teacher = CAM(self.teacher_dict, True)
        if self.args.cam_type == 'gradcam':
            self.cam_teacher = GradCAM(self.teacher_dict, True)
        elif self.args.cam_type == 'gradcampp':
            self.cam_teacher = GradCAMpp(self.teacher_dict, True)
        self.update_cam()

    def apply_filters(self, images, d_zero=80):
        """
        Apply low-pass filter to images
        Args:
            images: target images
            d_zero: ...??? # TODO
        """
        _, _, height, width = images.shape
        center_x, center_y = width // 2, height // 2

        Y, X = torch.meshgrid(torch.arange(height, device=images.device),
                              torch.arange(width, device=images.device), indexing='ij')
        D_square = (X - center_x) ** 2 + (Y - center_y) ** 2

        H = torch.exp(-D_square / (2 * (d_zero) ** 2))
        H = H[None, None, :, :]

        fft_images = torch.fft.fft2(images, dim=(-2, -1))
        fft_images = torch.fft.fftshift(fft_images, dim=(-2, -1))
        fft_images *= H
        fft_images = torch.fft.ifftshift(fft_images, dim=(-2, -1))

        filtered_images = torch.fft.ifft2(fft_images, dim=(-2, -1)).real
        return filtered_images

    def update_lr(self, epoch):
        """
        Update learning rate
        Args:
            epoch: current epoch
        """
        lr_S = self.lr_master_S.get_lr(epoch)
        lr_G = self.lr_master_G.get_lr(epoch)

        # update learning rate of model optimizer
        for param_group in self.optimizer_S.param_groups:
            param_group['lr'] = lr_S
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr_G

    def loss_fn_kd(self, output, labels, teacher_outputs):
        """
        Compute loss function for knowledge distillation
        Args:
            output: student output
            labels: labels
            teacher_outputs: teacher output
        """
        criterion_d = nn.CrossEntropyLoss().cuda()
        alpha = self.settings.alpha
        T = self.settings.temperature
        a = F.log_softmax(output / T, dim=1) + 1e-7
        b = F.softmax(teacher_outputs / T, dim=1)
        c = (alpha * T * T)
        d = criterion_d(output, labels)
        KD_loss = self.KLloss(a, b) * c
        return KD_loss, d

    def loss_fa(self):
        """
        Compute feature alignment loss
        """
        fa = torch.zeros(1).to(self.args.local_rank)
        for l in range(len(self.activation)):
            fa += (self.activation[l] - self.activation_teacher[l]).pow(2).mean()
        fa = self.settings.lam * fa
        return fa

    def update_cam(self):
        """
        Update class activation map (CAM)
        """
        if self.settings.dataset == "imagenet":
            self.student_dict = dict(type='resnet',
                                     arch=self.model,
                                     layer_name='stage4',
                                     input_size=(224, 224))
        else:
            self.student_dict = dict(type='resnet',
                                     arch=self.model,
                                     layer_name='stage4',
                                     input_size=(32, 32))

        if self.args.cam_type == 'gradcam':
            self.cam_student = GradCAM(self.student_dict, True)
        elif self.args.cam_type == 'cam':
            self.cam_student = CAM(self.teacher_dict, True)
        elif self.args.cam_type == 'gradcampp':
            self.cam_student = GradCAMpp(self.student_dict, True)

    def loss_cam(self, images):
        """
        Compute class activation map (CAM) loss
        Args:
            images: input images
        """
        assert not torch.isnan(images).any()
        mask_teacher, _ = self.cam_teacher(images, retain_graph=True)
        mask_student, _ = self.cam_student(images, retain_graph=True)

        assert not torch.isnan(mask_teacher).any(), "NaN values in mask_teacher"
        assert not torch.isnan(mask_student).any(), "NaN values in mask_student"
        loss = F.mse_loss(mask_teacher, mask_student)
        if torch.isnan(loss):
            print("NaN detected!")
            print("mask_teacher:", mask_teacher)
            print("mask_student:", mask_student)
            raise ValueError("NaN detected!")

        return F.mse_loss(mask_teacher, mask_student)

    def forward(self, images, teacher_outputs, labels=None, mask_selce=None):
        """
        Forward pass
        Args:
            images: input images
            teacher_outputs: teacher outputs
            labels: labels
            mask_selce: mask for selective cross-entropy  # TODO: Check this descriptsion
        """
        output = self.model(images)
        loss_KL, loss_CE = self.loss_fn_kd(output, labels, teacher_outputs)
        if mask_selce is not None:
            mask_selce = mask_selce.bool()
            masked_output = torch.where(mask_selce.unsqueeze(-1),
                                        output, torch.tensor(0.).to(output.device))
            masked_labels = torch.where(mask_selce, labels, torch.tensor(0).to(labels.device))
            loss_CE = self.criterion(masked_output, masked_labels)
        loss_FA = self.loss_fa()

        return output, loss_KL, loss_FA, loss_CE

    def backward_G(self, loss_G):
        """
        Backward pass for generator
        Args:
            loss_G: loss for generator
        """
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()

    def backward_S(self, loss_S):
        """
        Backward pass for student model
        Args:
            loss_S: loss for student model
        """
        self.optimizer_S.zero_grad()
        loss_S.backward()
        self.optimizer_S.step()

    def backward(self, loss):
        """
        Backward pass for both generator and student model
        Args:
            loss: loss
        """
        self.optimizer_G.zero_grad()
        self.optimizer_S.zero_grad()
        loss.backward()
        self.optimizer_G.step()
        self.optimizer_S.step()

    def reduce_minmax(self):
        """
        Reduce min and max values
        """
        for m in self.model.module.modules():
            if isinstance(m, QuantAct):
                dist.all_reduce(m.x_min, op=dist.ReduceOp.SUM)
                dist.all_reduce(m.x_max, op=dist.ReduceOp.SUM)
                m.x_min = m.x_min / dist.get_world_size()
                m.x_max = m.x_max / dist.get_world_size()

    def spatial_attention(self, x):
        """
        Compute spatial attention
        Args:
            x: input tensor
        """
        return F.normalize(x.pow(2).mean([1]).view(x.size(0), -1))

    def channel_attention(self, x):
        """
        Compute channel attention
        Args:
            x: input tensor
        """
        return F.normalize(x.pow(2).mean([2,3]).view(x.size(0), -1))

    def hook_activation_teacher(self, module, input, output):
        """
        Hook activation for teacher model
        Args:
            module: target module
            input: input
            output: output
        """
        self.activation_teacher.append(self.channel_attention(output.clone()))

    def hook_activation(self, module, input, output):
        """
        Hook activation
        Args:
            module: target module
            input: input
            output: output
        """
        self.activation.append(self.channel_attention(output.clone()))

    def hook_fn_forward(self,module, input, output):
        """
        Hook forward function
        Args:
            module: target module
            input: input
            output: output
        """
        input = input[0]
        mean = input.mean([0, 2, 3])
        # use biased var in train
        var = input.var([0, 2, 3], unbiased=False)

        self.mean_list.append(mean)
        self.var_list.append(var)
        self.teacher_running_mean.append(module.running_mean)
        self.teacher_running_var.append(module.running_var)

    def train(self, epoch, direct_dataload=None):
        """
        Train the model
        Args:
            epoch: current epoch
            direct_dataload: direct data loader
        """
        top1_error = utils.AverageMeter()
        top1_loss = utils.AverageMeter()
        top5_error = utils.AverageMeter()
        fp_acc = utils.AverageMeter()

        iters = 200
        self.update_lr(epoch)

        self.model.eval()
        self.model_teacher.eval()
        self.generator.train()

        start_time = time.time()
        end_time = start_time

        if epoch==0:
            for m in self.model_teacher.module.modules():
                if isinstance(m, nn.SyncBatchNorm):
                    handle = m.register_forward_hook(self.hook_fn_forward)
                    self.handle_list.append(handle)

        if epoch == 4:
            for handle in self.handle_list:
                handle.remove()
            self.reduce_minmax()

            for m in self.model_teacher.module.modules():
                if isinstance(m, ResUnit):
                    m.body.register_forward_hook(self.hook_activation_teacher)
                elif isinstance(m, DwsConvBlock):
                    m.pw_conv.bn.register_forward_hook(self.hook_activation_teacher)
                elif isinstance(m, LinearBottleneck):
                    m.conv3.register_forward_hook(self.hook_activation_teacher)

            for m in self.model.module.modules():
                if isinstance(m, ResUnit):
                    m.body.register_forward_hook(self.hook_activation)
                elif isinstance(m, DwsConvBlock):
                    m.pw_conv.bn.register_forward_hook(self.hook_activation)
                elif isinstance(m, LinearBottleneck):
                    m.conv3.register_forward_hook(self.hook_activation)

            self.generator = self.generator.cpu()
            self.optimizer_G.zero_grad()

        if direct_dataload is not None:
            direct_dataload.sampler.set_epoch(epoch)
            iterator = iter(direct_dataload)

        for i in range(iters):
            start_time = time.time()
            data_time = start_time - end_time

            self.update_cam()

            if epoch < 4:
                z = Variable(torch.randn(16,
                                         self.settings.latent_dim)).to(self.args.local_rank)
                labels = Variable(torch.randint(0, self.settings.nClasses,
                                                (16,))).to(self.args.local_rank)
                z = z.contiguous()
                labels = labels.contiguous()
                images = self.generator(z, labels)

                self.mean_list.clear()
                self.var_list.clear()
                output_teacher_batch = self.model_teacher(images)

                loss_one_hot = self.criterion(output_teacher_batch, labels)
                BNS_loss = torch.zeros(1).to(self.args.local_rank)
                for i in range(len(self.mean_list)):
                    BNS_loss += self.MSE_loss(self.mean_list[i], self.teacher_running_mean[i]) + \
                        self.MSE_loss(self.var_list[i], self.teacher_running_var[i])

                BNS_loss = BNS_loss / len(self.mean_list)
                loss_G = loss_one_hot + 0.1 * BNS_loss
                self.backward_G(loss_G)
                output = self.model(images.detach())
                loss_S = torch.zeros(1).to(self.args.local_rank)
            else:
                try:
                    images, labels = next(iterator)
                except:
                    iterator = iter(direct_dataload)
                    images, labels = next(iterator)
                if self.args.few_shot: # Real samples do not utilize low-pass filter
                    pass
                else:
                    images = self.apply_filters(images, self.args.d_zero)
                if torch.isnan(images).any():
                    images = torch.nan_to_num(images)
                images, labels = images.to(self.args.local_rank), labels.to(self.args.local_rank)

                self.activation_teacher.clear()
                self.activation.clear()

                images.requires_grad = True

                output_teacher_batch = self.model_teacher(images)

                if self.args.selce:
                    prob_true_label = torch.gather(F.softmax(output_teacher_batch, dim=1), 1, labels.unsqueeze(1)).squeeze()
                    difficulty = 1 - prob_true_label
                    mask_selce = (difficulty < self.args.tau_selce).long()
                else:
                    mask_selce = None

                output, loss_KL, loss_FA, loss_CE = self.forward(images, output_teacher_batch, labels, mask_selce)

                loss_S = loss_KL + self.args.lambda_ce * loss_CE + loss_FA 

                perturbation = torch.sgn(torch.autograd.grad(loss_S, images, retain_graph=True)[0])
                self.activation_teacher.clear()
                self.activation.clear()
                with torch.no_grad():
                    images_perturbed = images + self.settings.eps * perturbation
                    output_teacher_batch_perturbed = self.model_teacher(images_perturbed.detach())

                _, loss_KL_perturbed, loss_FA_perturbed, loss_CE_perturbed = self.forward(images_perturbed.detach(),
                                                                                          output_teacher_batch_perturbed.detach(),
                                                                                          labels,
                                                                                          mask_selce)

                loss_S_perturbed = loss_KL_perturbed + \
                                   self.args.lambda_ce * loss_CE_perturbed + \
                                   loss_FA_perturbed if self.settings.dataset == "imagenet" \
                                   else loss_KL_perturbed + self.args.lambda_ce * loss_CE_perturbed 

                loss_cam = self.loss_cam(images)

                loss_S = loss_S + \
                         self.args.lambda_pert * loss_S_perturbed + \
                         self.args.lambda_cam * loss_cam

                self.optimizer_S.zero_grad()
                loss_S.backward()
                self.optimizer_S.step()  

            single_error, single_loss, single5_error = utils.compute_singlecrop(
                outputs=output, labels=labels,
                loss=loss_S, top5_flag=True, mean_flag=True)

            top1_error.update(single_error, images.size(0))
            top1_loss.update(single_loss, images.size(0))
            top5_error.update(single5_error, images.size(0))

            end_time = time.time()

            gt = labels.data.cpu().numpy()
            d_acc = np.mean(np.argmax(output_teacher_batch.data.cpu().numpy(), axis=1) == gt)
            fp_acc.update(d_acc)

        if epoch < 4:
            log_message = (
                f"[Epoch {epoch + 1}/{self.settings.nEpochs}] [Batch {i + 1}/{iters}] "
                f"[train acc: {100 * fp_acc.avg:.4f}%] G loss: {loss_G.item():.2f} "
                f"One-hot loss: {loss_one_hot.item():.2f} BNS_loss: {BNS_loss.item():.2f}"
                )

        else:
            log_message = (
                f"[Epoch {epoch + 1}/{self.settings.nEpochs}] [Batch {i+1}/{iters}] "
                f"[train acc: {100 * fp_acc.avg:.4f}%] [loss: {loss_S.item():.2f}] "
                f"loss KL: {loss_KL.item():.2f} loss CE: {self.args.lambda_ce * loss_CE.item():.2f} "
                f"loss FA: {loss_FA.item():.2f} loss CAM: {self.args.lambda_cam * loss_cam:.2f} "
                f"loss KLp: {loss_KL_perturbed.item():.2f} loss CEp: {self.args.lambda_ce * loss_CE_perturbed.item():.2f} "
                f"loss FAp: {loss_FA_perturbed.item():.2f}"
            )

        self.logger.info(log_message)
        print(log_message)

        return top1_error.avg, top1_loss.avg, top5_error.avg


    def test(self, epoch):
        """
        Test the model
        Args:
            epoch: current epoch
        """
        top1_error = utils.AverageMeter()
        top1_loss = utils.AverageMeter()
        top5_error = utils.AverageMeter()

        self.model.eval()
        self.model_teacher.eval()

        iters = len(self.test_loader)
        start_time = time.time()
        end_time = start_time
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                start_time = time.time()

                labels = labels.to(self.args.local_rank)
                images = images.to(self.args.local_rank)
                output = self.model(images)
                loss = torch.ones(1)
                self.mean_list.clear()
                self.var_list.clear()

                single_error, single_loss, single5_error = utils.compute_singlecrop(
                    outputs=output, loss=loss,
                    labels=labels, top5_flag=True, mean_flag=True)

                top1_error.update(single_error, images.size(0))
                top1_loss.update(single_loss, images.size(0))
                top5_error.update(single5_error, images.size(0))

                end_time = time.time()
        self.logger.info(
            "[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%]" 
            % (epoch + 1, self.settings.nEpochs, i + 1, iters, (100.00-top1_error.avg))
        )

        print(
            "[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%]"
            % (epoch + 1, self.settings.nEpochs, i + 1, iters, (100.00-top1_error.avg))
        )

        self.run_count += 1

        return top1_error.avg, top1_loss.avg, top5_error.avg

    def test_teacher(self, epoch):
        """
        Test the teacher model
        Args:
            epoch: current epoch
        """
        top1_error = utils.AverageMeter()
        top1_loss = utils.AverageMeter()
        top5_error = utils.AverageMeter()

        self.model_teacher.eval()

        iters = len(self.test_loader)
        start_time = time.time()
        end_time = start_time

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                start_time = time.time()
                data_time = start_time - end_time

                labels = labels.to(self.args.local_rank)

                if self.settings.tenCrop:
                    image_size = images.size()
                    images = images.view(
                              image_size[0] * 10, image_size[1] / 10, image_size[2], image_size[3])
                    images_tuple = images.split(image_size[0])
                    output = None
                    for img in images_tuple:
                        if self.settings.nGPU == 1:
                            img = img.to(self.args.local_rank)
                        img_var = Variable(img, volatile=True)
                        temp_output, _ = self.forward(img_var)
                        if output is None:
                            output = temp_output.data
                        else:
                            output = torch.cat((output, temp_output.data))
                    single_error, single_loss, single5_error = utils.compute_tencrop(
                              outputs=output, labels=labels)
                else:
                    if self.settings.nGPU == 1:
                        images = images.to(self.args.local_rank)
                    self.activation_teacher.clear()
                    output = self.model_teacher(images)

                    loss = torch.ones(1)
                    self.mean_list.clear()
                    self.var_list.clear()

                    single_error, single_loss, single5_error = utils.compute_singlecrop(
                              outputs=output, loss=loss, labels=labels,
                              top5_flag=True, mean_flag=True)

                top1_error.update(single_error, images.size(0))
                top1_loss.update(single_loss, images.size(0))
                top5_error.update(single5_error, images.size(0))

                end_time = time.time()
                iter_time = end_time - start_time

        print(f"Teacher network: [Epoch {epoch + 1}/{self.settings.nEpochs}] "
              f"[Batch {i + 1}/{iters}] "
              f"[acc: {100.00 - top1_error.avg:.4f}%]")

        self.run_count += 1

        return top1_error.avg, top1_loss.avg, top5_error.avg

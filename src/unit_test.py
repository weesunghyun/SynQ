"""
[SW Starlab]
Zero-shot Quantization with SynQ (Synthesis-aware Fine-tuning for Zero-shot Quantization)

Author: Minjun Kim (minjun.kim@snu.ac.kr), Seoul National University
        Jongjin Kim (j2kim99@snu.ac.kr), Seoul National University
        U Kang (ukang@snu.ac.kr), Seoul National University

Version : 1.0
Date : Sep 6th, 2023
Main Contact: Minjun Kim
This software is free of charge under research purposes.
For commercial purposes, please contact the authors.

distill_data.py
    - codes for generating distilled data

This code is mainly based on
    - ZeroQ: https://github.com/amirgholami/ZeroQ
    - HAST: https://github.com/lihuantong/HAST
"""
import unittest
import os
import torch
import torch.nn as nn
import sys
from main_direct import Generator

sys.path.append(os.path.join(os.path.dirname(__file__), 'data_generate'))
from distill_data import check_path, LabelSmoothing, OutputHook, DistillData


class DummyOption:
    def __init__(self):
        self.num_classes = 10
        self.latent_dim = 100
        self.img_size = 32
        self.channels = 3


class TestDistillData(unittest.TestCase):

    def setUp(self):
        class Args:
            model = 'resnet18'
            save_path_head = './test_dir'
            batch_size = 4
            num_data = 1280
            lbns = False
            calib_centers = None

        self.args = Args()

        if not os.path.exists(self.args.save_path_head):
            os.makedirs(self.args.save_path_head)

        self.teacher_model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 16 * 16, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 10)
        ).cuda()

    def tearDown(self):
        if os.path.exists(self.args.save_path_head):
            import shutil
            shutil.rmtree(self.args.save_path_head)

    def test_check_path(self):
        """Test if check_path correctly creates directories."""
        test_path = './test_dir/test_file.pickle'
        check_path(test_path)
        self.assertTrue(os.path.exists(os.path.dirname(test_path)))
        os.rmdir(os.path.dirname(test_path))

    def test_label_smoothing(self):
        """Test LabelSmoothing forward pass."""
        criterion = LabelSmoothing(smoothing=0.1)
        inputs = torch.randn(5, 10).cuda()
        targets = torch.randint(0, 10, (5,)).cuda()
        loss = criterion(inputs, targets)
        self.assertTrue(isinstance(loss.item(), float))

    def test_output_hook(self):
        """Test OutputHook functionality."""
        model = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1).cuda()
        hook = OutputHook()
        model.register_forward_hook(hook.hook)
        input_tensor = torch.randn(1, 3, 32, 32).cuda()
        model(input_tensor)
        self.assertIsNotNone(hook.outputs)

    def test_hook_fn_forward(self):
        """Test DistillData's hook_fn_forward function for BatchNorm2d."""
        DD = DistillData(self.args)
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ).cuda()

        input_tensor = torch.randn(1, 3, 32, 32).cuda()
        for _, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.register_forward_hook(DD.hook_fn_forward)

        # Forward pass to trigger the hook
        model(input_tensor)

        # Check that mean and var have been captured
        self.assertTrue(len(DD.mean_list) > 0)
        self.assertTrue(len(DD.var_list) > 0)
        self.assertEqual(DD.mean_list[0].shape, torch.Size([64]))
        self.assertEqual(DD.var_list[0].shape, torch.Size([64]))

    def test_output_hook_clear(self):
        hook = OutputHook()
        hook.outputs = torch.tensor([1, 2, 3])
        hook.clear()
        self.assertIsNone(hook.outputs)

    # 추가된 Generator 관련 테스트
    def test_generator_creation(self):
        """Test Generator object creation."""
        options = DummyOption()
        generator = Generator(options=options)
        self.assertIsInstance(generator, Generator)

    def test_generator_forward_output_size(self):
        """Test the output size of the Generator's forward pass."""
        options = DummyOption()
        generator = Generator(options=options)
        z = torch.randn(1, options.latent_dim)
        labels = torch.randint(0, options.num_classes, (1,))
        output = generator(z, labels)
        self.assertEqual(output.shape, (1, options.channels, options.img_size, options.img_size))

    def test_generator_label_embedding(self):
        """Test the label embedding in the Generator."""
        options = DummyOption()
        generator = Generator(options=options)
        labels = torch.randint(0, options.num_classes, (1,))
        embedded_labels = generator.label_emb(labels)
        self.assertEqual(embedded_labels.shape, (1, options.latent_dim))

    def test_generator_intermediate_output_size(self):
        """Test intermediate output size in Generator's forward pass."""
        options = DummyOption()
        generator = Generator(options=options)
        z = torch.randn(1, options.latent_dim)
        labels = torch.randint(0, options.num_classes, (1,))
        out = generator.l1(torch.mul(generator.label_emb(labels), z))
        self.assertEqual(out.view(1, 128, options.img_size // 4, options.img_size // 4).shape, (1, 128, 8, 8))

    def test_generator_no_nan_in_output(self):
        """Test that the Generator output does not contain NaN values."""
        options = DummyOption()
        generator = Generator(options=options)
        z = torch.randn(1, options.latent_dim)
        labels = torch.randint(0, options.num_classes, (1,))
        output = generator(z, labels)
        self.assertFalse(torch.isnan(output).any())


if __name__ == '__main__':
    unittest.main()

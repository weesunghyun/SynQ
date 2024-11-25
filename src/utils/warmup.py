"""
[SW Starlab]
Zero-shot Quantization with SynQ (Synthesis-aware Fine-tuning for Zero-shot Quantization)

Author
    - Minjun Kim (minjun.kim@snu.ac.kr), Seoul National University
    - Jongjin Kim (j2kim99@snu.ac.kr), Seoul National University
    - U Kang (ukang@snu.ac.kr), Seoul National University

Version : 1.0

Date : Sep 6th, 2023

Main Contact: Minjun Kim

This software is free of charge under research purposes.

For commercial purposes, please contact the authors.

warmup.py
    - codes for warming up the dataset

This code is mainly based on
    - ZeroQ: https://github.com/amirgholami/ZeroQ
    - HAST: https://github.com/lihuantong/HAST
"""
# from torchlearning.mio import MIO

# train_dataset = MIO("/home/datasets/imagenet_mio/train/")
# test_dataset = MIO("/home/datasets/imagenet_mio/val/")

# for i in range(train_dataset.size):
#     print(i)
#     train_dataset.fetchone(i)

# for i in range(test_dataset.size):
#     print(i)
#     test_dataset.fetchone(i)

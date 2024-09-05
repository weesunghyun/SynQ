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

log_print.py
    - codes for logging the training and testing results

This code is mainly based on
    - ZeroQ: https://github.com/amirgholami/ZeroQ
    - HAST: https://github.com/lihuantong/HAST
"""
import datetime

import numpy as np


# __all__ = ["compute_remain_time", "print_result", "print_weight", "print_grad"]
__all__ = ["compute_remain_time", "print_result"]


class TimeTracker:
    """
        Time tracker
    """
    def __init__(self):
        self.single_train_time = 0
        self.single_test_time = 0
        self.single_train_iters = 0
        self.single_test_iters = 0

time_tracker = TimeTracker()

def compute_remain_time(epoch, num_epochs, count, iters, data_time, iter_time, mode="Train"):
    """
        Compute the remain time for training or testing
        Args:
            epoch: current epoch
            num_epochs: total epochs
            count: current iteration
            iters: total iterations
            data_time: data loading time
            iter_time: iteration time
            mode: train or test
        Returns:
            time_str: time string
            total_time: total time
            left_time: left time
    """
    # compute cost time
    if mode == "Train":
        time_tracker.single_train_time = time_tracker.single_train_time * \
                            0.95 + 0.05 * (data_time + iter_time)

        time_tracker.single_train_iters = iters
        train_left_iter = time_tracker.single_train_iters - count + \
                          (num_epochs - epoch - 1) * time_tracker.single_train_iters

        test_left_iter = (num_epochs - epoch) * time_tracker.single_test_iters
    else:
        time_tracker.single_test_time = time_tracker.single_test_time * \
                           0.95 + 0.05 * (data_time + iter_time)

        time_tracker.single_test_iters = iters
        train_left_iter = (num_epochs - epoch - 1) * time_tracker.single_train_iters
        test_left_iter = time_tracker.single_test_iters - count + \
                         (num_epochs - epoch - 1) * time_tracker.single_test_iters

    left_time = time_tracker.single_train_time * train_left_iter + \
                time_tracker.single_test_time * test_left_iter
    total_time = (time_tracker.single_train_time * time_tracker.single_train_iters +
                  time_tracker.single_test_time * time_tracker.single_test_iters) * num_epochs
    time_str = (
        f"TTime: {datetime.timedelta(seconds=total_time)}, "
        f"RTime: {datetime.timedelta(seconds=left_time)}"
    )

    return time_str, total_time, left_time


def print_result(epoch, num_epochs, count, iters, lr, data_time,
                 iter_time, error, loss, top5error=None,
                 mode="Train", logger=None):
    """
        Print the training or testing result
        Args:
            epoch: current epoch
            num_epochs: total epochs
            count: current iteration
            iters: total iterations
            lr: learning rate
            data_time: data loading time
            iter_time: iteration time
            error: error rate
            loss: loss
            top5error: top5 error rate
            mode: train or test
            logger: logger
        Returns:
            total_time: total time
            left_time: left time
        """

    log_str = (
        f">>> {mode}: [{epoch + 1:0>3d}|{num_epochs:0>3d}], Iter: [{count:0>3d}|{iters:0>3d}], "
        f"LR: {lr:.6f}, DataTime: {data_time:.4f}, IterTime: {iter_time:.4f},"
    )

    if isinstance(error, (list, np.ndarray)):
        for i, error_i in enumerate(error):
            log_str += f"Error_{i}: {error_i:.4f}, Loss_{i}: {loss[i]:.4f}, "

    else:
        log_str += f"Error: {error:.4f}, Loss: {loss:.4f}, "

    if top5error is not None:
        if isinstance(top5error, (list, np.ndarray)):
            for i, top5error_i in enumerate(top5error):
                log_str += f"Top5_Error_{i}: {top5error_i:.4f}, "

        else:
            log_str += f" Top5_Error: {top5error:.4f}, "

    time_str, total_time, left_time = compute_remain_time(epoch, num_epochs, count, iters,
                                                          data_time, iter_time, mode)

    logger.info(log_str + time_str)

    return total_time, left_time


# def print_weight(layers, logger):
#     """
#         Print the weight of the layers
#         Args:
#             layers: target layers
#             logger: logger
#     """
#     if isinstance(layers, MD.qConv2d):
#         logger.info(layers.weight)
#     elif isinstance(layers, MD.qLinear):
#         logger.info(layers.weight)
#         logger.info(layers.weight_mask)
#     logger.info("------------------------------------")


# def print_grad(m, logger):
#     """
#         Print the grad of the module
#         Args:
#             m: target module
#             logger: logger
#     """
#     if isinstance(m, MD.qLinear):
#         logger.info(m.weight.data)

import datetime

import numpy as np


# __all__ = ["compute_remain_time", "print_result", "print_weight", "print_grad"]
__all__ = ["compute_remain_time", "print_result"]


SINGLE_TRAIN_TIME = 0
SINGLE_TEST_TIME = 0
SINGLE_TRAIN_ITERS = 0
SINGLE_TEST_ITERS = 0


def compute_remain_time(epoch, nEpochs, count, iters, data_time, iter_time, mode="Train"):
    """
        Compute the remain time for training or testing
        Args:
            epoch: current epoch
            nEpochs: total epochs
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
    global SINGLE_TRAIN_TIME, SINGLE_TEST_TIME
    global SINGLE_TRAIN_ITERS, SINGLE_TEST_ITERS

    # compute cost time
    if mode == "Train":
        SINGLE_TRAIN_TIME = SINGLE_TRAIN_TIME * \
                            0.95 + 0.05 * (data_time + iter_time)

        SINGLE_TRAIN_ITERS = iters
        train_left_iter = SINGLE_TRAIN_ITERS - count + \
                          (nEpochs - epoch - 1) * SINGLE_TRAIN_ITERS

        test_left_iter = (nEpochs - epoch) * SINGLE_TEST_ITERS
    else:
        SINGLE_TEST_TIME = SINGLE_TEST_TIME * \
                           0.95 + 0.05 * (data_time + iter_time)

        SINGLE_TEST_ITERS = iters
        train_left_iter = (nEpochs - epoch - 1) * SINGLE_TRAIN_ITERS
        test_left_iter = SINGLE_TEST_ITERS - count + \
                         (nEpochs - epoch - 1) * SINGLE_TEST_ITERS

    left_time = SINGLE_TRAIN_TIME * train_left_iter + \
                SINGLE_TEST_TIME * test_left_iter
    total_time = (SINGLE_TRAIN_TIME * SINGLE_TRAIN_ITERS +
                  SINGLE_TEST_TIME * SINGLE_TEST_ITERS) * nEpochs
    time_str = (
        f"TTime: {datetime.timedelta(seconds=total_time)}, "
        f"RTime: {datetime.timedelta(seconds=left_time)}"
    )

    return time_str, total_time, left_time


def print_result(epoch, nEpochs, count, iters, lr, data_time,
                 iter_time, error, loss, top5error=None,
                 mode="Train", logger=None):
    """
        Print the training or testing result
        Args:
            epoch: current epoch
            nEpochs: total epochs
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
        f">>> {mode}: [{epoch + 1:0>3d}|{nEpochs:0>3d}], Iter: [{count:0>3d}|{iters:0>3d}], "
        f"LR: {lr:.6f}, DataTime: {data_time:.4f}, IterTime: {iter_time:.4f},"
    )

    if isinstance(error, list) or isinstance(error, np.ndarray):
        for i, error_i in enumerate(error):
            log_str += f"Error_{i}: {error_i:.4f}, Loss_{i}: {loss[i]:.4f}, "

    else:
        log_str += f"Error: {error:.4f}, Loss: {loss:.4f}, "

    if top5error is not None:
        if isinstance(top5error, list) or isinstance(top5error, np.ndarray):
            for i, top5error_i in enumerate(top5error):
                log_str += f"Top5_Error_{i}: {top5error_i:.4f}, "

        else:
            log_str += f" Top5_Error: {top5error:.4f}, "

    time_str, total_time, left_time = compute_remain_time(epoch, nEpochs, count, iters,
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

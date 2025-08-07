"""DSV-based data synthesis utilities."""
import os
import pickle

import torch
from torch import nn
from torch.nn import functional as F

from distill_data import check_path


def _determine_shape(model_name: str, batch_size: int):
    """Return input shape for the given model name."""
    if model_name in ['resnet20_cifar10', 'resnet20_cifar100', 'resnet34_cifar100']:
        return (batch_size, 3, 32, 32)
    else:
        return (batch_size, 3, 224, 224)


def _total_variation(x: torch.Tensor) -> torch.Tensor:
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w


def _image_norm(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).mean()


def generate_dsv_data(args, model: nn.Module):
    """Generate synthetic data via Deep Support Vectors.

    Args:
        args: command line arguments
        model: pretrained model (frozen)
    """
    device = next(model.parameters()).device
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # determine input shape and number of classes
    shape = _determine_shape(args.model, args.num_data)
    with torch.no_grad():
        dummy_out = model(torch.randn(1, *shape[1:], device=device))
        num_classes = dummy_out.shape[1]

    # initialize data and lambdas
    total = args.num_data
    labels = torch.arange(num_classes, device=device).repeat_interleave(total // num_classes + 1)[:total]
    x = torch.randn(total, *shape[1:], device=device, requires_grad=True)
    lambdas = torch.ones(total, device=device, requires_grad=True)

    theta = [p.detach().clone() for p in model.parameters()]
    optim = torch.optim.Adam([x, lambdas], lr=getattr(args, 'dsv_lr', 0.1))
    iters = getattr(args, 'dsv_iter', 200)

    for _ in range(iters):
        outputs = model(x)
        ce = F.cross_entropy(outputs, labels, reduction='none')
        preds = outputs.argmax(dim=1)
        primal_mask = (preds != labels).float()
        primal_loss = (ce * primal_mask).mean()

        weighted = (ce * lambdas.relu()).sum()
        grads = torch.autograd.grad(weighted, model.parameters(), create_graph=True)
        stat_loss = 0.0
        for th, g in zip(theta, grads):
            stat_loss = stat_loss + ((th + g) ** 2).mean()

        tv_loss = _total_variation(x)
        norm_loss = _image_norm(x)
        total_loss = stat_loss + args.beta * primal_loss + args.gamma * tv_loss + 1e-5 * norm_loss

        optim.zero_grad()
        total_loss.backward()
        optim.step()
        with torch.no_grad():
            lambdas.clamp_(min=0.0)

    data_path = os.path.join(args.save_path_head, f"{args.model}_dsv_data.pickle")
    label_path = os.path.join(args.save_path_head, f"{args.model}_dsv_labels.pickle")
    check_path(data_path)
    check_path(label_path)
    with open(data_path, 'wb') as f:
        pickle.dump([x.detach().cpu().numpy()], f)
    with open(label_path, 'wb') as f:
        pickle.dump([labels.detach().cpu().numpy()], f)

    print('****** DSV Data Generated ******')

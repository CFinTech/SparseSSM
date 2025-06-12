import math
import time
import torch
import torch.nn as nn
import transformers
import matplotlib.pyplot as plt

from utils.model_utils import setup_logger, find_layers


def magnitude_prune(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
    numel = weight.numel()
    k = int(numel * (1.0 - sparsity))
    if k <= 0:
        return torch.zeros_like(weight)
    if k >= numel:
        return weight

    flat = weight.view(-1)
    _, idx = torch.topk(flat.abs(), k, largest=True, sorted=False)
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[idx] = True
    mask = mask.view(weight.shape)
    return weight * mask


def mag_sequential(model, args, dev, logger):
    for i, layer in enumerate(model.layers):
        if not args.minlayer <= i < args.maxlayer:
            continue
        layer = layer.to(dev)
        A_log = layer.mixer.A_log.to(dev)
        A_log = magnitude_prune(A_log, args.sparsity)
        layer.mixer.A_log = torch.nn.Parameter(A_log.cpu())
        layer = layer.cpu()
        del layer


def mag_ffn(model, args, dev, logger):
    for i, layer in enumerate(model.layers):
        if not args.minlayer <= i < args.maxlayer:
            continue
        layer = layer.to(dev)
        full = find_layers(layer, args.target_module_classes)
        for name, module in full.items():
            logger.info(f"pruning module {name}")
            pruned_weight = magnitude_prune(module.weight.data, args.sparsity)
            module.weight = torch.nn.Parameter(pruned_weight)
        layer = layer.cpu()
        del layer

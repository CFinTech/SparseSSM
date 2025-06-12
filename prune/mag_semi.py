import math
import time
import torch
import torch.nn as nn
import transformers
import matplotlib.pyplot as plt

from utils.model_utils import setup_logger, find_layers


def magnitude_nm_prune(
    weight: torch.Tensor, n: int, m: int, dim: int = 0
) -> torch.Tensor:

    L = weight.shape[dim]

    perm = [dim] + [i for i in range(weight.ndim) if i != dim]
    w = weight.permute(*perm).contiguous()
    rest = w.shape[1:]
    K = int(torch.tensor(rest).prod().item())

    w_flat = w.view(L, -1)

    G = L // m
    w_grp = w_flat.view(G, m, K)

    _, topk_idx = torch.topk(w_grp.abs(), n, dim=1, largest=True)

    mask = torch.zeros_like(w_grp)
    g_idx = torch.arange(G, device=weight.device).view(G, 1, 1).expand(G, n, K)
    k_idx = torch.arange(K, device=weight.device).view(1, 1, K).expand(G, n, K)
    mask[g_idx, topk_idx, k_idx] = 1.0

    mask_flat = mask.reshape(L, K)
    pruned_flat = w_flat * mask_flat
    pruned_w = pruned_flat.view((L,) + rest)

    inv_perm = [perm.index(i) for i in range(weight.ndim)]
    return pruned_w.permute(*inv_perm)


def mag_ffn_semi(model, args, dev, logger):
    for i, layer in enumerate(model.layers):
        if not (args.minlayer <= i < args.maxlayer):
            continue
        layer.to(dev)
        modules = find_layers(layer, args.target_module_classes)
        for name, module in modules.items():
            w = module.weight.data
            pruned = magnitude_nm_prune(w, args.prunen, args.prunem, dim=0)
            module.weight = torch.nn.Parameter(pruned.cpu())
        layer.cpu()
        del layer


def mag_semi_sequential(model, args, dev, logger):
    for i, layer in enumerate(model.layers):
        if not args.minlayer <= i < args.maxlayer:
            continue
        layer = layer.to(dev)
        A_log = layer.mixer.A_log.to(dev)
        A_log = magnitude_nm_prune(A_log, args.prunen, args.prunem)
        layer.mixer.A_log = torch.nn.Parameter(A_log.cpu())
        layer = layer.cpu()
        del layer

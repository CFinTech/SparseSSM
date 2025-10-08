import math
import time
import torch
import torch.nn as nn
import transformers
import matplotlib.pyplot as plt
from einops import rearrange, repeat, einsum

from utils.model_utils import setup_logger


class SparseSSM:
    def __init__(self, weight, L):
        self.weight = weight
        self.nsamples = 0
        self.rows = weight.shape[0]
        self.columns = weight.shape[1]
        self.dev = weight.device
        self.L = L
        self.deltaA = torch.zeros(
            (1, self.L, self.rows, self.columns), device=self.dev
        )  # (1, L, D, N)
        self.scaler = torch.zeros(
            (self.L, self.rows, self.columns), device=self.dev
        )  # (L, D, N)

    def add_batch(self, inp, deltaA):
        tmp = inp.shape[0]
        self.scaler *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        self.scaler += torch.sum(inp[0] ** 2, dim=0) 
        self.deltaA += deltaA 

    def _time_weights_prefix(
        self,
        mode: str = "power",
        gamma: float = 0.9,
        k_frac: float = 0.25,
        power: float = 1.0,
    ):
        L = self.L
        dev = self.dev
        t = torch.arange(L, device=dev, dtype=torch.float)

        if mode == "discount":
            w = gamma ** t
        elif mode == "linear":
            w = (L - t)
        elif mode == "power":
            w = (t + 1.0).pow(-power)
        elif mode == "window":
            k = max(1, math.ceil(k_frac * L))
            w = torch.zeros(L, device=dev)
            w[:k] = 1.0
        else:
            w = torch.ones(L, device=dev)
        w = w / (w.sum() + 1e-12)
        return w  # (L,)

    def prune_time_weighted(
        self,
        sparsity: float,
        mode: str = "power",
        gamma: float = 0.9,
        k_frac: float = 0.25,
        power: float = 1.0,
    ):
        deltaA = self.deltaA.sum(dim=0)  # (L, D, N)
        metric = (deltaA ** 2) * self.scaler  # (L, D, N)

        w = self._time_weights_prefix(
            mode=mode, gamma=gamma, k_frac=k_frac, power=power
        )  # (L,)
        agg = (w.view(-1, 1, 1) * metric).sum(dim=0)  # (D, N)

        num_elements = self.rows * self.columns
        K = int(round(sparsity * num_elements))
        flat = agg.view(-1)
        _, idx = flat.topk(K, largest=False)
        mask = torch.ones_like(flat)
        mask[idx] = 0.0
        mask = mask.view(self.rows, self.columns)
        return self.weight * mask

    def _assert_nm(self, prunen: int, prunem: int):
        assert (
            prunen > 0 and prunem > 0
        ), "For semi-structured pruning, prunen and prunem must be > 0."
        assert (
            prunen < prunem
        ), f"prunen ({prunen}) must be smaller than prunem ({prunem})."
        assert (
            self.columns % prunem == 0
        ), f"columns ({self.columns}) must be divisible by prunem ({prunem})."

    def prune_time_weighted_semi(
        self,
        prunen: int,
        prunem: int,
        mode: str = "window",
        gamma: float = 0.9,
        k_frac: float = 0.25,
        power: float = 1.0,
    ):
        self._assert_nm(prunen, prunem)

        deltaA = self.deltaA.sum(dim=0)  # (L, D, N)
        metric = (deltaA ** 2) * self.scaler  # (L, D, N)
        w = self._time_weights_prefix(
            mode=mode, gamma=gamma, k_frac=k_frac, power=power
        )  # (L,)
        agg = (w.view(-1, 1, 1) * metric).sum(dim=0)  # (D, N)

        D, N = agg.shape
        G = N // prunem
        agg_g = agg.view(D, G, prunem)  # (D, G, M)

        _, idx_small = torch.topk(agg_g, prunen, dim=2, largest=False)
        mask_g = torch.ones_like(agg_g, dtype=agg_g.dtype, device=self.dev)
        mask_g.scatter_(2, idx_small, 0.0)

        mask = mask_g.view(D, N)
        return self.weight * mask

    def free(self):
        self.deltaA = None
        self.scaler = None
        torch.cuda.empty_cache()


def ts_semi_sequential(model, args, dev, logger, inps):
    prunen = getattr(args, "prunen", 0)
    prunem = getattr(args, "prunem", 0)
    assert prunen > 0 and prunem > 0, "Please set --prunen and --prunem for semi-structured pruning."

    for i, layer in enumerate(model.layers):
        logger.info(f"Layer {i}, Semi-structured pruning N:{prunen} of M:{prunem} …")
        if not (args.minlayer <= i < args.maxlayer):
            continue

        layer = layer.to(dev)
        A_log = layer.mixer.A_log.to(dev)
        logger.info(f"A_log original: {A_log.shape}")

        pruner = SparseSSM(A_log, model.seqlen)
        outs = torch.zeros_like(inps)

        with torch.no_grad():
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0))
                deltaA = layer.mixer.deltaA.to(dev)
                act = torch.stack(layer.mixer.h[0: model.seqlen], dim=1)
                pruner.add_batch(act.unsqueeze(0), deltaA)
                layer.mixer.h = []
                layer.mixer.deltaA = None

        A_log_pruned = pruner.prune_time_weighted_semi(
            prunen=prunen,
            prunem=prunem,
            mode=getattr(args, "tw_mode", "window"),
            gamma=getattr(args, "tw_gamma", 0.9),
            k_frac=getattr(args, "tw_k_frac", 0.25),
            power=getattr(args, "tw_power", 1.0),
        )

        logger.info(f"A_log pruned (semi): {A_log_pruned.shape}")
        layer.mixer.A_log = torch.nn.Parameter(A_log_pruned.cpu())
        pruner.free()

        layer = layer.to(dev)
        inps = inps.to(dev)
        outs = torch.zeros_like(inps).to(dev)
        with torch.no_grad():
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0))
                layer.mixer.h = []
                layer.mixer.deltaA = None

        layer = layer.cpu()
        del layer, pruner
        inps, outs = outs, inps

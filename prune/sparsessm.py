import math
import time
import torch
import torch.nn as nn
import transformers
import matplotlib.pyplot as plt

from utils.model_utils import setup_logger


class SparseSSM:
    def __init__(self, weight, L):
        self.weight = weight
        self.nsamples = 0
        self.rows = weight.shape[0]
        self.columns = weight.shape[1]
        self.dev = weight.device
        self.L = L
        self.scaler = torch.zeros(
            (self.L, self.rows, self.columns), device=self.dev
        )  # (L, D, N)

    def add_batch(self, inp):
        # * inp: (1, B, L, D, N)
        tmp = inp.shape[0]
        self.scaler *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        self.scaler += torch.sum(inp[0] ** 2, dim=0)  # (L, D, N)

    def prune(self, sparsity):
        num_elements = self.rows * self.columns
        K = int(round(sparsity * num_elements))

        counts = torch.zeros(num_elements, device=self.dev)

        for t in range(self.L):
            metric_t = (self.weight**2) * self.scaler[t]  # (D, N)
            _, idx_t = metric_t.view(-1).topk(K, largest=False)
            counts[idx_t] += 1
            mask_t = torch.ones(num_elements, device=self.dev)
            mask_t[idx_t] = 0.0
            mask_t = mask_t.view(self.rows, self.columns)

        _, idx_final = counts.topk(K, largest=True)
        mask_final = torch.ones(num_elements, device=self.dev)
        mask_final[idx_final] = 0.0
        mask_final = mask_final.view(self.rows, self.columns)

        return self.weight * mask_final

    def free(self):
        self.scaler = None
        torch.cuda.empty_cache()


def ts_sequential(model, args, dev, logger, inps):
    for i, layer in enumerate(model.layers):
        logger.info(f"Layer {i}, Pruning...")
        if not (args.minlayer <= i < args.maxlayer):
            continue

        layer = layer.to(dev)
        A_log = layer.mixer.A_log.to(dev)
        ssm = SparseSSM(A_log, model.seqlen)
        outs = torch.zeros_like(inps)
        with torch.no_grad():
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0))
                act = torch.stack(layer.mixer.h[0 : model.seqlen], dim=1)
                ssm.add_batch(act.unsqueeze(0))
                layer.mixer.h = []

        A_log = ssm.prune(args.sparsity)
        layer.mixer.A_log = torch.nn.Parameter(A_log.cpu())
        ssm.free()

        layer = layer.to(dev)
        inps = inps.to(dev)
        outs = torch.zeros_like(inps).to(dev)
        with torch.no_grad():
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0))
                layer.mixer.h = []

        layer = layer.cpu()
        del layer
        del ssm
        inps, outs = outs, inps

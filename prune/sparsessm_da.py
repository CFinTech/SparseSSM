import math
import time
import torch
import torch.nn as nn
import transformers
import matplotlib.pyplot as plt

from utils.model_utils import setup_logger


class SparseSSM_L2:
    def __init__(self, weight):
        self.weight = weight
        self.nsamples = 0
        self.rows = weight.shape[0]
        self.columns = weight.shape[1]
        self.dev = weight.device
        self.scaler = torch.zeros((self.rows, self.columns), device=self.dev)  # (D, N)

    def add_batch(self, h, deltaA):
        # * inp: (1, B, L, D, N)
        tmp = h.shape[0]
        self.scaler *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = h[0]  # (B, L, D, N)
        self.scaler += inp[0].norm(p=2, dim=(0, 1))

    def prune(self, sparsity, logger):
        K = int(round(sparsity * self.rows * self.columns))
        metric = self.scaler * (self.weight**2)  # (D, N)
        _, idx = metric.view(-1).topk(K, largest=False)
        mask = torch.ones(self.rows * self.columns, device=self.dev)  # (D*N)
        mask[idx] = 0.0
        mask = mask.view(self.rows, self.columns)
        return self.weight * mask

    def free(self):
        self.scaler = None
        torch.cuda.empty_cache()


def ts_real(model, args, dev, logger, inps):
    for i, layer in enumerate(model.layers):
        logger.info(f"Layer {i}, Pruning...")
        if not (args.minlayer <= i < args.maxlayer):
            continue

        layer = layer.to(dev)
        A_log = layer.mixer.A_log.to(dev)

        ssm = SparseSSM_L2(A_log)
        outs = torch.zeros_like(inps)
        with torch.no_grad():
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0))
                act = torch.stack(layer.mixer.h[0 : model.seqlen], dim=1)
                deltaA = layer.mixer.deltaA

                ssm.add_batch(act.unsqueeze(0), deltaA)
                layer.mixer.h = []
                layer.mixer.deltaA = None
        A_log = ssm.prune(args.sparsity, logger)
        ssm.free()

        logger.info(f"A_log_pruned: {A_log}")
        layer.mixer.A_log = torch.nn.Parameter(A_log.cpu())

        layer = layer.to(dev)
        inps = inps.to(dev)
        outs = torch.zeros_like(inps).to(dev)
        with torch.no_grad():
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0))
                layer.mixer.h = []
                layer.mixer.deltaA = None

        layer = layer.cpu()
        del layer
        del ssm
        inps, outs = outs, inps

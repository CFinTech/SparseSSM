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
            (1, self.L, self.rows, self.columns), device = self.dev
        )  # (L, D, N)
        self.scaler = torch.zeros(
            (self.L, self.rows, self.columns), device = self.dev
        )  # (L, D, N)

    def add_batch(self, inp, deltaA):
        # * inp: (1, B, L, D, N)
        tmp = inp.shape[0]
        self.scaler *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        self.scaler += torch.sum(inp[0] ** 2, dim=0)  # (L, D, N)
        self.deltaA += deltaA

    
    def _time_weights_prefix(self, mode: str = "power",
                             gamma: float = 0.9,
                             k_frac: float = 0.25,
                             power: float = 1.0):
        L = self.L
        dev = self.dev
        t = torch.arange(L, device=dev, dtype=torch.float)

        if mode == "discount":
            w = (gamma ** t) 
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
        return w 

    def prune_time_weighted(self, sparsity: float,
                                   mode: str = "power",
                                   gamma: float = 0.9,
                                   k_frac: float = 0.25,
                                   power: float = 1.0):
        deltaA = self.deltaA.sum(dim=0) 
        metric = (deltaA ** 2) * self.scaler  

        w = self._time_weights_prefix(mode=mode, gamma=gamma,
                                      k_frac=k_frac, power=power) 
        agg = (w.view(-1, 1, 1) * metric).sum(dim=0) 

        num_elements = self.rows * self.columns
        K = int(round(sparsity * num_elements))
        flat = agg.view(-1)
        _, idx = flat.topk(K, largest=False) 
        mask = torch.ones_like(flat)
        mask[idx] = 0.0
        mask = mask.view(self.rows, self.columns)

        return self.weight * mask
    
    def free(self):
        self.scaler = None
        torch.cuda.empty_cache()
    

def ts_dev(model, args, dev, logger, inps):
    for i, layer in enumerate(model.layers):
        logger.info(f"Layer {i}, Pruning...")
        if not (args.minlayer <= i < args.maxlayer):
            continue

        layer = layer.to(dev)
        A_log = layer.mixer.A_log.to(dev)
        logger.info(f"A log original: {A_log}")
        
        ssm = SparseSSM(A_log, model.seqlen)
        outs = torch.zeros_like(inps)
        
        with torch.no_grad():
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0))
                deltaA = layer.mixer.deltaA.to(dev)
                act = torch.stack(layer.mixer.h[0 : model.seqlen], dim=1)
                ssm.add_batch(act.unsqueeze(0), deltaA)
                layer.mixer.h = []
                layer.mixer.deltaA = None

        A_log = ssm.prune_time_weighted(args.sparsity)
        
        logger.info(f"A log pruned: {A_log}")
        layer.mixer.A_log = torch.nn.Parameter(A_log.cpu())
        ssm.free()

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

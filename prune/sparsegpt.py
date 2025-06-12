import math
import time

import torch
import torch.nn as nn
import transformers
from typing import Dict, List, Tuple
from utils.model_utils import *


DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class SparseGPT:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, torch.nn.modules.conv.Conv1d):
            if len(W.shape) == 3:
                W = W.reshape((-1, W.shape[-1]))
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]

        if isinstance(self.layer, nn.Linear) or isinstance(
            self.layer, torch.nn.modules.conv.Conv1d
        ):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
                if isinstance(self.layer, torch.nn.modules.conv.Conv1d):
                    inp = inp.t()
            inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def trace(self):
        return torch.sum(torch.diagonal(self.H) ** 2)

    def fasterprune(
        self, sparsity, logger, prunen=0, prunem=0, blocksize=128, percdamp=0.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, torch.nn.modules.conv.Conv1d):
            W = W.reshape((-1, W.shape[-1]))
            W = W.t()
        W = W.float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)

        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0:
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1**2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = (
                        W1[:, i : (i + prunem)] ** 2
                        / (torch.diag(Hinv1)[i : (i + prunem)].reshape((1, -1))) ** 2
                    )
                    mask1.scatter_(
                        1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True
                    )

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = W[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        logger.info(f"Elapsed time: {(time.time() - tick):.2f} s")
        logger.info(f"Total error: {torch.sum(Losses).item():.4f}")

        if isinstance(self.layer, torch.nn.modules.conv.Conv1d):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if DEBUG:
            print(
                f"real reconstruction loss: {torch.sum((self.layer(self.inp1) - self.out1) ** 2)}"
            )

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()


def ffn_sequential(
    model,
    args,
    dev,
    logger,
    inps,
):
    layers = model.layers

    outs = torch.zeros_like(inps)

    traces: Dict[Tuple[str, int], float] = {}
    sizes: Dict[Tuple[str, int], int] = {}
    param_counts: Dict[Tuple[str, int], int] = {}

    for i, layer_obj in enumerate(layers):
        if not (args.minlayer <= i < args.maxlayer):
            continue

        layer = layer_obj.to(dev)
        full = find_layers(layer, args.target_module_classes)

        gpts, handles = {}, []
        for name, module in full.items():
            g = SparseGPT(module)
            gpts[name] = g
            handles.append(
                module.register_forward_hook(
                    lambda m, _inp, _out, n=name: gpts[n].add_batch(
                        _inp[0].data, _out.data
                    )
                )
            )

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0))
            if hasattr(layer, "mixer") and hasattr(layer.mixer, "h"):
                layer.mixer.h = []

        for h in handles:
            h.remove()

        for name, g in gpts.items():
            traces[(name, i)] = g.trace()
            sizes[(name, i)] = g.H.shape[0]
            if hasattr(g, "layer") and hasattr(g.layer, "weight"):
                param_counts[(name, i)] = g.layer.weight.data.numel()
            else:
                param_counts[(name, i)] = 0
            g.free()

        layers[i] = layer.cpu()
        del gpts, layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    sparsity_alloc = allocate_sparsity(traces, sizes, args)

    total_params = 0
    total_pruned = 0.0
    for key, p_count in param_counts.items():
        s = sparsity_alloc.get(key, args.sparsity)
        total_params += p_count
        total_pruned += p_count * s

    if total_params > 0:
        overall_sparsity = total_pruned / total_params
        logger.info(
            f"Overall sparsity: {overall_sparsity:.4%} "
            f"({int(total_pruned)}/{total_params} parameters pruned)"
        )
    else:
        logger.info("No parameters found for pruning.")

    del traces, sizes, param_counts
    torch.cuda.empty_cache()

    for i, layer_obj in enumerate(layers):
        if not (args.minlayer <= i < args.maxlayer):
            continue

        layer = layer_obj.to(dev)
        full = find_layers(layer, args.target_module_classes)

        gpts = {}
        handles = []
        for name, module in full.items():
            if args.module != "None" and name != args.module:
                continue
            g = SparseGPT(module)
            gpts[name] = g
            handles.append(
                module.register_forward_hook(
                    lambda m, _inp, _out, n=name: gpts[n].add_batch(
                        _inp[0].data, _out.data
                    )
                )
            )

        for _ in range(args.nsamples):
            with torch.no_grad():
                layer(inps[_].unsqueeze(0))
            if hasattr(layer, "mixer") and hasattr(layer.mixer, "h"):
                layer.mixer.h = []

        for h in handles:
            h.remove()

        for name, g in gpts.items():
            s = sparsity_alloc.get((name, i), args.sparsity)
            logger.info(f"Layer {i}, module {name}: pruning at {s:.4%}")
            g.fasterprune(
                s,
                logger,
                prunen=args.prunen,
                prunem=args.prunem,
                percdamp=args.percdamp,
                blocksize=args.blocksize,
            )
            g.free()

        for sample_idx in range(args.nsamples):
            outs[sample_idx] = layer(inps[sample_idx].unsqueeze(0))
            if hasattr(layer, "mixer") and hasattr(layer.mixer, "h"):
                layer.mixer.h = []

        layers[i] = layer.cpu()
        del layer, full, gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps

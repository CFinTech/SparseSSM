import math
import time
import torch
import torch.nn as nn
import transformers
import matplotlib.pyplot as plt

from utils.model_utils import setup_logger

import torch


class SparseSSM_st:
    def __init__(self, weight: torch.Tensor, L: int):
        self.weight = weight
        self.nsamples = 0
        self.rows, self.columns = weight.shape
        self.dev = weight.device
        self.L = L

        self.scaler = torch.zeros(
            (L, self.rows, self.columns), device=self.dev
        )  # (L, D, N)

    def reset(self):
        self.nsamples = 0
        self.scaler.zero_()

    def add_batch(self, inp: torch.Tensor):
        batch_size = inp.shape[0]  # inp: (1, B, L, D, N)
        self.scaler *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size
        self.scaler += torch.sum(inp[0] ** 2, dim=0)

    def prune(
        self,
        sparsity: float,
        logger,
        axis: int = 1,
    ) -> torch.Tensor:
        assert axis in (0, 1), "axis must be 0 or 1"
        num_struct = self.rows if axis == 0 else self.columns
        k_struct = int(round(sparsity * num_struct))

        counts = torch.zeros(num_struct, device=self.dev)
        for t in range(self.L):
            metric_t = (self.weight**2) * self.scaler[t]
            if axis == 0:
                struct_metric = metric_t.sum(dim=1)
            else:
                struct_metric = metric_t.sum(dim=0)
            _, idx_t = struct_metric.view(-1).topk(k_struct, largest=False)
            counts[idx_t] += 1

        _, idx_final = counts.topk(k_struct, largest=True)
        mask_struct = torch.ones(num_struct, device=self.dev)
        mask_struct[idx_final] = 0.0

        if axis == 0:
            mask_final = mask_struct.view(self.rows, 1).expand(self.rows, self.columns)
        else:
            mask_final = mask_struct.view(1, self.columns).expand(
                self.rows, self.columns
            )

        self.pruned_indices = idx_final
        return self.weight * mask_final

    def free(self):
        self.scaler = None
        torch.cuda.empty_cache()


def st_ts_sequential(model, args, dev, logger, inps):
    for i, layer in enumerate(model.layers):
        logger.info(f"Layer {i}, pruning...")
        if not (args.minlayer <= i < args.maxlayer):
            continue

        layer = layer.to(dev)
        A_log_orig = layer.mixer.A_log.to(dev)
        ssm = SparseSSM_st(A_log_orig, model.seqlen)
        outs = torch.zeros_like(inps)
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0))
            act = torch.stack(layer.mixer.h[0 : model.seqlen], dim=1)
            ssm.add_batch(act.unsqueeze(0))
            layer.mixer.h = []
        A_log_masked = ssm.prune(args.sparsity, logger)
        ssm.free()

        pruned_idx = ssm.pruned_indices.detach().cpu().tolist()
        if pruned_idx:
            orig_n = A_log_masked.shape[1]
            keep_idx = sorted(set(range(orig_n)) - set(pruned_idx))
            new_n = len(keep_idx)
            A_log_new = A_log_masked[:, keep_idx]

            dt_rank = layer.mixer.args.dt_rank
            old_w = layer.mixer.x_proj.weight.data.cpu()
            new_out_dim = dt_rank + 2 * new_n
            new_x_proj = nn.Linear(old_w.shape[1], new_out_dim, bias=False)
            if dt_rank > 0:
                new_x_proj.weight.data[:dt_rank].copy_(old_w[:dt_rank])
            keep_idx_tensor = torch.tensor(keep_idx, dtype=torch.long)
            old_B = old_w[dt_rank : dt_rank + orig_n]
            old_C = old_w[dt_rank + orig_n : dt_rank + 2 * orig_n]
            new_x_proj.weight.data[dt_rank : dt_rank + new_n].copy_(
                old_B[keep_idx_tensor]
            )
            new_x_proj.weight.data[dt_rank + new_n : dt_rank + 2 * new_n].copy_(
                old_C[keep_idx_tensor]
            )
            layer.mixer.x_proj = new_x_proj
            layer.mixer.args.d_state = new_n

            A_log_final = A_log_new
            logger.info(
                f"move {len(pruned_idx)} columes, new shape: {A_log_final.shape}"
            )
        else:
            A_log_final = A_log_masked

        layer.mixer.A_log = torch.nn.Parameter(A_log_final.cpu())
        layer = layer.to(dev)
        inps = inps.to(dev)
        outs = torch.zeros_like(inps).to(dev)
        with torch.no_grad():
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0))
                layer.mixer.h = []
        layer = layer.cpu()
        del layer, ssm
        inps, outs = outs, inps

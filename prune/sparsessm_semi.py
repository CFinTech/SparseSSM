import math
import time
import torch
import torch.nn as nn
import transformers
import matplotlib.pyplot as plt

from utils.model_utils import setup_logger


class SparseSSM_semi:
    def __init__(self, weight: torch.Tensor, L: int):
        self.weight = weight  # (D, N)
        self.rows, self.columns = weight.shape
        self.dev = weight.device
        self.L = L
        self.nsamples = 0
        self.scaler = torch.zeros((self.L, self.rows, self.columns), device=self.dev)

    def add_batch(self, inp: torch.Tensor):
        tmp = inp.shape[0]
        self.scaler *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        self.scaler += torch.sum(inp[0] ** 2, dim=0)  # (L, D, N)

    def _semi_structured_vote(self, prunen: int, prunem: int) -> torch.Tensor:
        assert (
            self.columns % prunem == 0
        ), f"columns ({self.columns}) must be divisible by prunem ({prunem})"
        assert prunen < prunem, "prunen must be smaller than prunem"

        num_groups = self.columns // prunem
        votes = torch.zeros_like(self.weight, dtype=torch.int32, device=self.dev)

        for t in range(self.L):
            metric_t = (self.weight**2) * self.scaler[t]  # (D, N)
            metric_t = metric_t.view(self.rows, num_groups, prunem)  # (D, G, M)
            _, idx_small = torch.topk(metric_t, prunen, dim=2, largest=False)
            vote_block = torch.zeros_like(metric_t, dtype=torch.int32)
            vote_block.scatter_(2, idx_small, 1)
            votes += vote_block.view(self.rows, self.columns)

        votes_block = votes.view(self.rows, num_groups, prunem)
        _, idx_prune = torch.topk(votes_block, prunen, dim=2, largest=True)
        block_mask = torch.ones_like(votes_block, dtype=torch.bool)
        block_mask.scatter_(2, idx_prune, False)
        mask = block_mask.view(self.rows, self.columns).float()
        return mask

    def _unstructured_vote(self, K: int) -> torch.Tensor:
        counts = torch.zeros(self.rows * self.columns, device=self.dev)
        for t in range(self.L):
            metric_t = (self.weight**2) * self.scaler[t]
            _, idx_small = metric_t.view(-1).topk(K, largest=False)
            counts[idx_small] += 1
        _, idx_prune = counts.topk(K, largest=True)
        mask = torch.ones(self.rows * self.columns, device=self.dev)
        mask[idx_prune] = 0.0
        return mask.view(self.rows, self.columns)

    def prune(
        self,
        sparsity: float,
        logger,
        *,
        prunen: int = 0,
        prunem: int = 0,
        heatmap_path: str | None = None,
    ) -> torch.Tensor:
        num_elements = self.rows * self.columns

        if prunen > 0 and prunem > 0:
            mask = self._semi_structured_vote(prunen, prunem)
            return self.weight * mask

        K = int(round(sparsity * num_elements))
        mask = self._unstructured_vote(K)
        return self.weight * mask

    def free(self):
        self.scaler = None
        torch.cuda.empty_cache()


def ts_semi_sequential(model, args, dev, logger, inps):
    for i, layer in enumerate(model.layers):
        logger.info(f"Layer {i}: pruning …")
        if not (args.minlayer <= i < args.maxlayer):
            continue

        layer = layer.to(dev)
        A_log = layer.mixer.A_log.to(dev)
        pruner = SparseSSM_semi(A_log, model.seqlen)

        outs = torch.zeros_like(inps).to(dev)
        with torch.no_grad():
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0))
                act = torch.stack(layer.mixer.h[: model.seqlen], dim=1)
                pruner.add_batch(act.unsqueeze(0))
                layer.mixer.h = []

        A_log_pruned = pruner.prune(
            args.sparsity,
            logger,
            prunen=getattr(args, "prunen", 0),
            prunem=getattr(args, "prunem", 0),
        )
        layer.mixer.A_log = torch.nn.Parameter(A_log_pruned.cpu())
        pruner.free()

        layer.cpu()
        del layer, pruner
        inps, outs = outs, inps

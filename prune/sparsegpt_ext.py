import math
import time
import torch
import torch.nn as nn
import transformers

DEBUG = False


class SparseGPT_ext:
    def __init__(self, param_matrix=None):
        self.dev = param_matrix.device
        self.param_matrix = param_matrix
        W = param_matrix.clone()  # (D, N)
        self.D, self.N = W.shape

        self.H = torch.zeros((self.N, self.N), device=self.dev)  # (N, N)
        self.nsamples = 0

    def add_batch(self, inp, out=None):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out

        tmp = inp.shape[0]
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = inp.reshape((-1, inp.shape[-1]))
        inp = math.sqrt(2 / self.nsamples) * inp.float()  # (N, N)
        self.H += inp.matmul(inp.t())

    def fasterprune(
        self,
        sparsity,
        logger,
        percdamp: float = 0.01,
        blocksize: int = 128,
        prunen: int = 0,
        prunem: int = 0,
    ):
        W = self.param_matrix.clone().float()  # (D, N)
        H = self.H  # (N, N)
        del self.H

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        idx = torch.arange(H.shape[0], device=self.dev)
        H[idx, idx] += damp

        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        Hinv = torch.linalg.cholesky(H, upper=True)  # (N, N)

        Losses = torch.zeros(W.shape[0], device=self.dev)

        for i1 in range(0, H.shape[0], blocksize):
            i2 = min(i1 + blocksize, H.shape[0])
            cnt = i2 - i1

            W_blk = W[:, i1:i2].clone()
            Q_blk = torch.empty_like(W_blk)
            Err_blk = torch.empty_like(W_blk)
            Hinv11 = Hinv[i1:i2, i1:i2]
            Loss_blk = torch.zeros_like(W_blk)

            if prunen > 0 and prunem > 0:
                mask_blk = torch.zeros_like(W_blk, dtype=torch.bool)
                seglen = prunem
                keep = prunen

                for col in range(cnt):
                    w_abs = W_blk[:, col].abs()
                    D = w_abs.numel()
                    for g_start in range(0, D, seglen):
                        g_end = min(g_start + seglen, D)
                        segment = w_abs[g_start:g_end]

                        if segment.numel() <= keep:
                            continue
                        topk_idx = torch.topk(segment, keep, largest=True).indices
                        seg_mask = torch.ones_like(segment, dtype=torch.bool)
                        seg_mask[topk_idx] = False
                        mask_blk[g_start:g_end, col] = seg_mask
            else:
                salience = (W_blk**2) / (torch.diag(Hinv11).reshape((1, -1)) ** 2)
                thresh = torch.sort(salience.flatten())[0][
                    int(salience.numel() * sparsity)
                ]
                mask_blk = salience <= thresh

            for j in range(cnt):
                w_col = W_blk[:, j]
                d_ii = Hinv11[j, j]
                q_col = w_col.clone()
                q_col[mask_blk[:, j]] = 0
                Q_blk[:, j] = q_col

                Loss_blk[:, j] = (w_col - q_col) ** 2 / (d_ii**2)

                err_col = (w_col - q_col) / d_ii
                Err_blk[:, j] = err_col

                W_blk[:, j:] -= err_col.unsqueeze(1).matmul(Hinv11[j, j:].unsqueeze(0))

            W[:, i1:i2] = Q_blk
            Losses += torch.sum(Loss_blk, dim=1) / 2
            W[:, i2:] -= Err_blk.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        self.param_matrix = W
        logger.info(
            f"total 2-nd-order reconstruction error: " f"{torch.sum(Losses).item():.6f}"
        )
        return self.param_matrix

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()


def gpt_sequential(
    model,
    args,
    inps,
    dev,
    logger,
):
    for i, layer in enumerate(model.layers):
        logger.info(f"Layer {i}, Pruning...")
        if not (args.minlayer <= i < args.maxlayer):
            continue

        layer = layer.to(dev)
        A_log = layer.mixer.A_log.to(dev)
        logger.info(f"A_log_origin: {A_log}")
        logger.debug(f"inps: {inps}")
        ssm = SparseGPT_ext(A_log)
        outs = torch.zeros_like(inps)
        with torch.no_grad():
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0))
                h = torch.stack(layer.mixer.h[0 : model.seqlen], dim=1)
                for l in range(model.seqlen):
                    _h = h.permute(0, 1, 3, 2)  # (B, L, N, D)
                    ssm.add_batch(_h[0][l, :, :])  # (1, N, D)
                layer.mixer.h = []

        A_log = ssm.fasterprune(
            args.sparsity, logger, prunen=args.prunen, prunem=args.prunem
        )
        logger.info(f"A_log_pruned: {A_log}")
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

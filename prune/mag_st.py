import torch


def magnitude_prune(
    weight: torch.Tensor, sparsity: float, structured: bool = False, dim: int = 0
) -> torch.Tensor:
    if not structured:
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

    else:
        n_slices = weight.size(dim)
        k = int(n_slices * (1.0 - sparsity))
        if k <= 0:
            return torch.zeros_like(weight)
        if k >= n_slices:
            return weight
        slice_norms = weight.abs().sum(dim=1 if dim == 0 else 0)

        topk = torch.topk(slice_norms, k, largest=True)
        idx = topk.indices

        mask_slices = torch.zeros(n_slices, dtype=torch.bool, device=weight.device)
        mask_slices[idx] = True

        if dim == 0:
            mask = mask_slices.unsqueeze(1).expand_as(weight)
        else:
            mask = mask_slices.unsqueeze(0).expand_as(weight)

        return weight * mask


def mag_st_sequential(model, args, dev, logger):
    for i, layer in enumerate(model.layers):
        if not args.minlayer <= i < args.maxlayer:
            continue

        layer = layer.to(dev)
        A_log = layer.mixer.A_log.to(dev)

        pruned = magnitude_prune(
            A_log,
            sparsity=args.sparsity,
            structured=True,
            dim=getattr(args, "prune_dim", 0),
        )

        layer.mixer.A_log = torch.nn.Parameter(pruned.cpu())
        layer = layer.cpu()
        del layer

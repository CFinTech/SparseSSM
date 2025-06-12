import torch
import torch.nn as nn
import logging
import os

import matplotlib.pyplot as plt
import torch
from typing import Dict, Tuple


def get_mamba(model_path):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    from transformers import MambaForCausalLM, AutoTokenizer

    model = MambaForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    if hasattr(model.config, "max_seq_length"):
        model.seqlen = model.config.max_seq_length
    else:
        model.seqlen = 2048
    return model


def get_custom_mamba(model_path):
    from model.mamba import Mamba

    model = Mamba.from_pretrained(model_path)
    model.seqlen = 2048
    return model


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def setup_logger(name=None, log_file="train.log", log_dir="logs", to_console=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
        )

        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        if to_console:
            logger.addHandler(console_handler)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def allocate_sparsity(
    traces: Dict[Tuple[str, int], float], sizes: Dict[Tuple[str, int], int], args
) -> Dict[Tuple[str, int], float]:

    p = float(args.sparsity)
    alpha = float(getattr(args, "alpha", 0.0))
    default_ret = 1.0 - p
    L_ret = default_ret - alpha
    R_ret = default_ret + alpha

    sensitivity: Dict[Tuple[str, int], float] = {}
    for k, t in traces.items():
        sz = sizes.get(k, 0)
        if sz <= 0:
            raise ValueError(f"Missing or zero size for key {k}")
        sensitivity[k] = t / sz

    in_keys = [k for k in traces if k[0].endswith("in_proj")]
    out_keys = [k for k in traces if k[0].endswith("out_proj")]

    sparsity_alloc: Dict[Tuple[str, int], float] = {}
    for k in traces:
        sparsity_alloc[k] = p

    for group_keys in (in_keys, out_keys):
        m = len(group_keys)
        if m == 0:
            continue

        sens_tensor = torch.tensor([sensitivity[k] for k in group_keys])
        sorted_vals, sorted_idx = torch.sort(sens_tensor)
        ordered_keys = [group_keys[i] for i in sorted_idx.tolist()]

        n = int(getattr(args, "n", 24))
        n = min(n, m // 2)
        if n <= 0:
            continue

        if n == 1:
            bottom_rets = [(L_ret + default_ret) / 2.0]
        else:
            bottom_rets = [
                L_ret + i * (default_ret - L_ret) / (n - 1) for i in range(n)
            ]

        for i in range(n):
            k = ordered_keys[i]
            ret = bottom_rets[i]
            sparsity_alloc[k] = float(1.0 - ret)

        if n == 1:
            top_rets = [(default_ret + R_ret) / 2.0]
        else:
            top_rets = [
                default_ret + i * (R_ret - default_ret) / (n - 1) for i in range(n)
            ]

        for i in range(n):
            k = ordered_keys[-n + i]
            ret = top_rets[i]
            sparsity_alloc[k] = float(1.0 - ret)

    return sparsity_alloc

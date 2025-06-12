import os
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from utils.model_utils import *
from utils.data_utils import *

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

from model.mamba import Mamba


@torch.no_grad()
def mamba_eval(model_path, testenc, dev, dataset, log_dir, logger):
    print("Evaluating ...")
    model = Mamba.from_pretrained(model_path).to(dev).eval()
    input_ids = testenc.input_ids.to(dev)
    total_len = input_ids.numel()
    seqlen = 2048
    nsamples = total_len // seqlen
    input_ids = input_ids[:, : nsamples * seqlen]
    segments = input_ids.view(nsamples, seqlen)
    loss_fct = nn.CrossEntropyLoss(reduction="sum")
    nll_sum = 0.0
    token_count = 0

    for seg in tqdm(segments, desc="PPL Eval"):
        seg = seg.unsqueeze(0)
        model.args.ppl = True
        output = model(seg)
        logits = output

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = seg[:, 1:].contiguous()

        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        nll_sum += loss.item()
        token_count += shift_labels.numel()

    avg_nll = nll_sum / token_count
    ppl = float(np.exp(avg_nll))

    ppl_path = os.path.join(log_dir, "ppl.txt")
    with open(ppl_path, "a") as fout:
        fout.write(f"{dataset} perplexity: {ppl:.6f}\n")

    logger.info(f"Perplexity written to {ppl_path}")

    return ppl

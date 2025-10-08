import os
import time
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

from tqdm import tqdm
from datetime import datetime
from einops import rearrange, repeat, einsum
from eval.eval_zero_shot import evaluate_model
from eval.eval_ppl import mamba_eval

from utils.model_utils import *
from utils.options import *
from utils.data_utils import *

from prune import *

try:
    import wandb

    has_wandb = True
except:
    has_wandb = False


@torch.no_grad()
def mamba_sequential(model, dataloader, dev, logger):
    logger.info("Starting...")
    layers = model.layers
    model.embeddings = model.embeddings.to(dev)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.args.d_model),
        dtype=dtype,
        device=dev,
    )  # (S, L, D)

    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            raise ValueError

    layers[0] = layers[0].to(dev)
    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        try:
            model(batch[0].to(dev))  # (1, S)
        except ValueError:
            torch.cuda.empty_cache()
            pass
    layers[0] = layers[0].module.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)  # (S, L, D)

    str_to_class = {
        "nn.Linear": nn.Linear,
        "nn.Conv1d": nn.Conv1d,
        "nn.Conv2d": nn.Conv2d,
        "nn.LayerNorm": nn.LayerNorm,
    }

    try:
        args.target_module_classes = [
            str_to_class[name] for name in args.target_modules
        ]
    except KeyError as e:
        raise ValueError(
            f"Unsupported module type: {e}. Supported: {list(str_to_class.keys())}"
        )
    logger.info("Ready.")
    

    inps2 = inps.clone()
    if args.prune_A:
        if args.method == "magnitude":
            mag_sequential(model, args, dev, logger)
        elif args.method == "structure_ts":
            st_ts_sequential(model, args, dev, logger, inps)
        elif args.method == "ts_semi":
            ts_semi_sequential(model, args, dev, logger, inps)
        elif args.method == "mag_semi":
            mag_semi_sequential(model, args, dev, logger)
        elif args.method == "structure_mag":
            mag_st_sequential(model, args, dev, logger)
        elif args.method == "gpt_extend":
            gpt_sequential(model, args, inps, dev, logger)
        elif args.method == "sparsessm_dev":
            ts_dev(model, args, dev, logger, inps)
        else:
            raise ValueError(f"Method {args.method} not found!")
    if args.prune_layers:
        if args.method == "magnitude":
            mag_ffn(model, args, dev, logger)
        elif args.method == "sparsessm" or args.method == "sparsegpt" or args.method == "sparsessm_dev":
            ffn_sequential(model, args, dev, logger, inps2)
        else:
            raise ValueError(f"Method {args.method} not found!")


if __name__ == "__main__":
    args = parse_args()
    seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    exp_name = args.experiment_name
    run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.base_dir, exp_name, run_time)
    os.makedirs(log_dir, exist_ok=True)

    if args.log_wandb:
        assert has_wandb, "wandb not installed; try `pip install wandb`"
        wandb.init(
            name=exp_name + "_" + run_time,
            dir=log_dir,
            config=args,
        )
    logger = setup_logger(
        name="main", log_file="output.log", log_dir=log_dir, to_console=args.to_console
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_custom_mamba(args.model_path)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model_path,
        seqlen=model.seqlen,
    )

    if args.do_prune and (args.sparsity or args.prunen):
        tick = time.time()
        mamba_sequential(model, dataloader, device, logger)
        for n, p in model.named_parameters():
            logger.info(f"{n}, {torch.mean((p == 0).float())}")
        logger.info(f"Sequential pruning time:, {time.time() - tick}")

    if args.save:
        model.save_pretrained(model, save_directory=args.save)

    if args.ppl_datasets:
        for dataset in args.ppl_datasets:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model_path, seqlen=model.seqlen
            )
            logger.info(f"Dataset:, {dataset}")
            mamba_eval(args.save, testloader, device, dataset, log_dir, logger)

    if args.eval_zero_shot:
        evaluate_model(args.save, log_dir)


import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("model_path", type=str, help="Path to the Mamba model to load")
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Dataset for calibration and evaluation",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for calibration sampling"
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration samples"
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percentage of the average Hessian diagonal for dampening",
    )
    parser.add_argument("--sparsity", type=float, default=0, help="Target sparsity")
    parser.add_argument("--prunen", type=int, default=0, help="N for N:M pruning")
    parser.add_argument("--prunem", type=int, default=0, help="M for N:M pruning")
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="Blocksize for adaptive mask selection",
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, help="Prune all layers with id >= this"
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, help="Prune all layers with id < this"
    )
    parser.add_argument(
        "--save", type=str, default="", help="Path to save the pruned model"
    )
    parser.add_argument("--log_wandb", action="store_true", help="Log metrics to wandb")
    parser.add_argument("--log_file", type=str, default="logs/H_trace.log")
    parser.add_argument("--experiment_name", type=str, default="test_logging")
    parser.add_argument("--to_console", type=bool, default=False, help="Log to console")
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["nn.Linear", "nn.Conv1d"],
        help="List of module types to prune, e.g. Linear Conv1d",
    )
    parser.add_argument("--prune_layers", type=bool, default=False, help="Prune layers")
    parser.add_argument("--prune_A", type=bool, default=False, help="Prune A")
    parser.add_argument(
        "--method",
        type=str,
        choices=[
            "magnitude",
            "sparsegpt",
            "sparsessm",
            "structure_ts",
            "ts_real",
            "ts_semi",
            "mag_semi",
            "structure_mag",
            "gpt_extend",
            "sparsessm_dev",
            "sparsessm_stru_dev",
            "sparsessm_no_ts"
        ],
        help="Method for pruning",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.0, help="width of the varity of pruning ratio"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="experiments",
        help="Base dir of the experiments results",
    )
    parser.add_argument(
        "--module", type=str, default="None", help="choose which module to prune"
    )
    parser.add_argument(
        "--do_prune", action="store_true"
    )
    parser.add_argument(
        "--ppl_datasets", nargs="+", default=None
    )
    parser.add_argument(
        "--eval_zero_shot", action="store_true"
    )

    args = parser.parse_args()
    return args

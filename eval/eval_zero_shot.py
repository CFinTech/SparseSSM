import os
import json

from lm_eval import evaluator
from lm_eval.models.mamba_lm import MambaLMWrapper
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer, MambaForCausalLM

from utils.model_utils import *

TASKS = ["piqa", "arc_easy", "arc_challenge", "winogrande", "openbookqa"]


def evaluate_model(model_path, log_dir):
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    model = MambaLMHeadModel.from_pretrained(
        model_path, device="cuda", dtype=torch.float16
    )
    model.device = model.lm_head.weight.device
    logger = setup_logger(name="main", log_file="zero_shot.log", log_dir=log_dir)

    logger.info("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    lm = MambaLMWrapper(pretrained=model, tokenizer=tokenizer, batch_size=64)

    logging.info(f"Selected Tasks: {TASKS}")
    results = evaluator.simple_evaluate(lm, tasks=TASKS, log_samples=False)["results"]

    metric_vals = {}
    acc_values = []

    for task, result in results.items():
        res = (
            result["acc,none"]
            if task == "arc_easy"
            else result.get("acc_norm,none", result["acc,none"])
        )
        acc = round(res, 6) * 100
        metric_vals[task] = acc
        acc_values.append(acc)

        if task == "lambada_openai":
            metric_vals[task + "_ppl"] = result["perplexity,none"]

    mean_acc = round(sum(acc_values) / len(acc_values), 2)
    metric_vals["mean_accuracy"] = mean_acc
    logger.info("Evaluation Results:")
    logger.info(json.dumps(metric_vals, indent=4))

    result_file = os.path.join(log_dir, "evaluation_results.json")
    with open(result_file, "w") as f:
        json.dump(metric_vals, f, indent=4)
    logging.info(f"Results saved to {result_file}")

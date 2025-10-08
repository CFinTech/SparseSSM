import os
import json

from lm_eval import evaluator
from lm_eval.models.mamba_lm import MambaLMWrapper
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer, MambaForCausalLM

from utils.model_utils import *
from model.mamba_lmeval_adapter import MambaMinimalLM

TASKS = ["piqa", "arc_easy", "arc_challenge", "winogrande", "openbookqa"]


def evaluate_model(model_path, log_dir):
    logger = setup_logger(name="main", log_file="zero_shot.log", log_dir=log_dir)
    lm = MambaMinimalLM(
        pretrained=model_path,
        tokenizer="EleutherAI/gpt-neox-20b",
        device="cuda",
        batch_size=16,
        dtype="float32"
    )

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
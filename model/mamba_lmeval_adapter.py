from typing import List, Tuple, Any
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

try:
    from lm_eval.api.model import LM 
except Exception:
    from lm_eval.base import BaseLM as LM 

try:
    from lm_eval.api.registry import register_model
except Exception:
    def register_model(_name: str):
        def _decorator(cls):
            return cls
        return _decorator

from model.mamba import Mamba 

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _left_pad(ids: torch.Tensor, total_len: int, pad_id: int) -> torch.Tensor:
    pad_len = total_len - ids.size(0)
    if pad_len <= 0:
        return ids
    return F.pad(ids, (pad_len, 0), value=pad_id)


def _get_attr(obj: Any, name: str, default=None):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    return default


def _extract_loglik_req(req: Any) -> Tuple[str, str]:
    args = _get_attr(req, "arguments", None)
    if args is not None:
        if isinstance(args, (list, tuple)):
            if len(args) == 2:
                return str(args[0]), str(args[1])
        elif isinstance(args, dict):
            ctx = args.get("context", None)
            cont = args.get("continuation", None)
            if ctx is not None and cont is not None:
                return str(ctx), str(cont)

    ctx = _get_attr(req, "context", None)
    cont = _get_attr(req, "continuation", None)
    if ctx is not None and cont is not None:
        return str(ctx), str(cont)

    if isinstance(req, (list, tuple)) and len(req) == 2:
        return str(req[0]), str(req[1])

    if isinstance(req, dict) and "context" in req and "continuation" in req:
        return str(req["context"]), str(req["continuation"])

    raise TypeError(f"Unrecognized loglikelihood request format: {type(req)} -> {req}")


def _extract_gen_req(req: Any):
    args = _get_attr(req, "arguments", None)
    if args is not None:
        if isinstance(args, (list, tuple)):
            if len(args) >= 1:
                ctx = str(args[0])
                until = args[1] if len(args) >= 2 else []
                if isinstance(until, str):
                    until = [until]
                return ctx, list(until), _get_attr(req, "max_gen_toks", None)
        elif isinstance(args, dict):
            ctx = args.get("context", None)
            until = args.get("until", [])
            mgt = args.get("max_gen_toks", None)
            if isinstance(until, str):
                until = [until]
            if ctx is not None:
                return str(ctx), list(until), (int(mgt) if mgt is not None else None)

    ctx = _get_attr(req, "context", None)
    until = _get_attr(req, "until", [])
    mgt = _get_attr(req, "max_gen_toks", None)
    if ctx is not None:
        if isinstance(until, str):
            until = [until]
        return str(ctx), list(until), (int(mgt) if mgt is not None else None)

    if isinstance(req, (list, tuple)):
        if len(req) >= 1:
            ctx = str(req[0])
            until = req[1] if len(req) >= 2 else []
            if isinstance(until, str):
                until = [until]
            return ctx, list(until), None

    if isinstance(req, dict) and "context" in req:
        ctx = req["context"]
        until = req.get("until", [])
        if isinstance(until, str):
            until = [until]
        return str(ctx), list(until), req.get("max_gen_toks", None)

    raise TypeError(f"Unrecognized generate request format: {type(req)} -> {req}")


def _extract_roll_req(req: Any) -> str:
    args = _get_attr(req, "arguments", None)
    if args is not None:
        if isinstance(args, str):
            return args
        if isinstance(args, (list, tuple)) and len(args) >= 1:
            return str(args[0])
        if isinstance(args, dict):
            for k in ("text", "doc", "document", "context"):
                if k in args and isinstance(args[k], (str,)):
                    return str(args[k])
                
    for key in ("text", "doc", "document", "context"):
        val = _get_attr(req, key, None)
        if val is not None:
            return str(val)

    if isinstance(req, str):
        return req

    raise TypeError(f"Unrecognized rolling request format: {type(req)} -> {req}")


@register_model("mamba-minimal")
class MambaMinimalLM(LM):
    def __init__(
        self,
        pretrained: str,  
        tokenizer: str,
        device: str = "cuda",
        dtype: str = "bfloat16", 
        batch_size: int = 16,
        max_length: int = 2048,
        **kwargs,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer, use_fast=True, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        assert self.tokenizer.eos_token_id is not None, "tokenizer need eos_token_id"

        self.EOT_TOKEN_ID = int(self.tokenizer.eos_token_id)
        self.PAD_TOKEN_ID = int(self.tokenizer.pad_token_id)

        self.model = Mamba.from_pretrained(pretrained)
        self.model.eval()
        self.model.args.ppl = True

        if dtype == "float16":
            self.model.to(torch.float16)
        elif dtype == "bfloat16":
            self.model.to(torch.bfloat16)
        else:
            self.model.to(torch.float32)

        req_device = device
        if ("cuda" in device) and (not torch.cuda.is_available()):
            req_device = "cpu"
        self.device = torch.device(req_device)
        self.model.to(self.device)

        self._batch_size = int(batch_size)
        self.max_length = int(max_length)

        self._extra_kwargs = kwargs

    @property
    def eot_token_id(self):
        return self.EOT_TOKEN_ID

    @property
    def max_gen_toks(self):
        return 128

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device_name(self) -> str:
        return str(self.device)

    def tok_encode(self, s: str, add_special_tokens: bool = False) -> List[int]:
        return self.tokenizer.encode(s, add_special_tokens=add_special_tokens)

    def tok_decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t, skip_special_tokens=True)

    def _model_call(self, tok_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model(tok_batch.to(self.device))
        return logits

    def loglikelihood(self, requests: List[Any]) -> List[Tuple[float, bool]]:
        outputs: List[Tuple[float, bool]] = []

        for batch in _chunks(requests, self._batch_size):
            inps, targs = [], []

            for req in batch:
                context, continuation = _extract_loglik_req(req)
                ctx_ids = self.tok_encode(context)
                cont_ids = self.tok_encode(continuation)

                total_len = len(ctx_ids) + len(cont_ids)
                if total_len > self.max_length:
                    overflow = total_len - self.max_length
                    if overflow < len(ctx_ids):
                        ctx_ids = ctx_ids[overflow:]
                    else:
                        join_ids = (ctx_ids + cont_ids)[-self.max_length :]
                        cut = max(0, len(join_ids) - len(cont_ids))
                        ctx_ids, cont_ids = join_ids[:cut], join_ids[cut:]

                inp = torch.tensor(ctx_ids + cont_ids[:-1], dtype=torch.long)
                tgt = torch.tensor(cont_ids, dtype=torch.long)
                inps.append(inp)
                targs.append(tgt)

            max_len = max(x.size(0) for x in inps)
            pad_id = self.PAD_TOKEN_ID
            inps = [_left_pad(x, max_len, pad_id) for x in inps]
            inp_batch = torch.stack(inps, dim=0).to(self.device)

            logits = self._model_call(inp_batch)  
            logprobs = F.log_softmax(logits, dim=-1)

            for i, tgt in enumerate(targs):
                L = tgt.size(0)
                lp_last_L = logprobs[i, -L:, :]
                tgt_lp = lp_last_L.gather(-1, tgt.to(self.device).unsqueeze(-1)).squeeze(-1)
                total = float(tgt_lp.sum().item())
                greedy = bool((lp_last_L.argmax(dim=-1) == tgt.to(self.device)).all().item())
                outputs.append((total, greedy))

        return outputs

    def loglikelihood_rolling(self, requests: List[Any]) -> List[float]:
        results: List[float] = []
        for req in requests:
            text = _extract_roll_req(req)
            ids = self.tok_encode(text)
            if len(ids) <= 1:
                results.append(0.0)
                continue

            total_logprob = 0.0
            step = max(2, self.max_length - 1)
            for start in range(0, len(ids) - 1, step):
                chunk = ids[start : start + self.max_length]
                if len(chunk) < 2:
                    break
                inp_ids = chunk[:-1]
                tgt_ids = chunk[1:]

                inp = torch.tensor([inp_ids], dtype=torch.long, device=self.device)
                logits = self._model_call(inp)[0]
                logprobs = F.log_softmax(logits, dim=-1)
                tgt = torch.tensor(tgt_ids, dtype=torch.long, device=self.device)
                ll_chunk = logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
                total_logprob += float(ll_chunk)

            results.append(total_logprob)

        return results

    def generate_until(self, requests: List[Any]) -> List[str]:
        results: List[str] = []

        for req in requests:
            context, until, req_max_new = _extract_gen_req(req)
            ctx_ids = self.tok_encode(context)
            stop_seqs = until or []
            max_new = int(req_max_new or self.max_gen_toks)

            gen_ids: List[int] = []
            for _ in range(max_new):
                cur = ctx_ids + gen_ids
                if len(cur) >= self.max_length:
                    cur = cur[-self.max_length :]

                inp = torch.tensor([cur], dtype=torch.long, device=self.device)
                logits = self._model_call(inp) 
                next_id = int(logits[0, -1].argmax(dim=-1).item())
                gen_ids.append(next_id)

                if next_id == self.EOT_TOKEN_ID:
                    break
                txt = self.tok_decode(gen_ids)
                if any(s in txt for s in stop_seqs):
                    break

            results.append(self.tok_decode(gen_ids))

        return results

    def greedy_until(self, requests: List[Any]) -> List[str]:
        return self.generate_until(requests)

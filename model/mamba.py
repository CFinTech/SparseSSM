"""Simple, minimal implementation of Mamba in one file of PyTorch.

Suggest reading the following before/while reading the code:
    [1] Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Albert Gu and Tri Dao)
        https://arxiv.org/abs/2312.00752
    [2] The Annotated S4 (Sasha Rush and Sidd Karamcheti)
        https://srush.github.io/annotated-s4

Glossary:
    b: batch size                       (`B` in Mamba paper [1] Algorithm 2)
    l: sequence length                  (`L` in [1] Algorithm 2)
    d or d_model: hidden dim
    n or d_state: latent state dim      (`N` in [1] Algorithm 2)
    expand: expansion factor            (`E` in [1] Section 3.4)
    d_in or d_inner: d * expand         (`D` in [1] Algorithm 2)
    A, B, C, D: state space parameters  (See any state space representation formula)
                                        (B, C are input-dependent (aka selective, a key innovation in Mamba); A, D are not)
    Δ or delta: input-dependent step size
    dt_rank: rank of Δ                  (See [1] Section 3.6 "Parameterization of ∆")

"""

from __future__ import annotations
import os
import math
import json
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from safetensors.torch import save_file
from typing import Union


@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = "auto"
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    ppl = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple
                - self.vocab_size % self.pad_vocab_size_multiple
            )


class Mamba(nn.Module):
    def __init__(self, args: ModelArgs, is_ppl=False):
        """Full Mamba model."""
        super().__init__()
        self.args = args
        self.ppl = is_ppl

        self.embeddings = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)

        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.lm_head.weight = (
            self.embeddings.weight
        )  # Tie output projection to embedding weights.
        # See "Weight Tying" paper

    def forward(self, input_ids):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """
        x = self.embeddings(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        return logits

    # custom function
    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        if os.path.isdir(pretrained_model_name):
            config_path = os.path.join(pretrained_model_name, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(
                    f"in {pretrained_model_name} not found the config.json"
                )
            with open(config_path, "r") as f:
                config_data = json.load(f)
            args = ModelArgs(
                d_model=config_data.get("hidden_size", config_data.get("d_model")),
                n_layer=config_data["n_layer"],
                vocab_size=config_data["vocab_size"],
                d_state=config_data.get("ssm_cfg", {}).get("d_state", 16),
            )
            model = Mamba(args)

            shard_pattern = os.path.join(
                pretrained_model_name, "model-*-of-*.safetensors"
            )
            shard_files = sorted(glob.glob(shard_pattern))
            safetensor_path = os.path.join(pretrained_model_name, "model.safetensors")
            bin_path = os.path.join(pretrained_model_name, "pytorch_model.bin")

            if shard_files:
                try:
                    from safetensors.torch import load_file
                except ImportError as e:
                    raise ImportError(
                        "please run `pip install safetensors` first"
                    ) from e

                state_dict = {}
                for shard in shard_files:
                    shard_dict = load_file(shard, device="cpu")
                    state_dict.update(shard_dict)

            elif os.path.exists(safetensor_path):
                try:
                    from safetensors.torch import load_file
                except ImportError as e:
                    raise ImportError(
                        "please run `pip install safetensors` first"
                    ) from e
                state_dict = load_file(safetensor_path, device="cpu")

            elif os.path.exists(bin_path):

                state_dict = torch.load(bin_path, map_location="cpu")

            else:
                raise FileNotFoundError(
                    f"check {pretrained_model_name}, files not found"
                )

        elif pretrained_model_name.endswith("-hf"):
            hf_model_name = pretrained_model_name[:-3]
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(repo_id=hf_model_name, filename="config.json")
            with open(config_path, "r") as f:
                config_data = json.load(f)
            args = ModelArgs(
                d_model=config_data["hidden_size"],
                n_layer=config_data["n_layer"],
                vocab_size=config_data["vocab_size"],
            )
            model = Mamba(args)

            state_path = hf_hub_download(
                repo_id=hf_model_name, filename="pytorch_model.bin"
            )
            state_dict = torch.load(state_path, map_location="cpu")

        else:
            from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
            from transformers.utils.hub import cached_file

            def load_config_hf(model_name):
                resolved = cached_file(
                    model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False
                )
                return json.load(open(resolved, "r"))

            def load_state_dict_hf(model_name):
                resolved = cached_file(
                    model_name,
                    WEIGHTS_NAME,
                    _raise_exceptions_for_missing_entries=False,
                )
                return torch.load(resolved, map_location="cpu", mmap=True)

            config_data = load_config_hf(pretrained_model_name)
            args = ModelArgs(
                d_model=config_data.get("d_model"),
                n_layer=config_data["n_layer"],
                vocab_size=config_data["vocab_size"],
                d_state=config_data.get("ssm_cfg", {}).get("d_state", 16),
            )
            model = Mamba(args)
            state_dict = load_state_dict_hf(pretrained_model_name)

        new_state_dict = {}
        for key, val in state_dict.items():
            new_key = key.replace("backbone.", "")
            new_state_dict[new_key] = val
        model.load_state_dict(new_state_dict, strict=False)
        return model

    @staticmethod
    def save_pretrained(model, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)

        wrapped_state_dict = {}
        for k, v in model.state_dict().items():
            new_key = k if k.startswith("lm_head") else f"backbone.{k}"
            wrapped_state_dict[new_key] = v.cpu()

        model_file = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(wrapped_state_dict, model_file)
        print(f"Model weights saved to {model_file}")

        config = {
            "d_model": model.args.d_model,
            "n_layer": model.args.n_layer,
            "vocab_size": model.args.vocab_size,
            "ssm_cfg": {"d_state": model.args.d_state},
        }
        config_file = os.path.join(save_directory, "config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to {config_file}")


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """Simple block wrapping Mamba block with normalization and residual connection."""
        super().__init__()
        self.args = args
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)

        Official Implementation:
            Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297

            Note: the official repo chains residual blocks that look like
                [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
            where the first Add is a no-op. This is purely for performance reasons as this
            allows them to fuse the Add->Norm.

            We instead implement our blocks as the more familiar, simpler, and numerically equivalent
                [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....

        """
        output = self.mixer(self.norm(x)) + x

        return output


class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        self.h = []
        self.deltaA = None
        # self.delta = None
        # self.deltaB = None
        # self.C = None

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(
            args.d_inner, args.dt_rank + args.d_state * 2, bias=False
        )

        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), "n -> d n", d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].

        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d)

        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)

        (x, res) = x_and_res.split(
            split_size=[self.args.d_inner, self.args.d_inner], dim=-1
        )

        x = rearrange(x, "b l d_in -> b d_in l")
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, "b d_in l -> b l d_in")

        x = F.silu(x)

        y = self.ssm(x)

        y = y * F.silu(res)

        output = self.out_proj(y)

        return output

    def ssm(self, x):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311

        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)

        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)

        (self.delta, B, C) = x_dbl.split(
            split_size=[self.args.dt_rank, n, n], dim=-1
        )  # delta: (b, l, dt_rank). B, C: (b, l, n)
        self.delta = F.softplus(self.dt_proj(self.delta))  # (b, l, d_in)

        y = self.selective_scan(
            x, self.delta, A, B, C, D
        )  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).

        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)

        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.

        """
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"
        deltaA = torch.exp(einsum(delta, A, "b l d_in, d_in n -> b l d_in n"))
        if self.args.ppl:
            deltaA = deltaA * (A != -1).unsqueeze(0).unsqueeze(0)
        deltaB_u = einsum(delta, B, u, "b l d_in, b l n, b l d_in -> b l d_in n")

        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).

        x = torch.zeros((b, d_in, n), device=deltaA.device)
        if not self.args.ppl:
            self.h.append(x)
            self.deltaA = deltaA
        # self.deltaB = deltaB_u
        # self.C = C

        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], "b d_in n, b n -> b d_in")
            ys.append(y)
            if not self.args.ppl:
                self.h.append(x)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        y = y + u * D

        return y


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = (
            x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        )

        return output

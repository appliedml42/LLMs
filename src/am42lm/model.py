import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from am42lm.configs import ModelConfig
from torch.distributed import _functional_collectives as funcol


class SLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.lm_head = nn.Linear(config.n_embed, config.padded_vocab_size)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embed),
                h=nn.ModuleList(Block(config, i) for i in range(config.n_layer)),
                ln_f=nn.LayerNorm(config.n_embed, eps=config.norm_eps),
            )
        )
        self.max_seq_length = self.config.block_size
        self.mask_cache: torch.Tensor | None = None

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    def reset_parameters(self) -> None:
        self.max_seq_length = self.config.block_size

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        self._max_seq_length = value
        if not hasattr(self, "cos"):
            cos, sin = self.rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        elif value != self.cos.size(0):
            self.cos, self.sin = self.rope_cache(device=self.cos.device)

    def setup_kv_cache(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        for block in self.transformer.h:
            block.attn.build_kv_cache(batch_size, self.max_seq_length, device, dtype)

        max_seq_length = self.max_seq_length
        if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
            ones = torch.ones(
                (max_seq_length, max_seq_length), device=device, dtype=torch.bool
            )
            self.mask_cache = torch.tril(ones).view(
                1, 1, max_seq_length, max_seq_length
            )

    def forward(
        self, idx: torch.Tensor, input_pos: torch.Tensor | None = None
    ) -> torch.Tensor:
        if input_pos is None:
            T = idx.size(1)
            cos = self.cos[:T]
            sin = self.sin[:T]
            mask = None
        else:
            cos = self.cos.index_select(0, input_pos)
            sin = self.sin.index_select(0, input_pos)
            assert self.mask_cache is not None
            mask = self.mask_cache.index_select(2, input_pos)

        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x, cos, sin, mask, input_pos)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)

    def rope_cache(
        self, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        theta = 1.0 / (
            self.config.rope_base
            ** (
                torch.arange(0, self.config.rope_n_elem, 2, device=device).float()
                / self.config.rope_n_elem
            )
        )

        seq_idx = torch.arange(0, self.max_seq_length, device=device).float()

        idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

        return torch.cos(idx_theta), torch.sin(idx_theta)

    @staticmethod
    def from_name(name: str) -> "SLM":
        config = ModelConfig.get_config(name)
        return SLM(config)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embed, config.intermediate_size)
        self.proj = nn.Linear(config.intermediate_size, config.n_embed)
        self.config = copy.deepcopy(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x, approximate="tanh")
        return self.proj(x)

    def apply_tensor_parallel(self):
        rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        assert (
            world_size >= 2
        ), "world_size must be >= 2 for tensor parallelism to make sense."

        fc_weight = self.fc.weight
        fc_local_weight = torch.tensor_split(fc_weight, world_size, dim=0)[rank]
        self.fc.weight = nn.Parameter(fc_local_weight, requires_grad=False)

        if self.fc.bias is not None:
            fc_bias = self.fc.bias
            fc_local_bias = torch.tensor_split(fc_bias, world_size, dim=0)[rank]
            self.fc.bias = nn.Parameter(fc_local_bias, requires_grad=False)

        proj_weight = self.proj.weight
        proj_local_weight = torch.tensor_split(proj_weight, world_size, dim=1)[rank]
        self.proj.weight = nn.Parameter(proj_local_weight, requires_grad=False)

        if self.proj.bias is not None:
            proj_bias = self.proj.bias
            proj_local_bias = proj_bias * 1 / float(world_size)
            self.proj.bias = nn.Parameter(proj_local_bias, requires_grad=False)

        self.register_forward_hook(
            lambda _module, input, output: funcol.all_reduce(
                output, "sum", list(range(world_size))
            )
        )


class Block(nn.Module):
    def __init__(self, config: ModelConfig, index: int) -> None:
        super().__init__()
        self.norm_1 = nn.LayerNorm(config.n_embed, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.index = index

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n1 = self.norm_1(x)
        h = self.attn(n1, cos, sin, mask, input_pos)
        n2 = n1
        ffn = self.mlp(n2)
        return ffn + h + x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = copy.deepcopy(config)
        self.attn = nn.Linear(self.config.n_embed, 3 * self.config.n_embed)
        self.proj = nn.Linear(self.config.n_embed, self.config.n_embed)
        self.kv_cache: KVCache | None = None

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        B, T, _ = x.size()
        qkv = self.attn(x)
        q, k, v = qkv.split(self.config.n_embed, dim=-1)
        q = q.view(B, T, self.config.n_head, self.config.head_size).transpose(1, 2)
        k = k.view(B, T, self.config.n_head, self.config.head_size).transpose(1, 2)
        v = v.view(B, T, self.config.n_head, self.config.head_size).transpose(1, 2)

        q = q.view(B, self.config.n_head, T, self.config.head_size)
        k = k.view(B, self.config.n_head, T, self.config.head_size)
        v = v.view(B, self.config.n_head, T, self.config.head_size)

        q_roped = apply_rope(q[..., : self.config.rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., : self.config.rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., self.config.rope_n_elem :]), dim=-1)
        k = torch.cat((k_roped, k[..., self.config.rope_n_elem :]), dim=-1)

        if input_pos is not None:
            if self.kv_cache is None:
                raise RuntimeError("kv_cache is not initialized")
            k, v = self.kv_cache(input_pos, k, v)

        scale = 1.0 / math.sqrt(self.config.head_size)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, scale=scale, is_causal=mask is None
        )
        y = y.transpose(1, 2).contiguous().view(B, T, self.config.n_embed)
        return self.proj(y)

    def apply_tensor_parallel(self):
        rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        assert (
            world_size >= 2
        ), "world_size must be >= 2 for tensor parallelism to make sense."

        attn_weight = self.attn.weight
        qkv_dim = self.config.n_embed

        q_weight, k_weight, v_weight = attn_weight.split(
            [qkv_dim, qkv_dim, qkv_dim], dim=0
        )
        q_local_weight = torch.tensor_split(q_weight, world_size, dim=0)[rank]
        k_local_weight = torch.tensor_split(k_weight, world_size, dim=0)[rank]
        v_local_weight = torch.tensor_split(v_weight, world_size, dim=0)[rank]
        attn_local_weight = torch.cat(
            (q_local_weight, k_local_weight, v_local_weight), dim=0
        )
        self.attn.weight = nn.Parameter(attn_local_weight, requires_grad=False)

        if self.attn.bias is not None:
            attn_bias = self.attn.bias
            q_bias, k_bias, v_bias = attn_bias.split([qkv_dim, qkv_dim, qkv_dim], dim=0)
            q_local_bias = torch.tensor_split(q_bias, world_size, dim=0)[rank]
            k_local_bias = torch.tensor_split(k_bias, world_size, dim=0)[rank]
            v_local_bias = torch.tensor_split(v_bias, world_size, dim=0)[rank]
            attn_local_bias = torch.cat(
                (q_local_bias, k_local_bias, v_local_bias), dim=0
            )
            self.attn.bias = nn.Parameter(attn_local_bias, requires_grad=False)

        proj_weight = self.proj.weight
        proj_local_weight = torch.tensor_split(proj_weight, world_size, dim=1)[rank]
        self.proj.weight = nn.Parameter(proj_local_weight, requires_grad=False)
        if self.proj.bias is not None:
            proj_bias = self.proj.bias
            proj_local_bias = proj_bias * 1 / float(world_size)
            self.proj.bias = nn.Parameter(proj_local_bias, requires_grad=False)

        self.register_forward_hook(
            lambda _module, input, output: funcol.all_reduce(
                output, "sum", list(range(world_size))
            )
        )

        self.config.n_head = self.config.n_head // world_size
        self.config.n_embed = self.config.n_embed // world_size

    def build_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        v_shape = (
            batch_size,
            self.config.n_head,
            max_seq_length,
            self.config.head_size,
        )
        k_shape = v_shape

        self.kv_cache = KVCache(k_shape, v_shape, device=device, dtype=dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]
    x2 = x[..., head_size // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    roped = (x * cos) + (rotated * sin)
    return roped.type_as(x)


class KVCache(nn.Module):
    def __init__(
        self,
        k_shape: tuple[int, int, int, int],
        v_shape: tuple[int, int, int, int],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "k", torch.zeros(k_shape, device=device, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "v", torch.zeros(v_shape, device=device, dtype=dtype), persistent=False
        )

    def forward(
        self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.k = self.k.to(k.dtype)
        self.v = self.v.to(v.dtype)

        k = self.k.index_copy_(2, input_pos, k)
        v = self.v.index_copy_(2, input_pos, v)

        return k, v

    def register_parameter(self) -> None:
        torch.nn.init.zeros_(self.k)
        torch.nn.init.zeros_(self.v)

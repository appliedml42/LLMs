import itertools
import json
import os
import time

import torch
import torch._inductor.config
import torch.distributed as dist
from configs import ModelConfig
from jsonargparse import CLI
from model import SLM
from tokenizers import Tokenizer as HFTokenizer
from torch.nn import functional as F


def sample(logits, temperature: float = 1.0, top_k: int | None = None):
    logits = logits[0, -1] / max(temperature, 1e-5)
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("inf"), logits)
    probs = F.softmax(logits, dim=-1)

    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def decode_one_token(
    model: SLM, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs
):
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)


def decode_n_token(
    model: SLM,
    curr_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    **sampling_kwargs,
):
    new_tokens = []
    for i in range(num_new_tokens):
        new_token = decode_one_token(model, curr_token, input_pos, **sampling_kwargs)
        input_pos += 1
        new_tokens.append(new_token.clone())
        curr_token = new_token.view(1, -1)

    return new_tokens


def prefill(
    model: SLM, prompt: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs
) -> torch.Tensor:
    logits = model(prompt, input_pos)
    return sample(logits, **sampling_kwargs)[0]


def get_tokenizer(model_name: str, ckpt_dir: str):
    tokenizer_path = os.path.join(ckpt_dir, "tokenizer.json")
    tokenizer = HFTokenizer.from_file(tokenizer_path)

    special_tokens_path = os.path.join(ckpt_dir, "tokenizer_config.json")
    with open(special_tokens_path) as fp:
        config = json.load(fp)
    bos_token = config.get("bos_token")
    eos_token = config.get("eos_token")

    return tokenizer, bos_token, eos_token


def get_model(
    model_name: str,
    ckpt_dir: str,
    device: torch.device,
    precision: torch.dtype,
    use_tp: bool,
):
    config: ModelConfig = ModelConfig.get_config(model_name)
    model = SLM(config)

    checkpoint = torch.load(
        os.path.join(ckpt_dir, "am42_pytorch_model.bin"), mmap=True, weights_only=True
    )
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        # apply_tp(model)
        def apply_custom_method(module):
            if hasattr(module, "apply_tensor_parallel"):
                module.apply_tensor_parallel()

        model.apply(apply_custom_method)

    model = model.to(device=device, dtype=precision)
    return model.eval()


def encode(
    tokenizer: HFTokenizer, string: str, device: torch.device, bos: bool, bos_token: str
):
    tokens = tokenizer.encode(string).ids
    if bos:
        bos_id = tokenizer.token_to_id(bos_token)
        tokens = [bos_id] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)


@torch.no_grad()
def generate(
    model: SLM,
    prompt: torch.Tensor,
    max_new_tokens: int,
    **sampling_kwargs,
):
    T = prompt.size(0)
    T_new = T + max_new_tokens
    max_seq_length = min(T_new, model.config.block_size)
    model.max_seq_length = max_seq_length

    device, dtype = prompt.device, prompt.dtype
    with torch.device(device):
        model.setup_kv_cache(1, device)

    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device, dtype=torch.long)

    next_token = prefill(model, prompt.view(1, -1), input_pos, **sampling_kwargs)
    seq[T] = next_token
    input_pos = torch.tensor([T], device=device)

    generated_tokens = decode_n_token(
        model, next_token.view(1, -1), input_pos, max_new_tokens - 1, **sampling_kwargs
    )
    seq[T + 1 :] = torch.cat(generated_tokens)

    return seq


def maybe_init_dist() -> int | None:
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)

    if world_size < 2:
        return None

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank


def main(
    model_name: str,
    ckpt_dir: str,
    prompt: str = "Hello, my name is",
    compile: bool = False,
    compile_prefill: bool = False,
    num_samples: int = 5,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.0,
):
    global print
    rank = maybe_init_dist()
    use_tp = rank is not None
    if use_tp:
        if rank != 0:
            """
            Override print function to avoid printing on all ranks.
            """
            print = lambda *args, **kwargs: None

    device = torch.device("cuda")
    precision = torch.bfloat16

    model = get_model(model_name, ckpt_dir, device, precision, use_tp)
    tokenizer, bos_token, eos_token = get_tokenizer(model_name, ckpt_dir)

    encoded_prompt = encode(tokenizer, prompt, device, True, bos_token)
    prompt_length = encoded_prompt.size(0)
    model_size = sum(
        [
            p.numel() * p.dtype.itemsize
            for p in itertools.chain(model.parameters(), model.buffers())
        ]
    )
    if compile:
        global decode_one_token, prefill
        decode_one_token = torch.compile(
            decode_one_token, mode="reduce-overhead", fullgraph=True
        )

        if compile_prefill:
            prefill = torch.compile(prefill, dynamic=True, fullgraph=True)

    aggregate_metrics = {"token_per_sec": []}
    start = -1 if compile else 0

    for i in range(start, num_samples):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = generate(
            model,
            encoded_prompt,
            max_new_tokens,
            top_k=top_k,
            temperature=temperature,
        )
        torch.cuda.synchronize()
        t = time.perf_counter() - t0
        if i == -1:
            print(f"Compile time: {t:.02f} sec")
            continue
        if i == num_samples - 1:
            print(tokenizer.decode(y.tolist()))

        tokens_generated = y.size(0) - prompt_length
        token_sec = tokens_generated / t
        aggregate_metrics["token_per_sec"].append(token_sec)
        print(
            f"time for inference {i + 1}: {t:.02f} sec total, {token_sec:.02f} tokens/sec"
        )
        print(f"Bandwidth achieved: {model_size * token_sec / 1e9:.02f} GB/sec")
    print(
        f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['token_per_sec'])).item():.2f}"
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    CLI(main)

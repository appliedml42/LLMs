import functools
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass

import names
import torch
import torch.distributed as dist
from am42lm.configs import (
    AdamConfig,
    AdamWConfig,
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
)
from am42lm.model import SLM, Block
from am42lm.utils import bfloat_support
from datasets import concatenate_datasets, load_dataset
from jsonargparse import ActionConfigFile, ArgumentParser
from torch.distributed.fsdp import (
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics import RunningMean
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase as HFTokenizer

import wandb


@dataclass
class RunConfig:
    model_name: str
    train_config: str
    cache_dir: str
    ckpt_dir: str
    num_proc: int
    run_dir: str
    run_name: str
    world_size: int
    local_rank: int
    use_mixed_precision: bool
    use_fp16: bool


def setup(parser: ArgumentParser):
    args = parser.parse_args()

    model_name = args.model_name
    train_config = args.train_config
    cache_dir = args.cache_dir
    ckpt_dir = args.ckpt_dir
    experiment_dir = args.experiment_dir
    run_name = args.run_name
    num_proc: int = args.num_proc
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    use_mixed_precision = args.use_mixed_precision
    use_fp16 = args.use_fp16

    torch.manual_seed(42 + local_rank * 7)

    run_name = names.get_full_name().replace(" ", "_") if run_name is None else run_name
    run_dir = os.path.join(experiment_dir, run_name)

    if local_rank == 0 and not os.path.exists(run_dir):
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "args.yaml"), "w") as f:
            f.write(parser.dump(args))

    run_config = RunConfig(
        model_name=model_name,
        train_config=train_config,
        cache_dir=cache_dir,
        ckpt_dir=ckpt_dir,
        num_proc=num_proc,
        run_dir=run_dir,
        run_name=run_name,
        world_size=world_size,
        local_rank=local_rank,
        use_mixed_precision=use_mixed_precision,
        use_fp16=use_fp16,
    )

    main(run_config)


def maybe_init_dist():
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)

    if world_size < 2:
        return None
    else:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def get_policies(use_mixed_precision: bool, use_fp16: bool, rank: int):
    mixed_precision_policy = None

    if use_mixed_precision:
        bfloat_available = bfloat_support()
        if bfloat_available and not use_fp16:
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif use_fp16:
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )

            print(f"Using FP16 on rank {rank}.")

    model_auto_wrap = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls={Block}
    )

    return mixed_precision_policy, model_auto_wrap


def get_tokenizer(model_name: str, ckpt_dir: str, chat_template: str):
    model_path = os.path.join(ckpt_dir, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.chat_template is None:
        tokenizer.chat_template = chat_template

    return tokenizer


def loss_fn(logits: torch.Tensor, labels, mask):
    labels = labels.masked_fill(~mask, -100)
    return torch.nn.functional.cross_entropy(logits, labels)


def configure_dataloaders(
    dataset_configs: list[DatasetConfig],
    cache_dir: str,
    shuffle: bool,
    num_proc: int,
    tokenizer: HFTokenizer,
    micro_batch_size: int,
    local_rank: int,
    world_size: int,
    use_distributed_sampler: bool = True,
):
    datasets = []
    for dataset_config in dataset_configs:
        ds = load_dataset(
            dataset_config.name, cache_dir=cache_dir, split=dataset_config.split
        )
        datasets.append(ds.select(range(int(len(ds) * dataset_config.percent))))  # type: ignore
    combined_dataset = concatenate_datasets(datasets)

    def mapping_func(example):
        messages = example["messages"]
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})

        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    combined_dataset = combined_dataset.map(mapping_func, num_proc=num_proc)

    sampler = (
        DistributedSampler(
            combined_dataset,
            rank=local_rank,
            num_replicas=world_size,
            shuffle=shuffle,
        )
        if world_size > 1 and use_distributed_sampler
        else None
    )

    def collate_fn(batch):
        texts = [x["text"] for x in batch]
        tokens = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length + 1,
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        return input_ids, attention_mask

    return DataLoader(
        combined_dataset,  # type: ignore
        batch_size=micro_batch_size,
        collate_fn=collate_fn,
        num_workers=num_proc,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        drop_last=True,
        pin_memory=True,
    )


def get_model(run_config: RunConfig):
    model_name = run_config.model_name
    config: ModelConfig = ModelConfig.get_config(model_name)

    if os.path.exists(os.path.join(run_config.run_dir, "am42_pytorch_model.bin")):
        checkpoint_path = os.path.join(run_config.run_dir, "am42_pytorch_model.bin")
    else:
        checkpoint_path = os.path.join(
            run_config.ckpt_dir, run_config.model_name, "am42_pytorch_model.bin"
        )

    checkpoint = torch.load(checkpoint_path, mmap=True, weights_only=True)
    model = SLM(config)
    model.load_state_dict(checkpoint, strict=True)
    model = torch.compile(model)

    if run_config.world_size > 1:
        mixed_precision_policy, model_auto_wrap_policy = get_policies(
            use_mixed_precision=run_config.use_mixed_precision,
            use_fp16=run_config.use_fp16,
            rank=run_config.local_rank,
        )
        model = FSDP(
            model,  # type: ignore
            auto_wrap_policy=model_auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            use_orig_params=True,
        )
    else:
        print("Using single GPU.")
        model = model.to(torch.cuda.current_device())

    return model


def train(
    training_config: TrainingConfig,
    model: SLM,
    optimizer: torch.optim.Optimizer,
    run_dir: str,
    run_name: str,
    train_dl: DataLoader,
    tokenizer: HFTokenizer,
    local_rank: int,
    world_size: int,
    val_dl: DataLoader | None = None,
):
    total_iters = training_config.num_epochs * len(train_dl)
    total_steps = int(total_iters // training_config.gradient_accumulation_iters)

    warmup_steps = int(0.1 * total_steps)
    scheduler1 = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer, lr_lambda=lambda step: step / warmup_steps
    )
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=total_steps - warmup_steps
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[scheduler1, scheduler2],
        milestones=[warmup_steps],
    )

    if local_rank == 0:
        inner_pbar = tqdm(
            range(total_iters),
            colour="green",
        )

        wandb.login()
        config = dict()
        config.update(training_config.__dict__)
        config.update(model.config.__dict__)
        config["warmup_steps"] = warmup_steps

        wandb.init(
            project="Small Language Models",
            name=run_name,
            dir=run_dir,
            config=config,
            tags=[model.config.hf_repo_id, "full", "sft", "using_torch_compile"],
        )

    update_step = 0
    iter_num = 0
    running_loss = RunningMean(
        window=int(training_config.gradient_accumulation_iters), sync_on_compute=False
    ).to(torch.cuda.current_device())

    model.train()
    for epoch in range(training_config.num_epochs):
        if world_size > 1:
            train_dl.sampler.set_epoch(epoch)  # type: ignore

        for input_ids, attn_mask in train_dl:
            metrics = {}
            iter_t_start = time.perf_counter()

            is_accumulating = (
                iter_num % training_config.gradient_accumulation_iters != 0
            )

            x = input_ids[:, :-1].to(torch.cuda.current_device())
            y = input_ids[:, 1:].reshape(-1).to(torch.cuda.current_device())
            mask = attn_mask[:, 1:].reshape(-1).bool().to(torch.cuda.current_device())

            if world_size > 1 and is_accumulating:
                context_manager = model.no_sync()
            else:
                context_manager = nullcontext()

            with context_manager:
                logits = model(x)
                logits = logits.reshape(-1, logits.size(-1))
                loss = loss_fn(logits, y, mask)
                loss.backward(loss / training_config.gradient_accumulation_iters)

            if not is_accumulating:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                update_step += 1

                if update_step % training_config.val_log_step_interval == 0:
                    val_loss = validate(
                        model, tokenizer, local_rank, world_size, val_dl
                    )
                    if local_rank == 0:
                        metrics["val/loss"] = val_loss.item()  # type: ignore

            iter_num += 1
            if local_rank == 0:
                loss = loss.detach()
                running_loss.update(loss)

                metrics["train/loss"] = running_loss.compute().item()
                metrics["train/learning_rate"] = optimizer.param_groups[0]["lr"]
                metrics["train/update_step"] = update_step
                metrics["train/epoch"] = epoch
                metrics["train/tokens"] = (
                    iter_num
                    * training_config.micro_batch_size
                    * model.config.block_size
                    * world_size
                )
                metrics["train/iter_time"] = time.perf_counter() - iter_t_start
                metrics["train/percent_done"] = 100 * iter_num / total_iters
                wandb.log(metrics, step=iter_num, commit=True)

                inner_pbar.update(1)  # type: ignore
                inner_pbar.set_description(  # type: ignore
                    f"loss {metrics['train/loss']:.4f} epoch {epoch + 1}/{training_config.num_epochs} step {update_step}/{total_steps} accumulating {is_accumulating}"
                )
            if world_size > 1:
                dist.barrier()

    if local_rank == 0:
        wandb.finish()
        inner_pbar.close()  # type: ignore

    save_model(model, tokenizer, local_rank, run_dir)


@torch.no_grad()
def validate(
    model: SLM,
    tokenizer: HFTokenizer,
    local_rank: int,
    world_size: int,
    valid_dl: DataLoader | None = None,
):
    if valid_dl is None:
        return

    def mapping_func(example):
        messages = example["messages"]
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})

        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    total_steps = len(valid_dl)

    model.eval()
    losses = torch.zeros(len(valid_dl), device=torch.cuda.current_device())

    inner_pbar = None
    if local_rank == 0:
        inner_pbar = tqdm(
            range(total_steps),
            colour="yellow",
        )

    for i, (input_ids, attn_mask) in enumerate(valid_dl):
        x = input_ids[:, :-1].to(torch.cuda.current_device())

        y = input_ids[:, 1:]
        y = y.reshape(-1).to(torch.cuda.current_device())

        mask = attn_mask[:, 1:]
        mask = mask.reshape(-1).bool().to(torch.cuda.current_device())

        logits = model(x)

        logits = logits.reshape(-1, logits.size(-1))
        loss = loss_fn(logits, y, mask)

        losses[i] = loss.detach().item()
        if local_rank == 0:
            inner_pbar.update(1)  # type: ignore
            inner_pbar.set_description(  # type: ignore
                f"Step {i + 1}/{total_steps}"
            )

    if local_rank == 0:
        inner_pbar.close()  # type: ignore

    dist.all_reduce(losses, op=dist.ReduceOp.SUM)
    dist.barrier()
    num = losses.sum()  # type: ignore
    den = world_size * losses.size(0)  # type: ignore
    loss = num / den

    model.train()
    return loss


def save_model(model: SLM, tokenizer: HFTokenizer, local_rank: int, run_dir: str):
    fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
    ):
        cpu_state_dict = model.state_dict()

    if local_rank == 0:
        checkpoint_save_path = os.path.join(run_dir, "am42_pytorch_model.bin")
        torch.save(cpu_state_dict, checkpoint_save_path)

        tokenizer.save_pretrained(os.path.join(run_dir, "tokenizer"))


def main(run_config: RunConfig):
    _train_config = TrainingConfig.get_config(run_config.train_config)
    maybe_init_dist()
    tokenizer = get_tokenizer(
        run_config.model_name, run_config.ckpt_dir, _train_config.chat_template
    )

    train_dl = configure_dataloaders(
        _train_config.train_datasets,
        run_config.cache_dir,
        num_proc=run_config.num_proc,
        shuffle=True,
        tokenizer=tokenizer,
        micro_batch_size=_train_config.micro_batch_size,
        local_rank=run_config.local_rank,
        world_size=run_config.world_size,
    )

    val_dl = None
    if _train_config.val_datasets is not None:
        val_dl = configure_dataloaders(
            _train_config.val_datasets,
            run_config.cache_dir,
            num_proc=run_config.num_proc,
            shuffle=False,
            tokenizer=tokenizer,
            micro_batch_size=_train_config.micro_batch_size,
            local_rank=run_config.local_rank,
            world_size=run_config.world_size,
        )

    model = get_model(run_config)

    if isinstance(_train_config.optimizer, AdamConfig):
        optimizer = torch.optim.Adam(
            model.parameters(),
            **_train_config.optimizer.__dict__,
        )
    elif isinstance(_train_config.optimizer, AdamWConfig):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            **_train_config.optimizer.__dict__,
        )
    else:
        raise ValueError(f"Unknown optimizer: {_train_config.optimizer}")

    train(
        training_config=_train_config,
        model=model,  # type: ignore
        optimizer=optimizer,  # type: ignore
        train_dl=train_dl,
        val_dl=val_dl,
        run_dir=run_config.run_dir,
        run_name=run_config.run_name,
        tokenizer=tokenizer,
        local_rank=run_config.local_rank,
        world_size=run_config.world_size,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--train_config", type=str)
    parser.add_argument(
        "--cache_dir", type=str, default="/workspace/downloads/huggingface"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="/workspace/concepts/LLMs/microsoft/phi/checkpoints",
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="/workspace/concepts/LLMs/microsoft/phi/experiments",
    )
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--num_proc", type=int, default=32)
    parser.add_argument("--use_mixed_precision", action="store_true", default=True)
    parser.add_argument("--use_fp16", action="store_true", default=False)
    parser.add_argument("--config", action=ActionConfigFile)
    setup(parser)

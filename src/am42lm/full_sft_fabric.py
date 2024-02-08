import os
import time
from dataclasses import dataclass

import lightning as L
import names
import torch
import wandb
from configs import AdamConfig, AdamWConfig, DatasetConfig, ModelConfig, TrainingConfig
from datasets import concatenate_datasets, load_dataset
from jsonargparse import ActionConfigFile, ArgumentParser
from lightning.fabric.strategies import FSDPStrategy  # type: ignore
from model import SLM, Block
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics import RunningMean
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase as HFTokenizer


@dataclass
class RunConfig:
    model_name: str
    train_config: str
    cache_dir: str
    ckpt_dir: str
    num_proc: int
    num_devices: int
    precision: str
    run_dir: str
    run_name: str


def setup(parser: ArgumentParser):
    args = parser.parse_args()

    model_name = args.model_name
    train_config = args.train_config
    cache_dir = args.cache_dir
    ckpt_dir = args.ckpt_dir
    experiment_dir = args.experiment_dir
    run_name = args.run_name
    num_proc: int = args.num_proc
    num_devices: int = args.num_devices
    precision = args.precision

    run_name = names.get_full_name().replace(" ", "_") if run_name is None else run_name
    run_dir = os.path.join(experiment_dir, run_name)

    if num_devices > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=num_devices, strategy=strategy, precision=precision)
    if fabric.local_rank == 0 and not os.path.exists(run_dir):
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "args.yaml"), "w") as f:
            f.write(parser.dump(args))

    L.Fabric.seed_everything(42 + fabric.local_rank * 7)

    run_config = RunConfig(
        model_name=model_name,
        train_config=train_config,
        cache_dir=cache_dir,
        ckpt_dir=ckpt_dir,
        num_proc=num_proc,
        num_devices=num_devices,
        precision=precision,
        run_dir=run_dir,
        run_name=run_name,
    )
    fabric.launch(main, run_config)  # type: ignore


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
    fabric: L.Fabric,
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
            rank=fabric.local_rank,
            num_replicas=fabric.world_size,
            shuffle=shuffle,
        )
        if fabric.world_size > 1 and use_distributed_sampler
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
    )


def get_model(model_name: str, ckpt_dir: str, run_dir: str, fabric: L.Fabric):
    config: ModelConfig = ModelConfig.get_config(model_name)
    finetuned_model_path = os.path.join(run_dir, "model.pt")
    checkpoint_path = os.path.join(ckpt_dir, model_name, "am42_pytorch_model.bin")

    with fabric.init_module(empty_init=fabric.world_size > 1):
        model = SLM(config)
    model = torch.compile(model)
    model = fabric.setup_module(model)  # type: ignore
    if os.path.exists(finetuned_model_path):
        print(f"Loading finetuned model from {finetuned_model_path}")
        fabric.load(finetuned_model_path, {"model": model})
    else:
        print(f"Loading model from {checkpoint_path}")
        fabric.load_raw(checkpoint_path, model, strict=True)
    return model


def train(
    fabric: L.Fabric,
    training_config: TrainingConfig,
    model: SLM,
    optimizer: torch.optim.Optimizer,
    run_dir: str,
    run_name: str,
    train_dl: DataLoader,
    tokenizer: HFTokenizer,
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

    if fabric.local_rank == 0:
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
    ).to(fabric.device)

    for epoch in range(training_config.num_epochs):
        if fabric.world_size > 1:
            train_dl.sampler.set_epoch(epoch)  # type: ignore

        for input_ids, attn_mask in train_dl:
            metrics = {}
            iter_t_start = time.perf_counter()

            is_accumulating = (
                iter_num % training_config.gradient_accumulation_iters != 0
            )

            x = input_ids[:, :-1]
            x = fabric.to_device(x.pin_memory())

            y = input_ids[:, 1:]
            y = y.reshape(-1)
            y = fabric.to_device(y.pin_memory())

            mask = attn_mask[:, 1:]
            mask = mask.reshape(-1).bool()
            mask = fabric.to_device(mask.pin_memory())

            with fabric.no_backward_sync(model, enabled=is_accumulating):  # type: ignore
                logits = model(x)
                logits = logits.reshape(-1, logits.size(-1))
                loss = loss_fn(logits, y, mask)
                fabric.backward(loss / training_config.gradient_accumulation_iters)

            if not is_accumulating:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                update_step += 1

                if update_step % training_config.val_log_step_interval == 0:
                    val_loss = validate(fabric, model, tokenizer, val_dl)
                    if fabric.local_rank == 0:
                        metrics["val/loss"] = val_loss.item()  # type: ignore

            iter_num += 1
            if fabric.local_rank == 0:
                running_loss.update(loss.detach())

                metrics["train/loss"] = running_loss.compute().item()
                metrics["train/learning_rate"] = optimizer.param_groups[0]["lr"]
                metrics["train/update_step"] = update_step
                metrics["train/epoch"] = epoch
                metrics["train/tokens"] = (
                    iter_num
                    * training_config.micro_batch_size
                    * model.config.block_size
                    * fabric.world_size
                )
                metrics["train/iter_time"] = time.perf_counter() - iter_t_start
                metrics["train/percent_done"] = 100 * iter_num / total_iters
                wandb.log(metrics, step=iter_num, commit=True)

                inner_pbar.update(1)  # type: ignore
                inner_pbar.set_description(  # type: ignore
                    f"loss {metrics['train/loss']:.4f} epoch {epoch + 1}/{training_config.num_epochs} step {update_step}/{total_steps} accumulating {is_accumulating}"
                )
            fabric.barrier()

    if fabric.local_rank == 0:
        wandb.finish()
        inner_pbar.close()  # type: ignore

    save_path = os.path.join(run_dir, "model.pt")
    fabric.print(f"Saving model to {save_path}")
    fabric.save(save_path, {"model": model})


@torch.no_grad()
def validate(
    fabric: L.Fabric,
    model: SLM,
    tokenizer: HFTokenizer,
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
    losses = torch.zeros(len(valid_dl), device=fabric.device)

    inner_pbar = None
    if fabric.local_rank == 0:
        inner_pbar = tqdm(
            range(total_steps),
            colour="yellow",
        )

    for i, (input_ids, attn_mask) in enumerate(valid_dl):
        x = input_ids[:, :-1]
        x = fabric.to_device(x.pin_memory())

        y = input_ids[:, 1:]
        y = y.reshape(-1)
        y = fabric.to_device(y.pin_memory())

        mask = attn_mask[:, 1:]
        mask = mask.reshape(-1).bool()
        mask = fabric.to_device(mask.pin_memory())

        logits = model(x)

        logits = logits.reshape(-1, logits.size(-1))
        loss = loss_fn(logits, y, mask)

        losses[i] = loss.detach().item()
        if fabric.local_rank == 0:
            inner_pbar.update(1)  # type: ignore
            inner_pbar.set_description(  # type: ignore
                f"Step {i + 1}/{total_steps}"
            )

    if fabric.local_rank == 0:
        inner_pbar.close()  # type: ignore

    loss = fabric.all_reduce(losses, reduce_op="sum")
    fabric.barrier()
    num = loss.sum()  # type: ignore
    den = fabric.world_size * loss.size(0)  # type: ignore
    loss = num / den

    model.train()
    return loss


def main(fabric: L.Fabric, run_config: RunConfig):
    _train_config = TrainingConfig.get_config(run_config.train_config)
    tokenizer = get_tokenizer(
        run_config.model_name, run_config.ckpt_dir, _train_config.chat_template
    )

    train_dl = configure_dataloaders(
        _train_config.train_datasets,
        run_config.cache_dir,
        num_proc=run_config.num_proc,
        shuffle=True,
        tokenizer=tokenizer,
        fabric=fabric,
        micro_batch_size=_train_config.micro_batch_size,
    )

    val_dl = None
    if _train_config.val_datasets is not None:
        val_dl = configure_dataloaders(
            _train_config.val_datasets,
            run_config.cache_dir,
            num_proc=run_config.num_proc,
            shuffle=False,
            tokenizer=tokenizer,
            fabric=fabric,
            micro_batch_size=_train_config.micro_batch_size,
        )

    model = get_model(
        run_config.model_name, run_config.ckpt_dir, run_config.run_dir, fabric
    )

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

    optimizer = fabric.setup_optimizers(optimizer)

    train(
        fabric=fabric,
        training_config=_train_config,
        model=model,  # type: ignore
        optimizer=optimizer,  # type: ignore
        train_dl=train_dl,
        val_dl=val_dl,
        run_dir=run_config.run_dir,
        run_name=run_config.run_name,
        tokenizer=tokenizer,
    )


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
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--config", action=ActionConfigFile)

    setup(parser)

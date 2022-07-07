import os
from typing import Optional

import einops
import pytorch_lightning as plm
import torch.nn
import torch.optim
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from pytorch_lightning.utilities.cli import instantiate_class
from torch.nn import functional as F

from .transformer_modules import Embedding, Encoder


@MODEL_REGISTRY
class GPT(plm.LightningModule):
    def __init__(
            self,
            num_layers: int,
            vocab_size: int,
            num_heads: int,
            d_model: int,
            d_ff: int,
            max_seq_len: int,
            dropout: float,
            batch_size: int,
            optimizer_init: Optional[dict] = None,
            lr_scheduler_init: Optional[dict] = None,
            lr_scheduler_interval: Optional[str] = None,
    ):
        super(GPT, self).__init__()
        self.save_hyperparameters()
        self.embedding = Embedding(
            d_model, vocab_size, max_seq_len, enable_padding=True
        )
        self.encoder = Encoder(
            num_layers, num_heads, d_model, d_ff, dropout, max_seq_len, causal_mask=True
        )
        self.output = torch.nn.Linear(d_model, vocab_size)
        self.tokens_processed = 0.0

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.encoder(x, mask=mask)
        x = self.output(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y_true, mask, weight, _ = batch

        self.tokens_processed += torch.sum(weight)

        y_pred = einops.rearrange(
            self.forward(x, mask=mask), "batch seq_len vocab -> (batch seq_len) vocab"
        )
        y_true = einops.rearrange(y_true, "batch seq_len -> (batch seq_len)")

        loss_matrix = einops.rearrange(
            F.cross_entropy(y_pred, y_true, reduction="none"),
            "(batch seq_len) -> batch seq_len",
            batch=self.hparams.batch_size,
            seq_len=self.hparams.max_seq_len,
        )
        loss_matrix = loss_matrix * weight

        loss_overall = torch.sum(loss_matrix) / torch.sum(weight)
        ppl = torch.exp(loss_overall)

        self.log("Train/Loss", loss_overall, sync_dist=True, prog_bar=True)
        self.log("Train/PPL", ppl, sync_dist=True, prog_bar=True)
        self.log("Train/Tokens Processed", self.tokens_processed, sync_dist=True, prog_bar=True, reduce_fx="sum")

        return loss_overall

    def validation_step(self, batch, batch_idx):
        x, y_true, mask, weight, _ = batch

        y_pred = einops.rearrange(
            self.forward(x, mask=mask), "batch seq_len vocab -> (batch seq_len) vocab"
        )
        y_true = einops.rearrange(y_true, "batch seq_len -> (batch seq_len)")

        loss_matrix = einops.rearrange(
            F.cross_entropy(y_pred, y_true, reduction="none"),
            "(batch seq_len) -> batch seq_len",
            batch=self.hparams.batch_size,
            seq_len=self.hparams.max_seq_len,
        )
        loss_matrix = loss_matrix * weight

        loss_overall = torch.sum(loss_matrix) / torch.sum(weight)
        ppl = torch.exp(loss_overall)
        self.log("Validation/Loss", loss_overall, sync_dist=True, prog_bar=True)
        self.log("Validation/PPL", ppl, sync_dist=True, prog_bar=True)

        return loss_overall

    def configure_optimizers(self):
        optimizer = instantiate_class(self.parameters(), self.hparams.optimizer_init)
        scheduler_config = None

        if self.hparams.lr_scheduler_init is not None:
            scheduler = instantiate_class(optimizer, self.hparams.lr_scheduler_init)
            if self.hparams.lr_scheduler_interval is None:
                scheduler_config = [{"scheduler": scheduler}]
            else:
                scheduler_config = [
                    {
                        "scheduler": scheduler,
                        "interval": self.hparams.lr_scheduler_interval,
                    }
                ]

        if scheduler_config is None:
            return [optimizer]
        else:
            return [optimizer], scheduler_config

    def on_train_end(self) -> None:
        save_path = os.path.join(
            self.trainer.default_root_dir,
            f"epoch_{self.current_epoch}_{self.global_step}.ckpt",
        )
        self.trainer.save_checkpoint(save_path, weights_only=True)

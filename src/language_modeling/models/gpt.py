from collections import defaultdict
from typing import Optional

import einops
import pytorch_lightning as plm
import torch.nn
import torch.optim
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from pytorch_lightning.utilities.cli import instantiate_class
from torch.nn import functional as F
import numpy as np
from .transformer_modules import Embedding, Encoder


@MODEL_REGISTRY
class GPT(plm.LightningModule):
    def __init__(self,
                 num_layers: int,
                 vocab_size: int,
                 num_heads: int,
                 d_model: int,
                 max_seq_len: int,
                 dropout: float,
                 batch_size: int,
                 dataset_stats: Optional[dict] = None,
                 optimizer_init: Optional[dict] = None,
                 lr_scheduler_init: Optional[dict] = None,
                 lr_scheduler_interval: Optional[str] = None
                 ):
        super(GPT, self).__init__()
        self.save_hyperparameters()
        self.embedding = Embedding(d_model, vocab_size, max_seq_len, enable_padding=True)
        self.encoder = Encoder(num_layers, num_heads, d_model, dropout, max_seq_len, causal_mask=True)
        self.output = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.encoder(x, mask=mask)
        x = self.output(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y_true, mask, weight, _ = batch

        y_pred = einops.rearrange(self.forward(x, mask=mask), 'batch seq_len vocab -> (batch seq_len) vocab')
        y_true = einops.rearrange(y_true, 'batch seq_len -> (batch seq_len)')

        loss_matrix = einops.rearrange(F.cross_entropy(y_pred, y_true, reduction='none'),
                                       '(batch seq_len) -> batch seq_len',
                                       batch=self.hparams.batch_size,
                                       seq_len=self.hparams.max_seq_len)
        loss_matrix = loss_matrix * weight

        loss_overall = torch.sum(loss_matrix) / torch.sum(weight)
        self.log('Train/Loss', loss_overall)

        return loss_overall

    def validation_step(self, batch, batch_idx):
        dataset_stats = self.hparams.dataset_stats['val']
        x, y_true, mask, weight, datasets = batch

        y_pred = einops.rearrange(self.forward(x, mask=mask), 'batch seq_len vocab -> (batch seq_len) vocab')
        y_true = einops.rearrange(y_true, 'batch seq_len -> (batch seq_len)')

        loss_matrix = einops.rearrange(F.cross_entropy(y_pred, y_true, reduction='none'),
                                       '(batch seq_len) -> batch seq_len',
                                       batch=self.hparams.batch_size,
                                       seq_len=self.hparams.max_seq_len)

        # Make this more elegant :)
        losses = defaultdict(float)
        counts = defaultdict(float)
        loss_matrix = torch.sum(loss_matrix * weight, dim=1)
        count_matrix = torch.sum(weight, dim=1)
        for i, dataset in enumerate(datasets):
            losses[dataset] += loss_matrix[i]
            counts[dataset] += count_matrix[i]
        total = sum(v for k, v in counts.items())
        weights = {k: v / total for k, v in counts.items()}

        pile_loss = 0.0
        pile_bpb = 0.0
        output = {}
        for dataset, loss in losses.items():
            if counts[dataset] == 0:
                continue
            loss = loss / counts[dataset]
            bpb = loss * dataset_stats['num_tokens'][dataset] / float(dataset_stats['num_utf8_bytes'][dataset])
            pile_loss += weights[dataset] * loss
            pile_bpb += weights[dataset] * bpb
            output[dataset] = loss
            if self.trainer.is_global_zero:
                self.log(f'Validation/{dataset}/loss', loss, on_step=False, on_epoch=True,
                         sync_dist=False,
                         batch_size=self.hparams.batch_size, rank_zero_only=True)
                self.log(f'Validation/{dataset}/BPB', bpb / np.log(2), on_step=False, on_epoch=True,
                         sync_dist=False,
                         batch_size=self.hparams.batch_size, rank_zero_only=True)

        self.log('Validation/loss', pile_loss, on_step=False, on_epoch=True, sync_dist=True,
                 batch_size=self.hparams.batch_size)
        self.log('Validation/BPB', pile_bpb / np.log(2), on_step=False, on_epoch=True, sync_dist=True,
                 batch_size=self.hparams.batch_size)

    def configure_optimizers(self):
        optimizer = instantiate_class(self.parameters(), self.hparams.optimizer_init)
        scheduler_config = None
        if self.hparams.lr_scheduler_init is not None:
            scheduler = instantiate_class(optimizer, self.hparams.lr_scheduler_init)
            if self.hparams.lr_scheduler_interval is None:
                scheduler_config = [{'scheduler': scheduler}]
            else:
                scheduler_config = [
                    {
                        'scheduler': scheduler,
                        'interval': self.hparams.lr_scheduler_interval
                    }
                ]

        if scheduler_config is None:
            return [optimizer]
        else:
            return [optimizer], scheduler_config

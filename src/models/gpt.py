import math
from typing import Optional

import einops
import pytorch_lightning as plm
import torch.nn
import torch.optim
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from pytorch_lightning.utilities.cli import instantiate_class
from torch.nn import functional as F

from models.transformer_modules import Embedding, Encoder


@MODEL_REGISTRY
class GPT(plm.LightningModule):
    def __init__(self,
                 num_layers: int,
                 vocab_size: int,
                 num_heads: int,
                 d_model: int,
                 seq_len: int,
                 dropout: float,
                 batch_size: int,
                 optimizer_init: Optional[dict] = None,
                 lr_scheduler_init: Optional[dict] = None,
                 lr_scheduler_interval: Optional[str] = None
                 ):
        super(GPT, self).__init__()
        self.save_hyperparameters()
        self.embedding = Embedding(d_model, vocab_size, seq_len, enable_padding=True)
        self.encoder = Encoder(num_layers, num_heads, d_model, dropout, seq_len, causal_mask=True)
        self.output = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.encoder(x, mask=mask)
        x = self.output(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y_true, mask, weight = batch

        y_pred = einops.rearrange(self.forward(x, mask=mask), 'batch seq_len vocab -> (batch seq_len) vocab')
        y_true = einops.rearrange(y_true, 'batch seq_len -> (batch seq_len)')

        loss_matrix = einops.rearrange(F.cross_entropy(y_pred, y_true, reduction='none'),
                                       '(batch seq_len) -> batch seq_len',
                                       batch=self.hparams.batch_size,
                                       seq_len=self.hparams.seq_len)
        loss_matrix = loss_matrix * weight

        loss_overall = torch.sum(loss_matrix) / torch.sum(weight)

        self.log('Train/Loss', loss_overall)
        self.log('Train/PPL', math.pow(loss_overall / math.log(2), 2))

        for context_length in range(5, self.hparams.seq_len + 1, 50):
            loss_column = torch.sum(loss_matrix[:, context_length - 1]) / torch.sum(weight[:, context_length - 1])

            # Will happen if  that sequence index has no elements.
            if not torch.isnan(loss_column):
                self.log(f'Train/PPL@{context_length}', math.pow(loss_column / math.log(2), 2))
        return loss_overall

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

import pytorch_lightning as plm
import torch.nn
import torch.optim
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from torch.nn import functional as F
from typing import Optional
import einops
from models.transformer_modules import Embedding, Encoder
from models.optimizer_modules import CosineWarmupScheduler
from pytorch_lightning.loggers import WandbLogger
import math


@MODEL_REGISTRY
class GPT(plm.LightningModule):
    def __init__(self,
                 num_layers: int,
                 batch_size: int,
                 vocab_size: int,
                 num_heads: int,
                 d_model: int,
                 seq_len: int,
                 dropout: float,
                 lr: Optional[float] = -1,
                 warmup: Optional[float] = -1,
                 max_iters: Optional[float] = -1):
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

    '''def validation_step(self, batch, batch_idx):
        x, y_true, mask, weights = batch

        y_pred = einops.rearrange(self.forward(x, mask=mask), 'batch seq_len vocab -> batch (seq_len vocab)')
        y_true = einops.rearrange(y_true, 'batch seq_len -> (batch seq_len)')

        loss_matrix = F.cross_entropy(y_pred, y_true, reduction='none')
        loss_matrix = loss_matrix * weights

        loss = torch.sum(loss_matrix) / torch.sum(weights)
        ppl = math.pow(loss / math.log(2), 2)
        self.log('Validation/loss', loss)
        self.log('Validation/ppl', ppl)

        return loss'''

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
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)
        scheduler = CosineWarmupScheduler(optimizer=optimizer,
                                          warmup=self.hparams.warmup,
                                          max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

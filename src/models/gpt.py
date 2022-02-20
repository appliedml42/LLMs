from abc import ABC

import pytorch_lightning as plm
import torch.nn
from models.transformer_modules import Embedding, Encoder
from torch.nn import functional as F
import torch.optim
from pytorch_lightning.utilities.cli import MODEL_REGISTRY


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
                 lr: float):
        super(GPT, self).__init__()
        self.save_hyperparameters()
        self.embedding = Embedding(d_model, vocab_size, seq_len, enable_padding=True)
        self.encoder = Encoder(num_layers, num_heads, d_model, dropout)
        self.output = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.encoder(x, mask=mask)
        x = self.output(x)
        #x = F.softmax(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x, mask=None)
        loss = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), y_true.view(-1))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.hparams.lr)
        return [optimizer]

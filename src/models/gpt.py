import pytorch_lightning as plm
import torch.nn
import torch.optim
from pytorch_lightning.utilities.cli import MODEL_REGISTRY
from torch.nn import functional as F

from models.transformer_modules import Embedding, Encoder


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
        self.encoder = Encoder(num_layers, num_heads, d_model, dropout, seq_len, causal_mask=True)
        self.output = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.encoder(x, mask=mask)
        x = self.output(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y_true, mask = batch
        y_pred = self.forward(x, mask=mask)
        loss = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), y_true.contiguous().view(-1))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.hparams.lr)
        return [optimizer]

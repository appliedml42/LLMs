import pytorch_lightning as plm
import torch.nn
from models.transformer_modules import Embedding, Encoder
from torch.nn import functional as F
import torch.optim

class GPT(plm.LightningModule):
    def __init__(self,
                 num_layers,
                 num_heads,
                 d_model,
                 vocab_size,
                 seq_length,
                 dropout,
                 learning_rate,
                 warmup_steps,
                 tokenizer_fpath):
        super(GPT).__init__()
        self.save_hyperparameters()
        self.embedding = Embedding(d_model, vocab_size, seq_length, enable_padding=True)
        self.encoder = Encoder(num_layers, num_heads, d_model, dropout)
        self.output = torch.nn.Linear(d_model, vocab_size)

    def train_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def predict_dataloader(self):
        pass

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.encoder(x, mask=mask)
        x = self.output(x)
        x = F.softmax(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(x, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters())




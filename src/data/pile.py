import json
import os
from random import shuffle
from typing import Optional

import numpy as np
import sentencepiece as spm
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import IterableDataset, DataLoader
from pytorch_lightning.loggers import WandbLogger

class PileDataset(IterableDataset):
    def __init__(self, fpaths, seq_len, tokenizer: spm.SentencePieceProcessor):
        self.fpaths = fpaths
        self.seq_len = seq_len
        self.tokenizer = tokenizer

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files_per_worker = int(len(self.fpaths) / worker_info.num_workers)
        start_idx = worker_info.id * files_per_worker
        end_idx = start_idx + files_per_worker if worker_info.id < worker_info.num_workers - 1 else len(self.fpaths)
        fpaths = self.fpaths[start_idx:end_idx]
        shuffle(fpaths)

        for fpath in fpaths:
            with open(fpath) as reader:
                for line in reader:
                    text = json.loads(line)['text']
                    ids = self.tokenizer.EncodeAsIds(text)
                    for i in range(0, len(ids), self.seq_len + 1):
                        seq = ids[i:i + self.seq_len + 1]
                        weights = [1] * len(seq)
                        if len(seq) < self.seq_len + 1:
                            weights = weights + [0] * (self.seq_len + 1 - len(seq))
                            seq = seq + [self.tokenizer.pad_id()] * (self.seq_len + 1 - len(seq))
                        yield np.asarray(seq), np.asarray(weights)


@DATAMODULE_REGISTRY
class Pile(LightningDataModule):
    def __init__(self,
                 seq_len: int,
                 batch_size: int,
                 tokenizer_path: str,
                 path: str,
                 num_workers: int):
        super(Pile, self).__init__()
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)
        self.vocab_size = self.tokenizer.vocab_size()

        self.seq_len = seq_len

        self.batch_size = batch_size

        self.num_workers = num_workers

        self.path = path
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            train_path = os.path.join(self.path, 'fit', 'train')
            val_path = os.path.join(self.path, 'fit', 'val')

            self.train_paths = [os.path.join(train_path, x) for x in os.listdir(train_path) if x.endswith('jsonl')]
            shuffle(self.train_paths)
            self.train_dataset = PileDataset(self.train_paths, self.seq_len, self.tokenizer)

            self.val_paths = [os.path.join(val_path, x) for x in os.listdir(val_path) if x.endswith('jsonl')]
            shuffle(self.val_paths)
            self.val_dataset = PileDataset(self.val_paths, self.seq_len, self.tokenizer)

        if stage == 'test':
            self.test_paths = [os.path.join(self.path, x) for x in os.listdir(self.path) if x.endswith('jsonl')]
            shuffle(self.path)
            self.test_dataset = PileDataset(self.path, self.seq_len, self.tokenizer)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        raise ValueError('No prediction dataloader implemented')

    def on_after_batch_transfer(self, batch, dataloader_idx):
        batch, weights = batch
        x, y = batch[:, :-1], batch[:, 1:]
        mask, weights = weights[:, :-1] != 0, weights[:, 1:]
        return x, y, mask.type(torch.uint8), weights

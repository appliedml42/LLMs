import logging
import os
import sqlite3
from typing import Optional, List

import numpy as np
import sentencepiece as spm
import torch
import zstd
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from sortedcontainers import SortedList
from torch.utils.data import DataLoader, Dataset
import pickle
from .utils import compute_seq_and_weight


class PileRandomIODataset(Dataset):
    """
    Used for generating statistics and RandomIO index.
    """

    def __init__(self,
                 fpaths: List[str],
                 max_seq_len: int,
                 pad_id: int):
        self.fpaths = sorted(fpaths)
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

        self.length = 0
        self.index = []
        self.index_keys = SortedList()
        for i, fpath in enumerate(self.fpaths):
            self.index.append(fpath)
            # Store the global index that will be the last element of last db. For DB 0 this is -1.
            self.index_keys.add(self.length - 1)

            conn = sqlite3.connect(fpath)
            num_rows = conn.execute("SELECT COUNT(*) FROM rows").fetchall()[0][0]
            conn.close()

            self.length += num_rows
            logging.warning(f'DB {fpath} has {num_rows} rows. Total rows {self.length}')

    def __len__(self):
        return self.length

    def _get_db_and_idx(self, idx):
        # Identify which DB this key will belong to.
        key = self.index_keys.bisect_left(idx) - 1
        fpath = self.index[key]
        # Calculate key in the local DB.
        db_key = idx - (self.index_keys[key] + 1)
        return fpath, db_key

    def __getitem__(self, idx):
        try:
            fpath, db_idx = self._get_db_and_idx(idx)
            # Open connection for each call. This is not expensive and does not impact speed. Also, kills potential
            # complexity that can arise from keeping many connections open for a long time.
            conn = sqlite3.connect(fpath)
            seq, y_start, dataset = \
            conn.execute('SELECT tokens, y_start, dataset FROM rows WHERE idx == ?', (db_idx,)).fetchall()[0]
            seq = [int(x) for x in zstd.decompress(seq).decode(encoding='ASCII').split()]
            seq, weights = compute_seq_and_weight(seq, y_start, self.max_seq_len, self.pad_id)
            # Close connection
            conn.close()
            return np.asarray(seq), np.asarray(weights), dataset
        except:
            logging.error(f'idx: {idx} ')
            raise


@DATAMODULE_REGISTRY
class Pile(LightningDataModule):
    def __init__(self,
                 max_seq_len: int,
                 context_len: int,
                 batch_size: int,
                 tokenizer_path: str,
                 path: str):
        super(Pile, self).__init__()
        self.tokenizer_path = tokenizer_path
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)
        self.vocab_size = self.tokenizer.vocab_size()

        self.max_seq_len = max_seq_len
        self.context_len = context_len
        self.batch_size = batch_size
        self.path = path

        # Load Dataset Stats
        self.dataset_stats = {

        }
        with open(os.path.join(self.path, 'train', f'{self.max_seq_len}_{self.context_len}_stats.pickle'),
                  'rb') as handle:
            self.dataset_stats['train'] = pickle.load(handle)
        with open(os.path.join(self.path, 'val', f'{self.max_seq_len}_{self.context_len}_stats.pickle'),
                  'rb') as handle:
            self.dataset_stats['val'] = pickle.load(handle)

        self.train_dataset = None
        self.val_dataset = None

        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            train_path = os.path.join(self.path, 'train')
            val_path = os.path.join(self.path, 'val')

            train_paths = sorted(
                [os.path.join(train_path, x) for x in os.listdir(train_path) if
                 x.endswith(f'{self.max_seq_len}_{self.context_len}.db')])
            self.train_dataset = PileRandomIODataset(train_paths, self.max_seq_len, self.tokenizer.pad_id())

            val_paths = sorted(
                [os.path.join(val_path, x) for x in os.listdir(val_path) if
                 x.endswith(f'{self.max_seq_len}_{self.context_len}.db')])
            self.val_dataset = PileRandomIODataset(val_paths, self.max_seq_len, self.tokenizer.pad_id())

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=30,
                          drop_last=True,
                          shuffle=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=30,
                          drop_last=True)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        raise ValueError('No prediction dataloader implemented')

    def on_after_batch_transfer(self, batch, dataloader_idx):
        batch, weights, dataset = batch
        x, y = batch[:, :-1], batch[:, 1:]
        mask, weights = weights[:, :-1] != 0, weights[:, 1:]
        return x, y, mask.type(torch.uint8), weights, dataset
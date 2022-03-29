import json
import os
import pickle
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool
from random import shuffle
from typing import Optional

import numpy as np
import sentencepiece as spm
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import IterableDataset, DataLoader


def stat_worker(fpath, seq_len, tokenizer_path):
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)

    i = 0
    num_rows = defaultdict(float)
    num_nzt = defaultdict(float)
    num_t = defaultdict(float)

    dataset = PileDataset([fpath], seq_len, tokenizer)
    for _, _, non_zero_tokens, pile_set_name in dataset:
        i += 1
        num_rows[pile_set_name] += 1
        num_nzt[pile_set_name] += non_zero_tokens
        num_t[pile_set_name] += seq_len

    return dict(num_rows), dict(num_t), dict(num_nzt), dict(dataset.num_utf8_bytes)


class PileDataset(IterableDataset):
    def __init__(self, fpaths,
                 seq_len,
                 tokenizer: spm.SentencePieceProcessor):
        self.fpaths = fpaths
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.num_utf8_bytes = defaultdict(float)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            files_per_worker = int(len(self.fpaths) / worker_info.num_workers)
            start_idx = worker_info.id * files_per_worker
            end_idx = start_idx + files_per_worker if worker_info.id < worker_info.num_workers - 1 else len(self.fpaths)
            fpaths = self.fpaths[start_idx:end_idx]
        else:
            fpaths = self.fpaths

        shuffle(fpaths)

        for fpath in fpaths:
            writer = open(f'{fpath}.log', 'w')
            writer.write(f'Starting {fpath}\n')
            writer.flush()
            lines = 0
            with open(fpath) as reader:
                for line in reader:
                    obj = json.loads(line)
                    text = obj['text']
                    pile_set_name = obj['meta']['pile_set_name']
                    self.num_utf8_bytes[pile_set_name] += len(text.encode(encoding='utf-8'))
                    ids = self.tokenizer.EncodeAsIds(text) + [self.tokenizer.PieceToId('[eod]')]
                    for i in range(0, len(ids), self.seq_len + 1):
                        seq = ids[i:i + self.seq_len + 1]
                        weights = [1] * len(seq)
                        non_zero_tokens = len(seq)
                        if len(seq) < self.seq_len + 1:
                            weights = weights + [0] * (self.seq_len + 1 - len(seq))
                            seq = seq + [self.tokenizer.pad_id()] * (self.seq_len + 1 - len(seq))

                        yield np.asarray(seq), np.asarray(weights), non_zero_tokens, pile_set_name
                    lines += 1
                    if lines % 10000 == 0:
                        writer.write(f'Finished {lines} of {fpath} timestamp {datetime.now()}\n')
                        writer.flush()
                writer.write(f'Finished {fpath}\n')
                writer.close()


@DATAMODULE_REGISTRY
class Pile(LightningDataModule):
    def __init__(self,
                 seq_len: int,
                 batch_size: int,
                 tokenizer_path: str,
                 path: str,
                 num_workers: int):
        super(Pile, self).__init__()
        self.tokenizer_path = tokenizer_path
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
            self.test_paths = [os.path.join(self.path, 'test', x) for x in os.listdir(os.path.join(self.path, 'test'))
                               if x.endswith('jsonl')]
            shuffle(self.test_paths)
            self.test_dataset = PileDataset(self.test_paths, self.seq_len, self.tokenizer)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        raise ValueError('No prediction dataloader implemented')

    def generate_statistics(self, stage: Optional[str]):
        paths = None
        args = None
        if stage == 'train':
            dpath = os.path.join(self.path, 'fit', 'train')
            paths = [os.path.join(dpath, x) for x in os.listdir(dpath) if x.endswith('jsonl')]
            args = [(fpath, self.seq_len, self.tokenizer_path) for fpath in paths]
        elif stage == 'val':
            dpath = os.path.join(self.path, 'fit', 'val')
            paths = [os.path.join(dpath, x) for x in os.listdir(dpath) if x.endswith('jsonl')]
            args = [(fpath, self.seq_len, self.tokenizer_path) for fpath in paths]
        elif stage == 'test':
            dpath = os.path.join(self.path, 'test')
            paths = [os.path.join(dpath, x) for x in os.listdir(dpath) if x.endswith('jsonl')]
            args = [(fpath, self.seq_len, self.tokenizer_path) for fpath in paths]

        with Pool(len(paths)) as p:
            out = p.starmap(stat_worker, args)

            num_rows = defaultdict(float)
            num_tokens = defaultdict(float)
            num_nz_tokens = defaultdict(float)
            num_utf8_bytes = defaultdict(float)

            for nr, nt, nzt, nub in [(x[0], x[1], x[2], x[3]) for x in out]:
                for k, v in nr.items():
                    num_rows[k] += v

                for k, v in nt.items():
                    num_tokens[k] += v

                for k, v in nzt.items():
                    num_nz_tokens[k] += v

                for k, v in nub.items():
                    num_utf8_bytes[k] += v

            stats = {
                'stage': stage,
                'seq_len': self.seq_len,
                'tokenizer_path': self.tokenizer_path,
                'num_rows': dict(num_rows),
                'num_tokens': dict(num_tokens),
                'num_nz_tokens': dict(num_nz_tokens),
                'num_utf8_bytes': dict(num_utf8_bytes)
            }

            print(stats)

            with open(f'{stage}_{self.seq_len}_stats.pickle', 'wb') as handle:
                pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        batch, weights, non_zero_tokens, pile_set_name = batch
        x, y = batch[:, :-1], batch[:, 1:]
        mask, weights = weights[:, :-1] != 0, weights[:, 1:]
        return x, y, mask.type(torch.uint8), weights, non_zero_tokens, pile_set_name


pile = Pile(
    seq_len=1024,
    batch_size=128,
    path='/workspace/data/pile',
    tokenizer_path='/workspace/data/pile/tokenizer/50000.model',
    num_workers=1
)

pile.generate_statistics('test')
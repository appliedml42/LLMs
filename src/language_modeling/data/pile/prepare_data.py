import argparse
import json
import logging
import os
import pickle
import sqlite3
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool
from utils import get_rolling_token_windows
import sentencepiece as spm
import zstd
from torch.utils.data import IterableDataset


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, required=True)
    parser.add_argument("--context_len", type=int, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    return parser


def worker(fpath: str, max_seq_len: int, context_len: int, tokenizer_path: str):
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)

    i = 0
    num_tokens = defaultdict(float)
    num_utf8_bytes = defaultdict(float)
    num_rows = defaultdict(float)

    dataset = PileFileDataset(fpath, max_seq_len, context_len, tokenizer)
    for num_toks, num_utf8_byts, num_rws, dataset in dataset:
        i += 1
        num_tokens[dataset] += num_toks
        num_utf8_bytes[dataset] += num_utf8_byts
        num_rows[dataset] += num_rws

    return dict(num_rows), dict(num_tokens), dict(num_utf8_bytes),


class PileFileDataset(IterableDataset):
    """
    Used for generating statistics and RandomIO index.
    """

    def __init__(self, fpath: str, max_seq_len: int, context_len: int, tokenizer: spm.SentencePieceProcessor):
        self.fpath = fpath
        self.max_seq_len = max_seq_len
        self.context_len = context_len
        self.tokenizer = tokenizer
        self.con = sqlite3.connect(f'{fpath}_{self.max_seq_len}_{self.context_len}.db')

    def __iter__(self):
        logging.warning(f'Starting {self.fpath}\n')
        curr = self.con.cursor()
        create_cmd = "CREATE TABLE rows (" \
                     "idx INT PRIMARY KEY, " \
                     "local_idx INT, " \
                     "fpath TEXT, " \
                     "dataset TEXT, " \
                     "tokens BLOB, " \
                     "y_start INT)"
        curr.execute(create_cmd)
        insert_cmd = "INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?)"

        idx = 0
        lines = 0
        with open(self.fpath, 'r') as reader:
            fkey = '/'.join(self.fpath.split('/')[-2:])
            for line in reader:
                obj = json.loads(line)
                text = obj['text'].strip()
                dataset = obj['meta']['pile_set_name']
                ids = self.tokenizer.EncodeAsIds(text) + [self.tokenizer.PieceToId('[eod]')]
                if len(ids) <= 2:
                    continue
                gen = get_rolling_token_windows(token_list=ids,
                                                prefix_token=self.tokenizer.PieceToId('[eod]'),
                                                max_seq_len=self.max_seq_len,
                                                context_len=self.context_len)

                local_idx = 0
                for x, y in gen:
                    seq = x + [y[-1]]
                    y_start = len(seq) - len(y)
                    seq = ' '.join(str(x) for x in seq)
                    compressed_seq = zstd.compress(seq.encode(encoding='ASCII'))
                    curr.execute(insert_cmd, (idx, local_idx, fkey, dataset, compressed_seq, y_start))
                    local_idx += 1
                    idx += 1
                yield len(ids), len(text.encode(encoding='utf-8')), local_idx, dataset
                lines += 1
                if lines % 10000 == 0:
                    logging.warning(
                        f'Finished lines {lines} to produce rows {idx} of {self.fpath} timestamp {datetime.now()}\n')
                    self.con.commit()
            self.con.commit()
            self.con.close()
            logging.warning(f'Finished {self.fpath} with {lines}\n')


def prepare_data(stage, path, max_seq_len, context_len, tokenizer_path):
    paths = None
    dpath = None
    args = None
    if stage == 'train':
        dpath = os.path.join(path, 'train')
        paths = [os.path.join(dpath, x) for x in os.listdir(dpath) if x.endswith('jsonl')]
        args = [(fpath, max_seq_len, context_len, tokenizer_path) for fpath in paths]
    elif stage == 'val':
        dpath = os.path.join(path, 'val')
        paths = [os.path.join(dpath, x) for x in os.listdir(dpath) if x.endswith('jsonl')]
        args = [(fpath, max_seq_len, context_len, tokenizer_path) for fpath in paths]
    elif stage == 'test':
        dpath = os.path.join(path, 'test')
        paths = [os.path.join(dpath, x) for x in os.listdir(dpath) if x.endswith('jsonl')]
        args = [(fpath, max_seq_len, context_len, tokenizer_path) for fpath in paths]

    paths = paths
    with Pool(len(paths)) as p:
        out = p.starmap(worker, args)
        num_rows = defaultdict(float)
        num_tokens = defaultdict(float)
        num_utf8_bytes = defaultdict(float)

        for nr, nt, nub in [(x[0], x[1], x[2]) for x in out]:
            for k, v in nr.items():
                num_rows[k] += v

            for k, v in nt.items():
                num_tokens[k] += v

            for k, v in nub.items():
                num_utf8_bytes[k] += v

        stats = {
            'stage': stage,
            'max_seq_len': max_seq_len,
            'context_len': context_len,
            'tokenizer_path': tokenizer_path,
            'num_rows': dict(num_rows),
            'num_tokens': dict(num_tokens),
            'num_utf8_bytes': dict(num_utf8_bytes),
        }

        logging.info(stats)

        with open(os.path.join(dpath, f'{max_seq_len}_{context_len}_stats.pickle'), 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = get_args_parser()
    cmd = parser.parse_args()

    logging.warning('Preparing data for validation')
    prepare_data('val', cmd.path, cmd.max_seq_len, cmd.context_len, cmd.tokenizer_path)

    logging.warning('Preparing data for test')
    prepare_data('test', cmd.path, cmd.max_seq_len, cmd.context_len, cmd.tokenizer_path)

    logging.warning('Preparing data for train')
    prepare_data('train', cmd.path, cmd.max_seq_len, cmd.context_len, cmd.tokenizer_path)

import argparse
import json
import logging
import os
import pickle
import sqlite3
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool

import sentencepiece as spm
import zstd
from torch.utils.data import IterableDataset


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    return parser


def worker(fpath, seq_len, tokenizer_path):
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)

    i = 0
    num_rows = defaultdict(float)
    num_nzt = defaultdict(float)
    num_t = defaultdict(float)

    dataset = PileFileDataset(fpath, seq_len, tokenizer)
    for non_zero_tokens, pile_set_name in dataset:
        i += 1
        num_rows[pile_set_name] += 1
        num_nzt[pile_set_name] += non_zero_tokens
        num_t[pile_set_name] += seq_len

    return dict(num_rows), dict(num_t), dict(num_nzt), dict(dataset.num_utf8_bytes)


class PileFileDataset(IterableDataset):
    """
    Used for generating statistics and RandomIO index.
    """

    def __init__(self, fpath, seq_len, tokenizer: spm.SentencePieceProcessor):
        self.fpath = fpath
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.num_utf8_bytes = defaultdict(float)
        self.con = sqlite3.connect(f'{fpath}_{self.seq_len}.db')

    def __iter__(self):
        logging.warning(f'Starting {self.fpath}\n')
        curr = self.con.cursor()
        create_cmd = "CREATE TABLE rows (idx INT PRIMARY KEY, local_idx INT, fpath TEXT, dataset TEXT, tokens BLOB)"
        curr.execute(create_cmd)

        insert_cmd = "INSERT INTO rows VALUES (?, ?, ?, ?, ?)"

        idx = 0
        lines = 0
        with open(self.fpath, 'r') as reader:
            fkey = '/'.join(self.fpath.split('/')[-2:])
            for line in reader:
                obj = json.loads(line)
                text = obj['text']
                dataset = obj['meta']['pile_set_name']
                self.num_utf8_bytes[dataset] += len(text.encode(encoding='utf-8'))
                ids = self.tokenizer.EncodeAsIds(text) + [self.tokenizer.PieceToId('[eod]')]
                local_idx = 0
                for i in range(0, len(ids), self.seq_len + 1):
                    seq = ids[i:i + self.seq_len + 1]
                    seq = ' '.join(str(x) for x in seq)
                    compressed_seq = zstd.compress(seq.encode(encoding='ASCII'))
                    curr.execute(insert_cmd, (idx, local_idx, fkey, dataset, compressed_seq))
                    local_idx += 1
                    idx += 1
                    yield len(seq), dataset
                lines += 1
                if lines % 10000 == 0:
                    logging.warning(
                        f'Finished lines {lines} to produce rows {idx} of {self.fpath} timestamp {datetime.now()}\n')
                    self.con.commit()
            self.con.commit()
            self.con.close()
            logging.warning(f'Finished {self.fpath}\n')


def prepare_data(stage, path, seq_len, tokenizer_path):
    paths = None
    dpath = None
    args = None
    if stage == 'train':
        dpath = os.path.join(path, 'train')
        paths = [os.path.join(dpath, x) for x in os.listdir(dpath) if x.endswith('jsonl')]
        args = [(fpath, seq_len, tokenizer_path) for fpath in paths]
    elif stage == 'val':
        dpath = os.path.join(path, 'val')
        paths = [os.path.join(dpath, x) for x in os.listdir(dpath) if x.endswith('jsonl')]
        args = [(fpath, seq_len, tokenizer_path) for fpath in paths]
    elif stage == 'test':
        dpath = os.path.join(path, 'test')
        paths = [os.path.join(dpath, x) for x in os.listdir(dpath) if x.endswith('jsonl')]
        args = [(fpath, seq_len, tokenizer_path) for fpath in paths]

    with Pool(len(paths)) as p:
        out = p.starmap(worker, args)
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
            'seq_len': seq_len,
            'tokenizer_path': tokenizer_path,
            'num_rows': dict(num_rows),
            'num_tokens': dict(num_tokens),
            'num_nz_tokens': dict(num_nz_tokens),
            'num_utf8_bytes': dict(num_utf8_bytes),
        }

        logging.info(stats)

        with open(os.path.join(dpath, f'{seq_len}_stats.pickle'), 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = get_args_parser()
    cmd = parser.parse_args()

    logging.warning('Preparing data for validation')
    prepare_data('val', cmd.path, cmd.seq_len, cmd.tokenizer_path)

    logging.warning('Preparing data for test')
    #prepare_data('test', cmd.path, cmd.seq_len, cmd.tokenizer_path)

    logging.warning('Preparing data for train')
    # prepare_data('train', cmd.path, cmd.seq_len, cmd.tokenizer_path)

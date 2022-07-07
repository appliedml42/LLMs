import argparse
import io
import json
import logging
import os
import sqlite3
from multiprocessing import Pool
from typing import List, Optional

import sentencepiece as spm
import zstandard

from utils import get_rolling_token_windows

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s", level=logging.INFO
)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--read_directory", type=str, required=True)
    parser.add_argument("--write_directory", type=str, required=True)
    parser.add_argument("--max_seq_len", type=int, required=True)
    parser.add_argument("--context_len", type=int, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--datasets", type=str, required=False, default=None)

    return parser


def prepare_data_worker(
    read_path: str,
    write_path: str,
    max_seq_len: int,
    context_len: int,
    tokenizer_path: str,
    datasets: Optional[List[str]] = None,
) -> None:
    logging.info(f"Preparing data from {read_path}")

    connection = sqlite3.connect(write_path)
    cursor = connection.cursor()
    cursor.execute(
        "CREATE TABLE rows (id INTEGER PRIMARY KEY, dataset TEXT, seq BLOB, pred_start INTEGER)"
    )

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)

    with open(read_path, "rb") as fh:
        dctx = zstandard.ZstdDecompressor()
        data_stream = io.TextIOWrapper(dctx.stream_reader(fh), encoding="utf-8")
        compressor = zstandard.ZstdCompressor()
        idx = 0
        for line in data_stream:
            obj = json.loads(line)

            dataset = obj["meta"]["pile_set_name"]
            if datasets is not None and dataset not in datasets:
                continue

            text = obj["text"].strip()
            tokens = tokenizer.EncodeAsIds(text)
            if len(tokens) <= 5:
                continue

            for input_tokens, pred_tokens in get_rolling_token_windows(
                tokens, tokenizer.PieceToId("[eod]"), max_seq_len, context_len
            ):
                seq = input_tokens + pred_tokens[-1:]
                pred_start = len(seq) - len(pred_tokens)
                compressed_seq = compressor.compress(
                    " ".join([str(x) for x in seq]).encode(encoding="ASCII")
                )
                cursor.execute(
                    "INSERT INTO rows (id, dataset, seq, pred_start) VALUES (?, ?, ?, ?)",
                    (idx, dataset, compressed_seq, pred_start),
                )
                idx += 1
                if idx % 10000 == 0:
                    logging.info(f"Added {idx} rows from {read_path}")
    connection.commit()
    connection.close()
    logging.info(
        f"Finished preparing data from {read_path} at {write_path} and produced {idx} rows"
    )


def prepare_data(
    read_directory: str,
    write_directory: str,
    stage: str,
    max_seq_len: str,
    context_len: str,
    tokenizer_path: str,
    datasets: Optional[List[str]] = None,
):
    logging.info(f"Preparing data from {read_directory} and stage {stage}")

    read_directory = os.path.join(read_directory, stage)
    read_paths = [
        os.path.join(read_directory, x)
        for x in os.listdir(read_directory)
        if x.endswith(".jsonl.zst")
    ]

    write_directory = os.path.join(write_directory, stage)
    os.makedirs(write_directory, exist_ok=True)
    write_paths = [
        os.path.join(write_directory, f"{x}.db")
        for x in os.listdir(read_directory)
        if x.endswith("jsonl.zst")
    ]

    args = [
        (x, y, max_seq_len, context_len, tokenizer_path, datasets)
        for x, y in zip(read_paths, write_paths)
    ]

    with Pool(os.cpu_count()) as p:
        p.starmap(prepare_data_worker, args)

    logging.info(f"Finished preparing data from {read_directory} and stage {stage}")


if __name__ == "__main__":
    parser = get_args_parser()
    cfg = parser.parse_args()

    if cfg.datasets is not None:
        cfg.datasets = cfg.datasets.split(",")

    prepare_data(
        cfg.read_directory,
        cfg.write_directory,
        "train",
        cfg.max_seq_len,
        cfg.context_len,
        cfg.tokenizer_path,
        cfg.datasets,
    )

    prepare_data(
        cfg.read_directory,
        cfg.write_directory,
        "val",
        cfg.max_seq_len,
        cfg.context_len,
        cfg.tokenizer_path,
        cfg.datasets,
    )

    with open(os.path.join(cfg.write_directory, "config.json"), "w") as fh:
        json.dump(vars(cfg), fh)

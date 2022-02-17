import argparse
import logging
import os
from argparse import Namespace
import sentencepiece as spm
import pandas as pd
import torch
from torch.utils.data import Dataset


class Pile(Dataset):
    def __init__(self, *args):
        if len(args) == 2:
            generic_args, other_args = args
            self.parser = argparse.ArgumentParser()
            self.parser.add_argument('--dpath', type=str, required=True)
            self.params, _ = self.parser.parse_known_args(other_args, namespace=generic_args)
            self._initialize()
        elif isinstance(args[0], Namespace):
            self.params = args[0]

        self.tokenizer = None

    def _initialize(self):
        self.dataset = pd.read_json(os.path.join(self.params.dpath, 'df.json'), lines=True).reset_index()
        self.image_size = self.params.image_size
        self.text_seq_len = self.params.text_seq_len
        self.tokenizer = tokenizer
        self.params.vocab_size = self.tokenizer.vocab_size
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset)

    def tokenize(self, text):
        return self.tokenizer.tokenize(text,
                                       self.text_seq_len,
                                       truncate_text=True)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = None
        img_fpath = self.dataset.iloc[idx, 1]
        caption_text = self.dataset.iloc[idx, 2]
        bird = self.dataset.iloc[idx, 3]
        while img is None:
            try:
                with open(img_fpath, 'rb') as fpath:
                    img = Image.open(fpath)
                    img = self.transform(img)
                    caption = self.tokenize(caption_text).squeeze(0)
            except Exception as e:
                img = None
                rand_sample = self.dataset.sample()
                caption_text = rand_sample['caption'].values[0]
                img_fpath = rand_sample['image_fpath'].values[0]
                if self.params.gen_logs:
                    logging.error(e)
                    raise e

        if self.params.gen_logs:
            logging.info(f'idx:{idx} fpath:{img_fpath} bird:{bird}')

        return {
            'image': img,
            'caption': caption,
            'mask': caption != 0,
            'caption_text': caption_text
        }

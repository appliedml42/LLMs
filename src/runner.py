"""
Script to for training and evaluation.
"""
import sys
import os
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.utilities.cli import MODEL_REGISTRY, DATAMODULE_REGISTRY, LightningCLI

import data
import models


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.vocab_size", "model.vocab_size", apply_on='instantiate')
        parser.link_arguments("data.batch_size", "model.batch_size", apply_on='parse')
        parser.link_arguments("data.seq_len", "model.seq_len", apply_on='parse')


def main():
    MODEL_REGISTRY.register_classes(models, LightningModule, override=True)
    DATAMODULE_REGISTRY.register_classes(data, LightningDataModule, override=True)
    model = MODEL_REGISTRY[os.environ['MODEL']]
    datum = DATAMODULE_REGISTRY[os.environ['DATA']]
    cli = MyLightningCLI(model, datum)
    print(cli)


if __name__ == '__main__':
    main()

"""
Script to for training and evaluation.
"""

import os

from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.utilities.cli import MODEL_REGISTRY, DATAMODULE_REGISTRY, OPTIMIZER_REGISTRY, \
    LR_SCHEDULER_REGISTRY, LightningCLI

import data
import models


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.vocab_size", "model.vocab_size", apply_on='instantiate')
        parser.link_arguments("model.batch_size", "data.batch_size", apply_on='parse')
        parser.link_arguments("model.seq_len", "data.seq_len", apply_on='parse')

        parser.add_optimizer_args(
            OPTIMIZER_REGISTRY.classes,
            nested_key="optimizer",
            link_to="model.optimizer_init"
        )
        parser.add_lr_scheduler_args(
            LR_SCHEDULER_REGISTRY.classes,
            nested_key="lr_scheduler",
            link_to="model.lr_scheduler_init"
        )


def main():
    MODEL_REGISTRY.register_classes(models, LightningModule, override=True)
    DATAMODULE_REGISTRY.register_classes(data, LightningDataModule, override=True)
    model = MODEL_REGISTRY[os.environ['MODEL']]
    datum = DATAMODULE_REGISTRY[os.environ['DATA']]
    exp_path = os.environ['EXP_PATH']
    run_name = exp_path.split('/')[-1]

    try:
        cli = MyLightningCLI(model, datum, trainer_defaults={
            'logger': {
                'class_path': 'pytorch_lightning.loggers.WandbLogger',
                'init_args': {
                    'save_dir': exp_path,
                    'project': 'language_modeling',
                    'name': run_name
                }
            },
            'default_root_dir': exp_path
        })
    except:
        raise ValueError('Error in configurations')


if __name__ == '__main__':
    main()

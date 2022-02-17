"""
Script to for training and evaluation.
"""
import argparse
import collections.abc as container_abcs
import json
import logging
import os
import random
from argparse import Namespace

import horovod.torch as hvd
import numpy as np
import torch
from filelock import FileLock

from common.utils import get_model_class, get_dataset_class

random.seed(1334)
np.random.seed(1334)

logging.basicConfig(level=logging.INFO)


def get_args_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="actions")

    parser_train = subparsers.add_parser("train", add_help=False)
    parser_train.add_argument("--experiment_dpath", type=str, required=False, default=None,
                              help="Directory where training artifacts will be stored.")
    parser_train.add_argument("--dataset", type=str, required=True,
                              help="Fully qualified package name of the Dataset class.")
    parser_train.add_argument("--model", type=str, required=True,
                              help="Fully qualified package name of the Model class.")
    parser_train.add_argument("--notes", type=str, required=False, default="NA", help="Notes for experiment")
    parser_train.add_argument("--epochs", type=int, required=True, default=None)
    parser_train.add_argument("--use_horovod", action='store_true', default=False)
    parser_train.set_defaults(which="train")

    parser_classify_users = subparsers.add_parser("use", add_help=False)
    parser_classify_users.add_argument("--experiment_dpath", type=str, required=True)
    parser_classify_users.add_argument("--method", type=str, required=True)
    parser_classify_users.add_argument("--call_on", type=str, required=True)
    parser_classify_users.set_defaults(which="use")

    return parser


if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = get_args_parser()
    cmd, other_args = parser.parse_known_args()

    if cmd.which == "train":
        dataset_class = get_dataset_class(cmd)
        if cmd.use_horovod:
            hvd.init()
            torch.manual_seed(1334)
            torch.cuda.set_device(hvd.local_rank())
            torch.cuda.manual_seed(1334)
            torch.set_num_threads(1)

            with FileLock(os.path.expanduser('~/.horovod_lock')):
                dataset = dataset_class(cmd, other_args)
        else:
            dataset = dataset_class(cmd, other_args)

        model_class = get_model_class(cmd)
        model = model_class(cmd, other_args)

        with open(os.path.join(cmd.experiment_dpath, "config.json"), "w") as writer:
            json.dump(vars(cmd), writer)

        model.train(dataset)

    elif cmd.which == "use":
        with open(os.path.join(cmd.experiment_dpath, "config.json")) as reader:
            experiment_configuration = Namespace(**json.load(reader))
            experiment_configuration.which = cmd.which
            experiment_configuration.method = cmd.method
            experiment_configuration.call_on = cmd.call_on

        if experiment_configuration.call_on == 'model':
            model_class = get_model_class(experiment_configuration, False)
            model = model_class(experiment_configuration, other_args)
            method = getattr(model_class, experiment_configuration.method)
            method(model)

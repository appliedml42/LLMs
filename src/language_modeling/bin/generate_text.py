import argparse

import einops
import pytorch_lightning as plm
import sentencepiece as spm
import torch
from yaml import load, CLoader as Loader

from language_modeling.models import GPT

plm.seed_everything(2)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", type=str, required=True)
    parser.add_argument("--saved_model_path", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, required=True)
    return parser


def get_model(cmd) -> (GPT, spm.SentencePieceProcessor, int, int):
    model: GPT

    print("Loading tokenizer and model.")
    with open(cmd.model_config_path) as reader:
        config = load(reader, Loader=Loader)

    data_args = config["data"]
    model_args = config["model"]

    max_seq_len = model_args["max_seq_len"]
    context_len = data_args["context_len"]
    tokenizer_path = data_args["tokenizer_path"]

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    vocab_size = tokenizer.vocab_size()
    model_args["vocab_size"] = vocab_size

    model = GPT.load_from_checkpoint(cmd.saved_model_path)
    model.to("cuda")
    model.eval()
    print("Finished loading tokenizer and model.")

    return model, tokenizer, max_seq_len, context_len


def generate_text(cmd):
    model, tokenizer, max_seq_len, context_len = get_model(cmd)

    with open(cmd.prompt_path) as reader:
        prompt = reader.read()
    tokens = [tokenizer.PieceToId("[eod]")] + tokenizer.EncodeAsIds(prompt)[
        : max_seq_len - 1
    ]

    with torch.no_grad():
        while tokens[-1] != tokenizer.PieceToId("[eod]"):
            if len(tokens) == max_seq_len:
                tokens = tokens[-512]
            input = tokens + [tokenizer.pad_id()] * (max_seq_len - len(tokens))
            x = torch.as_tensor(input, device=model.device)
            x = einops.rearrange(
                x, "max_seq_len -> 1 max_seq_len", max_seq_len=max_seq_len
            )

            predictions = model(x)
            predictions = predictions[0, len(tokens) - 1]
            predictions = torch.softmax(predictions, dim=0)
            sorted_indices = torch.argsort(predictions, dim=0, descending=True)[:50]
            probs = predictions[sorted_indices]
            probs = probs / torch.sum(probs)

            sample = 0
            while tokenizer.IsControl(sample) or tokenizer.IsUnknown(sample):
                sample = int(
                    torch.multinomial(probs, num_samples=1).detach().cpu().numpy()[0]
                )
                sample = int(sorted_indices[sample].detach().cpu())

            tokens.append(sample)
            print("".join([tokenizer.IdToPiece(x) for x in tokens]))


if __name__ == "__main__":
    parser = get_args_parser()
    cmd = parser.parse_args()

    generate_text(cmd)

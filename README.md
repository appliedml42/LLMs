<h1 align="center">There are many like it, but this one is mine.</h1> 

There are various excellent open-source Transformer 
implementations. However, I decided to implement my own language models on the Pile dataset for the following reasons:
* Learn [Pytorch Lightning](https://www.pytorchlightning.ai), [Wandb](https://wandb.ai/site), and [Einops](http://einops.rocks).
* Gain experience with large-scale training by experimenting with tools like [Deepspeed](https://www.deepspeed.ai) and [Fairscale](https://fairscale.readthedocs.io/en/latest/).
* Enhance my understanding of Transformers.
* Justify buying [this](https://bizon-tech.com/bizon-z5000.html) at some point 🤑.

# Experiments & Documentation
- [Notes](https://appliedml85.github.io)
- [Wandb](https://wandb.ai/appliedml85/language_modeling?workspace=user-appliedml85)

# Tasks
I work on this when I have free time.
- [ ] Publish post, Perplexity: A Information Theoritic Viewpoint.
- [ ] Publish post, Transformer Training Stability: Role of Layer Normalization.
- [ ] Implement validation & Test step inspired by [Pile dataset details.](https://arxiv.org/pdf/2101.00027.pdf)
- [x] Implement first end2end run.
- [x] Integrate with WandB
- [x] Train sentencepiece tokenizer.
- [x] Add sentencepiece tokenizer CLI to Docker container.
- [x] Update Docker container with Einops and Jsonargparse.
- [x] Update Pile module to generate proper target variables. 
- [x] Standup Github Pages blog. 
- [x] Implement GPT, Pile, and CLI modules. Something that is working end2end and will serve as the foundation for future work.

# Tokenizers
## Pile
The tokenizer for Pile was trained using [Google SentencePiece](https://github.com/google/sentencepiece) with the following
configuration.

```
nohup spm_train --input input.txt --vocab_size 50000 --model_prefix 50000 --model_type bpe --pad_id=0 --unk_id=1 --bos_id=-1 --eos_id=-1 --pad_piece=[PAD] --unk_piece=[UNK] --input_sentence_size=100000000 --control_symbols=[eod] &
```

The input.txt is created by concatinating file [11](https://mystic.the-eye.eu/public/AI/pile/train/11.jsonl.zst) and 
[22](https://mystic.the-eye.eu/public/AI/pile/train/22.jsonl.zst) from the train section.

# References
- [Pytorch Lightning Transformer Tutorial](https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/05-transformers-and-MH-attention.html)
- [SentencePiece Python Tutorial](https://github.com/google/sentencepiece)

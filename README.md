<h1 align="center">There are many like it, but this one is mine.</h1> 

There are various excellent open-source Transformer 
implementations. However, I decided to implement my own language models on the Pile dataset for the following reasons:
* Learn [Pytorch Lightning](https://www.pytorchlightning.ai), [Wandb](https://wandb.ai/site), and [Einops](http://einops.rocks).
* Gain experience with large-scale training by experimenting with tools like [Deepspeed](https://www.deepspeed.ai) and [Fairscale](https://fairscale.readthedocs.io/en/latest/).
* Enhance my understanding of Transformers.
* Justify buying [this](https://bizon-tech.com/bizon-z5000.html) at some point 🤑.

# Experiments & Documentation
- [Wandb](https://wandb.ai/appliedml42/language_modeling?workspace=user-appliedml42)

# Tasks
I work on this when I have free time.
- [ ] Publish post, Perplexity: A Information Theoritic Viewpoint.
- [ ] Publish post, 1 TB Random Access Dataset using Sqlite3 and ZSTD.
- [x] Trigger training using mixed precision & gradient accumulation. 
- [ ] Publish post, Transformer Training Stability.
- [x] Generate BPB test metrics on Carolyn_Mayo and add to W&B.
- [x] Fork from EleutherAI/lm_perplexity and adjust to generate test metrics.
- [x] Implement text generation script.
- [x] Add context to partial documents. Partial documents arise when we split the original document to truncate.
- [x] Add log 2 adjustment to metrics computation.
- [x] Upgrade to Pytorch Lightning 1.6.0
- [x] Trigger first 4 GPU Run with validation step and DDP Sharded strategy.
- [x] Implement Pile Random Access Dataset. Needed for proper multi-gpu.
- [x] Implement validation step inspired by [Pile dataset details.](https://arxiv.org/pdf/2101.00027.pdf)
- [x] Implement first end2end run.
- [x] Integrate with WandB
- [x] Train sentencepiece tokenizer.
- [x] Add sentencepiece tokenizer CLI to Docker container.
- [x] Update Docker container with Einops and Jsonargparse.
- [x] Update Pile module to generate proper target variables. 
- [x] Standup Github Pages blog. 
- [x] Implement GPT, Pile, and CLI modules. Something that is working end2end and will serve as the foundation for future work.

# Dataset & Tokenizers
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

<h1 align="center">There are many like it, but this one is mine.</h1> 

There are various excellent open-source Transformer 
implementations. However, I decided to implement my own language models on the Pile dataset for the following reasons:
* Learn [Pytorch Lightning](https://www.pytorchlightning.ai), [Wandb](https://wandb.ai/site), and [Einops](http://einops.rocks).
* Gain experience with large-scale training by experimenting with tools like [Deepspeed](https://www.deepspeed.ai) and [Fairscale](https://fairscale.readthedocs.io/en/latest/).
* Refresh my understanding of Transformers.
* Justify buying [this](https://bizon-tech.com/bizon-z5000.html) at some point 🤑.

In addition, I will also document interesting learnings [here](https://appliedml85.github.io/blog/pytorch-lightning).

# Tasks
I work on this when I have free time.
- [ ] Train sentencepiece tokenizer.
- [ ] Implement [Pile](https://arxiv.org/pdf/2101.00027.pdf) bits per UTF-8 encoded byte (BPB) metric.
- [x] Update Docker container with Einops and Jsonargparse.
- [x] Update Pile module to generate proper target variables. 
- [x] Standup Github Pages blog. 
- [x] Implement GPT, Pile, and CLI modules. Something that is working end2end and will serve as the foundation for future work.

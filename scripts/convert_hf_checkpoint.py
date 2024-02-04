import os
from collections import defaultdict

import torch
from am42.language_modeling.configs import ModelConfig
from am42.language_modeling.model import SLM
from huggingface_hub import snapshot_download
from jsonargparse import CLI

WEIGHT_MAPS = {
    "microsoft/phi-1_5": {
        "model.embed_tokens.weight": "transformer.wte.weight",
        "model.layers.{}.input_layernorm.weight": "transformer.h.{}.norm_1.weight",
        "model.layers.{}.input_layernorm.bias": "transformer.h.{}.norm_1.bias",
        "model.layers.{}.self_attn.q_proj.weight": None,
        "model.layers.{}.self_attn.q_proj.bias": None,
        "model.layers.{}.self_attn.k_proj.weight": None,
        "model.layers.{}.self_attn.k_proj.bias": None,
        "model.layers.{}.self_attn.v_proj.weight": None,
        "model.layers.{}.self_attn.v_proj.bias": None,
        "model.layers.{}.self_attn.dense.weight": "transformer.h.{}.attn.proj.weight",
        "model.layers.{}.self_attn.dense.bias": "transformer.h.{}.attn.proj.bias",
        "model.layers.{}.mlp.fc1.weight": "transformer.h.{}.mlp.fc.weight",
        "model.layers.{}.mlp.fc1.bias": "transformer.h.{}.mlp.fc.bias",
        "model.layers.{}.mlp.fc2.weight": "transformer.h.{}.mlp.proj.weight",
        "model.layers.{}.mlp.fc2.bias": "transformer.h.{}.mlp.proj.bias",
        "model.final_layernorm.weight": "transformer.ln_f.weight",
        "model.final_layernorm.bias": "transformer.ln_f.bias",
        "lm_head.weight": "lm_head.weight",
        "lm_head.bias": "lm_head.bias",
    }
}


@torch.inference_mode()
def convert_hf_checkpoint(repo_id, model_dir):
    weight_map = WEIGHT_MAPS[repo_id]

    state_dict = torch.load(
        os.path.join(model_dir, "pytorch_model.bin"),
        weights_only=True,
        map_location="cpu",
        mmap=True,
    )

    new_state_dict = {}
    qkv_dict = defaultdict(dict)

    # Process the non QKV weights first
    for key, value in state_dict.items():
        if key.startswith("model.layers."):
            split = key.split(".")
            layer_num = int(split[2])
            split[2] = "{}"
            abstract_key = ".".join(split)
            new_key = weight_map[abstract_key]
            if new_key is not None:
                new_key = new_key.format(layer_num)
            elif "q" in key:
                weight_type = key.split(".")[-1]
                qkv_dict[layer_num][("q", weight_type)] = value
            elif "k" in key:
                weight_type = key.split(".")[-1]
                qkv_dict[layer_num][("k", weight_type)] = value
            elif "v" in key:
                weight_type = key.split(".")[-1]
                qkv_dict[layer_num][("v", weight_type)] = value
        else:
            new_key = weight_map[key]

        if new_key is not None:
            new_state_dict[new_key] = value

    # Process the QKV weights
    for layer_num, qkv in qkv_dict.items():
        q_weight = qkv[("q", "weight")]
        q_bias = qkv[("q", "bias")]
        k_weight = qkv[("k", "weight")]
        k_bias = qkv[("k", "bias")]
        v_weight = qkv[("v", "weight")]
        v_bias = qkv[("v", "bias")]

        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=1).transpose(0, 1)
        qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
        new_state_dict[f"transformer.h.{layer_num}.attn.attn.weight"] = qkv_weight
        new_state_dict[f"transformer.h.{layer_num}.attn.attn.bias"] = qkv_bias

    model_config = ModelConfig.get_config(repo_id)
    model = SLM(model_config)
    model.load_state_dict(new_state_dict, strict=True, assign=True)
    # Compile and then save the model. This is necessary because post compilation parameter names are changed.
    # Also, we use Fabric which prefers loading weights post compilation. So this makes life easy.
    model = torch.compile(model)
    torch.save(new_state_dict, os.path.join(model_dir, "am42_pytorch_model.bin"))


def main(
    repo_id: str,
    cache_dir: str,
    hf_local_dir: str,
    hf_token: str,
):
    model_dir = os.path.join(hf_local_dir, repo_id)
    snapshot_download(
        repo_id,
        cache_dir=cache_dir,
        local_dir=model_dir,
        token=hf_token,
    )

    convert_hf_checkpoint(repo_id, model_dir)


if __name__ == "__main__":
    CLI(main)

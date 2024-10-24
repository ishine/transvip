import os
import torch
import yaml
from fairseq2_011.nn.projection import TiedProjection

from .tokenizer import NllbTokenizer
from .builder import UnitYModel, unity_archs, create_unity_model


def load_s2st_model(
        model_path: str = None, 
        model_name: str = "m4t_valle",
        num_new_tokens: int = 0, # 1025 for codec
        spk_encoder_path: str = None,
        device=torch.device("cpu"), 
        dtype=torch.float32,
        **kwargs,
        ) -> UnitYModel:
    config = unity_archs.get_config(model_name)
    for k, v in kwargs.items():
        config.__setattr__(k, v)
    model = create_unity_model(config, device, dtype)
    if model_path is not None:
        state_dict = torch.load(model_path, map_location=device)['model']
        info = model.load_state_dict(state_dict, strict=False)
        print(info)

    if num_new_tokens != 0:
        old_num_tokens = model.text_encoder_frontend.embed.weight.shape[0]
        new_num_tokens = old_num_tokens + num_new_tokens # append codec tokens
        model = resize_token_embeddings(model, new_num_tokens)
    
    model.speech_encoder.requires_grad_(False)
    model.text_encoder.requires_grad_(False)

    if spk_encoder_path is not None:
        print("Loading spk_encoder from", spk_encoder_path)
        state_dict = torch.load(spk_encoder_path, map_location=device)['model']
        submodule_dict = {k.partition('spk_encoder.')[2]: v for k, v in state_dict.items() if k.startswith('spk_encoder.')}
        model.spk_encoder.load_state_dict(submodule_dict, strict=True)
        model.spk_encoder.requires_grad_(False)

    return model


def load_m4t_tokenizer(tokenizer_path: str) -> NllbTokenizer:
    with open(os.path.join(tokenizer_path, "tokenizer.yaml"), "r") as f:
        tokenizer_config = yaml.safe_load(f)
    tokenizer_path = os.path.join(tokenizer_path, "tokenizer.model")
    tokenizer = NllbTokenizer(
        tokenizer_path,
        langs=tokenizer_config["langs"],
        default_lang=tokenizer_config["default_lang"],
    )
    return tokenizer

def resize_token_embeddings(model: UnitYModel, new_num_tokens: int, random_init: bool = False):
    text_embed = model.text_encoder_frontend.embed
    old_embed_weight = text_embed.weight
    old_num_tokens, embedding_dim = old_embed_weight.shape

    new_embed_weight = torch.zeros(
        new_num_tokens, embedding_dim, dtype=old_embed_weight.dtype
    )

    torch.nn.init.normal_(new_embed_weight, mean=0, std=0.02)
    if not random_init:
        if new_num_tokens > old_num_tokens:
            new_embed_weight.data[:old_num_tokens, :] = old_embed_weight.data
        else:
            new_embed_weight.data = old_embed_weight.data[:new_num_tokens, :]
    text_embed.weight = torch.nn.Parameter(new_embed_weight)

    model.final_proj = TiedProjection(text_embed.weight)
    return model



    


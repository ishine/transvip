
import torch
import os
from .builder import create_nar_model, NarConfig, _12_layer

def load_nar_model(
    model_path: str = None, 
    model_config: NarConfig=None,
    embedding_path: str=None,
    device=torch.device('cpu'), 
    dtype=torch.float32,
    ):
    if model_config is None:
        model_config = _12_layer()
    model = create_nar_model(model_config, device=device, dtype=dtype)
    if model_path is not None:
        state_dict = torch.load(model_path, map_location=device)['model']
        model.load_state_dict(state_dict, strict=False)
    elif embedding_path is not None:
        embedding_weights = torch.load(embedding_path, map_location=device)
        model.decoder_frontend.s_embed.weight.data = embedding_weights['semantic'].data
        model.decoder_frontend.a_embeds[0].weight.data = embedding_weights['acoustic'].data
        print("Loaded embeddings from {}".format(embedding_path))
    
    return model
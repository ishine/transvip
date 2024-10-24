# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple, final, List


import torch
from torch import Tensor
from torch.nn import Dropout, Module, ModuleList
from torch.nn.functional import pad

from fairseq2_011.nn.embedding import Embedding
from fairseq2_011.nn.normalization import LayerNorm
from fairseq2_011.nn.position_encoder import PositionEncoder
from fairseq2_011.nn.transformer.layer_norm import (
    LayerNormFactory,
    create_default_layer_norm,
)
from fairseq2_011.nn.utils.mask import to_padding_mask
from fairseq2_011.typing import DataType, Device, finaloverride


LANGUAGE2ID_MAPPING = {
    'ab': 0,
    'af': 1,
    'am': 2,
    'ar': 3,
    'as': 4,
    'ast': 5,
    'az': 6,
    'ba': 7,
    'bas': 8,
    'be': 9,
    'bg': 10,
    'bn': 11,
    'br': 12,
    'ca': 13,
    'ckb': 14,
    'cnh': 15,
    'cs': 16,
    'cv': 17,
    'cy': 18,
    'da': 19,
    'de': 20,
    'dv': 21,
    'dyu': 22,
    'el': 23,
    'en': 24,
    'eo': 25,
    'es': 26,
    'et': 27,
    'eu': 28,
    'fa': 29,
    'fi': 30,
    'fr': 31,
    'fy-NL': 32,
    'ga-IE': 33,
    'gl': 34,
    'gn': 35,
    'ha': 36,
    'he': 37,
    'hi': 38,
    'hsb': 39,
    'hu': 40,
    'hy-AM': 41,
    'ia': 42,
    'id': 43,
    'ig': 44,
    'is': 45,
    'it': 46,
    'ja': 47,
    'ka': 48,
    'kab': 49,
    'kk': 50,
    'kmr': 51,
    'ko': 52,
    'ky': 53,
    'lg': 54,
    'lo': 55,
    'lt': 56,
    'lv': 57,
    'mdf': 58,
    'mhr': 59,
    'mk': 60,
    'ml': 61,
    'mn': 62,
    'mr': 63,
    'mrj': 64,
    'mt': 65,
    'myv': 66,
    'nan-tw': 67,
    'ne-NP': 68,
    'nl': 69,
    'nn-NO': 70,
    'oc': 71,
    'or': 72,
    'pa-IN': 73,
    'pl': 74,
    'ps': 75,
    'pt': 76,
    'quy': 77,
    'rm-sursilv': 78,
    'rm-vallader': 79,
    'ro': 80,
    'ru': 81,
    'rw': 82,
    'sah': 83,
    'sat': 84,
    'sc': 85,
    'sk': 86,
    'skr': 87,
    'sl': 88,
    'sq': 89,
    'sr': 90,
    'sv-SE': 91,
    'sw': 92,
    'ta': 93,
    'th': 94,
    'ti': 95,
    'tig': 96,
    'tk': 97,
    'tok': 98,
    'tr': 99,
    'tt': 100,
    'tw': 101,
    'ug': 102,
    'uk': 103,
    'ur': 104,
    'uz': 105,
    'vi': 106,
    'vot': 107,
    'yo': 108,
    'yue': 109,
    'zgh': 110,
    'zh-CN': 111,
    'zh-HK': 112,
    'zh-TW': 113,
}




class TransformerFrontend(Module, ABC):
    """Represents a Transformer encoder/decoder front-end."""

    model_dim: int

    def __init__(self, model_dim: int) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        """
        super().__init__()

        self.model_dim = model_dim

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"


@final
class ValleFrontend(TransformerFrontend):
    """Represents a Transformer encoder/decoder front-end as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    embed: Embedding
    scale: float
    pos_encoder: Optional[PositionEncoder]
    layer_norm: Optional[LayerNorm]
    dropout: Optional[Dropout]

    def __init__(
        self,
        acoustic_embeds: List[Embedding],
        pos_encoder: Optional[PositionEncoder] = None,
        no_scale: bool = False,
        layer_norm: bool = False,
        dropout_p: float = 0.1,
        layer_norm_fn: Optional[LayerNormFactory] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param embed:
            The token embedding table.
        :param pos_encoder:
            The position encoder.
        :param no_scale:
            If ``True``, does not scale embeddings by the square root of the
            embedding size.
        :param layer_norm:
            If ``True``, applies Layer Normalization to embeddings before
            dropout.
        :param dropout_p:
            The dropout probability on embeddings.
        :param layer_norm_fn:
            The factory to use to construct the Layer Normalization module.
        """
        model_dim = acoustic_embeds[0].embedding_dim

        super().__init__(model_dim)

        if layer_norm_fn is None:
            layer_norm_fn = create_default_layer_norm

        self.a_embeds = ModuleList(acoustic_embeds)
        self.total_steps = len(acoustic_embeds)

        self.scale = 1.0 if no_scale else math.sqrt(model_dim)

        if pos_encoder is not None:
            if pos_encoder.encoding_dim != model_dim:
                raise ValueError(
                    f"`encoding_dim` of `pos_encoder` and `embedding_dim` of `embed` must be equal, but are {pos_encoder.encoding_dim} and {model_dim} instead."
                )

            self.pos_encoder = pos_encoder
        else:
            self.register_module("pos_encoder", None)

        if layer_norm:
            self.layer_norm = layer_norm_fn(model_dim, device, dtype)
        else:
            self.register_module("layer_norm", None)

        if dropout_p > 0.0:
            self.dropout = Dropout(dropout_p)
        else:
            self.register_module("dropout", None)

        self.lang_embed = Embedding(
            num_embeddings=len(LANGUAGE2ID_MAPPING),
            embedding_dim=model_dim,
            pad_idx=None,
            scaled=False,
            device=device,
            dtype=dtype,
        )

    @finaloverride
    def forward(
        self,
        acoustic_prompts: List[Tensor],
        acoustic_tokens: List[Tensor],
        langs: List[str] = None,
    ):
        current_step = acoustic_tokens[0].shape[0]

        acoustic_prompt_embeds = [
            sum(
                self.a_embeds[i](prompt[i, :]) for i in range(self.total_steps)
            ) for prompt in acoustic_prompts
        ]
        acoustic_embeds = [
            sum(
                self.a_embeds[i](acoustic_token[i, :]) for i in range(current_step)
            ) for acoustic_token in acoustic_tokens
        ]
        embeds_seqs = [
            torch.cat((ap, a), dim=0)
            for ap, a in zip(
                acoustic_prompt_embeds, acoustic_embeds
            )
        ]
        embeds, padding_mask = self.pad_embeds_seqs(embeds_seqs)

        if langs is not None:
            lang_embeds = self.lang_embed(
                torch.tensor(
                    [LANGUAGE2ID_MAPPING[lang] for lang in langs],
                    device=embeds.device,
                    dtype=torch.long,
                ).unsqueeze(1)
            )
            embeds = torch.cat([lang_embeds, embeds], dim=1)
            if padding_mask is not None:
                padding_mask = torch.cat(
                    [
                        torch.zeros(lang_embeds.size(0), 1, dtype=torch.bool, device=embeds.device),
                        padding_mask,
                    ],
                    dim=1,
                )

        if self.scale != 1.0:
            embeds = embeds * self.scale

        if self.pos_encoder is not None:
            embeds = self.pos_encoder(embeds, padding_mask)

        if self.layer_norm is not None:
            embeds = self.layer_norm(embeds)

        if self.dropout is not None:
            embeds = self.dropout(embeds)
            
        return embeds, padding_mask, current_step - 1


    def pad_embeds_seqs(self, embeds_seqs: List[Tensor]) -> Tensor:
        seq_lens = torch.tensor(
            [seq.size(0) for seq in embeds_seqs], 
            device=embeds_seqs[0].device
            )
        max_seq_lens = max(seq.size(0) for seq in embeds_seqs)
        embeds_seqs = [pad(seq, (0, 0, 0, max_seq_lens - seq.size(0))) for seq in embeds_seqs]
        embeds = torch.stack(embeds_seqs, dim=0)
        padding_mask = to_padding_mask(embeds, seq_lens)
        return embeds, padding_mask



    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return s + ", no_scale=False" if self.scale != 1.0 else ""
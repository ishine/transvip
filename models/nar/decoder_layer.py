from abc import ABC, abstractmethod
from typing import Optional, Tuple, cast, final, List, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Dropout, Module

from fairseq2_011.nn.normalization import LayerNorm
from fairseq2_011.nn.transformer.ffn import FeedForwardNetwork
from fairseq2_011.nn.transformer.layer_norm import (
    LayerNormFactory,
    create_default_layer_norm,
)
from fairseq2_011.nn.transformer.multihead_attention import MultiheadAttention
from fairseq2_011.nn.transformer.decoder_layer import TransformerDecoderLayer
from fairseq2_011.nn.utils.module import check_model_dim
from fairseq2_011.nn.projection import Linear
from fairseq2_011.typing import DataType, Device, finaloverride


class ZeroAdaLN(nn.Module):
    def __init__(self, n_embd: int, total_steps: int, device=None, dtype=None):
        super().__init__()
        self.total_steps = total_steps
        self.timestep_emb = nn.Embedding(total_steps, n_embd, padding_idx=None, device=device, dtype=dtype) 
        self.silu = nn.SiLU()
        self.linear = Linear(n_embd, n_embd*6, bias=True, device=device, dtype=dtype)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, step_id: int):
        step_ids = torch.full_like(x[..., 0], step_id, dtype=torch.long)
        emb = self.timestep_emb(step_ids)
        emb = self.linear(self.silu(emb))
        return torch.chunk(emb, 6, dim=2)
    
@final
class NarTransformerDecoderLayer(TransformerDecoderLayer):
    """Represents a Transformer decoder layer as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    self_attn: MultiheadAttention
    self_attn_norm: Optional[LayerNorm]
    self_attn_dropout: Optional[Dropout]
    self_attn_layer_norm: LayerNorm
    ffn: FeedForwardNetwork
    ffn_dropout: Optional[Dropout]
    ffn_layer_norm: LayerNorm
    adaLN: Optional[ZeroAdaLN]

    def __init__(
        self,
        self_attn: MultiheadAttention,
        ffn: FeedForwardNetwork,
        dropout_p: float = 0.1,
        layer_norm_fn: Optional[LayerNormFactory] = None,
        adaLN: Optional[ZeroAdaLN] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param self_attn:
            The self attention layer.
        :param encoder_decoder_attn:
            The encoder-decoder attention layer.
        :param ffn:
            The feed-forward network.
        :param dropout_p:
            The dropout probability on outputs of the attention layers and the
            feed-forward network.
        :param norm_order:
            The Layer Normalization order to use.
        :param layer_norm_fn:
            The factory to use to construct the Layer Normalization modules.
        """
        model_dim = self_attn.model_dim

        super().__init__(model_dim)

        if layer_norm_fn is None:
            layer_norm_fn = create_default_layer_norm

        self_attn_layer_norm = layer_norm_fn(model_dim, device, dtype)


        self.self_attn_layer_norm = self_attn_layer_norm

        self.self_attn = self_attn

        if dropout_p > 0.0:
            self.self_attn_dropout = Dropout(dropout_p)
        else:
            self.register_module("self_attn_dropout", None)


        ffn_layer_norm = layer_norm_fn(model_dim, device, dtype)


        self.ffn_layer_norm = ffn_layer_norm

        self.ffn = ffn

        if dropout_p > 0.0:
            self.ffn_dropout = Dropout(dropout_p)
        else:
            self.register_module("ffn_dropout", None)

        if adaLN is not None:
            self.adaLN = adaLN
        else:
            self.register_module("adaLN", None)

        check_model_dim(self)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        if self.adaLN is not None:
            nn.init.zeros_(self.adaLN.linear.weight)

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[Tensor],
        self_attn_mask: Optional[Tensor] = None,
        timestep: Optional[int] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        if self.adaLN is not None:
            assert timestep is not None
            attn_norm_scale, attn_norm_bias, attn_final_scale, \
                ffn_norm_scale, ffn_norm_bias, ffn_final_scale = self.adaLN(seqs, timestep)
        else:
            attn_norm_scale = attn_norm_bias = attn_final_scale = None
            ffn_norm_scale = ffn_norm_bias = ffn_final_scale = None

        seqs = self._forward_self_attn(seqs, padding_mask, self_attn_mask,
            attn_norm_scale, attn_norm_bias, attn_final_scale)

        seqs = self._forward_ffn(seqs, ffn_norm_scale, ffn_norm_bias, ffn_final_scale)

        return seqs, padding_mask

    def _forward_self_attn(
        self,
        seqs: Tensor,
        padding_mask: Optional[Tensor],
        self_attn_mask: Optional[Tensor],
        norm_scale: Optional[Tensor] = None,
        norm_bias: Optional[Tensor] = None,
        final_scale: Optional[Tensor] = None,
    ) -> Tensor:
        residual = seqs

        seqs = self.self_attn_layer_norm(seqs)

        if norm_scale is not None:
            seqs = seqs * (1 + norm_scale)

        if norm_bias is not None:
            seqs = seqs + norm_bias

        seqs = self.self_attn(
            seqs,
            padding_mask,
            keys=seqs,
            values=seqs,
            attn_mask=self_attn_mask,
            key_padding_mask=padding_mask,
        )


        if self.self_attn_dropout is not None:
            seqs = self.self_attn_dropout(seqs)

        if final_scale is not None:
            seqs = seqs * final_scale

        seqs = seqs + residual

        return seqs


    def _forward_ffn(self, 
        seqs: Tensor,
        norm_scale: Optional[Tensor] = None,
        norm_bias: Optional[Tensor] = None,
        final_scale: Optional[Tensor] = None,
        ) -> Tensor:
        residual = seqs


        seqs = self.ffn_layer_norm(seqs)
        if norm_scale is not None:
            seqs = seqs * (1 + norm_scale)
        if norm_bias is not None:
            seqs = seqs + norm_bias

        seqs = self.ffn(seqs)

        if self.ffn_dropout is not None:
            seqs = self.ffn_dropout(seqs)

        if final_scale is not None:
            seqs = seqs * final_scale

        seqs = seqs + residual

        return seqs

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return s
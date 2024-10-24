# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Optional, Tuple, cast, final

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Dropout, Module
from torch.nn.parameter import Parameter

from fairseq2_011.nn.incremental_state import IncrementalStateBag
from fairseq2_011.nn.normalization import LayerNorm
from fairseq2_011.nn.transformer.ffn import FeedForwardNetwork
from fairseq2_011.nn.transformer.layer_norm import (
    LayerNormFactory,
    create_default_layer_norm,
)
from fairseq2_011.nn.transformer.multihead_attention import MultiheadAttention
from fairseq2_011.nn.transformer.norm_order import TransformerNormOrder
from fairseq2_011.nn.utils.module import check_model_dim
from fairseq2_011.typing import DataType, Device, finaloverride


class TransformerDecoderLayer(Module, ABC):
    """Represents a Transformer decoder layer."""

    model_dim: int

    def __init__(self, model_dim: int) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        """
        super().__init__()

        self.model_dim = model_dim

    @abstractmethod
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[Tensor],
        self_attn_mask: Optional[Tensor] = None,
        encoder_output: Optional[Tensor] = None,
        encoder_padding_mask: Optional[Tensor] = None,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        :param seqs:
            The sequences to process. *Shape:* :math:`(N,S,M)`, where :math:`N`
            is the batch size, :math:`S` is the sequence length, and :math:`M`
            is the dimensionality of the model.
        :param padding_mask:
            The float padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where
            :math:`N` is the batch size and :math:`S` is the sequence length.
        :param self_attn_mask:
            The float mask that will be added to the attention weights before
            computing the self attention. *Shape:* :math:`(S,S)`, where
            :math:`S` is the sequence length.
        :param encoder_output:
            The encoder output to use in encoder-decoder attention. *Shape:*
            :math:`(N,S_{enc},M_{enc})`, where :math:`N` is the batch size,
            :math:`S_{enc}` is the encoder output sequence length, and
            :math:`M_{enc}` is the dimensionality of the encoder.
        :param encoder_padding_mask:
            The float padding mask of ``encoder_output``. *Shape:*
            :math:`(N,S_{enc})`, where :math:`N` is the batch size and
            :math:`S_{enc}` is the encoder output sequence length.
        :param state_bag:
            The state bag to use for incremental evaluation.

        :returns:
            - The decoder layer output. *Shape:* Same as ``seqs``.
            - The float padding mask of the decoder layer output. *Shape:* Same
              as ``padding_mask``.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"


@final
class StandardTransformerDecoderLayer(TransformerDecoderLayer):
    """Represents a Transformer decoder layer as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    self_attn: MultiheadAttention
    self_attn_norm: Optional[LayerNorm]
    self_attn_dropout: Optional[Dropout]
    self_attn_layer_norm: LayerNorm
    encoder_decoder_attn: Optional[MultiheadAttention]
    encoder_decoder_dropout: Optional[Dropout]
    encoder_decoder_attn_layer_norm: Optional[LayerNorm]
    ffn: FeedForwardNetwork
    ffn_dropout: Optional[Dropout]
    residual_scale: Optional[Parameter]
    ffn_layer_norm: LayerNorm
    norm_order: TransformerNormOrder

    def __init__(
        self,
        self_attn: MultiheadAttention,
        encoder_decoder_attn: Optional[MultiheadAttention],
        ffn: FeedForwardNetwork,
        scale_residual: bool = False,
        dropout_p: float = 0.1,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        layer_norm_fn: Optional[LayerNormFactory] = None,
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
        :param scale_residual:
            If ``True``, scales residuals before adding them to the output of
            the feed-forward network. See
            :cite:t:`https://doi.org/10.48550/arxiv.2110.09456` for more
            information.
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

        if norm_order != TransformerNormOrder.POST:
            self.self_attn_layer_norm = self_attn_layer_norm

        self.self_attn = self_attn

        if norm_order == TransformerNormOrder.PRE_WITH_NORMFORMER:
            self.self_attn_norm = layer_norm_fn(model_dim, device, dtype)
        else:
            self.register_module("self_attn_norm", None)

        if dropout_p > 0.0:
            self.self_attn_dropout = Dropout(dropout_p)
        else:
            self.register_module("self_attn_dropout", None)

        if norm_order == TransformerNormOrder.POST:
            self.self_attn_layer_norm = self_attn_layer_norm

        if encoder_decoder_attn is None:
            self.register_module("encoder_decoder_attn", None)
            self.register_module("encoder_decoder_attn_layer_norm", None)
        else:
            encoder_decoder_attn_layer_norm = layer_norm_fn(model_dim, device, dtype)

            if norm_order != TransformerNormOrder.POST:
                self.encoder_decoder_attn_layer_norm = encoder_decoder_attn_layer_norm

            self.encoder_decoder_attn = encoder_decoder_attn

            if dropout_p > 0.0:
                self.encoder_decoder_attn_dropout = Dropout(dropout_p)
            else:
                self.register_module("encoder_decoder_attn_dropout", None)

            if norm_order == TransformerNormOrder.POST:
                self.encoder_decoder_attn_layer_norm = encoder_decoder_attn_layer_norm

        ffn_layer_norm = layer_norm_fn(model_dim, device, dtype)

        if norm_order != TransformerNormOrder.POST:
            self.ffn_layer_norm = ffn_layer_norm

        self.ffn = ffn

        if dropout_p > 0.0:
            self.ffn_dropout = Dropout(dropout_p)
        else:
            self.register_module("ffn_dropout", None)

        if scale_residual:
            self.residual_scale = Parameter(
                torch.empty((model_dim,), device=device, dtype=dtype)
            )
        else:
            self.register_parameter("residual_scale", None)

        if norm_order == TransformerNormOrder.POST:
            self.ffn_layer_norm = ffn_layer_norm

        self.norm_order = norm_order

        check_model_dim(self)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        if self.residual_scale is not None:
            nn.init.ones_(self.residual_scale)

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[Tensor],
        self_attn_mask: Optional[Tensor] = None,
        encoder_output: Optional[Tensor] = None,
        encoder_padding_mask: Optional[Tensor] = None,
        state_bag: Optional[IncrementalStateBag] = None,
        enc_dec_attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        seqs = self._forward_self_attn(seqs, padding_mask, self_attn_mask, state_bag)

        seqs = self._forward_encoder_decoder_attn(
            seqs, padding_mask, encoder_output, encoder_padding_mask, state_bag, enc_dec_attn_mask
        )

        seqs = self._forward_ffn(seqs)

        return seqs, padding_mask

    def _forward_self_attn(
        self,
        seqs: Tensor,
        padding_mask: Optional[Tensor],
        self_attn_mask: Optional[Tensor],
        state_bag: Optional[IncrementalStateBag],
    ) -> Tensor:
        residual = seqs

        if self.norm_order != TransformerNormOrder.POST:
            seqs = self.self_attn_layer_norm(seqs)

        seqs = self.self_attn(
            seqs,
            padding_mask,
            keys=seqs,
            values=seqs,
            attn_mask=self_attn_mask,
            key_padding_mask=padding_mask,
            state_bag=state_bag,
        )

        if self.self_attn_norm is not None:
            seqs = self.self_attn_norm(seqs)

        if self.self_attn_dropout is not None:
            seqs = self.self_attn_dropout(seqs)

        seqs = seqs + residual

        if self.norm_order == TransformerNormOrder.POST:
            seqs = self.self_attn_layer_norm(seqs)

        return seqs

    def _forward_encoder_decoder_attn(
        self,
        seqs: Tensor,
        padding_mask: Optional[Tensor],
        encoder_output: Optional[Tensor],
        encoder_padding_mask: Optional[Tensor],
        state_bag: Optional[IncrementalStateBag],
        enc_dec_attn_mask: Optional[Tensor],
    ) -> Tensor:
        if self.encoder_decoder_attn is None:
            if encoder_output is not None:
                raise ValueError(
                    "`encoder_output` must be `None` for decoder-only attention."
                )

            return seqs

        if encoder_output is None:
            raise ValueError(
                "`encoder_output` must not be `None` for encoder-decoder attention."
            )

        residual = seqs

        if self.norm_order != TransformerNormOrder.POST:
            seqs = cast(LayerNorm, self.encoder_decoder_attn_layer_norm)(seqs)

        seqs = self.encoder_decoder_attn(
            seqs,
            padding_mask,
            keys=encoder_output,
            values=encoder_output,
            key_padding_mask=encoder_padding_mask,
            attn_mask=enc_dec_attn_mask,
            state_bag=state_bag,
        )

        if self.encoder_decoder_attn_dropout is not None:
            seqs = self.encoder_decoder_attn_dropout(seqs)

        seqs = seqs + residual

        if self.norm_order == TransformerNormOrder.POST:
            seqs = cast(LayerNorm, self.encoder_decoder_attn_layer_norm)(seqs)

        return seqs

    def _forward_ffn(self, seqs: Tensor) -> Tensor:
        residual = seqs

        if self.norm_order != TransformerNormOrder.POST:
            seqs = self.ffn_layer_norm(seqs)

        seqs = self.ffn(seqs)

        if self.ffn_dropout is not None:
            seqs = self.ffn_dropout(seqs)

        if self.residual_scale is not None:
            residual = torch.mul(self.residual_scale, residual)

        seqs = seqs + residual

        if self.norm_order == TransformerNormOrder.POST:
            seqs = self.ffn_layer_norm(seqs)

        return seqs

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return s + f", norm_order={self.norm_order}"
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, List

from fairseq2_011.nn.embedding import Embedding
from fairseq2_011.nn.position_encoder import RotaryEncoder
from fairseq2_011.nn.transformer import (
    FeedForwardNetwork,
    MultiheadAttention,
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerNormOrder,
    create_default_sdpa,
)
from fairseq2_011.typing import DataType, Device

from .frontend import ValleFrontend, TransformerFrontend
from .model import NarModel, MultiStepTiedProjection
from .decoder import NarTransformerDecoder
from .decoder_layer import NarTransformerDecoderLayer, ZeroAdaLN


@dataclass
class NarConfig:
    """Holds the configuration of an NLLB model."""

    model_dim: int
    """The dimensionality of the model."""

    max_seq_len: int
    """The expected maximum sequence length."""

    total_steps: int
    """The total number of steps."""

    codec_size: int
    """The size of the codec."""

    num_decoder_layers: int
    """The number of Transformer decoder layers."""

    num_decoder_attn_heads: int
    """The number of attention heads in Transformer decoder layers."""

    ffn_inner_dim: int
    """The inner dimensionality of Transformer feed-forward networks."""

    dropout_p: float
    """The dropout probability in Transformer layers."""


def _12_layer() -> NarConfig:
    return NarConfig(
        model_dim=1024,
        max_seq_len=3000,
        total_steps=15,
        codec_size=1024,
        num_decoder_layers=12,
        num_decoder_attn_heads=16,
        ffn_inner_dim=1024 * 4,
        dropout_p=0.1,
    )


class NarBuilder:
    """Builds modules of an NLLB model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.2207.04672`.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    config: NarConfig
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: NarConfig,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param config:
            The configuration to use.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        self.config = config
        self.device = device
        self.dtype = dtype

    def build_model(self) -> NarModel:
        """Build a model."""
        a_embed = self.build_embedding()

        frontend = self.build_frontend(a_embed)

        decoder = self.build_decoder()

        final_proj = MultiStepTiedProjection([a.weight for a in a_embed[1:]])

        return NarModel(
            frontend, decoder, final_proj, target_pad_idx=None
        )

    def build_adaln(self) -> ZeroAdaLN:
        """Build an adaptive layer normalization layer."""
        return ZeroAdaLN(
            self.config.model_dim,
            self.config.total_steps,
            device=self.device,
            dtype=self.dtype,
        )

    def build_embedding(self):
        """Build an embedding table."""
        a_embed = [Embedding(
            num_embeddings=self.config.codec_size,
            embedding_dim=self.config.model_dim,
            pad_idx=None,
            scaled=True,
            device=self.device,
            dtype=self.dtype,
        ) for _ in range(self.config.total_steps + 1)]

        return  a_embed

    def build_frontend(self, a_embed: List[Embedding]) -> TransformerFrontend:
        """Build a Transformer encoder/decoder front-end."""

        # pos_encoder = SinusoidalPositionEncoder(
        #     self.config.model_dim,
        #     max_seq_len=self.config.max_seq_len,
        #     device=self.device,
        #     dtype=self.dtype,
        # )

        return ValleFrontend(
            a_embed,
            # pos_encoder=pos_encoder,
            dropout_p=self.config.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_decoder(self) -> TransformerDecoder:
        """Build a Transformer decoder."""
        num_layers = self.config.num_decoder_layers

        layers = [self.build_decoder_layer() for _ in range(num_layers)]

        return NarTransformerDecoder(
            layers,
            device=self.device,
            dtype=self.dtype,
        )


    def build_decoder_layer(self) -> TransformerDecoderLayer:
        """Build a Transformer decoder layer."""
        self_attn = self.build_attention(self.config.num_decoder_attn_heads)

        ffn = self.build_ffn()

        adaLN = self.build_adaln()

        return NarTransformerDecoderLayer(
            self_attn,
            ffn,
            adaLN=adaLN,
            dropout_p=self.config.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_attention(self, num_heads: int) -> MultiheadAttention:
        """Build a Transformer multi-head attention layer."""
        sdpa = create_default_sdpa(attn_dropout_p=self.config.dropout_p)
        pos_encoder = RotaryEncoder(
            encoding_dim=64,
            max_seq_len=self.config.max_seq_len,
        )

        return StandardMultiheadAttention(
            self.config.model_dim,
            num_heads,
            pos_encoder=pos_encoder,
            sdpa=sdpa,
            device=self.device,
            dtype=self.dtype,
        )

    def build_ffn(self) -> FeedForwardNetwork:
        """Build a Transformer feed-forward network."""
        return StandardFeedForwardNetwork(
            self.config.model_dim,
            self.config.ffn_inner_dim,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
        )


def create_nar_model(
    config: NarConfig,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> NarModel:
    """Create an NLLB model.

    :param config:
        The configuration to use.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    return NarBuilder(config, device, dtype).build_model()

if __name__ == '__main__':
    import torch
    config = _12_layer()
    model = create_nar_model(config, device=torch.device('cuda'))
    # print(model)
    s = [
        torch.ones(5, dtype=torch.long).cuda(),  # Example tensor with sequence length 5
        torch.ones(7, dtype=torch.long).cuda()*2,  # Example tensor with sequence length 7
        torch.ones(3, dtype=torch.long).cuda()*3   # Example tensor with sequence length 3
    ]

    a_p = [
        torch.ones(50, 8, dtype=torch.long).cuda(),  # Example tensor with sequence length 5
        torch.ones(70, 8, dtype=torch.long).cuda()*2,  # Example tensor with sequence length 7
        torch.ones(30, 8, dtype=torch.long).cuda()*3   # Example tensor with sequence length 3
    ]

    a = [
        torch.ones(50, 2, dtype=torch.long).cuda(),  # Example tensor with sequence length 5
        torch.ones(70, 2, dtype=torch.long).cuda()*2,  # Example tensor with sequence length 7
        torch.ones(30, 2, dtype=torch.long).cuda()*3   # Example tensor with sequence length 3
    ]

    class_ids = torch.tensor([1, 0, 1]).cuda()
    model(s, a_p, a, class_ids)
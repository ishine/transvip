# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, List

from fairseq2_011.data import VocabularyInfo
from fairseq2_011.nn.position_encoder import SinusoidalPositionEncoder
from fairseq2_011.nn.position_encoder import RotaryEncoder
from fairseq2_011.nn.transformer import (
    FeedForwardNetwork,
    MultiheadAttention,
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
    TransformerNormOrder,
    create_default_sdpa,
)
from fairseq2_011.typing import DataType, Device

from .spk_encoder import (
    TransformerFrontend, 
    SpeakerEncoderFrontend, 
    SpeakerEncoder, 
    SpeakerEncoderLayer
    )


@dataclass
class SpeakerEncoderConfig:
    """Holds the configuration of an NLLB model."""

    model_dim: int
    """The dimensionality of the model."""

    max_seq_len: int
    """The expected maximum sequence length."""

    num_layers: int
    """The number of Transformer encoder layers."""

    num_attn_heads: int
    """The number of attention heads in Transformer decoder layers."""

    ffn_inner_dim: int
    """The inner dimensionality of Transformer feed-forward networks."""

    dropout_p: float
    """The dropout probability in Transformer layers."""

    def update_vocabulary(self, info: VocabularyInfo) -> None:
        """Update vocabulary configuration from ``info``."""
        self.vocabulary_size, self.pad_idx = info.size, info.pad_idx

def _speaker_6_layer() -> SpeakerEncoderConfig:
    return SpeakerEncoderConfig(
        model_dim=1024,
        max_seq_len=500, # equal to 10 seconds
        num_layers=6,
        num_attn_heads=16,
        ffn_inner_dim=1024 * 4,
        dropout_p=0.1,
    )


class SpeakerEncoderBuilder:
    config: SpeakerEncoderConfig
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: SpeakerEncoderConfig,
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

    def build_model(self) -> SpeakerEncoder:
        """Build a Transformer encoder."""
        num_layers = self.config.num_layers

        frontend = self.build_frontend()

        layers = [self.build_layer() for _ in range(num_layers)]

        return SpeakerEncoder(
            frontend,
            layers,
            device=self.device,
            dtype=self.dtype,
        )

    def build_layer(self) -> SpeakerEncoderLayer:
        """Build a Transformer encoder layer."""
        self_attn = self.build_attention(self.config.num_attn_heads)

        ffn = self.build_ffn()

        return SpeakerEncoderLayer(
            self_attn,
            ffn,
            dropout_p=self.config.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_frontend(self) -> TransformerFrontend:
        """Build a Transformer encoder/decoder front-end."""
        pos_encoder = SinusoidalPositionEncoder(
            self.config.model_dim,
            self.config.max_seq_len,
            device=self.device,
            dtype=self.dtype,
        )

        frontend = SpeakerEncoderFrontend(
            self.config.model_dim,
            pos_encoder,
            dropout_p=self.config.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

        return frontend

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


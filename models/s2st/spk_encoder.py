from abc import ABC, abstractmethod
from typing import Optional, Tuple, cast, final, List, Union,  Iterable, Protocol


from torch import Tensor
from torch.nn import Dropout, Module

from fairseq2_011.typing import DataType, Device, finaloverride

from fairseq2_011.nn.module_list import ModuleList
from fairseq2_011.nn.position_encoder import PositionEncoder
from fairseq2_011.nn.utils.mask import to_padding_mask
from fairseq2_011.nn.utils.module import check_model_dim

from fairseq2_011.nn.transformer.multihead_attention import MultiheadAttention
from fairseq2_011.nn.transformer.encoder_layer import TransformerEncoderLayer
from fairseq2_011.nn.transformer.decoder_layer import TransformerDecoderLayer
from fairseq2_011.nn.transformer.encoder import TransformerEncoder
from fairseq2_011.nn.transformer.ffn import FeedForwardNetwork
from fairseq2_011.nn.transformer.layer_norm import (
    LayerNormFactory,
    create_default_layer_norm,
)
from fairseq2_011.nn.normalization import LayerNorm


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
class SpeakerEncoderFrontend(TransformerFrontend):

    def __init__(
        self,
        model_dim: int,
        pos_encoder: Optional[PositionEncoder] = None,
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

        super().__init__(model_dim)

        if layer_norm_fn is None:
            layer_norm_fn = create_default_layer_norm

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

    @finaloverride
    def forward(
        self,
        seqs,
        seq_lens,
    ):

        padding_mask = to_padding_mask(seqs, seq_lens)
        seqs = self.pos_encoder(seqs, padding_mask)

        if self.layer_norm is not None:
            seqs = self.layer_norm(seqs)

        if self.dropout is not None:
            seqs = self.dropout(seqs)
            
        return seqs, padding_mask



    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return s

@final
class SpeakerEncoderLayer(TransformerEncoderLayer):
    """Represents a Transformer decoder layer as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    self_attn: MultiheadAttention
    self_attn_norm: Optional[LayerNorm]
    self_attn_dropout: Optional[Dropout]
    self_attn_layer_norm: LayerNorm
    ffn: FeedForwardNetwork
    ffn_dropout: Optional[Dropout]
    ffn_layer_norm: LayerNorm

    def __init__(
        self,
        self_attn: MultiheadAttention,
        ffn: FeedForwardNetwork,
        dropout_p: float = 0.1,
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

        check_model_dim(self)

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[Tensor],
        self_attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        seqs = self._forward_self_attn(seqs, padding_mask, self_attn_mask)

        seqs = self._forward_ffn(seqs)

        return seqs, padding_mask

    def _forward_self_attn(
        self,
        seqs: Tensor,
        padding_mask: Optional[Tensor],
        self_attn_mask: Optional[Tensor],
    ) -> Tensor:
        residual = seqs

        seqs = self.self_attn_layer_norm(seqs)

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

        seqs = seqs + residual

        return seqs


    def _forward_ffn(self, 
        seqs: Tensor,
        ) -> Tensor:
        residual = seqs

        seqs = self.ffn_layer_norm(seqs)

        seqs = self.ffn(seqs)

        if self.ffn_dropout is not None:
            seqs = self.ffn_dropout(seqs)

        seqs = seqs + residual

        return seqs

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return s


@final
class SpeakerEncoder(TransformerEncoder):
    """Represents a Transformer decoder as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    layer_norm: Optional[LayerNorm]

    def __init__(
        self,
        frontend: TransformerFrontend,
        layers: Iterable[TransformerDecoderLayer],
        layer_drop_p: float = 0.0,
        layer_norm_fn: Optional[LayerNormFactory] = None,
        return_all_hiddens: bool = False,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param layers:
            The decoder layers.
        :param self_attn_mask_gen:
            The attention mask generator. If ``None``, an instance of
            :class:`CausalAttentionMaskGenerator` will be used.
        :param layer_drop_p:
            If greater than zero, applies LayerDrop to the decoder layers as
            described in :cite:t:`https://doi.org/10.48550/arxiv.1909.11556`.
        :param norm_order:
            The Layer Normalization order to use.
        :param layer_norm_fn:
            The factory to use to construct the Layer Normalization module.
        """
        layer_list = ModuleList(layers, layer_drop_p)
        if not layer_list:
            raise ValueError("`layers` must be non-empty.")

        model_dim = layer_list[0].model_dim

        super().__init__(model_dim)

        if layer_norm_fn is None:
            layer_norm_fn = create_default_layer_norm

        self.frontend = frontend

        self.layers = layer_list

        self.layer_norm = layer_norm_fn(model_dim, device, dtype)

        self.return_all_hiddens = return_all_hiddens

        check_model_dim(self)

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        seq_lens: Tensor,
        # padding_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        
        seqs, padding_mask = self.frontend(seqs, seq_lens)

        for layer_idx, layer in enumerate(self.layers.drop_iter()):
            seqs, padding_mask = layer(
                seqs,
                padding_mask,
                None, # None for NAR decoder
            )

        seqs = self.layer_norm(seqs)

        if self.return_all_hiddens:
            return seqs, padding_mask

        return seqs.mean(dim=1)

    def extra_repr(self) -> str:
        """:meta private:"""
        return super().extra_repr()

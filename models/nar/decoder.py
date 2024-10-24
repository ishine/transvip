from abc import ABC, abstractmethod
from typing import Iterable, Optional, Protocol, Tuple, final

from torch import Tensor

from fairseq2_011.nn.module_list import ModuleList
from fairseq2_011.nn.normalization import LayerNorm

from fairseq2_011.nn.transformer.decoder_layer import TransformerDecoderLayer
from fairseq2_011.nn.transformer.layer_norm import (
    LayerNormFactory,
    create_default_layer_norm,
)
from fairseq2_011.nn.transformer.decoder import TransformerDecoder
from fairseq2_011.nn.utils.module import check_model_dim
from fairseq2_011.typing import DataType, Device, finaloverride

@final
class NarTransformerDecoder(TransformerDecoder):
    """Represents a Transformer decoder as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    layer_norm: Optional[LayerNorm]

    def __init__(
        self,
        layers: Iterable[TransformerDecoderLayer],
        layer_drop_p: float = 0.0,
        layer_norm_fn: Optional[LayerNormFactory] = None,
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

        self.layers = layer_list

        self.layer_norm = layer_norm_fn(model_dim, device, dtype)

        self.output_residual = True

        check_model_dim(self)

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[Tensor],
        step: Optional[int] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        
        residual = seqs

        for layer_idx, layer in enumerate(self.layers.drop_iter()):
            seqs, padding_mask = layer(
                seqs,
                padding_mask,
                None, # None for NAR decoder
                step,
            )

        if self.output_residual:
            seqs = seqs - residual

        seqs = self.layer_norm(seqs)

        return seqs, padding_mask, step

    def extra_repr(self) -> str:
        """:meta private:"""
        return super().extra_repr()

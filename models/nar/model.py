# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, final, List

import torch
from torch import Tensor
from torch.nn import Module, ParameterList
from torch.nn.functional import linear
from torch.nn.parameter import Parameter
from typing import Optional, final
from dataclasses import dataclass

from fairseq2_011.models.decoder import DecoderModel
from fairseq2_011.nn.projection import Projection
from fairseq2_011.nn.transformer import TransformerDecoder
from fairseq2_011.nn.utils.module import check_model_dim
from fairseq2_011.typing import DataType, Device, finaloverride

from .frontend import ValleFrontend


@dataclass
class NarModelInput:
    ap: List[Tensor]
    a: List[Tensor]
    seq_lens: Tensor
    a_lens: Tensor
    langs: List[str]
    def __init__(
        self,
        ap: List[Tensor],
        a: List[Tensor],
        langs: List[str],
        seq_lens: Tensor = None,
        a_lens: Tensor = None,
    ) -> None:
        self.ap = ap
        self.a = a
        if seq_lens is None:
            seq_lens = torch.tensor(
                [a[i].shape[-1] + ap[i].shape[-1] for i in range(len(ap))],
                dtype=torch.long,
                device=ap[0].device,
                )
        if a_lens is None:
            a_lens = torch.tensor(
                [a[i].shape[-1] for i in range(len(a))],
                dtype=torch.long,
                device=a[0].device,
                )
        self.seq_lens = seq_lens + 1 # add 1 for the lang token
        self.a_lens = a_lens
        self.langs = langs


@dataclass
class NarModelOutput:
    logits: Tensor
    a_logits: List[Tensor]



class MultiStepTiedProjection(Module):
    """Applies a linear transformation to incoming data."""

    input_dim: int
    output_dim: int
    total_steps: int
    weights: ParameterList

    def __init__(
        self, 
        weights: List[Parameter]
        ) -> None:
        """
        :param input_dim:
            The dimensionality of inputs.
        :param output_dim:
            The dimensionality of projected outputs.
        """
        super().__init__()

        self.output_dim, self.input_dim = weights[0].shape
        self.weights = ParameterList(weights)
        self.total_steps = len(weights)

    def forward(self, x: Tensor, step: int) -> Tensor:
        return linear(x, self.weights[step])
    
    def extra_repr(self) -> str:
        """:meta private:"""
        return f"input_dim={self.input_dim}, output_dim={self.output_dim}"



@final
class NarModel(DecoderModel):
    """Represents a Transformer model as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    decoder_frontend: ValleFrontend
    decoder: TransformerDecoder
    final_proj: Projection
    target_pad_idx: Optional[int]

    def __init__(
        self,
        decoder_frontend: ValleFrontend,
        decoder: TransformerDecoder,
        final_proj: Projection,
        target_pad_idx: Optional[int],
    ) -> None:
        """
        :param encoder_frontend:
            The encoder frontend.
        :param encoder:
            The encoder.
        :param decoder_frontend:
            The decoder frontend.
        :param decoder:
            The decoder.
        :param final_proj:
            The projection to apply to decoder outputs to produce logits.
        :param target_pad_idx:
            The index of the pad symbol in the target vocabulary.
        """
        model_dim = decoder.model_dim

        super().__init__(model_dim)

        self.decoder_frontend = decoder_frontend
        self.decoder = decoder

        self.final_proj = final_proj

        self.target_pad_idx = target_pad_idx
        self.total_steps = len(self.decoder_frontend.a_embeds) - 1

        check_model_dim(self)

    @finaloverride
    def decode(
        self,
        inputs: NarModelInput
    ) -> Tuple[Tensor, Optional[Tensor]]:
        seqs, padding_mask, step = self.decoder_frontend(inputs.ap, inputs.a, inputs.langs)
        output, padding_mask, step =self.decoder( 
            seqs, padding_mask, step
        ) 

        return output, seqs, padding_mask, step
    
    @finaloverride
    def forward(self, inputs: NarModelInput) -> NarModelOutput:
        output_seq, input_seq, padding_mask, step = self.decode(inputs)

        return self.project(inputs, output_seq, step, padding_mask)

    @finaloverride
    def project(
        self, inputs: NarModelInput, 
        decoder_output: Tensor, step: int, 
        decoder_padding_mask: Optional[Tensor]
    ) -> NarModelOutput:
        logits = self.final_proj(decoder_output, step)
        acoustic_logits = [logits[i, inputs.seq_lens[i] - inputs.a_lens[i]:inputs.seq_lens[i], :] for i in range(len(inputs.seq_lens))]

        return NarModelOutput(logits=logits, a_logits=acoustic_logits)
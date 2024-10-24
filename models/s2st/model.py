# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, Tuple, final

from fairseq2_011.models.encoder_decoder import EncoderDecoderModel
from fairseq2_011.models.sequence import SequenceModelOutput
from fairseq2_011.models.transformer.frontend import TransformerFrontend
from fairseq2_011.models.seq2seq import Seq2SeqBatch
from fairseq2_011.nn.incremental_state import IncrementalStateBag
from fairseq2_011.nn.projection import Projection
from fairseq2_011.nn.transformer import TransformerDecoder, TransformerEncoder
from fairseq2_011.nn.position_encoder import LearnedPositionEncoder
from fairseq2_011.nn.utils.module import check_model_dim
from fairseq2_011.nn.embedding import Embedding
from overrides import final as finaloverride
import torch
from torch import Tensor
from torch import nn
from torch.nn import Module
import random


class LengthControlModule(Module):
    def __init__(
            self, 
            model_dim: int, 
            max_seq_len: int, 
            use_vad: bool = True,
            drop_initial: float = 0.5,
            drop_final: float = 0.0,
            drop_step: float = 10000,
            ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len

        self.drop_initial = drop_initial
        self.drop_final = drop_final
        self.drop_step = drop_step
        self.current_step = 0
        self.scale = 0.1

        self.reversed_pos_embed = Embedding(max_seq_len, model_dim)
        self.pos_embed = LearnedPositionEncoder(model_dim, max_seq_len)
        self.register_buffer('range_tensor', torch.arange(1, max_seq_len + 1, dtype=torch.long)) 
        if use_vad:
            self.vad_embed = Embedding(2, model_dim)
        else:
            self.register_buffer('vad_embed', None)

    def forward(self, x: Tensor, padding_mask: Tensor, target_length: Optional[Tensor] = None, vad_mask: Optional[Tensor] = None) -> Tensor:
        if self.training:
            self.current_step += 1
        p_drop = self.drop_initial + (self.drop_final - self.drop_initial) * min(self.current_step / self.drop_step, 1)
        if target_length is None or (self.training and torch.rand(1) < p_drop):
            return x, padding_mask
        assert torch.all(target_length > 0) and torch.all(target_length <= self.max_seq_len)
        target_length = target_length.flatten()
        bsz, dim = target_length.shape[0], self.model_dim
        reverse_pos_idx = (target_length.view(bsz, 1) - self.range_tensor) % self.max_seq_len
        lc_embed = self.reversed_pos_embed(reverse_pos_idx)
        lc_embed = lc_embed[:, :target_length.max(), :]
        if vad_mask is not None:
            lc_embed = lc_embed + self.vad_embed(vad_mask)
        lc_embed = self.pos_embed(lc_embed)

        if padding_mask is not None:
            lc_embed_mask = to_padding_mask(lc_embed, target_length)
            padding_mask_out = torch.cat([padding_mask, lc_embed_mask], dim=1)
        else:
            padding_mask_out = None
        out = torch.cat([x, lc_embed], dim=1)


        return out, padding_mask_out



def to_padding_mask(seqs: Tensor, seq_lens: Optional[Tensor]) -> Optional[Tensor]:
    if seq_lens is None:
        return None
    batch_size, mask_seq_len = seqs.shape[:2]
    indices = torch.arange(mask_seq_len, device=seq_lens.device).expand(batch_size, -1)
    bool_mask = indices >= seq_lens.unsqueeze(1).expand(-1, mask_seq_len)
    mask = seqs.new_zeros((batch_size, mask_seq_len))
    mask.masked_fill_(bool_mask, -torch.inf)
    return mask



@final
class UnitYModel(EncoderDecoderModel):
    """Represents a UnitY model as described in
    :cite:t`https://doi.org/10.48550/arxiv.2212.08055`.

    Note that this implementation is augmented with a text encoder to enable
    translating from text.
    """

    model_dim: int
    input_modality: str
    speech_encoder_frontend: TransformerFrontend
    speech_encoder: TransformerEncoder
    text_encoder_frontend: Optional[TransformerFrontend]
    text_encoder: Optional[TransformerEncoder]
    text_decoder_frontend: TransformerFrontend
    text_decoder: TransformerDecoder
    spk_encoder: Module
    length_control_module: Optional[LengthControlModule]
    final_proj: Projection
    pad_idx: Optional[int]

    def __init__(
        self,
        speech_encoder_frontend: TransformerFrontend,
        speech_encoder: TransformerEncoder,
        text_encoder_frontend: Optional[TransformerFrontend],
        text_encoder: Optional[TransformerEncoder],
        text_decoder_frontend: TransformerFrontend,
        text_decoder: TransformerDecoder,
        spk_encoder: Module,
        final_proj: Projection,
        length_control_module: Optional[LengthControlModule],
        pad_idx: Optional[int],
        input_modality: str = "speech",
        # use_transform: bool = True,
        use_transform: bool = False,
    ) -> None:
        model_dim = speech_encoder.model_dim

        super().__init__(model_dim)

        self.input_modality = input_modality

        self.spk_encoder = spk_encoder

        self.speech_encoder_frontend = speech_encoder_frontend
        self.speech_encoder = speech_encoder

        if text_encoder is not None:
            if text_encoder_frontend is None:
                raise ValueError(
                    "Both `text_encoder` and `text_encoder_frontend` must be specified, but `text_encoder_frontend` is `None`."
                )

            self.text_encoder_frontend = text_encoder_frontend
            self.text_encoder = text_encoder
        else:
            if text_encoder_frontend is not None:
                raise ValueError(
                    "Both `text_encoder` and `text_encoder_frontend` must be specified, but `text_encoder` is `None`."
                )

            self.register_module("text_encoder_frontend", None)
            self.register_module("text_encoder", None)

        self.text_decoder_frontend = text_decoder_frontend
        self.text_decoder = text_decoder

        self.final_proj = final_proj

        self.pad_idx = pad_idx
        if length_control_module is not None:
            self.length_control_module = length_control_module
        else:
            self.register_module("length_control_module", None)

        if use_transform:
            self.transform = nn.Linear(model_dim, model_dim)
        else:
            self.transform = nn.Identity()

        check_model_dim(self)


    @finaloverride
    def encode(
        self, seqs: Tensor, seq_lens: Optional[Tensor],
        target_length: Optional[Tensor] = None,
        vad_mask: Optional[Tensor] = None,
        prompts: Optional[Tensor] = None,
        prompt_lens: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if self.input_modality == "speech":
            encoder_output, encoder_padding_mask = self.encode_speech(seqs, seq_lens)
        elif self.input_modality == "text":
            encoder_output, encoder_padding_mask = self.encode_text(seqs, seq_lens)
        else:
            raise RuntimeError(
            f"`input_modality` must be 'speech' or 'text', but is '{self.input_modality}' instead."
        )
        if self.length_control_module is not None and target_length is not None:
            encoder_output, encoder_padding_mask = self.length_control_module(
                encoder_output, encoder_padding_mask, target_length, vad_mask
                )
        if prompts is not None:
            spk_embed = self.spk_encoder(prompts, prompt_lens)
            spk_embed = self.transform(spk_embed)
        else:
            spk_embed = None
        
        return encoder_output, encoder_padding_mask, spk_embed


    def encode_speech(
        self, seqs: Tensor, seq_lens: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        seqs, padding_mask = self.speech_encoder_frontend(seqs, seq_lens)

        return self.speech_encoder(seqs, padding_mask)  # type: ignore[no-any-return]

    def encode_text(
        self, seqs: Tensor, seq_lens: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if self.text_encoder is None:
            raise ValueError(
                "`encode_text()` requires a text encoder, but the current UnitY model does not have one."
            )

        assert self.text_encoder_frontend is not None

        seqs, padding_mask = self.text_encoder_frontend(seqs, seq_lens)

        return self.text_encoder(seqs, padding_mask)  # type: ignore[no-any-return]

    @finaloverride
    def decode(
        self,
        seqs: Tensor,
        seq_lens: Optional[Tensor],
        encoder_output: Tensor,
        encoder_padding_mask: Optional[Tensor],
        spk_embed: Optional[Tensor],
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        seqs, padding_mask = self.text_decoder_frontend(seqs, seq_lens, spk_embed, state_bag)

        return self.text_decoder(  # type: ignore[no-any-return]
            seqs, padding_mask, encoder_output, encoder_padding_mask, state_bag
        )

    @finaloverride
    def project(
        self, decoder_output: Tensor, decoder_padding_mask: Optional[Tensor]
    ) -> SequenceModelOutput:
        logits = self.final_proj(decoder_output)
        return SequenceModelOutput(logits, self.pad_idx)

    @finaloverride
    def forward(
        self, 
        batch: Seq2SeqBatch,
        target_length: Optional[Tensor] = None,
        vad_mask: Optional[Tensor] = None,
        prompts: Optional[Tensor] = None,
        prompt_lens: Optional[Tensor] = None,
        ) -> SequenceModelOutput:

        encoder_output, encoder_padding_mask, spk_embed = self.encode(
            batch.source_seqs, batch.source_seq_lens, target_length, vad_mask, prompts, prompt_lens
        )
        decoder_output, decoder_padding_mask = self.decode(
            batch.target_seqs,
            batch.target_seq_lens,
            encoder_output,
            encoder_padding_mask,
            spk_embed,
        )

        return self.project(decoder_output, decoder_padding_mask)



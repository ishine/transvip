# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, List, Tuple
import copy
from dataclasses import dataclass

import torch
from torch import Tensor
from fairseq2_011.data.text import TextTokenizer
from fairseq2_011.data.vocabulary_info import VocabularyInfo
from fairseq2_011.generation.logits_processor import LogitsProcessor
from .model import UnitYModel
from .fairseq2_custom.sequence_to_text_generator import SequenceToTextGenerator

@dataclass
class SequenceGeneratorOptions:
    """Holds the options to pass to a sequence generator."""

    beam_size: int = 5
    """The beam size."""

    min_seq_len: int = 1
    """The minimum length of generated sequences (including prefix sequence)."""

    soft_max_seq_len: Optional[Tuple[int, int]] = (1, 200)
    """The terms ``a`` and ``b`` of ``ax + b`` where ``x`` is the source
    sequence length. The generated sequences (including prefix sequence) will
    have the maximum length of ``min(hard_max_seq_len, ax + b)``. See also
    ``hard_max_seq_len``."""

    hard_max_seq_len: int = 1024
    """The hard limit on maximum length of generated sequences."""

    len_penalty: float = 1.0
    """The length penalty, where values less than 1.0 favor shorter, values
    greater than 1.0 favor longer sequences."""

    unk_penalty: float = 0.0
    """The unknown symbol penalty, where values less than 0 produce more UNKs,
    values greater than 0 produce fewer UNKs."""

    normalize_scores: bool = True
    """If ``True``, normalizes scores by the length of generated sequences."""

    search = None
    """The beam search algorithm to use."""

    logits_processor = None
    """Logits processor called before applying beam search step."""

class NGramRepeatBlockProcessor(LogitsProcessor):
    def __init__(self, no_repeat_ngram_size: int) -> None:
        self.no_repeat_ngram_size = no_repeat_ngram_size

    def __call__(self, seqs: Tensor, lprobs: Tensor) -> None:
        """Remove repeating n-gram tokens."""
        batch_size, beam_size, vocab_size = lprobs.size()
        step_nr = seqs.size(2) - 1
        # (N, B, S) -> (N * B, S)
        seqs = seqs.view(-1, seqs.size(2))
        # (N, B, V) -> (N * B, V)
        lprobs = lprobs.view(-1, vocab_size)
        self._no_repeat_ngram(seqs, lprobs, batch_size, beam_size, step_nr)

    def _no_repeat_ngram(
        self,
        seqs: Tensor,
        lprobs: Tensor,
        batch_size: int,
        beam_size: int,
        step_nr: int,
    ) -> Tensor:
        """For each hypothesis generate a list of previous ngrams
            and set associated lprobs to -inf

        :param seqs: The generated sequences of tokens for the first
            `step_nr` steps of decoding (N * B, step_nr + 1)
        :param lprobs: The next-step log probability reshaped to (N * B, V)
        :param batch_size: The batch size.
        :param beam_size: The beam size.
        :param step_nr: Step number for decoding.

        :returns:
            modified lprobs tensor with banned tokens set to -inf
        """
        banned_tokens = [[] for _ in range(batch_size * beam_size)]

        if step_nr + 2 - self.no_repeat_ngram_size >= 0:
            cpu_tokens: List[List[int]] = seqs.cpu().tolist()
            check_start_pos = step_nr + 2 - self.no_repeat_ngram_size
            for bbsz_idx in range(batch_size * beam_size):
                ngram_to_check = cpu_tokens[bbsz_idx][
                    -(self.no_repeat_ngram_size - 1) :
                ]
                for i in range(check_start_pos):
                    if (
                        ngram_to_check
                        == cpu_tokens[bbsz_idx][i : i + self.no_repeat_ngram_size - 1]
                    ):
                        banned_tokens[bbsz_idx].append(
                            cpu_tokens[bbsz_idx][i + self.no_repeat_ngram_size - 1]
                        )
        for bbsz_idx in range(batch_size * beam_size):
            lprobs[bbsz_idx, banned_tokens[bbsz_idx]] = -torch.inf
        return lprobs


@dataclass
class M4tValleGeneratorOutput:
    text_hyps: List[List[int]]
    codec_hyps: List[List[int]]
    text_scores: List[List[float]]
    codec_scores: List[List[float]]

def remove_consecutive_repeated_ngrams(
    sequence: List[int], min_size: int = 1, max_size: int = 40
):
    assert 1 <= min_size <= max_size
    drop_idx = set()  # indices that will be dropped from the sequence

    # start from the beginning, check if an ngram of size k (for k=max..min) is
    # followed by its copy, if so delete the first one, and start over after
    # the deleted ngram.
    start = 0
    while start < len(sequence):
        for k in range(max_size, min_size - 1, -1):
            if sequence[start : start + k] == sequence[start + k : start + k + k]:
                drop_idx |= set(range(start, start + k))
                start += k - 1  # assumes repeating subsequences don't overlap
                break
        start += 1
    return [token for idx, token in enumerate(sequence) if idx not in drop_idx]


class UnitYGenerator:
    """Generates text translations and speech units from a UnitY model."""

    model: UnitYModel
    generator: SequenceToTextGenerator

    def __init__(
        self,
        model: UnitYModel,
        text_tokenizer: TextTokenizer,
        target_lang: str,
        output_modality: str = "TextAndSpeech",
        text_opts: Optional[SequenceGeneratorOptions] = None,
    ) -> None:
        """
        :param model:
            The UnitY model to use for generation.
        :param text_tokenizer:
            The text tokenizer to use.
        :param unit_tokenizer:
            The unit tokenizer to use.
        :param target_lang:
            The target language.
        :param text_generator_opts:
            The options to pass to the underlying text :class:`Seq2SeqGenerator`.
        :param unit_generator_opts:
            The options to pass to the underlying unit :class:`Seq2SeqGenerator`.
        """

        model.eval()

        self.model = model
        self.output_modality = output_modality
        if output_modality == "TextOnly":
            text_tokenizer = copy.deepcopy(text_tokenizer)
            new_vocab_info = VocabularyInfo(
                size=text_tokenizer.vocab_info.size,
                pad_idx=text_tokenizer.vocab_info.pad_idx,
                bos_idx=text_tokenizer.vocab_info.bos_idx,
                eos_idx=2,
                unk_idx=text_tokenizer.vocab_info.unk_idx,
            )
            text_tokenizer.vocab_info = new_vocab_info # Stop generation when we see separator token.
        self.tokenizer = text_tokenizer

        self.generator = SequenceToTextGenerator(
            self.model, text_tokenizer, target_lang, text_opts
        )


    @torch.inference_mode()
    def __call__(
        self,
        source_seqs: Tensor,
        source_seq_lens: Optional[Tensor],
        ngram_filtering: bool = False,
        **kwargs,
    ):
        """
        :param source_seqs:
            The source sequences to use for generation. *Shape:* :math:`(N,S,*)`,
            where :math:`N` is the batch size, :math:`S` is the sequence length,
            and :math:`*` is any number of sequence-specific dimensions
            including none.
        :param source_seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``source_seqs``. *Shape:* :math:`(N)`, where
            :math:`N` is the batch size.
        :param input_modality:
            The type of modality to encode.
        :param output_modality:
            The type of modality to decode.

        :returns:
            - The output of the text generator.
            - The output of the unit generator.
        """


        text_output = self.generator.generate_ex(source_seqs, source_seq_lens,**kwargs)
        
        results = text_output.generator_output.results
        if self.output_modality in ['TextOnly', "TextOri"]:
            # text_hyps = [r[0].seq.flatten().detach().cpu().tolist() for r in results]
            text_hyps = []
            for r in results:
                try:
                    text_hyps.append(r[0].seq.flatten().detach().cpu().tolist())
                except:
                    print("Warning: No best hypothesis found in the output. Skipping codec generation.")
                    text_hyps.append([2])
            # text_scores = [r[0].step_scores.exp().flatten().detach().cpu().tolist() for r in results]
            codec_hyps = None
        else:
            text_hyps, codec_hyps, = self._split_text_codec(results)

        # We skip T2U when we only need to output text.
        return M4tValleGeneratorOutput(
            text_hyps=text_hyps,
            codec_hyps=codec_hyps,
            text_scores=None,
            codec_scores=None,
        )
    
    def _split_text_codec(self, gen_output):
        # best_hyps = [output[0].seq.flatten().detach().cpu().tolist() for output in gen_output]
        # scores = [output[0].step_scores.exp().flatten().detach().cpu().tolist() for output in gen_output]
        best_hyps = []
        for output in gen_output:
            try:
                best_hyps.append(output[0].seq.flatten().detach().cpu().tolist())
            except:
                print("Warning: No best hypothesis found in the output. Skipping codec generation.")
                best_hyps.append([2])

        text_hyps, codec_hyps = [], []
        # text_scores, codec_scores = [], []
        for hyp in best_hyps:
            hyp = [str(t) for t in hyp]
            hyp = " ".join(hyp).split(' 2 ')
            if len(hyp) != 2:
                print("Warning: No separator token found in the output. Skipping codec generation.")
                t, c = [2], [0] * 50
            else:
                t = [int(t) for t in hyp[0].split(' ')]
                c = [int(t) for t in hyp[1].split(' ')]
                if c[-1] == 3 or c[-1] == 2:
                    c = c[:-1]
                c = [idx - self.tokenizer.vocab_info.size for idx in c]
            text_hyps.append(t)
            codec_hyps.append(c)
        return text_hyps, codec_hyps



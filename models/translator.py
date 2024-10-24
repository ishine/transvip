import os
import os.path as osp
from typing import List, Optional

import torch
from torch.nn import Module
from torch import Tensor

from fairseq2.data import Collater
from fairseq2.data.audio import WaveformToFbankConverter

from sascodec import SASCodec
from silero_vad import load_silero_vad, get_speech_timestamps

import librosa
import soundfile as sf

if __name__ == "__main__":
    import sys
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(BASE_DIR)

from fairseq2_011.generation import SequenceGeneratorOptions
from models.s2st import UnitYModel, load_s2st_model, load_m4t_tokenizer, NllbTokenizer, UnitYGenerator, NGramRepeatBlockProcessor
from models.nar import NarModel, load_nar_model, NarModelInput
from models.nar.generater import NarGenerater


LANG_NARID_MAP = {
    'eng': 0,
    'fra': 1,
}

from audiotools import AudioSignal
from audiotools.data import transforms
from audiotools.core.util import prepare_batch

def build_transform():
    preprocess = transforms.Compose(
        transforms.Identity(),
        name="preprocess",
     )
    augment = transforms.Compose(
        transforms.Identity(),
        name="augment",
    )
    postprocess = transforms.Compose(  
        transforms.VolumeNorm(),
        transforms.RescaleAudio(),
        transforms.ShiftPhase(),
        name="postprocess",
    )
    transform = transforms.Compose(preprocess, augment, postprocess)
    return transform


class S2STranslatorlInput:
    wav: Optional[List[Tensor]] # bsz * len
    fbank: Optional[Tensor]
    fbank_lens: Optional[Tensor]
    src_codecs: Optional[List[List[int]]]
    target_length: Optional[Tensor]
    vad_mask: Optional[Tensor]
    target_langs: List[str]

    def __init__(
        self,
        wav: Optional[List[Tensor]] = None,
        fbank: Optional[Tensor] = None,
        fbank_lens: Optional[Tensor] = None,
        prompts: Optional[Tensor] = None,
        prompt_lens: Optional[Tensor] = None,
        src_codecs: Optional[List[List[int]]] = None,
        target_length: Optional[Tensor] = None,
        vad_mask: Optional[Tensor] = None,
        target_langs: List[str] = None,
    ) -> None:
        self.wav = wav
        self.fbank = fbank
        self.fbank_lens = fbank_lens
        self.src_codecs = src_codecs
        self.target_length = target_length
        self.vad_mask = vad_mask
        self.target_langs = target_langs
        self.prompt_lens = prompt_lens
        self.prompts = prompts

    def to(self, device):
        if self.wav is not None:
            self.wav = [wav.to(device) for wav in self.wav]
        if self.fbank is not None:
            self.fbank = self.fbank.to(device)
        if self.fbank_lens is not None:
            self.fbank_lens = self.fbank_lens.to(device)
        if self.target_length is not None:
            self.target_length = self.target_length.to(device)
        if self.vad_mask is not None:
            self.vad_mask = self.vad_mask.to(device)
        if self.prompt_lens is not None:
            self.prompt_lens = self.prompt_lens.to(device)
        if self.prompts is not None:
            self.prompts = self.prompts.to(device)
        return self
    
    def half(self):
        if self.fbank is not None:
            self.fbank = self.fbank.half()
        if self.prompts is not None:
            self.prompts = self.prompts.half()
        return self




class S2STranslator(Module):
    ar_model: UnitYModel
    nar_model: NarModel
    tokenizer: NllbTokenizer
    codec_model: SASCodec
    gen_opts: SequenceGeneratorOptions
    src_generator: UnitYGenerator
    generator: UnitYGenerator
    wav_to_fbank: WaveformToFbankConverter
    collator: Collater
    prompt_len: int
    sample_rate: int

    def __init__(
            self, 
            ar_path,
            nar_path,
            codec_path,
            tokenizer_path, 
            tgt_lang='eng', 
            prompt_len=10, 
            device='cuda'
            ) -> None:
        super().__init__()
        self.tokenizer = load_m4t_tokenizer(tokenizer_path=tokenizer_path)
        self.ar_model = load_s2st_model(model_path=ar_path).half()
        self.nar_model = load_nar_model(model_path=nar_path).half()
        self.codec_model = SASCodec.from_pretrained(codec_path)
        self.vad_model = load_silero_vad()

        self.gen_opts = SequenceGeneratorOptions(
            beam_size=10, soft_max_seq_len=(1, 500), len_penalty=1,
        )
        self.convert_to_fbank = WaveformToFbankConverter(
            num_mel_bins=80,
            waveform_scale=2**15,
            channel_last=True,
            standardize=True,
            )
        self.collator = Collater(
            pad_idx=self.tokenizer.vocab_info.pad_idx, pad_to_multiple=1
        )
        self.ngram_filtering = True
        if self.ngram_filtering:
            self.gen_opts.logits_processor = NGramRepeatBlockProcessor(
                no_repeat_ngram_size=3
            )

        self.sample_rate=16000
        self.src_generator = None
        self.generator = None
        self.token_decoder = self.tokenizer.create_decoder()
        self.transform = build_transform()
        self.nar_generator = NarGenerater(self.nar_model, None)        

        self.prompt_len = prompt_len
        self.to(device)
        self.reset(tgt_lang)
        

    def _extract_fbank(self, wave):
        audio = wave.flatten().unsqueeze(-1)
        decoded_audio = {
                "waveform": audio,
                "sample_rate": self.sample_rate,
                "format": -1,
            }
        inputs = self.convert_to_fbank(decoded_audio)
        return inputs['fbank']

    
    def reset(self, tgt_lang):
        self.generator = UnitYGenerator(
            self.ar_model,
            self.tokenizer,
            tgt_lang,
            output_modality="TextAndSpeech",
            text_opts=self.gen_opts,
        )

    def pre_process(self, audio_data, device='cuda'):
        audio_data = audio_data.unsqueeze(0)
        signal = AudioSignal(audio_data.clone(), self.sample_rate)
        kwargs = self.transform.batch_instantiate([0]*audio_data.shape[0], signal)
        kwargs = prepare_batch(kwargs, device)
        signal = self.transform(
            signal, **kwargs
        )
        return signal.audio_data.squeeze(0).squeeze(0)

    def ar_decode(self, inputs: S2STranslatorlInput):
        
        tgt_gen = self.generator(
            inputs.fbank,
            inputs.fbank_lens,
            ngram_filtering=self.ngram_filtering,
            target_length=inputs.target_length,
            vad_mask=inputs.vad_mask,
            prompts=inputs.prompts,
            prompt_lens=inputs.prompt_lens,
        )
        text_hyps = [str(self.token_decoder(torch.tensor(t))[0]) for t in tgt_gen.text_hyps]
        
        return {
            "text_hyps": text_hyps,
            "codec_hyps": tgt_gen.codec_hyps,
            "codec_scores": tgt_gen.codec_scores,
        }
    
    def nar_decode(self, inputs: NarModelInput):
        return self.nar_generator.generate(inputs)
    
    def gen_nar_batch(
        self,
        src_codecs: List[Tensor], # bsz * 8 * len
        tgt_codecs: List[Tensor], # bsz * 1 * len
        target_langs=List[str],
        ):

        if self.prompt_len > 0:
            p_len = int(50 * self.prompt_len)
            src_codecs = [src_codec[:, -p_len:] for src_codec in src_codecs]

        return NarModelInput(
            ap=src_codecs,
            a=tgt_codecs,
            langs=target_langs
        )

    def _get_vad(self, src_wav):
        src_wav = src_wav.flatten()
        speech_timestamps = get_speech_timestamps(src_wav, self.vad_model, sampling_rate=self.sample_rate)
        if len(speech_timestamps) > 0:
            start = speech_timestamps[0]['start']
            end = speech_timestamps[-1]['end']
            length = round((end - start) / self.sample_rate * 6.25)
        else:
            print('no speech detected')
            length = round(len(src_wav) / self.sample_rate * 6.25)
            start = 0
            end = len(src_wav)

        vad_mask = torch.zeros(length).long()
        
        if len(speech_timestamps) > 0:
            for t in speech_timestamps:
                s, e = round((t['start']-start) / self.sample_rate * 6.25), round((t['end']-start) / self.sample_rate * 6.25)
                vad_mask[s:e] = 1
        else:
            vad_mask[0:round(length / 8)] = 1
            
        return vad_mask, length, start, end
    
    def prepare_inputs(self, paths, target_langs, device):
        wavs = []
        for path in paths:
            wav, _ = librosa.load(path, sr=16000)
            wav = self.pre_process(torch.tensor(wav, dtype=torch.float32, device=device), device=device)
            wavs.append(wav)

        target_lengths = []
        starts = []
        ends = []
        vad_masks = []
        for i, wav in enumerate(wavs):
            vad_mask, length, start, end = self._get_vad(wav)
            vad_masks.append(vad_mask)
            starts.append(start)
            ends.append(end)
            target_lengths.append(length)

        fbank = [self._extract_fbank(wav) for wav in wavs]
        fbank = self.collator(fbank)
        fbank_lens = fbank['seq_lens']
        fbank = fbank['seqs']

        src_codecs = []
        prompts = []
        for wav, start, end in zip(wavs, starts, ends):
            wav = wav.flatten().unsqueeze(0).unsqueeze(0)
            codec = self.codec_model.encode(wav)
            codec = codec[:, :, int(start / self.sample_rate * 50):int(end / self.sample_rate * 50 + 25)]
            src_codecs.append(codec.squeeze(0))

        wavs_padded = self.collator(wavs)['seqs']
        wavs_padded = self.codec_model.preprocess(wavs_padded)
        prompts = self.codec_model.encoder(wavs_padded.unsqueeze(1)).transpose(1, 2)
        prompt_lens = torch.tensor([x.shape[-1] for x in wavs], dtype=torch.long, device=device)
        prompt_lens = (prompt_lens / 320).ceil().long()

        target_length = torch.tensor(
            target_lengths, 
            dtype=torch.long, device=device)

        vad_mask = self.collator(vad_masks)['seqs']

        if isinstance(target_langs, str):
            target_langs = [target_langs] * len(wavs)

        prompts = prompts[:, :500]
        prompt_lens = prompt_lens.clamp(0, 500)

        inputs = S2STranslatorlInput(
            wav=wavs,
            fbank=fbank,
            fbank_lens=fbank_lens,
            src_codecs=src_codecs,
            prompts=prompts,
            prompt_lens=prompt_lens,
            target_length=target_length,
            vad_mask=vad_mask,
            target_langs=target_langs,
        )

        return inputs.to(device).half()

    def forward(self, inputs: S2STranslatorlInput, join=False):
        device = inputs.wav[0].device
        ar_res = self.ar_decode(inputs)
        tgt_codecs = [torch.tensor(codec, dtype=torch.float32, device=device).unsqueeze(0).long() for codec in ar_res['codec_hyps']]
        nar_inputs = self.gen_nar_batch(
            src_codecs=inputs.src_codecs,
            tgt_codecs=tgt_codecs,
            target_langs=inputs.target_langs,
        )
        if join:
            clip_lens = [x.shape[-1] for x in nar_inputs.a]
            nar_inputs = NarModelInput(ap=nar_inputs.ap[-1:], a=[torch.cat(nar_inputs.a, dim=-1)], langs=nar_inputs.langs[:1])
        tgt_codecs = self.nar_decode(nar_inputs)
        wav_hyps = []
        for codec in tgt_codecs:
            codec = codec.unsqueeze(0).cuda()
            self.codec_model.cuda()
            wav = self.codec_model.decode(codec).detach().cpu()[0][0].numpy()
            wav_hyps.append(wav)
        if join:
            new_wav_hyps = []
            current = 0
            for i, clip_len in enumerate(clip_lens):
                new_wav_hyps.append(wav_hyps[0][current*320:(current+clip_len)*320])
                current += clip_len
            wav_hyps = new_wav_hyps
        return {
            'text_hyps': ar_res['text_hyps'],
            'codec_hyps': ar_res['codec_hyps'],
            'wav_hyps': wav_hyps,
        }
        
if __name__ == "__main__":
    MODEL_PATH = '/data/pretrained/translator'
    translator = S2STranslator(MODEL_PATH, device=torch.device('cuda')).eval()

    paths = ["PATH_TO_AUDIO_FILE"]
    inputs = translator.prepare_inputs(
        paths,
        target_langs='en',
        device=torch.device('cuda')
    )
    res = translator(inputs, join=True)
    print(res)
    wav_outputs = res['wav_hyps']
    for i, wav_output in enumerate(wav_outputs):
        sf.write(f'OUT_PATH_{i}.wav', wav_output, 16000)







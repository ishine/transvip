import os.path as osp
import datasets
from itertools import accumulate
from bisect import bisect
import random
import pandas as pd
import torch
import librosa
import numpy as np

from fairseq2.data.audio import WaveformToFbankConverter
from torch.utils.data import Dataset
from .tokenizer import NllbTokenizer

CVSS2M4T_MAP = {
    'en': 'eng',
    'fr': 'fra',
    'zh': 'cmn',
    'zh-CN': 'cmn',
}
# M4T2CVSS_MAP = {v: k for k, v in CVSS2M4T_MAP.items()}

def normalize_text(text: str):
    return text.replace('–', " ").replace('​', " ").replace('„', '"').replace('ǃ', "!").replace('Ų', "ų").replace("Ҫ", "ҫ").replace('’', "'").replace('‘', "'").replace('‘', "'").replace('”', '"').replace("“", '"').replace("«", '"').replace("»", '"').replace('‑', '-').replace('ﷺ', '').replace('ﷻ', '').replace('—', ' ').replace('   ', ' ').replace('  ', ' ').replace('…', '...').replace('�', '')

def read_table(path):
    return pd.read_table(path, on_bad_lines='error', quoting=3, doublequote=False, encoding='utf-8', engine="python")

def random_clip_wav(wav, frac_min=0.2, frac_max=0.5, sample_rate=16000, start=None, end=None):
    length = int((end - start) * sample_rate)

    target_length = random.randint(int(length*frac_min), int(length*frac_max))
    s = random.randint(0, length - target_length) + int(start * sample_rate)
    e = s + target_length
    return wav[s:e], s/sample_rate, e/sample_rate


def load_seamless_record(records_path, audio_root, src_lang, tgt_lang, **kwargs):
    data_pair_lists = []
    data_pair = read_table(records_path)
    data_pair['src_lang'] = CVSS2M4T_MAP[src_lang]
    data_pair['tgt_lang'] = CVSS2M4T_MAP[tgt_lang]
    data_pair['src_path'] = data_pair['path'].apply(lambda x: osp.join(audio_root, src_lang, x))
    data_pair['tgt_path'] = data_pair['path'].apply(lambda x: osp.join(audio_root, tgt_lang, x))
    data_pair_lists.append(data_pair)
    print(f"Loaded {len(data_pair)} seamless fr to en data pairs.")

    # reverse
    data_pair = data_pair.rename(
        columns={
        'src_lang': 'tgt_lang',
        'tgt_lang': 'src_lang',
        'sentence': 'translation',
        'translation': 'sentence',
        'src_codec': 'tgt_codec',
        'tgt_codec': 'src_codec',
        'src_vad': 'tgt_vad',
        'tgt_vad': 'src_vad',
        'src_path': 'tgt_path',
        'tgt_path': 'src_path',
        'src_offset': 'tgt_offset',
        'tgt_offset': 'src_offset',
        },
        inplace=False
    )
    data_pair_lists.append(data_pair)
    print(f"Loaded {len(data_pair)} seamless {src_lang} to {tgt_lang} data pairs.")

    return pd.concat(data_pair_lists, ignore_index=True)


def load_cvss_record(records_path, src_audio_root, tgt_audio_root, language, **kwargs):
    data_pair_lists = []
    data_pair = read_table(records_path)

    # src to tgt
    data_pair['src_lang'] = CVSS2M4T_MAP[language]
    data_pair['tgt_lang'] = 'eng'

    data_pair['src_path'] = data_pair['path'].apply(lambda x: osp.join(src_audio_root, x))

    data_pair['tgt_path'] = data_pair['path'].apply(lambda x: (osp.join(tgt_audio_root, x)+'.wav'))
    data_pair = data_pair.dropna()
    data_pair_lists.append(data_pair)
    print(f"Loaded {len(data_pair)} cvss {language} to english data pairs.")

    # # tgt to src
    data_pair = data_pair.rename(
        columns={
        'src_lang': 'tgt_lang', 
        'tgt_lang': 'src_lang',
        'sentence': 'translation',
        'translation': 'sentence',
        'src_codec': 'tgt_codec',
        'tgt_codec': 'src_codec',
        'src_vad': 'tgt_vad',
        'tgt_vad': 'src_vad',
        'src_path': 'tgt_path',
        'tgt_path': 'src_path',
        'src_offset': 'tgt_offset',
        'tgt_offset': 'src_offset',
        }, 
        inplace=False
    )
        
    data_pair_lists.append(data_pair)
    print(f"Loaded {len(data_pair)} cvss english to {language} data pairs.")

    return pd.concat(data_pair_lists, ignore_index=True)


def load_asr_record(dataset_configs):
    data_pair_lists = []
    for config in dataset_configs:
        data_pair = read_table(config['records_path'])
        data_pair['lang'] = CVSS2M4T_MAP[config['language']]
        data_pair['src_path'] = data_pair['path'].apply(lambda x: osp.join(config['language'], x))
        data_pair = data_pair.dropna()
        data_pair_lists.append(data_pair)
        print(f"Loaded {len(data_pair)} asr {config['language']} data pairs.")
    return pd.concat(data_pair_lists, ignore_index=True)


load_data_func = {
    'seamless': load_seamless_record,
    'cvss': load_cvss_record,
}

def load_s2st_records(dataset_configs) -> pd.DataFrame:
    data_pair_lists = []
    for dataset_config in dataset_configs:
        dataset_name = dataset_config['name']
        if dataset_name in load_data_func:
            records = load_data_func[dataset_name](**dataset_config)
            data_pair_lists.append(records)
    return pd.concat(data_pair_lists, ignore_index=True)


class S2STDataset(Dataset):
    def __init__(
        self, 
        dataset_configs: list,
        tokenizer: NllbTokenizer, 
        split: str = 'train',
        device=torch.device('cpu'), 
        dtype=torch.float32
        ) -> None:
        super().__init__()

        audio_info_list = load_s2st_records(dataset_configs, split)
        self.audio_info_list = audio_info_list.to_dict("records")
        if not split == 'train':
            self.audio_info_list = self.audio_info_list[:3000]
        self.split = split
        self.tokenizer = tokenizer
        self.sample_rate = 16000
        self.max_length = 1000
        self.device = device
        self.dtype = dtype
        self.convert_to_fbank = WaveformToFbankConverter(
            num_mel_bins=80,
            waveform_scale=2**15,
            channel_last=True,
            standardize=True,
            device=device,
            dtype=dtype,
        )

    def __len__(self):
        return len(self.audio_info_list)
        # return len(self.audio_info_list) * 2 if self.split == 'train' else len(self.audio_info_list)
    
    def _load_fbank(self, audio_path):
        audio = librosa.load(audio_path, sr=self.sample_rate)[0]
        audio = torch.tensor(
            audio, 
            dtype=self.dtype, 
            device=self.device).unsqueeze(-1)
        decoded_audio = {
                "waveform": audio,
                "sample_rate": self.sample_rate,
                "format": -1,
            }
        inputs = self.convert_to_fbank(decoded_audio)
        return inputs['fbank']
    
    def _codec_str2list(self, codec_str):
        codec_list = [int(c) for c in codec_str.split(" ")]
        codec_list = (np.array(codec_list, dtype=np.int64) + self.tokenizer.vocab_info.size).tolist()
        return codec_list

    def _get_vad_mask(self, vad_str, total_length, disturb_factor):
        starts = []
        ends = []
        disturb_factor = disturb_factor.item()
        for vad in vad_str.split(" "):
            start, end = vad.split(":")
            start, end = float(start), float(end)
            starts.append(round(start * 6.25 * disturb_factor)) # 100 / 16
            ends.append(round(end * 6.25 * disturb_factor))
        vad_mask = torch.zeros(total_length).long()
        for start, end in zip(starts, ends):
            vad_mask[start:end] = 1
        return vad_mask


    def __getitem__(self, id):
        record = self.audio_info_list[id % len(self.audio_info_list)]

        audio_path = record['src_path']
        src_lang = record["src_lang"]
        tgt_lang = record["tgt_lang"]
        
        enc_tokenizer = self.tokenizer.create_encoder(
            task="translation", lang=src_lang, mode="source", device=self.device
        )

        text_inputs = enc_tokenizer(record['sentence'])
        speech_inputs = self._load_fbank(audio_path)
        if torch.isnan(speech_inputs).any():
            return self.__getitem__(id+1)
        dec_tokenizer = self.tokenizer.create_encoder(
            task="translation", lang=tgt_lang, mode="target_sep", device=self.device
        )

        text_label = record['translation'] 
        codec_label = record['tgt_codec']

        text_label = normalize_text(text_label)
        text_label = dec_tokenizer(text_label)
        codec_label = self._codec_str2list(codec_label)
        codec_label = torch.tensor(codec_label + [3], dtype=torch.long, device=self.device) # 3 is eos

        label = torch.cat([text_label, codec_label])
        text_mask = torch.tensor([1] * len(text_label) + [0] * len(codec_label))
        codec_mask = torch.tensor([0] * len(text_label) + [1] * len(codec_label))
        
        dec_inputs = label[:-1].clone()
        label = label[1:]
        text_mask = text_mask[1:]
        codec_mask = codec_mask[1:]

        duration = codec_label.shape[0] * 0.02
        offset = record['tgt_offset']

        tgt_wav = librosa.load(record['tgt_path'], sr=self.sample_rate)[0]
        tgt_wav = torch.tensor(
            tgt_wav,
            dtype=self.dtype,
            device=self.device
        )
        if self.split == 'train':
            tgt_wav, prompt_start, prompt_end = random_clip_wav(tgt_wav, start=offset, end=offset+duration)
            prompt_start = max(0, prompt_start - offset)
            prompt_end = max(0, prompt_end - offset)
            prompt_token_start = round(prompt_start * 50)
            prompt_token_end = round(prompt_end * 50)

            codec_start = len(text_label) - 1
            label[codec_start+prompt_token_start:codec_start+prompt_token_end] = 0
            codec_mask[codec_start+prompt_token_start:codec_start+prompt_token_end] = 0

        target_length = round((codec_label.shape[0]-1) / 8)
        target_length = torch.tensor([target_length])
        length_disturb_factor = torch.normal(mean=torch.ones(1), std=torch.ones(1)*0.1).clamp(min=0.8, max=1.2) if self.split == 'train' else torch.tensor([1])
        target_length = (target_length * length_disturb_factor).long()
        vad = record['tgt_vad']
        
        vad_mask = self._get_vad_mask(vad, target_length, length_disturb_factor)

        if label.shape[0] > self.max_length:
            label = label[:self.max_length]
            dec_inputs = dec_inputs[:self.max_length]
            text_mask = text_mask[:self.max_length]
            codec_mask = codec_mask[:self.max_length]
       

        return {
            "text_length": len(text_label) - 1,
            "speech_inputs": speech_inputs,
            "text_inputs": text_inputs,
            "dec_inputs": dec_inputs,
            "label": label,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "text_mask": text_mask,
            "codec_mask": codec_mask,
            "target_length": target_length,
            "vad_mask": vad_mask,
            "audio_path": audio_path,
            "text_ref": record['translation'],
            "tgt_wav": tgt_wav,
        }
    

class MTDataset(Dataset):
    def __init__(self, datasets_paths, tokenizer):
        self.datasets = [datasets.load_from_disk(path) for path in datasets_paths]
        self.lengths = [x.num_rows for x in self.datasets]
        self.cum_lengths = [0] + list(accumulate(self.lengths))
        self.total_length = sum(self.lengths)
        self.tokenizer = tokenizer
        self.device = torch.device('cpu')
        self.max_length = 300
    def __len__(self):
        return self.total_length * 2
    def __getitem__(self, i):
        idx = i % self.total_length
        dataset_idx = bisect(self.cum_lengths, idx) - 1
        sample_idx = idx - self.cum_lengths[dataset_idx]
        record = self.datasets[dataset_idx][sample_idx]['translation']
        keys = list(record.keys())
        lang1 = keys[0] if keys[0] != 'en' else keys[1]
        lang2 = 'en'
        src_lang = lang1 if i < self.total_length else lang2
        tgt_lang = lang2 if i < self.total_length else lang1
        text_label = record[tgt_lang]
        enc_tokenizer = self.tokenizer.create_encoder(
            task="translation", lang=CVSS2M4T_MAP[src_lang], mode="source", device=self.device
        )
        text_inputs = enc_tokenizer(record[src_lang])
        dec_tokenizer = self.tokenizer.create_encoder(
            task="translation", lang=CVSS2M4T_MAP[tgt_lang], mode="target_sep", device=self.device
        )
        text_label = normalize_text(text_label)
        text_label = dec_tokenizer(text_label)
        label = text_label[1:]
        dec_inputs = text_label[:-1].clone()
        if text_inputs.shape[0] > self.max_length:
            return self.__getitem__(i+1)
        if label.shape[0] > self.max_length:
            label = label[:self.max_length]
            dec_inputs = dec_inputs[:self.max_length]
        
        return {
            "text_inputs": text_inputs,
            "dec_inputs": dec_inputs,
            "label": label,
            "src_lang": CVSS2M4T_MAP[src_lang],
            "tgt_lang": CVSS2M4T_MAP[tgt_lang],
            "text_ref": record[tgt_lang],
        }

class ASRDataset(Dataset):
    def __init__(
        self, 
        dataset_configs: list,
        tokenizer: NllbTokenizer, 
        device=torch.device('cpu'), 
        dtype=torch.float32
        ) -> None:
        super().__init__()

        self.audio_info_list = load_asr_record(**dataset_configs).to_dict("records")
        self.tokenizer = tokenizer
        self.sample_rate = 16000
        self.max_length = 800
        self.device = device
        self.dtype = dtype
        self.convert_to_fbank = WaveformToFbankConverter(
            num_mel_bins=80,
            waveform_scale=2**15,
            channel_last=True,
            standardize=True,
            device=device,
            dtype=dtype,
        )

    def __len__(self):
        return len(self.audio_info_list)
    
    def _load_fbank(self, audio_path):
        audio = librosa.load(audio_path, sr=self.sample_rate)[0]
        audio = torch.tensor(
            audio, 
            dtype=self.dtype, 
            device=self.device).unsqueeze(-1)
        decoded_audio = {
                "waveform": audio,
                "sample_rate": self.sample_rate,
                "format": -1,
            }
        inputs = self.convert_to_fbank(decoded_audio)
        return inputs['fbank']
    
    def _codec_str2list(self, codec_str):
        codec_list = [int(c) for c in codec_str.split(" ")]
        codec_list = (np.array(codec_list, dtype=np.int64) + self.tokenizer.vocab_info.size).tolist()
        return codec_list

    def _get_vad_mask(self, vad_str, total_length, disturb_factor):
        starts = []
        ends = []
        disturb_factor = disturb_factor.item()
        for vad in vad_str.split(" "):
            start, end = vad.split(":")
            start, end = float(start), float(end)
            starts.append(round(start * 6.25 * disturb_factor)) # 100 / 16
            ends.append(round(end * 6.25 * disturb_factor))
        vad_mask = torch.zeros(total_length).long()
        for start, end in zip(starts, ends):
            vad_mask[start:end] = 1
        return vad_mask


    def __getitem__(self, id):
        record = self.audio_info_list[id % len(self.audio_info_list)]

        audio_path = record['src_path']
        lang = record["lang"]
        
        speech_inputs = self._load_fbank(audio_path)
        if torch.isnan(speech_inputs).any():
            return self.__getitem__(id+1)
        dec_tokenizer = self.tokenizer.create_encoder(
            task="translation", lang=lang, mode="target_sep", device=self.device
        )

        text_label = record['sentence'] 
        codec_label = record['codecs']
        text_label = normalize_text(text_label)
        text_label = dec_tokenizer(text_label)
        codec_label = self._codec_str2list(codec_label)
        codec_label = torch.tensor(codec_label + [3], dtype=torch.long, device=self.device) # 3 is eos

        label = torch.cat([text_label, codec_label])
        text_mask = torch.tensor([1] * len(text_label) + [0] * len(codec_label))
        codec_mask = torch.tensor([0] * len(text_label) + [1] * len(codec_label))
        
        dec_inputs = label[:-1].clone()
        label = label[1:]
        text_mask = text_mask[1:]
        codec_mask = codec_mask[1:]

        duration = codec_label.shape[0] * 0.02
        offset = record['offset']

        tgt_wav = librosa.load(record['src_path'], sr=self.sample_rate)[0]
        tgt_wav = torch.tensor(
            tgt_wav,
            dtype=self.dtype,
            device=self.device
        )

        tgt_wav, prompt_start, prompt_end = random_clip_wav(tgt_wav, start=offset, end=offset+duration)
        prompt_start = max(0, prompt_start - record['offset'])
        prompt_end = max(0, prompt_end - record['offset'])
        prompt_token_start = round(prompt_start * 50)
        prompt_token_end = round(prompt_end * 50)

        codec_start = len(text_label) - 1
        label[codec_start+prompt_token_start:codec_start+prompt_token_end] = 0
        codec_mask[codec_start+prompt_token_start:codec_start+prompt_token_end] = 0

        target_length = round((codec_label.shape[0]-1) / 8)
        if codec_label.shape[0] > 2000:
            return self.__getitem__(id+1)
        target_length = torch.tensor([target_length])
        length_disturb_factor = torch.normal(mean=torch.ones(1), std=torch.ones(1)*0.1).clamp(min=0.6, max=1.4)
        target_length = (target_length * length_disturb_factor).long()
        
        vad_mask = self._get_vad_mask(record['vad'], target_length, length_disturb_factor)

        if label.shape[0] > self.max_length:
            label = label[:self.max_length]
            dec_inputs = dec_inputs[:self.max_length]
            text_mask = text_mask[:self.max_length]
            codec_mask = codec_mask[:self.max_length]

        return {
            "text_length": len(text_label) - 1,
            "speech_inputs": speech_inputs,
            "dec_inputs": dec_inputs,
            "label": label,
            "lang": lang,
            "text_mask": text_mask,
            "codec_mask": codec_mask,
            "target_length": target_length,
            "vad_mask": vad_mask,
            "audio_path": audio_path,
            "text_ref": record['sentence'],
            "tgt_wav": tgt_wav,
        }


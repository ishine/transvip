import glob
import random
import os

import torch
import librosa
from torch.utils.data import Dataset
import torch.nn.functional as F
import math
from itertools import accumulate
from bisect import bisect

def load_audio_list(data_path):
    lines = []
    with open(data_path, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines

class BaseDataset(Dataset):
    """Dataset to load NSynth data."""

    def __init__(self, audio_root, audio_file, sample_rate=16000):
        super().__init__()
        self.audio_root = audio_root
        self.filenames = load_audio_list(audio_file)
        print(len(self.filenames))
        self.sr = sample_rate
        self.max_len = sample_rate * 20
        self.downsample_rate = 320

    def __len__(self):
        return len(self.filenames)

    def _load_audio(self, filename):
        filename = os.path.join(self.audio_root, filename)
        duration = librosa.get_duration(path=filename)
        offset = 0
        if duration > 20:
            new_duration = random.uniform(3, 20)
            offset = random.uniform(0, duration - new_duration)
            duration = new_duration
        audio, sr = librosa.load(filename, sr=self.sr, offset=offset, duration=duration)
        audio = torch.tensor(audio).flatten().unsqueeze(0)
        return audio

    def _clip_audio(self, audio):
        if audio.shape[1] > self.max_len:
            st = random.randint(0, audio.shape[1] - self.max_len - 1)
            ed = st + self.max_len
            return audio[:, st:ed]
        else:
            ans = torch.zeros(1, self.max_len)
            ans[:, :audio.shape[1]] = audio
            return ans
        
    def _lang(self, idx):
        pass

    def __getitem__(self, index):
        ans = torch.zeros(1, self.max_len)
        try:
            audio = self._load_audio(self.filenames[index])
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.filenames) - 1))
        # lang = self.filenames[index].split("/")[0]
        lang = self._lang(index)
        if audio.shape[1] > self.max_len:
            st = random.randint(0, audio.shape[1] - self.max_len - 1)
            ed = st + self.max_len
            ans = audio[:, st:ed]
        else:
            ans[:, :audio.shape[1]] = audio

        seq_lens = min(math.ceil(audio.shape[1] / self.downsample_rate), self.max_len // self.downsample_rate)
        return ans, seq_lens, lang
    


class CommonVoiceDataset(BaseDataset):
    """Dataset to load NSynth data."""
    def _lang(self, idx):
        return self.filenames[idx].split("/")[0]

class LibrilightDataset(BaseDataset):
    """Dataset to load NSynth data."""
    def _lang(self, idx):
        return 'en'

class WenetSpeechDataset(BaseDataset):
    """Dataset to load NSynth data."""
    def __len__(self):
        return len(self.filenames) * 30
    
    def __getitem__(self, index):
        return super().__getitem__(index // 30)

    def _lang(self, idx):
        return 'zh-CN'

def build_dataset(name, **args):
    if name == "commonvoice":
        return CommonVoiceDataset(**args)
    elif name == "wenetspeech":
        return WenetSpeechDataset(**args)
    elif name == "librilight":
        return LibrilightDataset(**args)
    else:
        raise ValueError(f"Dataset {name} not supported")

class JoinedDataset(Dataset):
    def __init__(self, dataset_args):
        self.datasets = []
        for name, args in dataset_args.items():
            self.datasets.append(build_dataset(name, **args))
        self.lengths = [len(x) for x in self.datasets]
        self.cum_lengths = [0] + list(accumulate(self.lengths))
        self.total_length = sum(self.lengths)
        print(self.lengths)
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, index):
        dataset_idx = bisect(self.cum_lengths, index) - 1
        sample_idx = index - self.cum_lengths[dataset_idx]
        return self.datasets[dataset_idx][sample_idx]




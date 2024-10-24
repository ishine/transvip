import glob
import os.path as osp

from tqdm import tqdm
import pandas as pd

import librosa
import torch
import math
import os
import multiprocessing
from silero_vad import load_silero_vad, get_speech_timestamps
import whisper
from sascodec import SASCodec

if __name__ == "__main__":
    import sys
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(BASE_DIR)


os.environ["TOKENIZERS_PARALLELISM"] = "false"

AUDIO_DIR = '/PATH/TO/seamless_align' if "SRC_AUDIO_DIR" not in os.environ else os.environ["AUDIO_DIR"]
CODEC_CKPT_PATH = '' if "CODEC_CKPT_PATH" not in os.environ else os.environ["CODEC_CKPT_PATH"]
OUT_DIR = '' if "OUT_DIR" not in os.environ else os.environ["OUT_DIR"]
SRC_LANG = 'fr' if "SRC_LANG" not in os.environ else os.environ["SRC_LANG"]
TGT_LANG = 'en' if "TGT_LANG" not in os.environ else os.environ["TGT_LANG"]

def load_model(device):
    asr_model = whisper.load_model("large", device=device,)
    codec_model = SASCodec.from_pretrained(CODEC_CKPT_PATH)
    codec_model.eval().to(device)
    vad_model = load_silero_vad()
    vad_model = vad_model.to(device)

    return asr_model, codec_model, vad_model

def extract_codec(vad_model, codec_model, path, device):
    wav, sr = librosa.load(path, sr=codec_model.sample_rate)

    # remove silence at the beginning and end
    speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=sr)
    if len(speech_timestamps) == 0:
        speech_timestamps = [{'start': 0, 'end': len(wav)}]
    start = speech_timestamps[0]['start']
    end = speech_timestamps[-1]['end']
    wav = wav[start:end]

    vad_str = " ".join([f"{((s['start'])/sr):.2f}:{((s['end'])/sr):.2f}" for s in speech_timestamps])

    wav = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        codes = codec_model.encode(wav).squeeze(0)
    codes = codes.detach().cpu().tolist()
    codes_str = " ".join([str(c) for c in codes[0]]) 
    offset = start / sr

    return codes_str, vad_str, offset

def asr(model, paths, lang):
    mels = []
    for path in paths:
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(path)
        audio = whisper.pad_or_trim(audio)
        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        mels.append(mel)
    mels = torch.stack(mels)

    # decode the audio
    options = whisper.DecodingOptions(language=lang, beam_size=5, without_timestamps=True, fp16=True)
    result = whisper.decode(model, mels, options)
    return [r.text.strip() for r in result]

def check_duration(path):
    src_path = f'{AUDIO_DIR}/{SRC_LANG}/{path}'
    tgt_path = f'{AUDIO_DIR}/{TGT_LANG}/{path}'
    src_duration = librosa.get_duration(path=src_path)
    tgt_duration = librosa.get_duration(path=tgt_path)
    return 1 <= src_duration <= 30 and 1 <= tgt_duration <= 30

class Loader:
    def __init__(self, paths, batch_size):
        self.paths = paths
        self.batch_size = batch_size
    def __iter__(self):
        for i in range(0, len(self.paths), self.batch_size):
            paths = self.paths[i:i+self.batch_size]
            yield paths
    def __len__(self):
        return math.ceil(len(self.paths) / self.batch_size)

def process_on_gpu(paths_list, device_id):
    device = torch.device(f"cuda:{device_id}")
    asr_model, codec_model, vad_model = load_model(device)
    loader = Loader(paths_list, 5)
    results = []
    print(f'gpu {device_id} start')
    for paths in tqdm(loader):
        paths = [path for path in paths if check_duration(path)]
        if len(paths) == 0:
            continue
        try:
            src_paths = [f'{AUDIO_DIR}/{SRC_LANG}/{path}' for path in paths]
            tgt_paths = [f'{AUDIO_DIR}/{TGT_LANG}/{path}' for path in paths]
            sentence = asr(asr_model, src_paths, SRC_LANG)
            translation = asr(asr_model, tgt_paths, TGT_LANG)
            src_codecs, src_vad, src_offset = extract_codec(vad_model,codec_model, src_paths, device)
            tgt_codecs, tgt_vad, tgt_offset = extract_codec(vad_model, codec_model, tgt_paths, device)
            results.append((paths, sentence, translation, src_codecs, tgt_codecs, src_vad, tgt_vad, src_offset, tgt_offset))
        except Exception as e:
            print(e)
            continue
    return results

def main():
    NUM_GPUS = torch.cuda.device_count()
    src_file_list = glob.glob(f"{AUDIO_DIR}/{SRC_LANG}/*")
    src_file_list = [osp.basename(f) for f in src_file_list]
    tgt_file_list = glob.glob(f"{AUDIO_DIR}/{TGT_LANG}/*")
    tgt_file_list = [osp.basename(f) for f in tgt_file_list]
    joined_file_list = list(set(src_file_list) & set(tgt_file_list))
    
    output_path = f'{OUT_DIR}/seamless_train.tsv'

    header = 'path\tsentence\ttranslation\tsrc_codec\ttgt_codec\tsrc_vad\ttgt_vad\tsrc_offset\ttgt_offset\n'
    global finished_paths
    if not osp.exists(output_path):
        # write header
        with open(output_path, 'w') as f:
            f.write(header)
        finished_paths = set()
        print('created new file')
    else:
        finished_paths = pd.read_csv(output_path, sep='\t', on_bad_lines='error', quoting=3, doublequote=False, encoding='utf-8')['path'].tolist()
        finished_paths = set(finished_paths)
        print('loaded existing file')

    joined_file_list = [path for path in joined_file_list if path not in finished_paths]
    print(len(joined_file_list))
    chunked_file_list = [joined_file_list[i::NUM_GPUS] for i in range(NUM_GPUS)]
    with multiprocessing.Pool(NUM_GPUS) as pool:
        all_results = pool.starmap(process_on_gpu, [(chunk, i) for i, chunk in enumerate(chunked_file_list)])
    all_results = [item for sublist in all_results for item in sublist]
    df = pd.DataFrame(all_results)
    df.to_csv(output_path, index=False, mode='a', header=False, sep='\t',  quoting=3, doublequote=False, encoding='utf-8')
    
if __name__ == '__main__':
    main()




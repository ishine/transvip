import requests
import librosa
import soundfile as sf
import pandas as pd
import os
import os.path as osp
from tqdm import tqdm
import traceback
from pathlib import Path
from multiprocessing import Pool, cpu_count
import tempfile

TABLE_PATH = f'/PATH/TO/seamless.dataset.metadata.public.enA-frA.processed.tsv' if "TABLE_PATH" not in os.environ else os.environ["TABLE_PATH"]
OUT_DIR = f'/PATH/TO/seamless_audio' if "OUT_DIR" not in os.environ else os.environ["OUT_DIR"]


def download_audio(url):
    response = requests.get(url, verify=False, timeout=100)
    if response.status_code == 200:
        return response.content
    else:
        raise ConnectionError("Failed to download audio")


def save_audio(data, path, sr):
    sf.write(path, data, sr)


def process_audio(url, start_frame, end_frame, output_path):
    audio_data = download_audio(url)
    file_type = url.split('.')[-1]

    with tempfile.NamedTemporaryFile(suffix=f".{file_type}", delete=True) as temp_file:
        temp_file.write(audio_data)
        temp_filename = temp_file.name

        audio, sr = librosa.load(temp_filename, sr=16000)

    clipped_audio = audio[start_frame:end_frame]

    save_audio(clipped_audio, output_path, sr)

    return clipped_audio.shape


def process_record(record):
    try:
        audio_url, start_frame, end_frame = record['url'].split(' ')
        audio_index = record['index']
        language_side = record['side'][:2]
        shape=0
        Path(osp.join(OUT_DIR, language_side)).mkdir(parents=True, exist_ok=True)
        if not osp.exists(osp.join(OUT_DIR, record['side'][:2], f'{record["index"]}.wav')):
            shape = process_audio(audio_url, int(start_frame), int(end_frame), osp.join(OUT_DIR, language_side, f'{audio_index}.wav'))
        return (audio_index, True, "", shape)
    except Exception as e:
        return (audio_index, False, f"Failed to process {audio_index}: {str(e)}\n{traceback.format_exc()}", 0)


if __name__ == "__main__":
    table = pd.read_csv(TABLE_PATH, sep='\t', names=['none1', 'none2', 'url', 'none3', 'none4', 'none5', 'none6', 'score', 'direction', 'side', 'index'])[['index', 'url', 'side', 'direction', 'score']]
    
    start_index = 0 * 100000
    end_index = 1 * 100000
    table = table[(table['index'] >= start_index) & (table['index'] < end_index)]
    records = table.to_dict('records')

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    failed_index = set()

    with Pool(processes=cpu_count()) as pool:
        for audio_index, success, error_message, audio_shape in tqdm(pool.imap_unordered(process_record, records), total=len(records), desc=f"num_failed={len(failed_index)}"):
            if not success:
                failed_index.add(audio_index)
                print(error_message)

    
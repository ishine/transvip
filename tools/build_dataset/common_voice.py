import librosa
import pandas as pd
import torch
from tqdm import tqdm
import argparse
import os
import os.path as osp
from silero_vad import load_silero_vad, get_speech_timestamps
from sascodec import SASCodec
import multiprocessing

save_interval = 100

CODEC_CKPT_PATH = '' if "CODEC_CKPT_PATH" not in os.environ else os.environ["CODEC_CKPT_PATH"]
CV_ROOT = '' if "CV4_ROOT" not in os.environ else os.environ["CV4_ROOT"]
OUT_DIR = '' if "OUT_DIR" not in os.environ else os.environ["OUT_DIR"]

def load_tsv(path, **kwargs):
    table = pd.read_table(
        path, 
        on_bad_lines='error', 
        quoting=3, 
        doublequote=False, 
        encoding='utf-8', 
        **kwargs
        )
    return table

def load_cv_record(cvss_code):
    # cvss and codec table path
    cv_path = osp.join(CV_ROOT, f"{cvss_code}", f"validated.tsv")
    cv_table = load_tsv(cv_path)
    cv_table['abs_path'] = cv_table['path'].apply(lambda x: osp.join(CV_ROOT,  f"{cvss_code}", "clips", x))
    return cv_table


def load_model(device):
    codec_model = SASCodec.from_pretrained(CODEC_CKPT_PATH)
    codec_model.eval().to(device)
    vad_model = load_silero_vad()
    vad_model = vad_model.to(device)

    return codec_model, vad_model

def extract_codec(vad_model, codec_model, path, device):
        
    wav, sr = librosa.load(path, sr=codec_model.sample_rate)

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


def process_on_gpu(data_record_list, device_id):
    NUM_GPUS = torch.cuda.device_count()
    device_id = device_id % NUM_GPUS
    device = torch.device(f"cuda:{device_id}")
    codec_model, vad_model = load_model(device)
    results = []
    print(f'gpu {device_id} start')
    for record in tqdm(data_record_list):
        paths = record['path']
        sentence = record['sentence']
        try:
            codecs, vad, offset = extract_codec(vad_model, codec_model, record['abs_path'], device)
        except Exception as e:
            print(f"Error processing {record['abs_path']}: {e}")
            continue
        results.append((paths, sentence, codecs, vad, offset))
    return results

def main(lang_code):
    table = load_cv_record(lang_code)
    data_records = table.to_dict('records')
    output_path = f"{OUT_DIR}/cv_train.tsv"
    NUM_PROCESS = torch.cuda.device_count() * 4
    header = 'path\tsentence\tcodecs\tvad\toffset\n'
    chunked_data_record = [data_records[i::NUM_PROCESS] for i in range(NUM_PROCESS)]
    with multiprocessing.Pool(NUM_PROCESS) as pool:
        all_results = pool.starmap(process_on_gpu, [(chunk, i) for i, chunk in enumerate(chunked_data_record)])
    all_results = [item for sublist in all_results for item in sublist]
    assert not osp.exists(output_path)
    with open(output_path, 'w') as f:
        f.write(header)
    print('created new file')
    df = pd.DataFrame(all_results)
    df.to_csv(output_path, index=False, mode='a', header=False, sep='\t',  quoting=3, doublequote=False, encoding='utf-8')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--lang', type=str, default='en')
    args = parser.parse_args()
    main(args.lang)
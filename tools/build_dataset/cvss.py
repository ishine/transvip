import librosa
import pandas as pd
import torch
from tqdm import tqdm
import argparse
import os
import os.path as osp
import multiprocessing
from silero_vad import load_silero_vad, get_speech_timestamps
from sascodec import SASCodec


save_interval = 100

CODEC_CKPT_PATH = '' if "CODEC_CKPT_PATH" not in os.environ else os.environ["CODEC_CKPT_PATH"]
CV4_ROOT = '' if "CV4_ROOT" not in os.environ else os.environ["CV4_ROOT"]
CoVoST_ROOT = '' if "CoVoST_ROOT" not in os.environ else os.environ["CoVoST_ROOT"]
CVSS_ROOT = '' if "CVSS_ROOT" not in os.environ else os.environ["CVSS_ROOT"]
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

def load_cvss_record(split, cvss_code):
    # cvss and codec table path
    covost_path = osp.join(CoVoST_ROOT, f"{cvss_code}_en", f"covost_v2.{cvss_code}_en.{split}.tsv")
    cvss_path = osp.join(CVSS_ROOT, f"{cvss_code}_en", f"{split}.tsv")
    covost_table = load_tsv(covost_path)[['path', 'sentence', 'translation' ]]
    cvss_table = load_tsv(cvss_path, header=None, names=['path', 'null'])[['path']]
    merged_table = pd.merge(covost_table, cvss_table, on='path', how='inner')
    merged_table['src_path'] = merged_table['path'].apply(lambda x: osp.join(CV4_ROOT, cvss_code, 'clips', x))
    merged_table['tgt_path'] = merged_table['path'].apply(lambda x: osp.join(CVSS_ROOT, f"{cvss_code}_en", f"{split}", f"{x}.wav"))
    return merged_table


def load_model(device):
    codec_model = SASCodec.from_pretrained(CODEC_CKPT_PATH)
    codec_model.eval().to(device)
    vad_model = load_silero_vad()
    vad_model = vad_model.to(device)

    return codec_model, vad_model

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
        translation = record['translation']
        src_codecs, src_vad, src_offset = extract_codec(vad_model,codec_model, record['src_path'], device)
        tgt_codecs, tgt_vad, tgt_offset = extract_codec(vad_model, codec_model, record['tgt_path'], device)
        results.append((paths, sentence, translation, src_codecs, tgt_codecs, src_vad, tgt_vad, src_offset, tgt_offset))
    return results

def main(split, lang_code):
    table = load_cvss_record(split, lang_code)
    data_records = table.to_dict('records')
    output_path = f"{OUT_DIR}/cvss_{split}.tsv"
    NUM_PROCESS = torch.cuda.device_count() * 8
    header = 'path\tsentence\ttranslation\tsrc_codec\ttgt_codec\tsrc_vad\ttgt_vad\tsrc_offset\ttgt_offset\n'
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
    parser.add_argument('-s', '--split', type=str, default='train')
    parser.add_argument('-t', '--lang', type=str, default='fr')
    args = parser.parse_args()
    main(args.split, args.lang)
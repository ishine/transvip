from models.translator import *
import argparse

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--ar_path', type=str, default='/path/to/ar.pt')
    args.add_argument('--nar_path', type=str, default='/path/to/nar.pt')
    args.add_argument('--codec_path', type=str, default='/path/to/codec.pt')
    args.add_argument('--tokenizer_path', type=str, default='/path/to/tokenizer')
    args.add_argument('--target_lang', type=str, default='en')
    args.add_argument('--device', type=str, default='cuda')
    args.add_argument('-o', '--output', type=str, default='/path/to/output.wav')
    args.add_argument('-i', '--input', type=str, default='/path/to/audio.wav')

    translator = S2STranslator(
        ar_path=args.ar_path,
        nar_path=args.nar_path,
        codec_path=args.codec_path,
        tokenizer_path=args.tokenizer_path,
        tgt_lang=args.target_lang,
        device=torch.device(args.device)
    )

    paths = [args.input]
    inputs = translator.prepare_inputs(
        paths,
        target_langs='en',
        device=torch.device('cuda')
    )
    res = translator(inputs, join=True)
    # print(res)
    wav_output = res['wav_hyps'][0]
\
    sf.write(args.output, wav_output, 16000)

    
import torch
import os
import pickle
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from pytorch_lightning.loggers import WandbLogger

from fairseq2.data import Collater

from torchmetrics.text import WordErrorRate, SacreBLEUScore
from sascodec import SASCodec

if __name__ == "__main__":
    import sys
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(BASE_DIR)

from fairseq2_011.generation import SequenceGeneratorOptions
from models.s2st import (
    UnitYModel, 
    load_s2st_model, 
    load_m4t_tokenizer, 
    NllbTokenizer, 
    UnitYGenerator, 
    NGramRepeatBlockProcessor,
    S2STCriteria,
    S2STDataset, 
    MTCriteria,
    MTDataset,
    ASRDataset,
    ASRCriteria,
    STCriteria
)
from models.optimizer import configure_optimizer_schedular
from tools.data.text_normalizer import BasicTextNormalizer

MATRICDICT = {
    'wer': WordErrorRate(),
    'bleu': SacreBLEUScore()
}

textNormalizer = BasicTextNormalizer()

DATASET_DICT = {
    's2st': S2STDataset,
    'mt': MTDataset,
    'asr': ASRDataset
}


class M4tValleModule(LightningModule):
    model: UnitYModel
    tokenizer: NllbTokenizer
    s2st_criterion: S2STCriteria
    collate: Collater
    generator: UnitYGenerator

    def __init__(self, cfg) -> None:
        super().__init__()
        model_cfg = cfg['model_cfg']
        self.tokenizer = load_m4t_tokenizer(model_cfg['tokenizer_path'])
        self.model = load_s2st_model(**model_cfg)
        self.cfg = cfg     
        self.collate = Collater(
            pad_idx=self.tokenizer.vocab_info.pad_idx, pad_to_multiple=1
        )
        self.s2st_criterion = S2STCriteria(self.model)
        if 'mt' in cfg['train_dataset_cfg']:
            self.mt_criterion = MTCriteria(self.model)
        else:
            self.mt_criterion = None
        if 'asr' in cfg['train_dataset_cfg']:
            self.asr_criterion = ASRCriteria(self.model)
        else:
            self.asr_criterion = None
        if 'st' in cfg['train_dataset_cfg']:
            self.st_criterion = STCriteria(self.model)
        else:
            self.st_criterion = None

        self.gen_opts = SequenceGeneratorOptions(
            beam_size=5, soft_max_seq_len=(1, 500), len_penalty=1,
        )
        self.ngram_filtering = False

        self.gen_opts.logits_processor = NGramRepeatBlockProcessor(
            no_repeat_ngram_size=3
        )
        self.generator = None

        self.val_metric = SacreBLEUScore()
        self.test_metric = None
        self.token_decoder = self.tokenizer.create_decoder()

        self.wav_in = SASCodec.from_pretrained(model_cfg['codec_path']).encoder.eval()
        self.wav_in.requires_grad_(False)
        
        if 'init_path' in cfg:
            self.load_state_dict(torch.load(cfg['init_path'])['state_dict'], strict=False)
            print(f"Loaded model from {cfg['init_path']}.")

    def forward(self, **kwargs):
        return self.model(**kwargs)
        

    def on_train_start(self) -> None:
        sch = self.lr_schedulers()
        sch.step()

    def on_load_checkpoint(self, checkpoint) -> None:
        """Fix the checkpoint loading issue for deepspeed."""
        if self._trainer is not None:
            return
        if "state_dict" in checkpoint:
            return
        state_dict = checkpoint['module']
        state_dict = {k.partition('module.')[2]: state_dict[k] for k in state_dict.keys()}
        checkpoint['state_dict'] = state_dict
        return

    def preprocess_batch(self, batch):
        if 'tgt_wav' in batch:
            batch['prompt'] = {}
            # dtype = batch['tgt_wav']['seqs'].dtype
            with torch.no_grad():
                self.wav_in.eval()
                batch['prompt']['seqs'] = self.wav_in(batch['tgt_wav']['seqs'].unsqueeze(1)).transpose(1, 2)
                batch['prompt']['seq_lens'] = torch.ceil(batch['tgt_wav']['seq_lens'] / 320).long()
        return batch

    def training_step(self, batch, batch_idx):
        # N = 1
        if not 's2st' in batch:
            batch = {'s2st': batch}
        batch = {k: self.preprocess_batch(v) for k, v in batch.items()}
        loss, logging_output, _ = self.s2st_criterion(batch['s2st'])
        if 'mt' in batch:
            loss_mt = self.mt_criterion(batch['mt'])
            loss += loss_mt
            logging_output['loss_mt'] = loss_mt
        if 'asr' in batch:
            loss_asr, logging_output_asr, _ = self.asr_criterion(batch['asr'])
            loss += loss_asr 
            logging_output = {**logging_output, **logging_output_asr}
        if 'st' in batch:
            loss_st, logging_output_st, _ = self.st_criterion(batch['st'])
            loss += loss_st
            logging_output = {**logging_output, **logging_output_st}
        for log_item, data in logging_output.items():
            if data is not None:
                self.log(f"train_{log_item}", data, on_step=True, prog_bar=False, logger=True, on_epoch=False,
                    sync_dist=True)
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
        
        return loss

    def on_validation_epoch_start(self):
        self.model.eval()
        self.model.input_modality = "speech"
        self.val_metric.set_dtype(torch.float32)


    def validation_step(self, batch, batch_id, dataloader_idx=None):
        if 's2st' in batch:
            batch = batch['s2st']
        batch = self.preprocess_batch(batch)
        rank_zero = self._trainer is None or self._trainer.global_rank == 0
        if batch_id == 0 and rank_zero:
            self._reset_generator(tgt_lang="eng", input_modality="speech", mode=0)
        elif batch_id == 1 or (batch_id == 0 and not rank_zero):
            self._reset_generator(tgt_lang="eng", input_modality="speech", mode=1)
        loss, logging_output, _ = self.s2st_criterion(batch)
        for log_item, data in logging_output.items():
            if data is not None:
                self.log(f"val_{log_item}", data, on_step=False, prog_bar=False, logger=True, on_epoch=True,
                    sync_dist=True)
        self.log("val_loss", loss, on_step=False, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
        self.model.input_modality = "speech"
        gen_output = self.generator(
            batch['speech_inputs']["seqs"],
            batch['speech_inputs']["seq_lens"],
            ngram_filtering=self.ngram_filtering,
            target_length=batch['target_length']["seqs"],
            vad_mask=batch['vad_mask']["seqs"],
            prompts=batch['prompt']['seqs'],
            prompt_lens=batch['prompt']['seq_lens'],
        )
        text_hyps = []
        for t in gen_output.text_hyps:
            try:
                text_hyps.append(str(self.token_decoder(torch.tensor(t))[0]))
            except:
                text_hyps.append('')
                print('decode error')
        text_ref = [str(t) for t in batch['text_ref']]
        
        self.val_metric(text_hyps, [text_ref])
        self.log("val_bleu", self.val_metric, on_step=True, on_epoch=True)

        if batch_id == 0 and rank_zero:
            codec_hyps = [','.join([str(c) for c in codec]) for codec in gen_output.codec_hyps]
            if self.logger is not None and isinstance(self.logger, WandbLogger):
                columns = ['no.', 'text_label', "text", "code"]
                data = []
                for i, (t_l, t, c) in enumerate(zip(text_ref, text_hyps, codec_hyps)):
                    data.append([i, t_l, t, c])
                self.logger.log_table(key="val_audio", columns=columns, data=data, step=self.global_step)

        return logging_output

    

    def _reset_generator(
            self, 
            batch=None, 
            tgt_lang=None,
            input_modality=None,
            mode=0,
            ):
        output_modality = 'TextAndSpeech' if mode == 0 else 'TextOnly'
        if tgt_lang is None:
            tgt_lang = str(batch['tgt_lang'][0])
        if input_modality is None:
            input_modality = str(batch['input_modality'][0])
        self.model.input_modality = input_modality
        self.generator = UnitYGenerator(
            self.model,
            self.tokenizer,
            tgt_lang,
            output_modality=output_modality,
            text_opts=self.gen_opts,
        )

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        pass


    def configure_optimizers(self):
        optimizer, scheduler = configure_optimizer_schedular(
            cfg=self.cfg,
            params_generator=self.named_parameters,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        self.optimizer = optimizer
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def train_dataloader(self):
        loader_cfg = self.cfg['loader_cfg']
        dataset_cfg = self.cfg['train_dataset_cfg']
        datasets = {}
        
        datasets['s2st'] = S2STDataset(
                tokenizer=self.tokenizer, 
                dataset_configs=dataset_cfg['s2st'],
                split='train'
            )
        if 'mt' in dataset_cfg:
            datasets['mt'] = MTDataset(
                    dataset_cfg['mt']['paths'],
                    tokenizer=self.tokenizer, 
                )
        if 'asr' in dataset_cfg:
            datasets['asr'] = ASRDataset(
                    tokenizer=self.tokenizer, 
                    dataset_configs=dataset_cfg['asr'],
                )
        dataloaders = {k: DataLoader(
                v,
                batch_size=loader_cfg['batch_size'],
                shuffle=True,
                num_workers=loader_cfg['num_worker'],
                drop_last=True,
                collate_fn=self.collate,
            ) for k, v in datasets.items()}
        assert len(dataloaders) > 0
        combined_loader = CombinedLoader(dataloaders, 'max_size_cycle')
        return combined_loader

    def val_dataloader(self):
        loader_cfg = self.cfg['loader_cfg']
        dataset = S2STDataset(
            tokenizer=self.tokenizer, 
            dataset_configs=self.cfg['val_dataset_cfg'],
            split='dev'
            )

        dataloader = DataLoader(
                dataset,
                batch_size=loader_cfg['val_batch_size'],
                drop_last=False,
                shuffle=False,
                num_workers=loader_cfg['num_worker'],
                collate_fn=self.collate,
            ) 

        return dataloader

    def test_dataloader(self):
        pass


if __name__ == "__main__":
    from config.parse_yaml_args import parse_args_and_yaml
    import torch

    def deep_to_device(obj, device):
        if isinstance(obj, torch.Tensor):
            obj = obj.to(device)
            if obj.dtype == torch.float32:
                obj = obj.half()
            return obj
        if isinstance(obj, dict):
            return {k: deep_to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [deep_to_device(v, device) for v in obj]

        return obj 

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cfg = parse_args_and_yaml(config_path="../config/exp_spec/m4t_valle4_local.yaml")

    cfg['loader_cfg']['batch_size'] = cfg['loader_cfg']['val_batch_size'] = 3
    cfg['loader_cfg']['num_worker'] = 0
    cfg['use_deepspeed'] = False

    module = M4tValleModule(cfg).half().cuda().eval()
    print(module.model)

    # optimizer, scheduler = configure_optimizer_schedular(
    #     cfg=module.cfg,
    #     params_generator=module.named_parameters,
    #     num_training_steps=100000
    # )


    with torch.no_grad():
        # loader = module.train_dataloader()
        loader = module.val_dataloader()
        for i, b in enumerate(loader):
            if i >= 1:
                break
            b = deep_to_device(b, 'cuda')
            loss = module.training_step(b, 0)
            print(loss, '\n')
            module.on_validation_epoch_start()
            res = module.validation_step(b, 0, 0)
            for k, v in res.items():
                print(k, v, '\n')
                if k == 'codec_hyps':
                    codec_hyps = v
            
            
            print('\n\n')
        module.on_validation_epoch_end()



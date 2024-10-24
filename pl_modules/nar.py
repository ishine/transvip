import torch
import os
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
import soundfile as sf
from pytorch_lightning.loggers import WandbLogger
import wandb
import random
import copy
from sascodec import SASCodec


if __name__ == "__main__":
    import sys
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(BASE_DIR)

from models.nar import NarModel, load_nar_model, NarModelInput, NarConfig
from models.nar.criterion import NarCriteria
from models.optimizer import configure_optimizer_schedular
from models.nar.dataset import JoinedDataset


class NarModule(LightningModule):
    model: NarModel
    criterion: NarCriteria

    def __init__(self, cfg) -> None:
        super().__init__()
        model_cfg = cfg['model_cfg']
        codec_model = SASCodec.from_pretrained(model_cfg['codec_model_path'])
        codec_model.eval().requires_grad_(False)
        self.codec_model = codec_model
        
        self.sample_rate = 16000
        
        model_config = NarConfig(**model_cfg['nar_config']) if 'nar_config' in model_cfg else None

        self.model = load_nar_model(
            model_config=model_config,
            )

        self.cfg = cfg

        self.criterion = NarCriteria(self.model)

        if 'init_path' in cfg:
            self.load_state_dict(torch.load(cfg['init_path'])['state_dict'], strict=False)
            print(f"Loaded model from {cfg['init_path']}.")

    def forward(self, **kwargs):
        
        return self.model(**kwargs)

    def decode(self, inputs: NarModelInput):
        assert inputs.a[0].shape[0] == 1
        inputs = copy.deepcopy(inputs)
        for _ in range(15):
            outputs = self.model(inputs)
            # x = torch.argmax
            a_hyps = [a.argmax(-1, keepdim=True).transpose(0,1) for a in outputs.a_logits]
            inputs.a = [torch.cat([a, h], dim=0) for a, h in zip(inputs.a, a_hyps)]
        return inputs

    def on_train_start(self) -> None:
        sch = self.lr_schedulers()
        sch.step()

    def prepare_code_from_wav(self, batch, step=None, split='random'):
        wav, seq_lens, langs = batch
        self.codec_model.eval().float()
        a = []
        ap = []
        a_lens = []
        label = []
        with torch.no_grad():
            codec = self.codec_model.encode(wav.float())
        if step is None:
            step = random.randint(1, 15)
        source = codec[:, :step]
        for s, t, l in zip(source, codec, seq_lens):
            if split == 'random':
                p_split = random.randint(1, l-1)
            elif split == 'middle':
                p_split = l // 2
            a.append(s[:,p_split:l])
            ap.append(t[:, :p_split])
            a_lens.append(l - p_split)
            label.append(t[step, p_split:l])
        return NarModelInput(a=a, ap=ap, a_lens=a_lens, seq_lens=seq_lens, langs=langs), label

    def training_step(self, batch, batch_idx):
        inputs, label = self.prepare_code_from_wav(batch)
        # N = 1
        loss, logging_output, net_output = self.criterion(inputs, label)
        for log_item, data in logging_output.items():
            if data is not None:
                self.log(f"train_{log_item}", data, on_step=True, prog_bar=True, logger=True, on_epoch=False,
                    sync_dist=True)
        return loss


    def validation_step(self, batch, batch_id, dataloader_idx=None):
        b, label = self.prepare_code_from_wav(batch, step=1, split='middle')
        loss, logging_output, net_output = self.criterion(b, label)
        for log_item, data in logging_output.items():
            if data is not None:
                self.log(f"val_{log_item}", data, on_step=False, prog_bar=False, logger=True, on_epoch=True,
                    sync_dist=True)

        if self.logger is not None and isinstance(self.logger, WandbLogger) and batch_id < 1:
            reconstr = self.decode(b)
            prompt = reconstr.ap
            pred = reconstr.a
            columns = ['no.', "type", "audio"]
            data = []
            for i, (a, p) in enumerate(zip(pred, prompt)):
                try:
                    self.codec_model.eval().float()
                    wav_pred = self.codec_model.decode(a.unsqueeze(0)).detach().cpu().float().squeeze(1).squeeze(0).numpy()
                    wav_prompt = self.codec_model.decode(p.unsqueeze(0)).detach().cpu().float().squeeze(1).squeeze(0).numpy()
                    data.append([i, 'pred', wandb.Audio(wav_pred, sample_rate=self.sample_rate)])
                    data.append([i, 'prompt', wandb.Audio(wav_prompt, sample_rate=self.sample_rate)])
                except Exception as e:
                    print(e)
                    print(a.shape, p.shape)
                if i >= 20:
                    break
            self.logger.log_table(key="val_audio", columns=columns, data=data, step=self.global_step)

        return logging_output


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
        dataset = JoinedDataset(
            self.cfg['train_dataset_cfg'],
            )
        dataloader = DataLoader(
                dataset,
                batch_size=loader_cfg['batch_size'],
                shuffle=True,
                num_workers=loader_cfg['num_worker'],
                drop_last=True,
            ) 
        return dataloader

    def val_dataloader(self):
        loader_cfg = self.cfg['loader_cfg']
        dataset = JoinedDataset(
            self.cfg['val_dataset_cfg'],
            )

        dataloader = DataLoader(
                dataset,
                batch_size=loader_cfg['val_batch_size'],
                drop_last=False,
                shuffle=False,
                num_workers=loader_cfg['num_worker'],
            ) 

        return dataloader

    def test_dataloader(self):
        pass


def deep_to_device(obj, device):
        if isinstance(obj, torch.Tensor):
            obj = obj.to(device)
            if obj.dtype == torch.float32:
                obj = obj.half()
            return obj
        elif isinstance(obj, dict):
            return {k: deep_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [deep_to_device(v, device) for v in obj]
        else:
            return obj 



if __name__ == "__main__":
    from config.parse_yaml_args import parse_args_and_yaml
    import pandas as pd
    import librosa
    import soundfile as sf


    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cfg = parse_args_and_yaml(config_path="../config/exp_spec/nar2_local.yaml")

    cfg['loader_cfg']['batch_size'] = cfg['loader_cfg']['val_batch_size'] = 1
    cfg['loader_cfg']['num_worker'] = 0
    cfg['use_deepspeed'] = False

    module = NarModule(cfg).half().cuda().eval()
    print(module)
    
    optimizer, scheduler = configure_optimizer_schedular(
        cfg=module.cfg,
        params_generator=module.named_parameters,
        num_training_steps=100000
    )

    loader = module.train_dataloader()
    loader = module.val_dataloader()
    # loader = module.test_dataloader()
    with torch.no_grad():
        for i, b in enumerate(loader):
            if i >= 4:
                break

            b = deep_to_device(b, 'cuda')
            loss = module.training_step(b, 0)
            print(loss, '\n')
            val_res = module.validation_step(b, 0, 0)
            for k, v in val_res.items():
                print(k, v, '\n')
            print('\n\n\n')


                




        

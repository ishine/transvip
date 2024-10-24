import torch
import os
import random
import librosa
from torch.utils.data import Dataset
from tqdm import tqdm
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import functional as F
import os.path as osp
from pytorch_lightning.loggers import WandbLogger
from audiotools import AudioSignal
from audiotools.data import transforms
from audiotools.core.util import prepare_batch
from speechtokenizer import SpeechTokenizer
from transformers import Wav2Vec2ForPreTraining
try:
    import wandb
except ImportError:
    pass

from sascodec import SASCodec, Discriminator, MultiScaleSTFTLoss, MelSpectrogramLoss, GANLoss, L1Loss


def build_transform(
    transforms_list: list = ["Identity"],
):
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


def get_all_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def load_commonvoice_list(path, split="train"):
    file_list_path = os.path.join(path, f"{split}.txt")
    lines = []
    line_count = {'train': 21101675, 'test': 8000, 'dev': 2000}
    line_count = line_count[split]
    with open(file_list_path, "r") as f:
        for line in tqdm(f, total=line_count, leave=False):
            lines.append(line.strip())
    return lines


class NSynthDataset(Dataset):
    """Dataset to load NSynth data."""

    def __init__(self, audio_dir, sample_rate=16000, split="train"):
        super().__init__()
        self.audio_dir = audio_dir
        self.filenames = load_commonvoice_list(audio_dir, split=split)
        print(len(self.filenames))
        self.sr = sample_rate
        self.max_len = sample_rate * 2

    def __len__(self):
        return len(self.filenames)

    def _load_audio(self, filename):
        filename = os.path.join(self.audio_dir, filename)
        audio, sr = librosa.load(filename, sr=self.sr)
        audio = torch.tensor(audio).flatten().unsqueeze(0)
        if audio.shape[1] > sr * 30:
            audio = audio[:, :sr * 30]
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

    def __getitem__(self, index):
        ans = torch.zeros(1, self.max_len)
        audio = self._load_audio(self.filenames[index])
        if audio.shape[1] > self.max_len:
            st = random.randint(0, audio.shape[1] - self.max_len - 1)
            ed = st + self.max_len
            ans = audio[:, st:ed]
        else:
            ans[:, :audio.shape[1]] = audio
            st = 0
        ans_teacher = F.pad(ans, (50, 50), 'constant', 0).squeeze(0)
        return ans, ans_teacher


class SASCModule(LightningModule):
    model: SASCodec

    def __init__(self, cfg, eval_only=False) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        model_cfg=cfg['model_cfg']
        
        self.model = SASCodec(**model_cfg['generator'])
        self.descriminator = Discriminator(**model_cfg['discriminator'])
        self.losses = torch.nn.ModuleDict({
            'mel': MelSpectrogramLoss(**model_cfg['losses'].get('mel', {})),
            'stft': MultiScaleSTFTLoss(**model_cfg['losses'].get('stft', {})),
            'gan': GANLoss(self.descriminator),
            'l1': L1Loss(),
            'cos' : torch.nn.CosineSimilarity(dim=-1, eps=1e-6),
            })
        self.loss_weights = model_cfg['losses']['weights']
        self.sample_rate = model_cfg['sample_rate']
        if not eval_only:
            self.semantic_teacher = Wav2Vec2ForPreTraining.from_pretrained(
                cfg['model_cfg']['semantic_path']
                )
            self.semantic_teacher.eval()
            self.semantic_teacher.requires_grad_(False)
            self.semantic_teacher.config.output_hidden_states = True
        self.transform = build_transform(
                ['VolumeNorm', 'RescaleAudio', 'ShiftPhase']
            )

        self.automatic_optimization = False
        self.layer_outputs = []

        init_model_path = cfg['init_model_path']
        self.speechtokenizer = SpeechTokenizer.load_from_checkpoint(
            osp.join(init_model_path, 'config.json'),
            osp.join(init_model_path, 'SpeechTokenizer.pt'),
            )
        self.speechtokenizer.requires_grad_(False)
        print(self.model.encoder.load_state_dict(self.speechtokenizer.encoder.state_dict(), strict=False))
        print(self.model.decoder.load_state_dict(self.speechtokenizer.decoder.state_dict(), strict=False))

        
        if "init_path" in cfg:
            state = torch.load(cfg['init_path'], map_location='cpu')
            print(self.load_state_dict(state['state_dict'], strict=False))
            print("Load model from {}".format(cfg['init_path']))


    def forward(self, **kwargs):
        return self.model(**kwargs)

    def pre_process(self, audio_data):
        signal = AudioSignal(audio_data.clone(), self.sample_rate)
        kwargs = self.transform.batch_instantiate([0]*audio_data.shape[0], signal)
        kwargs = prepare_batch(kwargs, self.device)
        signal = self.transform(
            signal, **kwargs
        )
        return signal


    def training_step(self, batch, batch_idx):
        x, x_teacher = batch

        signal = self.pre_process(x)
        output = {}
        optimizer_g, optimizer_d = self.optimizers()

        out = self.model(signal.audio_data)
        recons = AudioSignal(out["audio"], self.sample_rate)
        output["adv/disc_loss"] = self.losses['gan'].discriminator_loss(recons, signal)
        
        # Discriminator
        optimizer_d.zero_grad()
        self.manual_backward(output["adv/disc_loss"])
        self.clip_gradients(optimizer_d, gradient_clip_val=10, gradient_clip_algorithm="norm")

        has_nan = False
        for param in self.descriminator.parameters():
            if param.grad is not None:
                has_nan = has_nan or torch.isnan(param.grad).any()
        if has_nan:
            print("NAN in descriminator gradients")
            optimizer_d.zero_grad()

        optimizer_d.step()

        # Generator
        with torch.cuda.amp.autocast():
            self.semantic_teacher.eval()
            semantic_teacher_out = self.semantic_teacher(x_teacher)
            sementic_states = torch.stack(semantic_teacher_out.hidden_states).mean(dim=0)
        
        self.speechtokenizer.eval()
        with torch.no_grad():
            semantic_signal = self.speechtokenizer(x, n_q=1)[0]
        signal_zero = AudioSignal(semantic_signal, self.sample_rate)
        recon_zero = AudioSignal(out['audio_0'], self.sample_rate)

        output["stft/loss"] = self.losses['stft'](recons, signal)
        output["mel/loss"] = self.losses['mel'](recons, signal)
        output["mel/loss_zero"] = self.losses['mel'](recon_zero, signal_zero)
        output["waveform/loss"] = self.losses['l1'](recons, signal)
        output["adv/gen_loss"], output["adv/feat_loss"] = self.losses['gan'].generator_loss(recons, signal)
        output["vq/commit_loss"] = out["commit_loss"]
        output["vq/codebook_loss"] = out["codebook_loss"]
        output["semantic/distill_loss"] = 1 - F.cosine_similarity(out['semantic'], sementic_states, dim=1).mean()
        output["loss"] = sum([v * output[k] for k, v in self.loss_weights.items() if k in output])
        
        optimizer_g.zero_grad()
        self.manual_backward(output["loss"])
        self.clip_gradients(optimizer_g, gradient_clip_val=1e3, gradient_clip_algorithm="norm")

        has_nan = False
        for param in self.model.parameters():
            if param.grad is not None:
                has_nan = has_nan or torch.isnan(param.grad).any()
        if has_nan:
            print("NAN in gradients")
            optimizer_g.zero_grad()

        optimizer_g.step()

        for k, v in sorted(output.items()):
            if k == "loss":
                self.log("train/" + k, v, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            else:
                self.log("train/" + k, v, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)

        lr_scheduler_g, lr_scheduler_d = self.lr_schedulers()
        lr_scheduler_g.step()
        lr_scheduler_d.step()
        return output


    def validation_step(self, batch, batch_id, dataloader_idx=None):
        x, x_teacher = batch

        signal = self.pre_process(x)
        output = {}

        out = self.model(signal.audio_data)
        recons = AudioSignal(out["audio"], self.sample_rate)

        output["vq/commit_loss"] = out["commit_loss"]
        output["vq/codebook_loss"] = out["codebook_loss"]
        output["mel/loss"] = self.losses['mel'](recons, signal)
        output['loss'] = output["mel/loss"]

        for k, v in sorted(output.items()):
            self.log("val/" + k, v, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        if self.logger is not None and isinstance(self.logger, WandbLogger) and batch_id < 1:
            audio_wav = out["audio"].detach().cpu().float().squeeze(1).numpy()
            columns = ['type', "no.", "audio"]
            data = [['all', i, wandb.Audio(audio, sample_rate=self.sample_rate)] for i, audio in enumerate(audio_wav)]
            audio_zero = out["audio_0"].detach().cpu().float().squeeze(1).numpy()
            data += [['zero', i, wandb.Audio(audio, sample_rate=self.sample_rate)] for i, audio in enumerate(audio_zero)]
            self.logger.log_table(key="val_audio", columns=columns, data=data, step=self.global_step)
        return output


    def configure_optimizers(self):
        optim_cfg = self.cfg['optim_cfg']
        lr = optim_cfg['learning_rate']
        betas = optim_cfg['adam_betas']
        gamma = optim_cfg['decay_gamma']
        optimizer_g = torch.optim.AdamW(
            self.model.parameters(), lr=lr, betas=betas
            )
        lr_scheduler_g = ExponentialLR(optimizer_g, gamma=gamma)
        optimizer_d = torch.optim.AdamW(
            self.descriminator.parameters(),
            lr=lr,
            betas=betas
            )
        lr_scheduler_d = ExponentialLR(optimizer_d, gamma=gamma)

        return [optimizer_g, optimizer_d], [lr_scheduler_g, lr_scheduler_d]


    def train_dataloader(self):
        loader_cfg = self.cfg['loader_cfg']
        dataset = NSynthDataset(
            **self.cfg['train_dataset_cfg'],
            split='train',
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
        dataset = NSynthDataset(
            **self.cfg['val_dataset_cfg'],
            split='dev',
            )

        dataloader = DataLoader(
                dataset,
                batch_size=loader_cfg['val_batch_size'],
                drop_last=False,
                shuffle=False,
                num_workers=loader_cfg['num_worker'],
            ) 

        return dataloader


if __name__ == "__main__":
    from config.parse_yaml_args import parse_args_and_yaml
    import torch

    def deep_to_device(obj, device):
        if isinstance(obj, torch.Tensor):
            obj = obj.to(device)
            return obj
        elif isinstance(obj, dict):
            return {k: deep_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [deep_to_device(v, device) for v in obj]
        else:
            return obj 

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cfg = parse_args_and_yaml(config_path="../config/exp_spec/sasc2_local.yaml")

    cfg['loader_cfg']['batch_size'] = cfg['loader_cfg']['val_batch_size'] = 2
    cfg['loader_cfg']['num_worker'] = 0
    cfg['use_deepspeed'] = False


    module = SASCModule(cfg, True).cuda()
    module.model = SASCodec.from_pretrained('/data/pretrained/sasc/v1.pt').cuda()

    with torch.no_grad():
        loader = module.train_dataloader()
    loader = module.val_dataloader()
    for i, b in enumerate(loader):
        if i >= 5:
            break

        b = deep_to_device(b, 'cuda')
        loss = module.training_step(b, 0)
        print(loss, '\n')
        module.on_validation_epoch_start()
        res = module.validation_step(b, 0, 0)
        for k, v in res.items():
            print(k, v, '\n')            
        
        print('\n\n')
    module.on_validation_epoch_end()


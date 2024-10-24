import torch.nn.functional as F
from copy import deepcopy
import torch
import random

from fairseq2_011.models.seq2seq import Seq2SeqBatch
from .model import UnitYModel
from .fairseq2_custom.nllb.multihead_attention import StoreAttentionWeights

class S2STCriteria:
    def __init__(self, model: UnitYModel) -> None:
        self.model = model
        self.ignore_index = 0 # pad
        self.label_smoothing = 0.0 # fix to 0 for now
        self.speech_weight = 1.0
        self.text_weight = 1.0
        self.kld_weight = 1.0

    def _ce_loss(self, logits, label, reduction='mean'):
        label = label.reshape(-1)
        if label.max() >= logits.shape[-1]:
            raise ValueError(f"Label max {label.max()} is larger than logits shape {logits.shape}")
        loss = F.cross_entropy(
            logits.reshape((-1, logits.shape[-1])),
            label,
            reduction=reduction,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )
        return loss
    
    def _kld_loss(self, logits_stu, logits_tea, token_mask, reduction):
        b, t, c = logits_stu.shape
        logits_stu = logits_stu.reshape((-1, logits_stu.shape[-1])).float()
        logits_tea = logits_tea.reshape((-1, logits_tea.shape[-1])).detach().float()
        loss = F.kl_div(
            F.log_softmax(logits_stu, dim=-1),
            F.softmax(logits_tea, dim=-1),
            reduction='none',
        ).sum(-1)
        loss = loss.reshape((b, t)) * token_mask.reshape((b, t))
        loss = loss.clamp(max=10, min=0)
        # if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
        #     raise ValueError(f"KLD loss is nan")
        if reduction == 'mean':
            loss = loss.sum() / token_mask.sum()
        elif reduction == 'sum':
            loss = loss.sum()
        return loss

    def _kd_loss(self, logits_stu, logits_tea, token_mask, reduction):
        b, t, c = logits_stu.shape
        dtype = logits_stu.dtype
        logits_stu = logits_stu.reshape((-1, logits_stu.shape[-1])).float()
        logits_tea = logits_tea.reshape((-1, logits_tea.shape[-1])).detach().float()
        loss = -(F.log_softmax(logits_stu, dim=-1) * F.softmax(logits_tea, dim=-1)).sum(-1)
        loss = loss.reshape((b, t)) * token_mask.reshape((b, t)).long()
        loss = loss.to(dtype)
        if reduction == 'mean':
            loss = loss.sum() / token_mask.sum()
        elif reduction == 'sum':
            loss = loss.sum()
        return loss

    def __call__(self, batch):
        text_mask = batch['text_mask']['seqs']
        codec_mask = batch['codec_mask']['seqs']

        # src2tgt_mask = batch['src2tgt_mask']
        text_length = batch["text_length"]
        max_text_length = max(text_length)

        # speech
        self.model.input_modality = "speech"
        model_inputs = Seq2SeqBatch(
            source_seqs=batch["speech_inputs"]['seqs'],
            source_seq_lens=batch["speech_inputs"]['seq_lens'],
            target_seqs=batch["dec_inputs"]['seqs'],
            target_seq_lens=batch["dec_inputs"]['seq_lens'],
        )   
        speech_output = self.model(
            model_inputs, 
            target_length=batch["target_length"]['seqs'], 
            vad_mask=batch["vad_mask"]['seqs'],
            prompts=batch['prompt']['seqs'],
            prompt_lens=batch['prompt']['seq_lens'],
            )

        speech_ce_loss = self._ce_loss(speech_output.logits, batch["label"]['seqs'], reduction='none')
        mean_s2t_ce_loss = (speech_ce_loss * text_mask.reshape(-1)).sum() / text_mask.sum()
        mean_s2c_ce_loss = (speech_ce_loss * codec_mask.reshape(-1)).sum() / codec_mask.sum()

        # text
        self.model.input_modality = "text"
        model_inputs = Seq2SeqBatch(
            source_seqs=batch["text_inputs"]['seqs'],
            source_seq_lens=batch["text_inputs"]['seq_lens'],
            target_seqs=batch["dec_inputs"]['seqs'],
            target_seq_lens=batch["dec_inputs"]['seq_lens'],
        )
        text_output = self.model(
            model_inputs,
            target_length=batch["target_length"]['seqs'],
            vad_mask=batch["vad_mask"]['seqs'],
            prompts=batch['prompt']['seqs'],
            prompt_lens=batch['prompt']['seq_lens'],
        )

        text_ce_loss = self._ce_loss(text_output.logits, batch["label"]['seqs'], reduction='none')
        mean_t2t_ce_loss = (text_ce_loss * text_mask.reshape(-1)).sum() / text_mask.sum()
        mean_t2c_ce_loss = (text_ce_loss * codec_mask.reshape(-1)).sum() / codec_mask.sum()

        token_mask = text_mask | codec_mask
    


        text_mask = ~(batch["label"]['seqs'] >=256206 ) | (batch["label"]['seqs'] == 0) | (batch["label"]['seqs'] == 3)
        speech_text_kd_loss = self._kld_loss(speech_output.logits[:, :max_text_length], text_output.logits[:, :max_text_length], text_mask[:, :max_text_length], reduction='None')
        speech_text_kd_loss = speech_text_kd_loss.sum() / text_mask[:, :max_text_length].sum()


        speech_ce_loss = speech_ce_loss.sum() / token_mask.sum()
        text_ce_loss = text_ce_loss.sum() / token_mask.sum()

        loss = self.speech_weight * speech_ce_loss + self.text_weight * text_ce_loss 
        if speech_text_kd_loss is not None:
            loss += speech_text_kd_loss * self.kld_weight
        
        logged_output = {
            f"s2t_ce_loss": mean_s2t_ce_loss,
            f"s2c_ce_loss": mean_s2c_ce_loss,
            f"t2t_ce_loss": mean_t2t_ce_loss,
            f"t2c_ce_loss": mean_t2c_ce_loss,
            f"kd_loss": speech_text_kd_loss if speech_text_kd_loss is not None else torch.tensor(0.0),
        }

        model_outputs = {
            "speech_logits": speech_output.logits,
            "text_logits": text_output.logits,
        }

        if torch.isnan(loss):
            print('S2ST loss is nan')

        return loss, logged_output, model_outputs

class MTCriteria:
    def __init__(self, model: UnitYModel) -> None:
        self.model = model
        self.ignore_index = 0 # pad
        self.label_smoothing = 0.0 # fix to 0 for now

    def _ce_loss(self, logits, label, reduction='mean'):
        label = label.reshape(-1)
        if label.max() >= logits.shape[-1]:
            raise ValueError(f"Label max {label.max()} is larger than logits shape {logits.shape}")
        loss = F.cross_entropy(
            logits.reshape((-1, logits.shape[-1])),
            label,
            reduction=reduction,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )
        return loss

    def __call__(self, batch):
        self.model.input_modality = "text"
        model_inputs = Seq2SeqBatch(
            source_seqs=batch["text_inputs"]['seqs'],
            source_seq_lens=batch["text_inputs"]['seq_lens'],
            target_seqs=batch["dec_inputs"]['seqs'],
            target_seq_lens=batch["dec_inputs"]['seq_lens'],
        )
        text_output = self.model(
            model_inputs,
        )

        return self._ce_loss(text_output.logits, batch["label"]['seqs'], reduction='mean')

class ASRCriteria:
    def __init__(self, model: UnitYModel) -> None:
        self.model = model
        self.ignore_index = 0 # pad
        self.label_smoothing = 0.0

    def _ce_loss(self, logits, label, reduction='mean'):
        label = label.reshape(-1)
        if label.max() >= logits.shape[-1]:
            raise ValueError(f"Label max {label.max()} is larger than logits shape {logits.shape}")
        loss = F.cross_entropy(
            logits.reshape((-1, logits.shape[-1])),
            label,
            reduction=reduction,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )
        return loss

    def __call__(self, batch):
        text_mask = batch['text_mask']['seqs']
        codec_mask = batch['codec_mask']['seqs']

        # speech
        self.model.input_modality = "speech"
        model_inputs = Seq2SeqBatch(
            source_seqs=batch["speech_inputs"]['seqs'],
            source_seq_lens=batch["speech_inputs"]['seq_lens'],
            target_seqs=batch["dec_inputs"]['seqs'],
            target_seq_lens=batch["dec_inputs"]['seq_lens'],
        )   
        speech_output = self.model(
            model_inputs, 
            target_length=batch["target_length"]['seqs'], 
            vad_mask=batch["vad_mask"]['seqs'],
            prompts=batch['prompt']['seqs'],
            prompt_lens=batch['prompt']['seq_lens'],
            )

        loss = self._ce_loss(speech_output.logits, batch["label"]['seqs'], reduction='none')
        mean_s2t_ce_loss = (loss * text_mask.reshape(-1)).sum() / text_mask.sum()
        mean_s2c_ce_loss = (loss * codec_mask.reshape(-1)).sum() / codec_mask.sum()

        token_mask = text_mask | codec_mask

        # loss = loss.sum() / token_mask.sum()

        loss = mean_s2c_ce_loss
        if torch.isnan(loss):
            print('ASR loss is nan')
            loss = torch.tensor(0.0, device=loss.device)
        
        logged_output = {
            f"s2t_sr_loss": mean_s2t_ce_loss,
            f"s2c_sr_loss": mean_s2c_ce_loss,
        }

        model_outputs = {
            "speech_logits": speech_output.logits,
        }

        return loss, logged_output, model_outputs


class STCriteria:
    def __init__(self, model: UnitYModel) -> None:
        self.model = model
        self.ignore_index = 0 # pad
        self.label_smoothing = 0.0 # fix to 0 for now
        self.speech_weight = 1.0
        self.text_weight = 1.0
        self.kld_weight = 1.0

    def _ce_loss(self, logits, label, reduction='mean'):
        label = label.reshape(-1)
        if label.max() >= logits.shape[-1]:
            raise ValueError(f"Label max {label.max()} is larger than logits shape {logits.shape}")
        loss = F.cross_entropy(
            logits.reshape((-1, logits.shape[-1])),
            label,
            reduction=reduction,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )
        return loss
    
    def _kld_loss(self, logits_stu, logits_tea, token_mask, reduction):
        b, t, c = logits_stu.shape
        logits_stu = logits_stu.reshape((-1, logits_stu.shape[-1])).float()
        logits_tea = logits_tea.reshape((-1, logits_tea.shape[-1])).detach().float()
        loss = F.kl_div(
            F.log_softmax(logits_stu, dim=-1),
            F.softmax(logits_tea, dim=-1),
            reduction='none',
        ).sum(-1)
        loss = loss.reshape((b, t)) * token_mask.reshape((b, t))
        loss = loss.clamp(max=10, min=0)
        # if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
        #     raise ValueError(f"KLD loss is nan")
        if reduction == 'mean':
            loss = loss.sum() / token_mask.sum()
        elif reduction == 'sum':
            loss = loss.sum()
        return loss

    def __call__(self, batch):
        # speech
        self.model.input_modality = "speech"
        model_inputs = Seq2SeqBatch(
            source_seqs=batch["speech_inputs"]['seqs'],
            source_seq_lens=batch["speech_inputs"]['seq_lens'],
            target_seqs=batch["dec_inputs"]['seqs'],
            target_seq_lens=batch["dec_inputs"]['seq_lens'],
        )   
        speech_output = self.model(
            model_inputs, 
            )
        speech_ce_loss = self._ce_loss(speech_output.logits, batch["label"]['seqs'], reduction='mean')

        # text
        self.model.input_modality = "text"
        model_inputs = Seq2SeqBatch(
            source_seqs=batch["text_inputs"]['seqs'],
            source_seq_lens=batch["text_inputs"]['seq_lens'],
            target_seqs=batch["dec_inputs"]['seqs'],
            target_seq_lens=batch["dec_inputs"]['seq_lens'],
        )
        text_output = self.model(
            model_inputs,
        )
        text_ce_loss = self._ce_loss(text_output.logits, batch["label"]['seqs'], reduction='mean')

        token_mask = (batch["label"]['seqs'] != 0).long()
        speech_text_kd_loss = self._kld_loss(speech_output.logits, text_output.logits, token_mask, reduction='mean')

        loss = self.speech_weight * speech_ce_loss + self.text_weight * text_ce_loss 
        if speech_text_kd_loss is not None:
            loss += speech_text_kd_loss * self.kld_weight
        
        logged_output = {
            f"s2t_st_loss": speech_ce_loss,
            f"t2t_st_loss": text_ce_loss,
            f"kd_st_loss": speech_text_kd_loss if speech_text_kd_loss is not None else torch.tensor(0.0),
        }

        model_outputs = {
            "speech_logits": speech_output.logits,
            "text_logits": text_output.logits,
        }

        if torch.isnan(loss):
            print('ST loss is nan')
            loss = torch.tensor(0.0, device=loss.device)

        return loss, logged_output, model_outputs
import torch
import torch.nn.functional as F
import copy
from models.nar.model import NarModelInput, NarModelOutput, NarModel
from models.sasc import SASC

class NarGenerater:
    def __init__(self, model: NarModel, codec_model: SASC):
        self.model = model
        # self.codec_model = codec_model

    @torch.no_grad()
    def generate(self, inputs: NarModelInput, beam_size=10) -> NarModelOutput:
        inputs_ori = copy.deepcopy(inputs)
        bsz = len(inputs.ap)
        gen_outputs = []
        for batch_idx in range(bsz):
            inputs = NarModelInput(
                ap=[inputs_ori.ap[batch_idx]] * beam_size,
                a=[inputs_ori.a[batch_idx]] * beam_size,
                seq_lens=inputs_ori.seq_lens[batch_idx].unsqueeze(0).expand(beam_size)-1,
                a_lens=inputs_ori.a_lens[batch_idx].unsqueeze(0).expand(beam_size),
                langs=[inputs_ori.langs[batch_idx]] * beam_size
            )
            lprob_total = torch.zeros(beam_size, beam_size).to(inputs.ap[0].device)
            # spk_embeds_ref = self.model.spk_encoder(inputs.prompts[:1], None)
            for _ in range(15):
                outputs:NarModelOutput = self.model(inputs)
                logits = torch.stack(outputs.a_logits, dim=0) # (beam_size, seq_len, vocab_size)
                lprobs = F.log_softmax(logits, dim=-1)
                topk_values, topk_indices = torch.topk(logits, 5, dim=-1)
                topk_probs = F.softmax(topk_values, dim=-1)
                candidates = torch.zeros(beam_size, beam_size, logits.shape[1]).long().to(logits.device)
                for i in range(beam_size):
                    samples = torch.multinomial(topk_probs.reshape(-1, 5), 1).reshape(-1, logits.shape[1])
                    sampled_indices = torch.gather(topk_indices, -1, samples.unsqueeze(-1)) # (beam_size, seq_len)
                    sampled_lprobs = torch.gather(lprobs, -1, sampled_indices).squeeze(-1) # (beam_size, seq_len)
                    avg_lprobs = torch.mean(sampled_lprobs, dim=-1)
                    candidates[:, i] = sampled_indices.squeeze(-1)
                    lprob_total = lprob_total.clone()
                    lprob_total[:, i] = lprob_total[:, i] + avg_lprobs

                hyps = []
                for i in range(beam_size):
                    a = inputs.a[i]
                    for j in range(beam_size):
                        h = candidates[i, j].unsqueeze(0)
                        hyps.append(torch.cat([a, h], dim=0))
                hyps = torch.stack(hyps, dim=0)
                # wavs = self.codec_model.decode(hyps)
                # feature = self.codec_model.encoder(wavs).half().transpose(1, 2)
                # spk_embeds = self.model.spk_encoder(feature[:, :500], None)
                # similarity = F.cosine_similarity(spk_embeds, spk_embeds_ref.expand(beam_size*beam_size, -1), dim=-1)
                # similarity = (similarity.reshape(beam_size, beam_size) + 1) / 2
                # lprob_total = lprob_total + torch.log(similarity) * 0.3



                values, flat_indices = torch.topk(lprob_total.flatten(), beam_size)
                rows = flat_indices // beam_size
                cols = flat_indices % beam_size
                lprob_total = values.reshape(beam_size, 1).expand(-1, beam_size)
                next_inputs = []
                for i, (row, col) in enumerate(zip(rows, cols)):
                    a = inputs.a[row]
                    h = candidates[row, col].unsqueeze(0)
                    next_inputs.append(torch.cat([a, h], dim=0))
                inputs.a = next_inputs
            gen_outputs.append(inputs.a[0])
        return gen_outputs


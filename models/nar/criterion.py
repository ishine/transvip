from .model import NarModel
import torch.nn.functional as F
import torch

class NarCriteria:
    model: NarModel

    def __init__(self, model: NarModel) -> None:
        self.model = model
        self.label_smoothing = 0.0 # fix to 0 for now

    def _ce_loss(self, logits, label, reduction='mean'):
        label = label.reshape(-1)
        if label.max() >= logits.shape[-1]:
            raise ValueError(f"Label max {label.max()} is larger than logits shape {logits.shape}")
        loss = F.cross_entropy(
            logits.reshape((-1, logits.shape[-1])),
            label,
            reduction=reduction,
            label_smoothing=self.label_smoothing,
        )
        acc = (logits.argmax(dim=-1) == label).float().mean()
        return loss, acc

    def __call__(self, inputs, labels):

        model_outputs = self.model(inputs)
        a_logits = torch.cat(model_outputs.a_logits, dim=0)
        label = torch.cat(labels, dim=0)
        loss, acc = self._ce_loss(a_logits, label, reduction='mean')
        
        logged_output = {
            "loss": loss,
            "acc": acc
        }

        return loss, logged_output, model_outputs
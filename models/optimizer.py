# from transformers import get_polynomial_decay_schedule_with_warmup
from fairseq2_011.optim.lr_scheduler import MyleLR, PolynomialDecayLR
import torch
try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    deepspeed_available = True
except ImportError:
    deepspeed_available = False


def configure_optimizer_schedular(cfg, params_generator, num_training_steps=None):
    optim_cfg = cfg['optim_cfg']
    no_decay_group = optim_cfg.get("no_decay", [])
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in params_generator()
                       if (not any(nd in n for nd in no_decay_group) and p.requires_grad)],
            "weight_decay": optim_cfg['weight_decay'],
        },
        {
            "params": [p for n, p in params_generator()
                       if any(nd in n for nd in no_decay_group) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    if cfg['use_deepspeed'] and deepspeed_available and cfg['deepspeed_cfg']['offload_optimizer']:
        optimizer = DeepSpeedCPUAdam(
            optimizer_grouped_parameters,
            lr=optim_cfg['learning_rate'],
            eps=optim_cfg['adam_epsilon'],
            betas=optim_cfg['adam_betas'],
        )
    else:
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=optim_cfg['learning_rate'],
            eps=optim_cfg['adam_epsilon'],
            betas=optim_cfg['adam_betas'],
        )

    scheduler_cfg = cfg['scheduler_cfg']

    if scheduler_cfg['name'].lower() == "myle":
        scheduler = MyleLR(
            optimizer,
            num_warmup_steps=scheduler_cfg.get("warmup_steps", 0),
        )
    elif scheduler_cfg['name'].lower() == "poly":
        assert num_training_steps is not None
        scheduler = PolynomialDecayLR(
            optimizer,
            num_steps=num_training_steps,
            num_warmup_steps=scheduler_cfg.get("warmup_steps", 0),
            power=scheduler_cfg.get("power", 1.0),
            final_lr=scheduler_cfg.get("lr_end", 1e-7)
        )
    else:
        raise NotImplementedError(f"Scheduler {scheduler_cfg['name']} must be one of myle, poly.")

    return optimizer, scheduler

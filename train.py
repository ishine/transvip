from pathlib import Path
import os
import os.path as osp
import logging

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy

from pl_modules import get_module, get_module_class
from config.parse_yaml_args import load_config

import yaml

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
import torch



if __name__ == "__main__":

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # cfg = load_config()
    cfg = load_config()

    ckpt_cfg = cfg['ckpt_cfg']

    seed_everything(cfg['seed'])

    checkpoint_callback = ModelCheckpoint(
        **ckpt_cfg,
        auto_insert_metric_name=False,
        save_top_k=-1,  # all model save
        save_last=True,
    )

    try:
        logger = WandbLogger(
            project=cfg['train_name'],
            name=f"{cfg['train_name']}-{cfg['train_id']}",
            version=cfg['train_id'],
            save_dir=osp.join(cfg['log_dir'], cfg['train_name'], cfg['train_id']),
            log_model=False
        )
    except ModuleNotFoundError:
        logger = TensorBoardLogger(
            save_dir=cfg['log_dir'],
            name=cfg['train_name'],
            version=cfg['train_id']
        )

    callback_list = [
        checkpoint_callback, 
        LearningRateMonitor(logging_interval="step")
        ]
    if 'resume_from_path' in cfg:
        module_cls = get_module_class(cfg)
        module_cfg = cfg['module_cfg']
        module = module_cls.load_from_checkpoint(
            cfg['resume_from_path'],
            cfg=cfg,
            **module_cfg,
            map_location='cpu'
        )
        print('resume from path:', cfg['resume_from_path'])
    else:
        module = get_module(cfg)


    if cfg['use_deepspeed']:
        strategy = DeepSpeedStrategy(
            **cfg['deepspeed_cfg'],
            logging_level=logging.WARN,
        )
    else:
        strategy = DDPStrategy(find_unused_parameters=True)

    trainer = Trainer(
        **cfg['trainer_cfg'],
        accelerator="gpu",
        logger=logger,
        callbacks=callback_list,
        strategy=strategy,
        )

    # if cfg.get('val_first', False):
    # trainer.validate(module)

    if cfg['mode'] == "test_last":
        trainer.test(module, ckpt_path="last")
    elif cfg['mode'] == "test":
        trainer.test(module)
    else:
        trainer.fit(module, ckpt_path='last')



import yaml
import argparse
from pathlib import Path
import os
import os.path as osp
import torch.distributed as dist

default_config_parser = parser = argparse.ArgumentParser(
    description='Training Config', add_help=False)

parser.add_argument(
    '-c',
    '--config',
    default='sasc_local.yaml',
    type=str,
    metavar='FILE',
    help='YAML config file specifying default arguments')

def rank_zero():
    return (dist.is_initialized() and dist.get_rank() == 0) or ('RANK' in os.environ and os.environ['RANK'] == '0')


def parse_args_and_yaml(given_parser=None, config_path=None):
    if given_parser is None:
        given_parser = default_config_parser
    given_configs, remaining = given_parser.parse_known_args()
    file_name = given_configs.config if "yaml" in given_configs.config else given_configs.config + ".yaml"
    if config_path is None:
        config_path = "config/exp_spec/" + file_name if '/' not in file_name else file_name
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        # given_parser.set_defaults(**cfg)
    return cfg

def load_config(config_path=None):
    cfg = parse_args_and_yaml(config_path=config_path)

    ckpt_cfg = cfg['ckpt_cfg']
    log_dir = osp.join(ckpt_cfg['dirpath'], 'logs')
    Path(log_dir).mkdir(exist_ok=True)
    dir_path = osp.join(
        ckpt_cfg['dirpath'], 
        f"ckpt_{cfg['train_name']}_{cfg['train_id']}"
        )  
    config_path = osp.join(dir_path, "config.yaml")
    # if cfg['mode'] in ["test_last", "resume"] and osp.exists(config_path):
    #     old_cfg = parse_args_and_yaml(config_path=config_path)
    #     old_cfg['mode'] = cfg['mode']
    #     cfg = old_cfg
    #     print(f"Loading config from {config_path}")
    if rank_zero():
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, default_flow_style=False)
        print(f"Saving config to {config_path}")
    else:
        print("Not doing anything with config")
    cfg['ckpt_cfg']['dirpath'] = dir_path
    cfg['log_dir'] = log_dir
    return cfg



# if __name__ == "__main__":
#     args, args_text = _parse_args_and_yaml()
#     print(args_text)
#     print(args.cache_dir)

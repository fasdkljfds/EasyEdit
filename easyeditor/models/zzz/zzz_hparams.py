from dataclasses import dataclass
from typing import List
from typing import Optional

import yaml
from omegaconf import DictConfig

from ...util.hparams import HyperParams


@dataclass
class ZZZHyperParams(HyperParams):
    # 算法和模型参数
    alg_name: str
    model_name: str
    device: int

    # 内部参数设置
    inner_params: List[str]
    norm_constraint: float

    # 随机种子
    seed: int

    embedding: DictConfig
    clustering: DictConfig
    
    batch_size: int = 1
    max_length: int = 30
    model_parallel: bool = False
    use_chat_template: bool = False

    save_path: Optional[str] = None
    load_path: Optional[str] = None
    
    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'
    
        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)
        
        # 验证算法名称是否正确
        assert (config and config['alg_name'] == 'ZZZ'), \
            f'ZZZHyperParams can not load from {hparams_name_or_path}. alg_name is {config["alg_name"]}'

        return cls(**config)        
    
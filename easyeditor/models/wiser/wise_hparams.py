from dataclasses import dataclass
from typing import List, Union
from ...util.hparams import HyperParams
import yaml


@dataclass
class WISEHyperParams(HyperParams):
    # Experiments

    edit_lr: float
    n_iter: int
    # Method
    objective_optimization: str
    mask_ratio: float
    alpha: float    # act_margin[0]
    beta: float  # act_margin[1]
    gamma: float  # act_margin[2]
    act_ratio: float
    merge_freq: int
    retrieve: bool
    replay: bool
    save_freq: Union[int, None]
    merge_alg: str
    norm_constraint: float
    # Module templates
    inner_params: List[str]
    weights: Union[float, None]
    densities: Union[float, None]

    device: int
    alg_name: str
    model_name: str

    # 新增参数：语义因果解耦相关
    use_causal_intervention: bool = False  # 是否使用因果干预
    concept_dim: int = 32  # 概念空间的维度
    
    # 新增参数：正交性保持相关
    use_orthogonality: bool = False  # 是否使用正交性约束
    orthogonality_weight: float = 0.1  # 正交性损失的权重
        
    # 新增参数：自适应稀疏相关
    use_adaptive_sparsification: bool = False  # 是否使用自适应稀疏策略
    conflict_threshold: float = 0.6  # 冲突检测阈值

    # Defaults
    batch_size: int = 1
    max_length: int = 30
    model_parallel: bool = False
    use_chat_template: bool = False

    # Save and Load
    save_path: str = None
    load_path: str = None

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert config['merge_freq'] % config['save_freq'] == 0, 'merge_freq need to be divisible by save_freq (like 1000 / 500)'
        assert len(config['act_margin']) == 3
        config['alpha'], config['beta'], config['gamma'] = config['act_margin'][0], config['act_margin'][1], config['act_margin'][2]
        config.pop('act_margin')

        # 设置默认值 - 支持新增参数
        if 'use_causal_intervention' not in config:
            config['use_causal_intervention'] = False
        if 'concept_dim' not in config:
            config['concept_dim'] = 32
        if 'use_orthogonality' not in config:
            config['use_orthogonality'] = False
        if 'orthogonality_weight' not in config:
            config['orthogonality_weight'] = 0.1
        if 'use_adaptive_sparsification' not in config:
            config['use_adaptive_sparsification'] = False
        if 'conflict_threshold' not in config:
            config['conflict_threshold'] = 0.6

        assert (config and config['alg_name'] == 'WISE'), \
            f'WISEHyperParams can not load from {hparams_name_or_path}. alg_name is {config["alg_name"]}'
        return cls(**config)
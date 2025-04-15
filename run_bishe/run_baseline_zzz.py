import os
import os.path as path
import json
import random
import sys
import argparse
from typing import List, Any, Union
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import yaml # 用于保存临时 hparams 文件


sys.path.append(os.getcwd()+'/EasyEdit')
sys.path.append(os.getcwd()+'/EasyEdit/run_bishe')

try:
    from EasyEdit.easyeditor import (
        ZZZHyperParams
        )

    from EasyEdit.easyeditor import BaseEditor
    from EasyEdit.easyeditor.models.ike import encode_ike_facts
    from sentence_transformers import SentenceTransformer
    from EasyEdit.easyeditor import KnowEditDataset
    from EasyEdit.easyeditor.models.zzz.router import KnowRouter

except ImportError:
    from easyeditor import (
        ZZZHyperParams
        )

    from easyeditor import BaseEditor
    from easyeditor.models.ike import encode_ike_facts
    from sentence_transformers import SentenceTransformer
    from easyeditor import KnowEditDataset
    from easyeditor.models.zzz.router import KnowRouter
















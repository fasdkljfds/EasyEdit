# 测试4.16的路由策略能否区分相近表述
# 4.16

import sys
import os

sys.path.append(os.getcwd()+'/EasyEdit')
sys.path.append(os.getcwd()+'/EasyEdit/run_bishe')

from multiarea_dataset import MultiAreaDataset
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

dataset_configs = {
    'business_industry': 50,
    'human_scientist': 50,
    'event_sport': 50,
    'geography_forest': 50,
    'place_landmark': 50
}

multiarea_dataset = MultiAreaDataset(
    root_dir='EasyEdit/data/output_meta_llama_3_8b_instruct',
    dataset_configs=dataset_configs,
    seed=42,  # 只有随机采样时有用
    random_sample=False
)

editing_hparams = ZZZHyperParams
hparams = editing_hparams.from_hparams('EasyEdit/hparams/ZZZ/llama3.2-1b.yaml')
router = KnowRouter(cfg=hparams)

prompts, rephrase_prompts, target_new, subjects, locality_inputs, _ = multiarea_dataset.to_edit_dataset()

router.build_route_table(prompt_list=prompts)

print("路由表构建完成")


locality_prompts = locality_inputs['neighborhood']['prompt']








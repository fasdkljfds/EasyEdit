# 运行ZZZ
# 适配counterfact
# 确实是适配counterfact的4.20

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


class DataHandler:
    def __init__(self):
        pass

    
    def load_counterfact_zzz(self, data_dir: str, ds_size: int):
        datas = KnowEditDataset(data_dir, size=ds_size)

        prompts = [data['prompt'] for data in datas]
        subjects = [data['subject'] for data in datas]
        target_new = [data['target_new'] for data in datas]

        portability_r = [data['portability_r'] for data in datas]
        portability_s = [data['portability_s'] for data in datas]
        portability_l = [data['portability_l'] for data in datas]

        portability_reasoning_prompts = []
        portability_reasoning_ans = []
        portability_Logical_Generalization_prompts = []
        portability_Logical_Generalization_ans = []
        portability_Subject_Aliasing_prompts = []
        portability_Subject_Aliasing_ans = []

        portability_data = [portability_r, portability_s, portability_l]
        portability_prompts = [portability_reasoning_prompts, portability_Subject_Aliasing_prompts, portability_Logical_Generalization_prompts]
        portability_answers = [portability_reasoning_ans, portability_Subject_Aliasing_ans, portability_Logical_Generalization_ans]
        for data, portable_prompts, portable_answers in zip(portability_data, portability_prompts, portability_answers):
            for item in data:
                if item is None:
                    portable_prompts.append(None)
                    portable_answers.append(None)
                else:
                    temp_prompts = []
                    temp_answers = []
                    for pr in item:
                        prompt = pr["prompt"]
                        an = pr["ground_truth"]
                        while isinstance(an, list):
                            an = an[0]
                        if an.strip() == "":
                            continue
                        temp_prompts.append(prompt)
                        temp_answers.append(an)
                    portable_prompts.append(temp_prompts)
                    portable_answers.append(temp_answers)
        assert len(prompts) == len(portability_reasoning_prompts) == len(portability_Logical_Generalization_prompts) == len(portability_Subject_Aliasing_prompts)

        locality_rs = [data['locality_rs'] for data in datas]
        locality_f = [data['locality_f'] for data in datas]
        locality_Relation_Specificity_prompts = []
        locality_Relation_Specificity_ans = []
        locality_Forgetfulness_prompts = []
        locality_Forgetfulness_ans = []

        locality_data = [locality_rs, locality_f]
        locality_prompts = [locality_Relation_Specificity_prompts, locality_Forgetfulness_prompts]
        locality_answers = [locality_Relation_Specificity_ans, locality_Forgetfulness_ans]
        for data, local_prompts, local_answers in zip(locality_data, locality_prompts, locality_answers):
            for item in data:
                if item is None:
                    local_prompts.append(None)
                    local_answers.append(None)
                else:
                    temp_prompts = []
                    temp_answers = []
                    for pr in item:
                        prompt = pr["prompt"]
                        an = pr["ground_truth"]
                        while isinstance(an, list):
                            an = an[0]
                        if an.strip() == "":
                            continue
                        temp_prompts.append(prompt)
                        temp_answers.append(an)
                    local_prompts.append(temp_prompts)
                    local_answers.append(temp_answers)
        assert len(prompts) == len(locality_Relation_Specificity_prompts) == len(locality_Forgetfulness_prompts)
        locality_inputs = {}
        portability_inputs = {}

        locality_inputs = {
            'Relation_Specificity': {
                'prompt': locality_Relation_Specificity_prompts,
                'ground_truth': locality_Relation_Specificity_ans
            },
            'Forgetfulness': {
                'prompt': locality_Forgetfulness_prompts,
                'ground_truth': locality_Forgetfulness_ans
            }
        }
        portability_inputs = {
            'Subject_Aliasing': {
                'prompt': portability_Subject_Aliasing_prompts,
                'ground_truth': portability_Subject_Aliasing_ans
            },
            'reasoning': {
                'prompt': portability_reasoning_prompts,
                'ground_truth': portability_reasoning_ans
            },
            'Logical_Generalization': {
                'prompt': portability_Logical_Generalization_prompts,
                'ground_truth': portability_Logical_Generalization_ans
            }
        }

        return prompts, target_new, subjects, locality_inputs, portability_inputs


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--datatype', default=None, type=str)
    parser.add_argument('--router_save_path', default='./router', type=str)
    parser.add_argument('--router_load_path', default='./router', type=str)

    parser.add_argument('--sequential_edit', default=True, type=str2bool) # 是否使用顺序编辑
     
    args = parser.parse_args()

    hparams = ZZZHyperParams.from_hparams(args.hparams_dir)
    
    # --- 准备数据集 ---
    print('准备数据集...')
    data_handler = DataHandler()

    if args.datatype == 'counterfact':
        print(f'数据类型: {args.datatype}; 数据长度: {args.ds_size}')
        prompts, target_new, subjects, locality_inputs, portability_inputs = data_handler.load_counterfact_zzz(
            data_dir=args.data_dir,
            ds_size=args.ds_size
        )
    else:
        raise NotImplementedError('只实现了counterfact')
    # ----------------

    # --- 训练路由器 ---
    if args.router_load_path:
        try:
            router = KnowRouter.load(args.router_load_path)
            print(f"成功从 {args.router_load_path} 加载路由器")
            print(f"现有聚类数量: {router.get_num_clusters()}")
        except Exception as e:
            print(f"加载路由器失败: {str(e)}")
            print("将重新训练路由器...")
            router = KnowRouter(cfg=hparams)
            router.build_route_table(prompt_list=prompts)
    else:
        router = KnowRouter(cfg=hparams)
        print(hparams.clustering)
        router.build_route_table(prompt_list=prompts)
        if args.router_save_path:
            try:
                router.save(args.router_save_path)
                print(f"路由器已保存到 {args.router_save_path}")
            except Exception as e:
                print(f"保存路由器失败: {str(e)}")
      
    print(f"聚类数量: {router.get_num_clusters()}")
      

    # --- 执行知识编辑 ---

    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        target_new=target_new,
        subject=subjects,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        keep_original_weight=True,
        sequential_edit=args.sequential_edit,
        router=router,
        
        # train_ds=train_ds, # 没甚用处
        # pre_file=args.pre_file, # 没甚用处
        # pre_edit = pre_edit, # 没甚用处
        # test_generation=True, # 测ppl的
    )

    if not os.path.exists(args.metrics_save_dir):
        os.makedirs(args.metrics_save_dir)
    result_path = os.path.join(args.metrics_save_dir, f'{args.editing_method}_{args.datatype}_{hparams.model_name.split("/")[-1]}_results.json')
    json.dump(metrics, open(result_path, 'w'), indent=4)
    print('Done')

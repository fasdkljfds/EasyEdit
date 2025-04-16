# 评估zzz
# 适配multiarea
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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_dataset_configs(config_str, all_files=None):
    """
    支持两种格式：
    1. 明确配置模式(例如: "business_industry:50, human_scientist:50, event_sport:50")
    注意，:两边不能有空格
    2. 总量自动平均模式（例如: "ALL:300"），需传入 all_files 列表
    """
    if all_files is None:
        all_files = [
            "art_sculpture", "business_brand", "business_corporation",
            "business_industry", "entertainment_anime", "entertainment_music_genre",
            "entertainment_song", "event_film", "event_history",
            "event_sport", "geography_forest", "geography_glacier",
            "geography_volcano", "health_disease", "health_medication",
            "health_symptom", "human_athlete", "human_entrepreneur",
            "human_scientist", "human_writer", "places_city",
            "places_country", "places_landmark", "technology_database",
            "technology_programming_language", "technology_software"
        ]
    config_dict = {}
    if not config_str:
        return config_dict

    config_str = config_str.strip()

    if config_str.startswith("ALL:"):
        if not all_files:
            raise ValueError("使用 'ALL:<total>' 模式时，必须提供 all_files 列表！")
        total = int(config_str.split(":")[1])
        num_files = len(all_files)
        base = total // num_files
        remainder = total % num_files

        for i, name in enumerate(all_files):
            k = base + (1 if i < remainder else 0)
            filename = name if name.endswith(".json") else name + ".json"
            config_dict[filename] = k
    else:
        for entry in config_str.split(','):
            if ':' not in entry:
                raise ValueError(f"配置项格式错误: '{entry}'，应该是 'filename[:count]' 或 'ALL:total'")
            filename, k = entry.split(':')
            if '.json' not in filename:
                filename = filename + '.json'
            k = int(k) if k != "None" else None
            config_dict[filename.strip()] = k

    return config_dict


if __name__ == "__main__":
    # --- 解析参数 ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)

    parser.add_argument('--data_dir', required=True, type=str)  # 数据集目录
    parser.add_argument('--data_configs', type=str, required=True)  # 数据集配置
    parser.add_argument('--random_sample', default=False, type=str2bool)  # 默认顺序采样
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    parser.add_argument('--output_dir', default='./outputs', type=str)
    parser.add_argument('--router_save_path', default='./router', type=str)  # 路由器保存路径
    parser.add_argument('--router_load_path', default='./router', type=str)  # 路由器加载路径
    parser.add_argument('--retrain', default=False, type=str2bool)  # 是否重新训练路由器


    parser.add_argument('--sequential_edit', default=True, type=str2bool)  # 是否使用顺序编辑 默认为是

    args = parser.parse_args()

    if args.editing_method == 'ZZZ':
        editing_hparams = ZZZHyperParams
    else:
        raise NotImplementedError

    # --- 准备数据集 ---
    dataset_configs = parse_dataset_configs(args.data_configs)

    multiarea_dataset = MultiAreaDataset(
        root_dir=args.data_dir,
        dataset_configs=dataset_configs,
        seed=42,  # 只有随机采样时有用
        random_sample=args.random_sample
    )

    prompts, rephrase_prompts, target_new, subjects, locality_inputs, _ = multiarea_dataset.get_data()

    # --- 训练路由器 ---
    hparams = editing_hparams.from_hparams(args.hparams_dir)

    if args.router_load_path and not args.retrain:
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
        print("训练路由器...")
        router = KnowRouter(cfg=hparams)
        router.build_route_table(prompt_list=prompts)
        if args.router_save_path:
            try:
                router.save(args.router_save_path)
                print(f"路由器已保存到 {args.router_save_path}")
            except Exception as e:
                print(f"保存路由器失败: {str(e)}")
      
    print(f"聚类数量: {router.get_num_clusters()}")  


    # --- 准备编辑器 ---
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f'{hparams.model_name.split("/")[-1]}_{args.editing_method}_Sequential={args.sequential_edit}.json'
    )

    print("See results at: ", output_file)
    if args.editing_method == 'WISE':
        loc_filepath = 'EasyEdit/data/wise/ZsRE/zsre_mend_train.json'
        loc_data = json.load(
            open(loc_filepath, 'r', encoding='utf-8')
        )[:len(multiarea_dataset)]
        loc_prompts = [edit_data_['loc'] + ' ' + edit_data_['loc_ans'] for edit_data_ in loc_data]
        print('Len of loc_prompts: ', len(loc_prompts))
    else:
        loc_prompts = None
    
    # --- 执行知识编辑 ---
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        subject=subjects,
        locality_inputs=locality_inputs,
        sequential_edit=args.sequential_edit,
        loc_prompts=loc_prompts,  # only for WISE
        router=router
    )

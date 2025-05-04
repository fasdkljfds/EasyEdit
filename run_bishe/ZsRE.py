# 用来启动wise、ft、rome在counterfact上编辑，token_em评估

import argparse
import os.path
import sys
import json
import datetime

sys.path.append(os.getcwd() + '/EasyEdit')
sys.path.append(os.getcwd()+'/EasyEdit/run_bishe')

from easyeditor import (
    WISEHyperParams,
    FTHyperParams,
    ROMEHyperParams,
    MEMITHyperParams,
    summary_metrics,
    GraceHyperParams,
    ZZZHyperParams
)

from easyeditor import BaseEditor
from easyeditor import BaseEditor
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from easyeditor import KnowEditDataset
from easyeditor.models.zzz.router import KnowRouter

from multiarea_dataset import MultiAreaDataset

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

def preprocess_ZsRE(edit_filepath, loc_filepath, N):
    edit_data = json.load(
        open(edit_filepath, 'r', encoding='utf-8')
    )[:N]
    loc_data = json.load(
        open(loc_filepath, 'r', encoding='utf-8')
    )[:N]
    loc_prompts = [edit_data_['loc'] + ' ' + edit_data_['loc_ans'] for edit_data_ in loc_data]

    prompts = [edit_data_['src'] for edit_data_ in edit_data]
    subject = [edit_data_['subject'] for edit_data_ in edit_data]
    rephrase_prompts = [edit_data_['rephrase'] for edit_data_ in edit_data]
    target_new = [edit_data_['alt'] for edit_data_ in edit_data]
    locality_prompts = [edit_data_['loc'] for edit_data_ in edit_data]
    locality_ans = [edit_data_['loc_ans'] for edit_data_ in edit_data]
    locality_inputs = {
        'neighborhood': {
            'prompt': locality_prompts,
            'ground_truth': locality_ans
        },
    }
    return prompts, subject, rephrase_prompts, target_new, locality_inputs, loc_prompts


def preprocess_coutnerfact(edit_filepath, loc_filepath, N):
    edit_data = json.load(
        open(edit_filepath, 'r', encoding='utf-8')
    )[:N]
    # loc_data = json.load(
    #     open(loc_filepath, 'r', encoding='utf-8')
    # )[:N]
    loc_data = json.load(
        open(loc_filepath, 'r', encoding='utf-8')
    )[:N]
    loc_prompts = [edit_data_['loc'] + ' ' + edit_data_['loc_ans'] for edit_data_ in loc_data]

    prompts = [edit_data_['prompt'] for edit_data_ in edit_data]
    subject = [edit_data_['subject'] for edit_data_ in edit_data]
    rephrase_prompts = [edit_data_['rephrase_prompt'] for edit_data_ in edit_data]
    target_new = [edit_data_['target_new'] for edit_data_ in edit_data]
    locality_prompts = [edit_data_['locality_prompt'] for edit_data_ in edit_data]
    locality_ans = [edit_data_['locality_ground_truth'] for edit_data_ in edit_data]

    locality_inputs = {
        'neighborhood': {
            'prompt': locality_prompts,
            'ground_truth': locality_ans
        },
    }
    return prompts, subject, rephrase_prompts, target_new, locality_inputs, loc_prompts


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
    hyperparams_maps = {
        'WISE': WISEHyperParams,
        'FT': FTHyperParams,
        'ROME': ROMEHyperParams,
        'MEMIT': MEMITHyperParams,
        'GRACE': GraceHyperParams,
        'TSR': ZZZHyperParams
    }

    data_processor_maps = {
        'ZsRE': preprocess_ZsRE,
        'counterfact': preprocess_coutnerfact
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--sequential_edit', default=True, type=str2bool)  # 是否使用顺序编辑

    # 编辑用数据集
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--data_type', default=None, type=str)

    parser.add_argument('--evaluation_type', type=str)  # llm 维新派 tranditional 传统派
    parser.add_argument('--api_key', default=None, type=str)

    parser.add_argument('--output_dir', default='./outputs', type=str)

    # --- For TSR ---
    # parser.add_argument('--data_configs', type=str)  # 数据集配置
    parser.add_argument('--random_sample', default=False, type=str2bool)  # 默认顺序采样
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--router_save_path', default='./router', type=str)  # 路由器保存路径
    parser.add_argument('--router_load_path', default='./router', type=str)  # 路由器加载路径
    parser.add_argument('--retrain', default=False, type=str2bool)  # 是否重新训练路由器
    
    parser.add_argument('--sbert_path', default='sentence-transformers/all-MiniLM-L6-v2', type=str)  # 领域路由使用的嵌入模型

    # for update
    parser.add_argument('--two_stages', default=True, type=str2bool)  # 使用使用二阶段
    parser.add_argument('--boundary_model_name', default=None, type=str)  # 语义路由使用的微调后嵌入模型路径
    parser.add_argument('--boundary_threshold', default=None, type=float)   # 语义路由的正阈值

    # for 消融实验
    parser.add_argument('--use_clustering', default=True, type=str2bool)  # 是否使用聚类
    parser.add_argument('--use_multi_ffn', default=True, type=str2bool)  # 是否使用多FFN

    args = parser.parse_args()

    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    hparams = hyperparams_maps[args.editing_method].from_hparams(args.hparams_dir)

    if args.data_type == 'ZsRE' or args.data_type == 'counterfact':
        prompts, subject, rephrase_prompts, target_new, locality_inputs, loc_prompts = data_processor_maps[args.data_type](
            edit_filepath=args.data_dir,
            loc_filepath='EasyEdit/data/wise/ZsRE/zsre_mend_train.json',
            N=args.ds_size
        )

    # elif args.data_type == 'multiarea':
    #     dataset_configs = parse_dataset_configs(args.data_configs)

    #     multiarea_dataset = MultiAreaDataset(
    #         root_dir=args.data_dir,
    #         dataset_configs=dataset_configs,
    #         seed=42,  # 只有随机采样时有用
    #         random_sample=args.random_sample
    #     )

    #     prompts, rephrase_prompts, target_new, subject, locality_inputs, _ = multiarea_dataset.to_edit_dataset()

    #     loc_filepath = 'EasyEdit/data/wise/ZsRE/zsre_mend_train.json'

    #     loc_data = json.load(
    #         open(loc_filepath, 'r', encoding='utf-8')
    #     )[args.ds_size]
    #     loc_prompts = [edit_data_['loc'] + ' ' + edit_data_['loc_ans'] for edit_data_ in loc_data]
    else:
        raise NotImplementedError


    if args.editing_method == 'TSR':
        hparams.two_stages = args.two_stages
        hparams.boundary_model_name = args.boundary_model_name
        hparams.boundary_threshold = args.boundary_threshold
        hparams.use_clustering = args.use_clustering
        hparams.use_multi_ffn = args.use_multi_ffn

        if args.sbert_path != 'sentence-transformers/all-MiniLM-L6-v2':
            hparams.sbert_path = args.sbert_path
            hparams.embedding.model_name = args.sbert_path

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
            print(hparams.clustering)
            router.build_route_table(prompt_list=prompts)
            if args.router_save_path:
                try:
                    router.save(args.router_save_path)
                    print(f"路由器已保存到 {args.router_save_path}")
                except Exception as e:
                    print(f"保存路由器失败: {str(e)}")

        print(f"聚类数量: {router.get_num_clusters()}")

    editor = BaseEditor.from_hparams(hparams)
    if args.editing_method == 'WISE':
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            rephrase_prompts=rephrase_prompts,
            target_new=target_new,
            subject=subject,
            locality_inputs=locality_inputs,
            sequential_edit=args.sequential_edit,

            loc_prompts=loc_prompts,
        )
    elif args.editing_method == 'TSR':
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            rephrase_prompts=rephrase_prompts,
            target_new=target_new,
            subject=subject,
            locality_inputs=locality_inputs,
            sequential_edit=args.sequential_edit,
            router=router
        )
    else:
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            rephrase_prompts=rephrase_prompts,
            target_new=target_new,
            subject=subject,
            locality_inputs=locality_inputs,
            sequential_edit=args.sequential_edit,
        )

    end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('Method: {}'.format(args.editing_method))
    print('Data: {}'.format(args.data_type))
    print('Size: {}'.format(args.ds_size))
    print('Model: {}'.format(hparams.model_name.split("/")[-1]))
    print('Evaluation: {}'.format(args.evaluation_type))
    print('Sequential: {}'.format(args.sequential_edit))
    print('from {} to {}'.format(start_time, end_time))

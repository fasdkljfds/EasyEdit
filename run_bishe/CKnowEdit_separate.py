import os
import os.path
import sys
import json
import random

sys.path.append(os.getcwd() + '/EasyEdit')  # 假设脚本在EasyEdit的父目录运行
sys.path.append('..')  # 假设脚本在EasyEdit的某个子目录运行

from easyeditor import (
    FTHyperParams,
    IKEHyperParams,
    KNHyperParams,
    MEMITHyperParams,
    ROMEHyperParams,
    LoRAHyperParams,
    GraceHyperParams,
    MENDHyperParams,
    SERACHparams,  # 注意：这里可能是 SERACPERparams 或 SERACHparams，根据实际情况调整
    WISEHyperParams,
    ZZZHyperParams,
)
from easyeditor import BaseEditor
# from easyeditor.models.ike import encode_ike_facts # 这行在原始代码中未被使用，暂时注释
# from sentence_transformers import SentenceTransformer # 这行在原始代码中未被使用，暂时注释
from easyeditor import CKnowEditDataset
from easyeditor.models.zzz.router import KnowRouter

import argparse

all_subset = [
    'ancient_poetry_reviewed',
    'classical_chinese_results_reviewed',
    'geography_results_reviewed',
    'history',
    'phonetic_notation_results_reviewed',
    'proverb_results_reviewed'
]


def load_cknowedit(filepath, ds_size):
    datas = CKnowEditDataset(filepath, ds_size)
    prompts = [data['prompt'] for data in datas]
    target_new = [data['target_new'] for data in datas]
    ground_truth = [data['target_old'] for data in datas]  # This is target_old
    subject = [data['subject'] for data in datas]
    rephrase_prompts = [data['rephrase'] for data in datas]
    portability_data = [data['portability'] for data in datas]
    locality_data = [data['locality'] for data in datas]

    portability_prompts = []
    portability_answers = []
    for item in portability_data:
        if item is None or len(item) == 0:
            portability_prompts.append(None)
            portability_answers.append(None)
        else:
            temp_prompts = []
            temp_answers = []
            for pr in item:
                prompt = pr['prompt']
                an = pr['answer']
                temp_prompts.append(prompt)
                temp_answers.append(an)
            portability_prompts.append(temp_prompts)
            portability_answers.append(temp_answers)
    # assert len(prompts) == len(portability_prompts) == len(portability_answers) # This assert might fail if a subset is empty

    locality_prompts = []
    locality_answers = []
    for item in locality_data:
        if item is None or len(item) == 0:
            locality_prompts.append(None)
            locality_answers.append(None)
        else:
            temp_prompts = []
            temp_answers = []
            for pr in item:
                if 'prompt' in pr.keys():  # Ensure 'prompt' key exists
                    prompt = pr["prompt"]
                    an = pr["answer"]
                    temp_prompts.append(prompt)
                    temp_answers.append(an)
            locality_prompts.append(temp_prompts)
            locality_answers.append(temp_answers)
    # assert len(prompts) == len(locality_prompts) == len(locality_answers) # This assert might fail if a subset is empty

    locality_inputs = {
        'loc_hop': {
            'prompt': locality_prompts,
            'ground_truth': locality_answers
        }
    }
    portability_inputs = {
        'por_hop': {
            'prompt': portability_prompts,
            'ground_truth': portability_answers
        }
    }

    return prompts, target_new, ground_truth, subject, rephrase_prompts, locality_inputs, portability_inputs

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str, help="Directory containing cknowedit subset json files")
    parser.add_argument('--ds_size', default=None, type=int, help="Total number of samples to load from cknowedit subsets")
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    # 添加 datatype 和 chinese_ds_type 参数以匹配输出文件名格式
    parser.add_argument('--datatype', default='CKnowEdit', type=str, help="General datatype identifier for filename")
    parser.add_argument('--chinese_ds_type', default='all_subsets_sampled', type=str, help="Specific CKnowEdit subset type for filename")

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

    # 初始化用于合并数据的列表
    shuffled_subsets = all_subset[:]

    for i, subset_name in enumerate(shuffled_subsets):
        subset_filepath = os.path.join(args.data_dir, subset_name + '.json')

        print(f"Loading data from {subset_name} (file: {subset_filepath}), aiming for {args.ds_size} samples...")

        prompts, target_new, ground_truth_old, subject, rephrase_prompts, \
            locality_inputs, portability_inputs = load_cknowedit(subset_filepath, args.ds_size)

        print(f"Loaded {len(prompts)} samples from {subset_name}")

        # 选择超参数类
        if args.editing_method == 'FT':
            editing_hparams = FTHyperParams
        elif args.editing_method == 'ROME':
            editing_hparams = ROMEHyperParams
        elif args.editing_method == 'GRACE':
            editing_hparams = GraceHyperParams
        elif args.editing_method == 'WISE':
            editing_hparams = WISEHyperParams
        elif args.editing_method == 'TSR':
            editing_hparams = ZZZHyperParams
        else:
            raise NotImplementedError

        loc_prompts = None  # 默认为None，仅WISE需要
        if args.editing_method == 'WISE':
            loc_filepath = 'EasyEdit/data/wise/ZsRE/zsre_mend_train.json'

            loc_data = json.load(
                open(loc_filepath, 'r', encoding='utf-8')
            )[:int(len(prompts))]
            loc_prompts = [edit_data_['loc'] + ' ' + edit_data_['loc_ans'] for edit_data_ in loc_data]

        hparams = editing_hparams.from_hparams(args.hparams_dir)
        editor = BaseEditor.from_hparams(hparams)

        print(f"\nStarting editing with method: {args.editing_method}")

        print(f"Number of edits to perform: {len(prompts)}")

        if args.editing_method == 'TSR':
            hparams.two_stages = args.two_stages
            if hparams.two_stages:
                print("使用二阶段路由")
            hparams.boundary_model_name = args.boundary_model_name
            if hparams.boundary_model_name:
                print(f"使用微调后的嵌入模型: {hparams.boundary_model_name}")
            hparams.boundary_threshold = args.boundary_threshold
            if hparams.boundary_threshold:
                print(f"使用正阈值: {hparams.boundary_threshold}")
            hparams.use_clustering = args.use_clustering
            if args.use_clustering:
                print("使用聚类")
            hparams.use_multi_ffn = args.use_multi_ffn
            if args.use_multi_ffn:
                print("使用多FFN")

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

        if args.editing_method == 'WISE':
            metrics, edited_model, _ = editor.edit(
                prompts=prompts,
                target_new=target_new,
                ground_truth=target_new,
                rephrase_prompts=rephrase_prompts,
                portability_inputs=portability_inputs,
                subject=subject,
                keep_original_weight=True,
                sequential_edit=True,  # 假设对于所有方法都使用顺序编辑
                loc_prompts=loc_prompts  # WISE 特有的参数
            )
        elif args.editing_method == 'TSR':
            metrics, edited_model, _ = editor.edit(
                prompts=prompts,
                target_new=target_new,
                ground_truth=target_new,
                rephrase_prompts=rephrase_prompts,
                portability_inputs=portability_inputs,
                subject=subject,
                keep_original_weight=True,
                sequential_edit=True,  # 假设对于所有方法都使用顺序编辑
                router=router
            )

        else:
            metrics, edited_model, _ = editor.edit(
                prompts=prompts,
                target_new=target_new,
                ground_truth=target_new,
                rephrase_prompts=rephrase_prompts,
                portability_inputs=portability_inputs,
                subject=subject,
                keep_original_weight=True,
                sequential_edit=True,
            )

        print('subset_name:', subset_name)
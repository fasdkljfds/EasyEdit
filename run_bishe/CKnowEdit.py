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
)
from easyeditor import BaseEditor
# from easyeditor.models.ike import encode_ike_facts # 这行在原始代码中未被使用，暂时注释
# from sentence_transformers import SentenceTransformer # 这行在原始代码中未被使用，暂时注释
from easyeditor import CKnowEditDataset

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

    args = parser.parse_args()

    # 初始化用于合并数据的列表
    combined_prompts = []
    combined_target_new = []
    combined_ground_truth_old = []
    combined_subject = []
    combined_rephrase_prompts = []

    combined_locality_inputs = {'loc_hop': {'prompt': [], 'ground_truth': []}}
    combined_portability_inputs = {'por_hop': {'prompt': [], 'ground_truth': []}}

    num_subsets = len(all_subset)

    shuffled_subsets = all_subset[:]

    base_samples_per_subset = args.ds_size // num_subsets
    remainder_samples = args.ds_size % num_subsets
    samples_per_subset_list = [base_samples_per_subset] * num_subsets
    for i in range(remainder_samples):
        samples_per_subset_list[i] += 1

    for i, subset_name in enumerate(shuffled_subsets):  # 使用打乱后的列表
        current_subset_size_to_load = samples_per_subset_list[i]

        subset_filepath = os.path.join(args.data_dir, subset_name + '.json')

        num_to_load_str = "all" if current_subset_size_to_load is None else str(current_subset_size_to_load)
        print(f"Loading data from {subset_name} (file: {subset_filepath}), aiming for {num_to_load_str} samples...")

        s_prompts, s_target_new, s_ground_truth_old, s_subject, s_rephrase_prompts, \
            s_locality_inputs, s_portability_inputs = load_cknowedit(subset_filepath, current_subset_size_to_load)

        combined_prompts.extend(s_prompts)
        combined_target_new.extend(s_target_new)
        combined_ground_truth_old.extend(s_ground_truth_old)
        combined_subject.extend(s_subject)
        combined_rephrase_prompts.extend(s_rephrase_prompts)


        # 合并 portability_inputs
        if s_portability_inputs['por_hop']['prompt']:  # 检查是否有数据
            combined_portability_inputs['por_hop']['prompt'].extend(s_portability_inputs['por_hop']['prompt'])
            combined_portability_inputs['por_hop']['ground_truth'].extend(s_portability_inputs['por_hop']['ground_truth'])

        print(f"Loaded {len(s_prompts)} samples from {subset_name}. Total samples so far: {len(combined_prompts)}")

    # 将合并后的数据赋值给原变量名
    prompts = combined_prompts
    target_new = combined_target_new
    # ground_truth_old = combined_ground_truth_old # 这实际上是 target_old, 编辑器调用时 ground_truth=target_new
    subject = combined_subject
    rephrase_prompts = combined_rephrase_prompts
    locality_inputs = combined_locality_inputs
    portability_inputs = combined_portability_inputs


    print(f"\nTotal CKnowEdit samples loaded across all subsets: {len(prompts)}")

    # 选择超参数类
    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'GRACE':
        editing_hparams = GraceHyperParams
    elif args.editing_method == 'WISE':
        editing_hparams = WISEHyperParams
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

# 评估GRACE和MEMIT在counterfact上的表现 4.19

import os
import os.path as path
import json
import random
import sys

sys.path.append(os.getcwd()+'/EasyEdit')

print(sys.path)
try:
    from EasyEdit.easyeditor import (
        MEMITHyperParams,
        GraceHyperParams,
        WISEHyperParams
        )

    from EasyEdit.easyeditor import BaseEditor
    from EasyEdit.easyeditor.models.ike import encode_ike_facts
    from sentence_transformers import SentenceTransformer
    from EasyEdit.easyeditor import KnowEditDataset

except ImportError:
    from easyeditor import (
        MEMITHyperParams,
        GraceHyperParams,
        WISEHyperParams
        )

    from easyeditor import BaseEditor
    from easyeditor.models.ike import encode_ike_facts
    from sentence_transformers import SentenceTransformer
    from easyeditor import KnowEditDataset

import argparse
import numpy as np


def eval(result_path):
    """专门用来适配knowedit的评估函数"""
    if path.exists(result_path):

        with open(result_path, 'r') as file:
            datas = json.load(file)
        # data_rome_counterfact['post'].keys()  dict_keys(['rewrite_acc', 'locality', 'portability'])
        Edit_Succ_list = [data_rome_counterfact['post']['rewrite_acc'][0] for data_rome_counterfact in datas]
        Edit_Succ = sum(Edit_Succ_list) / len(Edit_Succ_list) * 100
        print('Edit_Succ:', Edit_Succ)

        Portability_list = []
        for data_rome_counterfact in datas:
            case_list = []
            for key in data_rome_counterfact['post']['portability'].keys():
                case_list.append(sum(data_rome_counterfact['post']['portability'][key]) / len(data_rome_counterfact['post']['portability'][key]) * 100)
            if len(case_list) != 0:
                Portability_list.append(np.mean(case_list))
        Overall_portability = np.mean(Portability_list)
        print('Overall_portability:', Overall_portability)

        Locality_list = []
        for data_rome_counterfact in datas:
            case_list = []
            for key in data_rome_counterfact['post']['locality'].keys():
                case_list.append(sum(data_rome_counterfact['post']['locality'][key]) / len(data_rome_counterfact['post']['locality'][key]) * 100)
            if len(case_list) != 0:
                Locality_list.append(np.mean(case_list))
        Overall_locality = np.mean(Locality_list)
        print('Overall_locality:', Overall_locality)

        Fluency_list = [x['post']['fluency']['ngram_entropy'] for x in datas]
        Fluency = sum(Fluency_list) / len(Fluency_list) * 100
        print('Fluency:', Fluency)

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
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    parser.add_argument('--datatype', default=None,type=str)

    parser.add_argument('--sequential_edit', default=True, type=str2bool) # 是否使用顺序编辑
    
    args = parser.parse_args()

    if args.editing_method == 'MEMIT':
        editing_hparpzams = MEMITHyperParams
    elif args.editing_method == 'GRACE':
        editing_hparams = GraceHyperParams
    elif args.editing_method == 'WISE':
        editing_hparams = WISEHyperParams
    else:
        raise NotImplementedError('只实现了MEjMIT和GRACE')

    datas = KnowEditDataset(args.data_dir, size=args.ds_size)

    if args.datatype == 'counterfact':
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
    else:
        raise NotImplementedError('只实现了counterfact')

    hparams = editing_hparams.from_hparams(args.hparams_dir)

    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        target_new=target_new,
        subject=subjects,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        keep_original_weight=True,
        sequential_edit=args.sequential_edit,
    )

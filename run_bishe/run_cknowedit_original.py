
# 评估grace在cknowedit上的表

import os
import os.path
import sys
import json
import random
sys.path.append(os.getcwd()+'/EasyEdit')

sys.path.append('..')
from easyeditor import (
    FTHyperParams,
    IKEHyperParams,
    KNHyperParams,
    MEMITHyperParams,
    ROMEHyperParams,
    LoRAHyperParams,
    GraceHyperParams,
    MENDHyperParams,
    SERACHparams,
    WISEHyperParams,
)
from easyeditor import BaseEditor
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
from easyeditor import CKnowEditDataset

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--metrics_save_dir', default='./output', type=str)
    args = parser.parse_args()
    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'KN':
        editing_hparams = KNHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    elif args.editing_method == 'GRACE':
        editing_hparams = GraceHyperParams
    elif args.edtiting_method == 'WISE':
        editing_hparams = WISEHyperParams
    else:
        raise NotImplementedError

    loc_filepath = 'EasyEdit/data/wise/ZsRE/zsre_mend_train.json'
    loc_data = json.load(
        open(loc_filepath, 'r', encoding='utf-8')
    )[args.ds_size]
    loc_prompts = [edit_data_['loc'] + ' ' + edit_data_['loc_ans'] for edit_data_ in loc_data]

    datas = CKnowEditDataset(args.data_dir, size=args.ds_size)
    prompts = [data['prompt'] for data in datas]
    target_new = [data['target_new'] for data in datas]
    ground_truth = [data['target_old'] for data in datas]
    subject = [data['subject'] for data in datas]
    rephrase_prompts = [data['rephrase'] for data in datas]
    portability_data = [data['portability'] for data in datas]
    locality_data = [data['locality'] for data in datas]

    portability_prompts = []
    portability_answers = []
    for item in portability_data:
        if item is None:
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
    assert len(prompts) == len(portability_prompts) == len(portability_answers)

    locality_prompts = []
    locality_answers = []
    for item in locality_data:
        if item is None:
            locality_prompts.append(None)
            locality_answers.append(None)
        else:
            temp_prompts = []
            temp_answers = []
            for pr in item:
                if 'prompt' in pr.keys():
                    prompt = pr["prompt"]
                    an = pr["answer"]
                    temp_prompts.append(prompt)
                    temp_answers.append(an)
            locality_prompts.append(temp_prompts)
            locality_answers.append(temp_answers)
    assert len(prompts) == len(locality_prompts) == len(locality_answers)

    locality_inputs = {}
    portability_inputs = {}
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

    hparams = editing_hparams.from_hparams(args.hparams_dir)
    editor = BaseEditor.from_hparams(hparams)

    if args.editing_method == 'WISE':
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            target_new=target_new,
            ground_truth=target_new,
            rephrase_prompts=rephrase_prompts,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            subject=subject,
            keep_original_weight=True,
            sequential_edit=True,
            loc_prompts=loc_prompts
        )
    else:
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            target_new=target_new,
            ground_truth=target_new,
            rephrase_prompts=rephrase_prompts,
            locality_inputs=locality_inputs,
            portability_inputs=portability_inputs,
            subject=subject,
            keep_original_weight=True,
            sequential_edit=True,
        )


    if not os.path.exists(args.metrics_save_dir):
        os.makedirs(args.metrics_save_dir)
    json.dump(metrics, open(os.path.join(args.metrics_save_dir, f'{args.editing_method}_{args.datatype}_{hparams.model_name.split("/")[-1]}_{args.chinese_ds_type}_results.json'), 'w'), indent=4)

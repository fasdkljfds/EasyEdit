# 用来启动wise、ft、rome在counterfact上编辑，token_em评估

import argparse
import os.path
import sys
import json
import datetime
sys.path.append(os.getcwd() + '/EasyEdit')

from easyeditor import (
    WISEHyperParams,
    FTHyperParams,
    ROMEHyperParams,
    MEMITHyperParams,
    summary_metrics
)

from easyeditor import BaseEditor

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
    loc_prompts = [edit_data_['locality_prompt'] + ' ' + edit_data_['locality_ground_truth'] for edit_data_ in edit_data]
    
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
    }

    data_processor_maps = {
        'ZsRE': preprocess_ZsRE,
        'counterfact': preprocess_coutnerfact
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)
    parser.add_argument('--sequential_edit', default=True, type=str2bool) # 是否使用顺序编辑

    # 编辑用数据集
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--data_type', default=None, type=str)

    parser.add_argument('--evaluation_type', type=str) # llm 维新派 tranditional 传统派
    parser.add_argument('--api_key', default=None, type=str)

    parser.add_argument('--output_dir', default='./outputs', type=str)
    args = parser.parse_args()

    start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    hparams = hyperparams_maps[args.editing_method].from_hparams(args.hparams_dir)

    prompts, subject, rephrase_prompts, target_new, locality_inputs, loc_prompts = data_processor_maps[args.data_type](
        edit_filepath=args.data_dir,
        loc_filepath='EasyEdit/data/wise/ZsRE/zsre_mend_train.json',
        N=args.ds_size
    )

    if args.evaluation_type == 'traditional':
        editor = BaseEditor.from_hparams(hparams)
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            rephrase_prompts=rephrase_prompts,
            target_new=target_new,
            loc_prompts=loc_prompts,
            subject=subject,
            locality_inputs=locality_inputs,
            sequential_edit=args.sequential_edit,
        )
    elif args.evaluation_type == 'llm':
        hparams.evaluation_type = 'LLM-judge'
        hparams.api_key = args.api_key 
        editor = BaseEditor.from_hparams(hparams)
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            rephrase_prompts=rephrase_prompts,
            target_new=target_new,
            loc_prompts=loc_prompts,
            subject=subject,
            locality_inputs=locality_inputs,
            sequential_edist=args.sequential_edit,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f'{hparams.model_name.split("/")[-1]}_{args.editing_method}_N={args.ds_size}_Sequential={args.sequential_edit}.json'
    )

    print("See results at: ", output_file)

    with open(output_file, 'w') as f: json.dump(metrics, f, indent=4)


    if len(metrics) > 0:
        summary_metrics(metrics)

    end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('Method: {}'.format(args.editing_method))
    print('Data: {}'.format(args.data_type))
    print('Size: {}'.format(args.ds_size))
    print('Model: {}'.format(hparams.model_name.split("/")[-1]))
    print('Evaluation: {}'.format(args.evaluation_type))
    print('Sequential: {}'.format(args.sequential_edit))
    print('from {} to {}'.format(start_time, end_time))

    if len(metrics) > 0:
        summary_metrics(metrics)
    
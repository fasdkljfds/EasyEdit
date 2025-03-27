import argparse
import os.path
import sys
import json


from EasyEdit.easyeditor import (
    WISEHyperParams,
    summary_metrics
)

from EasyEdit.easyeditor import BaseEditor

sys.path.append(os.getcwd() + '/EasyEdit')


def preprocess_ZsRE(edit_filepath, loc_filepath, N):
    edit_data = json.load(
        open(edit_filepath, 'r', encoding='utf-8')
    )[:N]
    loc_data = json.load(
        open(loc_filepath, 'r', encoding='utf-8')
    )
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


if __name__ == '__main__':
    hyperparams_maps = {
        'WISE': WISEHyperParams
    }

    data_processor_maps = {
        'ZsRE': preprocess_ZsRE
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--editing_method', required=True, type=str)
    parser.add_argument('--hparams_dir', required=True, type=str)

    # 编辑用数据集
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--output_dir', default='./outputs', type=str)
    parser.add_argument('--ds_size', default=None, type=int)
    parser.add_argument('--datatype', default=None, type=str)
    parser.add_argument('--sequential_edit', default=True, type=bool)
    parser.add_argument('--evaluation_type', type=str)
    parser.add_argument('--api_key', default=None, type=str)
    args = parser.parse_args()

    hparams = hyperparams_maps[args.editing_method].from_hparams(args.hparams_dir)

    prompts, subject, rephrase_prompts, target_new, locality_inputs, loc_prompts = data_processor_maps[args.data_type](
        edit_filepath=args.data_dir,
        loc_filepath='EasyEdit/data/wise/ZsRE/zsre_mend_train.json',
        N=args.ds_size
    )

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

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f'{hparams.model_name.split("/")[-1]}_{args.editing_method}_N={args.ds_size}_Sequential={args.sequential_edit}.json'
    )

    print("See results at: ", output_file)
    
    with open(output_file, 'w') as f: json.dump(metrics, f, indent=4)

    if len(metrics) > 0:
        summary_metrics(metrics)

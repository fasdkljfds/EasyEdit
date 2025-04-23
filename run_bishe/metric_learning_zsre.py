# 实现对counterfact和multi-area的度量学习

import os
import sys
from typing import List, Dict, Optional, Any, Tuple

from sentence_transformers.evaluation import SentenceEvaluator
from sklearn.model_selection import train_test_split
from multiarea_dataset import MultiAreaDataset
# 导入 sentence-transformers 核心库
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments  # 用于更精细的训练配置 (可选)
import math
import json
import datasets
from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLossDistanceFunction  # TripletLoss 相关
# 1
sys.path.append(os.getcwd() + '/EasyEdit')
sys.path.append(os.getcwd() + '/EasyEdit/run_bishe')

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

def prepare_triplet_data(root_dir: str,
                         dataset_configs: Dict[str, int],
                         validation_split_ratio: float = 0.1,
                         seed: int = 42,
                         random_sample: bool = False) -> tuple[datasets.Dataset, datasets.Dataset]: # <--- 修改返回类型
    """
    准备用于三元组损失训练的数据集，使用 Hugging Face datasets 格式。

    Args:
        root_dir (str): MultiAreaDataset 的根目录。
        dataset_configs (Dict[str, int]): 数据集配置文件和样本数量。
        validation_split_ratio (float): 验证集分割比例。
        seed (int, optional): 随机种子. Defaults to 42.
        random_sample (bool, optional): 是否随机采样. Defaults to False.

    Returns:
        tuple[datasets.Dataset, datasets.Dataset]: 训练集和验证集 (Hugging Face Dataset 对象)。
    """
    print(f"开始准备 MultiArea 数据集，根目录: {root_dir}")

    multiarea_dataset = MultiAreaDataset(
        root_dir=root_dir,
        dataset_configs=dataset_configs,
        seed=seed,
        random_sample=random_sample
    )
    print("MultiAreaDataset 实例创建成功。")

    prompts, rephrase_prompts, _, _, locality_inputs, _ = multiarea_dataset.to_edit_dataset()
    locality_prompts = locality_inputs['neighborhood']['prompt']
    print(f"成功提取到 {len(prompts)} 个编辑问题 (anchors)。")
    print(f"成功提取到 {len(rephrase_prompts)} 个等价改写问题 (positives)。")
    print(f"成功提取到 {len(locality_prompts)} 个局部性问题 (negatives)。")

    min_len = min(len(prompts), len(rephrase_prompts), len(locality_prompts))
    if not (len(prompts) == len(rephrase_prompts) == len(locality_prompts)):
        print(f"警告：提取的 Anchors ({len(prompts)}), Positives ({len(rephrase_prompts)}), Negatives ({len(locality_prompts)}) 数量不一致！")
        print(f"将使用前 {min_len} 条数据构建三元组。")
        prompts = prompts[:min_len]
        rephrase_prompts = rephrase_prompts[:min_len]
        locality_prompts = locality_prompts[:min_len]

    if min_len == 0:
        print("错误：没有有效的三元组数据可以用于创建数据集！")
        # 返回空的 Dataset 对象，或者抛出错误，取决于你希望如何处理
        empty_data = {'anchor': [], 'positive': [], 'negative': []}
        return datasets.Dataset.from_dict(empty_data), datasets.Dataset.from_dict(empty_data)

    train_anchors, val_anchors, \
        train_positives, val_positives, \
        train_negatives, val_negatives = train_test_split(
        prompts,
        rephrase_prompts,
        locality_prompts,
        test_size=validation_split_ratio,
        random_state=seed,
        shuffle=True
    )
    print(f"数据已分割：训练集 {len(train_anchors)} 条，验证集 {len(val_anchors)} 条。")

    # --- 使用 datasets.Dataset.from_dict 创建数据集 ---
    train_data_dict = {
        'anchor': train_anchors,
        'positive': train_positives,
        'negative': train_negatives
    }
    val_data_dict = {
        'anchor': val_anchors,
        'positive': val_positives,
        'negative': val_negatives
    }

    # features 参数是可选的，但有助于明确数据类型
    features = datasets.Features({
        'anchor': datasets.Value('string'),
        'positive': datasets.Value('string'),
        'negative': datasets.Value('string')
    })

    train_dataset = datasets.Dataset.from_dict(train_data_dict, features=features)
    val_dataset = datasets.Dataset.from_dict(val_data_dict, features=features)
    # --- 创建完成 ---

    if train_dataset:  # 确保数据集非空
        train_dataset.info.dataset_name = "multi_area_triplet_train"
        train_dataset.info.description = "Training dataset for multi-area triplet loss fine-tuning."  # 可选，添加描述

        # 为验证集添加名称 (如果存在且非空)
    if val_dataset:
        val_dataset.info.dataset_name = "multi_area_triplet_validation"
        val_dataset.info.description = "Validation dataset for multi-area triplet loss fine-tuning."  # 可选

    print("Hugging Face 训练数据集创建成功:")
    print(train_dataset)
    print("Hugging Face 验证数据集创建成功:")
    print(val_dataset)
    return train_dataset, val_dataset


def prepare_triplet_data_zsre(
        data_dir,
        ds_size
):
    K = ds_size
    edit_data = json.load(open(f'{data_dir}/ZsRE/zsre_mend_edit.json', 'r', encoding='utf-8'))[:K]
    loc_data = json.load(open(f'{data_dir}/ZsRE/zsre_mend_train.json', 'r', encoding='utf-8'))[:K]
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

    return prompts, rephrase_prompts, loc_prompts


# --- 主流程函数 ---
def finetune_sentence_transformer(
        # 数据相关参数
        data_root_dir: str,
        data_configs: Dict[str, int],

        # 模型相关参数
        base_model_name: str,
        output_model_dir: str,
        final_model_subdir: str,  # 最终模型保存在 output_model_dir 下的子目录名

        # 损失函数相关参数
        distance_metric_name: str,  # "COSINE" 或 "EUCLIDEAN" 或 "MANHATTAN"
        triplet_margin: float,

        # 训练超参数
        num_train_epochs: int,
        train_batch_size: int,
        learning_rate: float,
        warmup_ratio: float,  # 使用比例计算 warmup steps
        weight_decay: float,

        # 其他训练参数
        logging_steps: int,
        save_strategy: str,
        evaluation_strategy: str,
        eval_steps: int,
) -> Optional[str]:
    """
    执行 Sentence Transformer 模型微调的完整流程，用于知识编辑任务中的度量学习。
    """
    print("=" * 30)
    print("开始 Sentence Transformer 微调流程")
    print("=" * 30)

    # --- 1. 准备数据、加载模型--
    train_dataset, eval_dataset = prepare_triplet_data(root_dir=data_root_dir, dataset_configs=data_configs)

    model = SentenceTransformer(base_model_name)
    print(f'基础模型 {base_model_name} 加载成昆')

    # --- 2. 定义损失函数 ---
    if distance_metric_name.upper() == "COSINE":
        distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE
    elif distance_metric_name.upper() == "EUCLIDEAN":
        distance_metric = losses.SiameseDistanceMetric.EUCLIDEAN
    else:
        print(f"警告：未知的距离度量 '{distance_metric_name}'，将使用默认的 COSINE。")
        distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE

    loss_func = losses.TripletLoss(model=model, distance_metric=distance_metric, triplet_margin=triplet_margin)

    # --- 3. 配置训练参数 ---
    # 计算总训练步数和预热步数
    steps_per_epoch = math.ceil(len(train_dataset) / train_batch_size)
    total_steps = steps_per_epoch * num_train_epochs
    warmup_steps = math.ceil(total_steps * warmup_ratio)
    print(f"总训练步数: {total_steps}, 预热步数: {warmup_steps}")

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_model_dir,

        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,

        logging_dir=os.path.join(output_model_dir, 'logs'),  # 日志放在输出目录下
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        evaluation_strategy=evaluation_strategy if eval_dataset else "no",
        eval_steps= eval_steps if evaluation_strategy == "steps" and eval_dataset else None,
        load_best_model_at_end=True if eval_dataset else False,

        report_to="tensorboard",
    )

    # --- 4. 执行训练 ---
    # 注意：TripletLoss 不需要特殊的 evaluator，除非你想在验证集上评估 triplet loss 本身或相关指标

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss_func,
        # eval_dataset=eval_dataset,  # 如果有评估集，可以传入
        # evaluator=evaluator  # 如果有评估器，可以传入
    )

    trainer.train()

    # --- 5. 保存模型 ---
    final_model_path = os.path.join(output_model_dir, final_model_subdir)

    os.makedirs(final_model_path, exist_ok=True)  # 确保目录存在
    model.save(final_model_path)

    print("=" * 30)
    print("微调流程成功结束。")
    print(f"最终模型已保存至: {final_model_path}")
    print("=" * 30)
    return final_model_path

# --- 主程序入口 ---
if __name__ == "__main__":

    # 定义数据集配置
    # dataset_configs = {
    #     'business_industry.json': 50,
    #     'human_scientist.json': 50,
    #     'event_sport.json': 50,
    #     'geography_forest.json': 50,
    #     'places_landmark.json': 50
    # }

    ALL_FILES = [
        "art_sculpture.json", "business_brand.json", "business_corporation.json",
        "business_industry.json", "entertainment_anime.json", "entertainment_music_genre.json",
        "entertainment_song.json", "event_film.json", "event_history.json",
        "event_sport.json", "geography_forest.json", "geography_glacier.json",
        "geography_volcano.json", "health_disease.json", "health_medication.json",
        "health_symptom.json", "human_athlete.json", "human_entrepreneur.json",
        "human_scientist.json", "human_writer.json", "places_city.json",
        "places_country.json", "places_landmark.json", "technology_database.json",
        "technology_programming_language.json", "technology_software.json"
    ]

    dataset_configs = {file: 1000 for file in ALL_FILES}  # 使用所有文件

    multi_area_root_dir = 'EasyEdit/data/output_meta_llama_3_8b_instruct'

    train_hparams = {
        # 模型相关参数
        "base_model_name": 'sentence-transformers/all-MiniLM-L6-v2',
        "output_model_dir": './finetuned_sbert_triplet',
        "final_model_subdir": 'final_model_1',
        # 损失函数相关参数
        "distance_metric_name": "COSINE",
        "triplet_margin": 0.3,
        # 训练参数
        "num_train_epochs": 3,
        "train_batch_size": 8,
        "learning_rate": 1e-5,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,

        # 其他训练参数
        "logging_steps": 50,
        "save_strategy": "epoch",
        "evaluation_strategy": "epoch",
        "eval_steps": 100,  # 走个形式
    }
    # 调用主流程函数
    final_model_path = finetune_sentence_transformer(
        data_root_dir=multi_area_root_dir,
        data_configs=dataset_configs,
        **train_hparams
    )

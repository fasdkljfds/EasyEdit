# 实现BatchHardSoftMarginTripletLoss的度量学习 5.3

import datasets

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
from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLossDistanceFunction
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



def prepare_data_for_batch_hard_soft_margin(root_dir: str,
                                            dataset_configs: Dict[str, int],
                                            validation_split_ratio: float = 0.1,
                                            seed: int = 42,
                                            random_sample: bool = False) -> tuple[datasets.Dataset, datasets.Dataset]:
    print(f"开始准备 MultiArea 数据集 (用于 BatchHardSoftMarginTripletLoss)，根目录: {root_dir}")

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
        print(f"将使用前 {min_len} 条数据。")
        prompts = prompts[:min_len]
        rephrase_prompts = rephrase_prompts[:min_len]
        locality_prompts = locality_prompts[:min_len]

    if min_len == 0:
        print("错误：没有有效的数据可以用于创建数据集！")
        empty_data = {'sentence': [], 'label': []}
        return datasets.Dataset.from_dict(empty_data), datasets.Dataset.from_dict(empty_data)

    all_sentences = []
    all_labels = []
    current_positive_label = 0
    # 为了确保 locality_prompts 的标签与 (prompt, rephrase_prompt) 对的标签不同，
    # 我们可以给 locality_prompts 的标签一个较大的偏移量。
    # 假设我们最多有 N 个 (prompt, rephrase_prompt) 对，那么 locality_prompts 的标签可以从 N 开始。
    # 更安全的方法是，为每个 locality_prompt 分配一个与任何正例标签都不同的唯一标签。

    # 收集所有 (prompt, rephrase_prompt) 对
    positive_pairs_sentences = []
    positive_pairs_labels = []
    for i in range(min_len):
        positive_pairs_sentences.append(prompts[i])
        positive_pairs_labels.append(current_positive_label)
        positive_pairs_sentences.append(rephrase_prompts[i])
        positive_pairs_labels.append(current_positive_label)
        current_positive_label += 1

    # 收集 locality_prompts
    # 为每个 locality_prompt 分配一个独特的标签，且该标签与之前的 positive_label 不同
    # 例如，可以从 current_positive_label (即原始概念的数量) 开始编号
    negative_sentences = []
    negative_labels = []
    current_negative_label_offset = current_positive_label # 确保负例标签不与正例标签冲突
    for i in range(min_len):
        negative_sentences.append(locality_prompts[i])
        # 每个 locality_prompt 形成自己的类，或者你可以有更复杂的策略
        # 这里简单地给每个 locality_prompt 一个新的、唯一的标签
        negative_labels.append(current_negative_label_offset + i)

    # 方案2: 分别分割正例对和负例，然后合并 (更推荐，确保正负例比例，并尝试保持原始对的完整性)
    # 首先分割正例对的索引
    indices = list(range(min_len)) # 每个索引代表一个 (prompt, rephrase, locality) 原始组
    train_indices, val_indices = train_test_split(indices, test_size=validation_split_ratio, random_state=seed, shuffle=True)

    train_sentences_list = []
    train_labels_list = []
    val_sentences_list = []
    val_labels_list = []

    # 为训练集构建
    for idx in train_indices:
        train_sentences_list.append(prompts[idx])
        train_labels_list.append(idx) # 使用原始索引作为类别标签
        train_sentences_list.append(rephrase_prompts[idx])
        train_labels_list.append(idx)
        train_sentences_list.append(locality_prompts[idx])
        # 负例标签需要不同于正例标签。可以给它一个大的偏移或唯一的新标签。
        # 为简单起见，让每个 locality_prompt 成为一个独立的类，其标签不同于任何正例。
        train_labels_list.append(min_len + idx) # 确保与0到min_len-1的标签不同

    # 为验证集构建
    for idx in val_indices:
        val_sentences_list.append(prompts[idx])
        val_labels_list.append(idx)
        val_sentences_list.append(rephrase_prompts[idx])
        val_labels_list.append(idx)
        val_sentences_list.append(locality_prompts[idx])
        val_labels_list.append(min_len + idx)


    print(f"数据已分割：训练集 {len(train_sentences_list)} 条，验证集 {len(val_sentences_list)} 条。")

    train_data_dict = {'sentence': train_sentences_list, 'label': train_labels_list}
    val_data_dict = {'sentence': val_sentences_list, 'label': val_labels_list}

    features = datasets.Features({
        'sentence': datasets.Value('string'),
        'label': datasets.Value('int64') # 标签通常是整数
    })

    train_dataset = datasets.Dataset.from_dict(train_data_dict, features=features)
    val_dataset = datasets.Dataset.from_dict(val_data_dict, features=features)

    if train_dataset:
        train_dataset.info.dataset_name = "multi_area_batchhard_train"
    if val_dataset:
        val_dataset.info.dataset_name = "multi_area_batchhard_validation"

    print("Hugging Face 训练数据集创建成功:")
    print(train_dataset)
    print("Hugging Face 验证数据集创建成功:")
    print(val_dataset)
    return train_dataset, val_dataset


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
    train_dataset, eval_dataset = prepare_data_for_batch_hard_soft_margin(root_dir=data_root_dir, dataset_configs=data_configs, validation_split_ratio=0.1)
    
    model = SentenceTransformer(base_model_name)
    print(f'基础模型 {base_model_name} 加载成昆')
    from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLossDistanceFunction

    # --- 2. 定义损失函数 ---
    if distance_metric_name.upper() == "COSINE":
        distance_metric = BatchHardTripletLossDistanceFunction.cosine_distance # <--- 新的，正确的
    elif distance_metric_name.upper() == "EUCLIDEAN":
        distance_metric = BatchHardTripletLossDistanceFunction.eucledian_distance # <--- 新的，正确的
    elif distance_metric_name.upper() == "MANHATTAN":
        distance_metric = BatchHardTripletLossDistanceFunction.manhattan_distance # <--- 新的，正确的 (如果存在，否则需要自定义或检查可用性)

    loss_func = losses.BatchHardSoftMarginTripletLoss(model=model, distance_metric=distance_metric)

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
        "triplet_margin": 0.9,
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
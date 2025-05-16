import os
import shutil
import math
import json
import random
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import datasets
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.readers import InputExample

# 确保 multiarea_dataset.py 在同一目录或 Python 路径中
try:
    from multiarea_dataset import MultiAreaDataset
except ImportError:
    print("错误: multiarea_dataset.py 未找到。请确保它与此脚本在同一目录中。")
    exit()


# === Helper function for preparing triplet data ===
def prepare_triplet_data_for_experiment(
        root_dir: str,
        dataset_configs: Dict[str, int],
        total_triplets_for_this_run: Optional[int] = None,
        validation_split_ratio: float = 0.1,
        seed: int = 42
) -> tuple[datasets.Dataset, datasets.Dataset]:
    print(f"开始准备 MultiArea 数据集，根目录: {root_dir}")
    # print(f"原始数据集配置: {dataset_configs}") # 打印信息可能过多

    multiarea_dataset = MultiAreaDataset(
        root_dir=root_dir,
        dataset_configs=dataset_configs,
        seed=seed,
        random_sample=True
    )
    # print("MultiAreaDataset 实例创建成功。")

    prompts, rephrase_prompts, _, _, locality_inputs, _ = multiarea_dataset.to_edit_dataset()
    # x_loc (negative) comes from locality_inputs['neighborhood']['prompt'] which are paired with x_e (prompts)
    locality_prompts_paired_with_prompts = locality_inputs['neighborhood']['prompt']

    # print(f"从 MultiAreaDataset 提取到 {len(prompts)} 个编辑问题 (anchors)。")
    # print(f"从 MultiAreaDataset 提取到 {len(rephrase_prompts)} 个等价改写问题 (positives)。")
    # print(f"从 MultiAreaDataset 提取到 {len(locality_prompts_paired_with_prompts)} 个局部性问题 (negatives)。")

    # Ensure all lists have the same length by design of MultiAreaDataset's to_edit_dataset
    # but double check for safety.
    min_len = min(len(prompts), len(rephrase_prompts), len(locality_prompts_paired_with_prompts))
    if not (len(prompts) == len(rephrase_prompts) == len(locality_prompts_paired_with_prompts)):
        print(f"警告：提取的 Anchors ({len(prompts)}), Positives ({len(rephrase_prompts)}), "
              f"Negatives ({len(locality_prompts_paired_with_prompts)}) 数量不一致！")
        print(f"将使用前 {min_len} 条数据构建三元组。")
        prompts = prompts[:min_len]
        rephrase_prompts = rephrase_prompts[:min_len]
        locality_prompts_paired_with_prompts = locality_prompts_paired_with_prompts[:min_len]

    if min_len == 0:
        print("错误：没有有效的三元组数据可以用于创建数据集！")
        empty_data = {'anchor': [], 'positive': [], 'negative': []}
        return datasets.Dataset.from_dict(empty_data), datasets.Dataset.from_dict(empty_data)

    all_triplets_data = []
    for i in range(min_len):
        if prompts[i] and rephrase_prompts[i] and locality_prompts_paired_with_prompts[i]:
            all_triplets_data.append({
                'anchor': prompts[i],  # x_e
                'positive': rephrase_prompts[i],  # x_gen
                'negative': locality_prompts_paired_with_prompts[i]  # x_loc
            })
        # else:
        # print(f"警告: 跳过索引 {i} 处的三元组，因其包含空字符串。")

    if not all_triplets_data:
        print("错误：过滤空字符串后，没有有效的三元组数据！")
        empty_data = {'anchor': [], 'positive': [], 'negative': []}
        return datasets.Dataset.from_dict(empty_data), datasets.Dataset.from_dict(empty_data)

    # print(f"过滤空字符串后，总共有 {len(all_triplets_data)} 个有效三元组。")

    if total_triplets_for_this_run is not None and total_triplets_for_this_run < len(all_triplets_data):
        # print(f"需要 {total_triplets_for_this_run} 条三元组，从 {len(all_triplets_data)} 条中随机抽样。")
        random.seed(seed)
        all_triplets_data = random.sample(all_triplets_data, total_triplets_for_this_run)
        # print(f"抽样后得到 {len(all_triplets_data)} 条三元组。")

    if validation_split_ratio > 0 and len(all_triplets_data) > 1:  # Need at least 2 samples to split
        train_data_list, val_data_list = train_test_split(
            all_triplets_data,
            test_size=validation_split_ratio,
            random_state=seed,
            shuffle=True
        )
    else:
        train_data_list = all_triplets_data
        val_data_list = []

    # print(f"数据已分割：训练集 {len(train_data_list)} 条，验证集 {len(val_data_list)} 条。")

    features = datasets.Features({
        'anchor': datasets.Value('string'),
        'positive': datasets.Value('string'),
        'negative': datasets.Value('string')
    })

    train_dataset = datasets.Dataset.from_list(train_data_list, features=features) if train_data_list else datasets.Dataset.from_dict({'anchor': [], 'positive': [], 'negative': []}, features=features)
    val_dataset = datasets.Dataset.from_list(val_data_list, features=features) if val_data_list else datasets.Dataset.from_dict({'anchor': [], 'positive': [], 'negative': []}, features=features)

    if train_dataset: train_dataset.info.dataset_name = "multi_area_triplet_train"
    if val_dataset: val_dataset.info.dataset_name = "multi_area_triplet_validation"

    # print("Hugging Face 训练数据集创建成功:")
    # print(train_dataset)
    # print("Hugging Face 验证数据集创建成功:")
    # print(val_dataset)
    return train_dataset, val_dataset


# === Main finetuning function ===
def finetune_sentence_transformer(
        train_dataset: datasets.Dataset,
        eval_dataset: Optional[datasets.Dataset],
        base_model_name: str,
        output_model_dir: str,
        final_model_subdir: str,
        distance_metric_name: str,
        triplet_margin: float,
        num_train_epochs: int,
        train_batch_size: int,
        learning_rate: float,
        warmup_ratio: float,
        weight_decay: float,
        logging_steps: int,
        save_strategy: str,
        evaluation_strategy: str,
        eval_steps: int,
        use_auth_token: Optional[str] = None
) -> Optional[str]:
    print("=" * 30)
    print(f"开始SBERT微调: {output_model_dir}")
    print("=" * 30)

    if os.path.exists(output_model_dir):
        print(f"警告: 输出目录 {output_model_dir} 已存在。将删除并重建。")
        shutil.rmtree(output_model_dir)
    os.makedirs(output_model_dir, exist_ok=True)

    model = SentenceTransformer(base_model_name, use_auth_token=use_auth_token)
    print(f'基础模型 {base_model_name} 加载成功。')

    if distance_metric_name.upper() == "COSINE":
        distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE
    elif distance_metric_name.upper() == "EUCLIDEAN":
        distance_metric = losses.SiameseDistanceMetric.EUCLIDEAN
    else:
        distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE

    loss_func = losses.TripletLoss(model=model, distance_metric=distance_metric, triplet_margin=triplet_margin)

    steps_per_epoch = math.ceil(len(train_dataset) / train_batch_size) if len(train_dataset) > 0 else 0
    total_steps = steps_per_epoch * num_train_epochs
    warmup_steps = math.ceil(total_steps * warmup_ratio)
    # print(f"总训练步数: {total_steps}, 预热步数: {warmup_steps}")

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_model_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir=os.path.join(output_model_dir, 'logs'),
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        evaluation_strategy=evaluation_strategy if eval_dataset and len(eval_dataset) > 0 else "no",
        eval_steps=eval_steps if evaluation_strategy == "steps" and eval_dataset and len(eval_dataset) > 0 else None,
        load_best_model_at_end=True if eval_dataset and len(eval_dataset) > 0 else False,
        report_to="tensorboard",
        save_total_limit=1  # Only keep the best model
    )

    evaluator = None
    if eval_dataset and len(eval_dataset) > 0:
        eval_examples = [InputExample(texts=[item['anchor'], item['positive'], item['negative']]) for item in eval_dataset]
        if eval_examples:
            evaluator = TripletEvaluator.from_input_examples(eval_examples, name='multi-area-val', show_progress_bar=False)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if evaluator else None,
        loss=loss_func,
        evaluator=evaluator
    )

    if len(train_dataset) == 0:
        print("错误：训练数据为空，跳过训练。")
        final_model_path = os.path.join(output_model_dir, final_model_subdir)
        os.makedirs(final_model_path, exist_ok=True)
        print(f"空模型目录已创建: {final_model_path}")
        return final_model_path

    trainer.train()

    final_model_path = os.path.join(output_model_dir, final_model_subdir)
    os.makedirs(final_model_path, exist_ok=True)
    model.save(final_model_path)

    print(f"微调流程成功结束: {output_model_dir}")
    print(f"最终模型已保存至: {final_model_path}")
    return final_model_path


# === Experiment Configuration ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# !! 用户需要根据实际情况修改此路径 !!
MULTI_AREA_ROOT_DIR = 'EasyEdit/data/output_meta_llama_3_8b_instruct'
BASE_MODEL_NAME = 'BAAI/bge-m3'
OUTPUT_BASE_DIR = './exp_sbert_finetuning_output'  # 主输出目录

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
# 快速测试时可以减少文件和每个文件的样本数
# ALL_FILES = ["art_sculpture.json", "business_brand.json", "human_scientist.json"]
# MAX_SAMPLES_PER_FILE_FULL = 50 # For faster testing

initial_dataset_configs = {file: 1000 for file in ALL_FILES}

print("步骤1: 准备全局测试集和完整训练池...")
full_training_pool, global_test_dataset = prepare_triplet_data_for_experiment(
    root_dir=MULTI_AREA_ROOT_DIR,
    dataset_configs=initial_dataset_configs,
    total_triplets_for_this_run=None,
    validation_split_ratio=0.1,  # 10% for global test set
    seed=SEED
)

if not global_test_dataset or len(global_test_dataset) == 0:
    print("错误：全局测试集为空！实验无法继续。请检查数据源和路径。")
    exit()
if not full_training_pool or len(full_training_pool) == 0:
    print("错误：完整训练池为空！实验无法继续。请检查数据源和路径。")
    exit()

print(f"全局测试集准备完毕，包含 {len(global_test_dataset)} 条样本。")
print(f"完整训练池准备完毕，包含 {len(full_training_pool)} 条样本。")

N_full_actual = len(full_training_pool)
print(f"基于实际加载和过滤，N_full_actual (训练池大小) = {N_full_actual}")

# 目标数据量，如果实际数据不足，则使用实际数据量
TARGET_SMALL_DATA = 100
TARGET_MEDIUM_DATA = 500
TARGET_LARGE_DATA = 1000
TARGET_FULL_DATA = 2000

data_size_configs_training = {
    "Small-Data": min(TARGET_SMALL_DATA, N_full_actual),
    "Medium-Data": min(TARGET_MEDIUM_DATA, N_full_actual),
    "Large-Data": min(TARGET_LARGE_DATA, N_full_actual),
    "Full-Data": min(TARGET_FULL_DATA, N_full_actual)
}
# Ensure Small and Medium are not larger than Full after min()
data_size_configs_training["Medium-Data"] = min(data_size_configs_training["Medium-Data"], data_size_configs_training["Full-Data"])
data_size_configs_training["Small-Data"] = min(data_size_configs_training["Small-Data"], data_size_configs_training["Medium-Data"])

# Filter out zero-size configurations to avoid issues
data_size_configs_training = {k: v for k, v in data_size_configs_training.items() if v > 0}

if not data_size_configs_training:
    print("错误：所有目标训练数据量均为0或过小，无法进行训练。")
    exit()

print(f"将使用的训练数据量: {data_size_configs_training}")

COMMON_TRAIN_HPARAMS = {
    "base_model_name": BASE_MODEL_NAME,
    "final_model_subdir": 'final_model',
    "distance_metric_name": "COSINE",
    "triplet_margin": 0.9,  # Margin for TripletLoss
    "num_train_epochs": 3,
    "train_batch_size": 32,  # Adjusted batch size
    "learning_rate": 1e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "logging_steps": 50,
    "save_strategy": "epoch",
    "evaluation_strategy": "epoch",
    "eval_steps": 50,
}

trained_model_paths = {}

print("\n步骤2: 开始模型训练循环...")
for size_label, num_triplets in data_size_configs_training.items():
    print(f"\n--- 开始为 {size_label} ({num_triplets} 条三元组) 进行训练 ---")

    # Sample from the full_training_pool. Ensure `replace=False`.
    # Need to handle cases where num_triplets might be larger than N_full_actual due to min() logic,
    # though current logic for data_size_configs_training should prevent this.
    actual_num_to_sample = min(num_triplets, N_full_actual)
    if actual_num_to_sample == 0:
        print(f"跳过 {size_label}，因为采样数量为0。")
        trained_model_paths[size_label] = None
        continue

    current_train_indices = np.random.choice(N_full_actual, size=actual_num_to_sample, replace=False)
    current_train_dataset = full_training_pool.select(current_train_indices)

    print(f"为 {size_label} 准备的训练集大小: {len(current_train_dataset)}")
    print(f"将使用全局测试集进行评估，大小: {len(global_test_dataset)}")

    output_dir_for_size = os.path.join(OUTPUT_BASE_DIR, "trained_models", size_label)

    final_model_path = finetune_sentence_transformer(
        train_dataset=current_train_dataset,
        eval_dataset=global_test_dataset,
        output_model_dir=output_dir_for_size,
        **COMMON_TRAIN_HPARAMS
    )
    trained_model_paths[size_label] = final_model_path

print("\n\n=== 训练完成 ===")
print("所有训练好的模型路径:")
for label, path in trained_model_paths.items():
    print(f"{label}: {path}")

experiment_tracker_path = os.path.join(OUTPUT_BASE_DIR, "experiment_summary.json")
global_test_dataset_save_path = os.path.join(OUTPUT_BASE_DIR, "global_test_dataset.hf")

experiment_tracker = {
    "trained_model_paths": trained_model_paths,
    "global_test_dataset_path": global_test_dataset_save_path if global_test_dataset and len(global_test_dataset) > 0 else None,
    "base_model_name": BASE_MODEL_NAME,
    "output_base_dir": OUTPUT_BASE_DIR,  # Main output directory for the experiment
    "seed": SEED
}

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)  # Ensure base output dir exists

if global_test_dataset and len(global_test_dataset) > 0:
    global_test_dataset.save_to_disk(global_test_dataset_save_path)
    print(f"全局测试集已保存到: {global_test_dataset_save_path}")

with open(experiment_tracker_path, "w") as f:
    json.dump(experiment_tracker, f, indent=4)
print(f"实验摘要已保存到: {experiment_tracker_path}")

print("\n训练脚本执行完毕。下一步是运行评估和可视化脚本。")
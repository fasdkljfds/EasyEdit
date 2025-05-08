import os
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, classification_report # 添加 classification_report
import numpy as np
from typing import List, Tuple, Dict
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




# --- 配置区 ---
MODEL_PATH = './finetuned_sbert_triplet/final_model_1' # 替换为你的模型路径
# MultiAreaDataset 配置 (你需要根据你的实际情况修改)
MULTI_AREA_ROOT_DIR = 'EasyEdit/data/output_meta_llama_3_8b_instruct' # 你的数据根目录

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

DATASET_CONFIGS = {file: 20 for file in ALL_FILES}
TAU_POSITIVE = 0.8 # 固定阈值，你可以根据需要调整


# --- 语义边界路由算法实现 (与之前相同) ---
def R_boundary(x_new: str, E_texts: List[str], model: SentenceTransformer, tau_pos: float, E_embeddings: np.ndarray = None) -> bool:
    emb_new = model.encode([x_new], convert_to_tensor=False, show_progress_bar=False)
    decision = False
    if E_embeddings is None:
        E_embeddings = model.encode(E_texts, convert_to_tensor=False, show_progress_bar=False)

    for emb_e in E_embeddings:
        similarity = cosine_similarity(emb_new, emb_e.reshape(1, -1))[0, 0]
        if similarity > tau_pos:
            decision = True
            break
    return decision

# --- 主测试流程 ---
def evaluate_model_with_multiarea(model: SentenceTransformer,
                                  multiarea_root_dir: str,
                                  dataset_configs: Dict[str, int],
                                  tau_pos: float) -> Dict[str, float]:
    """
    使用 MultiAreaDataset 生成的数据评估模型性能。
    E (已编辑问题) 将是 multiarea_dataset.prompts
    测试样本 x_new 将是:
        - multiarea_dataset.rephrase_prompts (预期为 True)
        - multiarea_dataset.locality_prompts (预期为 False)
    """
    print(f"\n开始使用 MultiAreaDataset 进行评估，固定阈值 τ_pos = {tau_pos:.4f}")

    # 1. 加载并准备 MultiAreaDataset 数据
    dataset = MultiAreaDataset(root_dir=multiarea_root_dir, dataset_configs=dataset_configs)
    prompts, rephrase_prompts, _, _, locality_inputs, _ = dataset.to_edit_dataset()
    locality_prompts = locality_inputs['neighborhood']['prompt']

    if not prompts:
        print("错误: MultiAreaDataset 未能提供 prompts (E)。无法评估。")
        return {"accuracy": 0.0, "report": "No prompts from MultiAreaDataset."}

    edited_questions_E = prompts
    print(f"使用 {len(edited_questions_E)} 条 prompts 作为已编辑问题 (E)。")

    # 2. 构建测试样本 (x_new, true_label)
    test_samples: List[Tuple[str, bool]] = []
    # 正样本: rephrase_prompts, 期望与对应的 prompt 相似
    for rp in rephrase_prompts:
        test_samples.append((rp, True)) # 标签 True，期望落入边界

    # 负样本: locality_prompts, 期望不与 E 中的任何 prompt 高度相似
    # 注意：这里的逻辑是，locality_prompt 不应该与 *任何* E 中的 prompt 相似度超过 tau_pos。
    # 这与 triplet loss 的目标（locality_prompt 与其 *特定* anchor 的距离远于 positive）略有不同，但符合 R_boundary 的定义。
    for lp in locality_prompts:
        test_samples.append((lp, False)) # 标签 False，期望不落入边界

    if not test_samples:
        print("错误:未能从 MultiAreaDataset 构建测试样本。")
        return {"accuracy": 0.0, "report": "No test samples constructed."}
    print(f"构建了 {len(test_samples)} 条测试样本 ({len(rephrase_prompts)} 正, {len(locality_prompts)} 负)。")


    y_true = []
    y_pred = []

    # 预计算 E 的嵌入以提高效率
    print("正在预计算已编辑问题 (E) 的嵌入...")
    E_embeddings = model.encode(edited_questions_E, convert_to_tensor=False, show_progress_bar=True)
    if E_embeddings is None or E_embeddings.size == 0:
        print("错误: 无法计算 E 的嵌入。")
        return {"accuracy": 0.0, "report": "Failed to compute E embeddings."}
    print(f"已编辑问题嵌入计算完成，形状: {E_embeddings.shape}")


    print("正在处理测试样本...")
    for i, (x_new, true_label) in enumerate(test_samples):
        if (i + 1) % 50 == 0: # 每50个样本打印一次进度
            print(f"  处理中: {i+1}/{len(test_samples)}")

        predicted_decision = R_boundary(x_new, edited_questions_E, model, tau_pos, E_embeddings=E_embeddings)
        y_true.append(true_label)
        y_pred.append(predicted_decision)

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['Not in Boundary (False)', 'In Boundary (True)'], zero_division=0)

    print(f"评估完成。")
    print(f"  样本总数: {len(test_samples)}")
    print(f"  预测为True的数量: {sum(y_pred)}")
    print(f"  实际为True的数量: {sum(y_true)}")
    print(f"  准确率 (Accuracy): {accuracy:.4f}")
    print("\n详细分类报告:")
    print(report)

    return {"accuracy": accuracy, "report": report}


if __name__ == "__main__":
    # 1. 加载模型
    print(f"正在加载模型从: {MODEL_PATH}...")
    try:
        tuned_model = SentenceTransformer(MODEL_PATH)
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请确保 MODEL_PATH 指向一个有效的 Sentence Transformer 模型目录。")
        exit()

    # 2. 使用 MultiAreaDataset 进行评估
    results = evaluate_model_with_multiarea(
        model=tuned_model,
        multiarea_root_dir=MULTI_AREA_ROOT_DIR,
        dataset_configs=DATASET_CONFIGS,
        tau_pos=TAU_POSITIVE
    )

    print("\n测试完成。")

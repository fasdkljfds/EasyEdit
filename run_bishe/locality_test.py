# 测试4.16的路由策略能否区分相近表述
# 4.21 这个脚本需要手动改参数，
# 4.21 这个脚本实际上变为，给定一组参数，观察KnowRouter的工作效果

import sys
import os

sys.path.append(os.getcwd()+'/EasyEdit')
sys.path.append(os.getcwd()+'/EasyEdit/run_bishe')

import yaml
from dataclasses import dataclass, field # 引入 field 以便为字典设置默认工厂
from typing import List, Optional, Dict, Any # 引入 Any
from omegaconf import DictConfig, OmegaConf # 明确引入 OmegaConf

from multiarea_dataset import MultiAreaDataset
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

import optuna
import numpy as np
from omegaconf import DictConfig, OmegaConf
import copy


# --- 0. 准备数据 ---
dataset_configs = {
    'business_industry.json': 50,
    'human_scientist.json': 50,
    'event_sport.json': 50,
    'geography_forest.json': 50,
    'places_landmark.json': 50
}

multiarea_dataset = MultiAreaDataset(
    root_dir='EasyEdit/data/output_meta_llama_3_8b_instruct',
    dataset_configs=dataset_configs,
    seed=42,  # 只有随机采样时有用
    random_sample=False
)


# --- 0.5 创建路由器 ---
editing_hparams = ZZZHyperParams
hparams = editing_hparams.from_hparams('EasyEdit/hparams/ZZZ/llama3.2-1b.yaml')


config = {
    "use_umap": True,
    "random_seed": 42,
    "umap_params": {
        "n_neighbors": 5,
        "min_dist": 0.1,
        "n_components": 50,
        "metric": "cosine"
    },
    "hdbscan_params": {
        "min_cluster_size": 5,
        "min_samples": 3,
        "metric": "euclidean",
        "cluster_selection_method": "eom",
        "allow_single_cluster": False
    }
}

hparams.clustering = config
hparams.embedding.model_name = './finetuned_sbert_triplet/final_model_1'


print(hparams.clustering)
router = KnowRouter(cfg=hparams)

prompts, rephrase_prompts, target_new, subjects, locality_inputs, _ = multiarea_dataset.to_edit_dataset()
locality_prompts = locality_inputs['neighborhood']['prompt']  # 这个loc数据要单独拿出来

router.build_route_table(prompt_list=prompts)

print("路由表构建完成")


# --- 1. 测试locality_prompts的路由情况 ---
# 给出对应的prompt的路由目标、locality目标和置信度

print("\n--- Locality Prompts Routing Test ---")
correct_locality_routing = 0
total_locality = len(prompts)
for i in range(total_locality):
    original_prompt = prompts[i]
    locality_prompt = locality_prompts[i]

    # 获取原始 prompt 的目标 cluster ID (来自路由表)
    original_cluster_id = router.route_table.get(original_prompt, -99)
    if original_cluster_id == -99:
        print(f"错误：原始 prompt '{original_prompt}' 不在路由表中！")
        continue

    # 预测 locality prompt 的 cluster ID 和置信度
    predicted_locality_cluster_id, locality_confidence = router.route_with_confidence(locality_prompt)

    # 理想情况下，locality prompt 不应路由到 original_cluster_id
    is_correct = (predicted_locality_cluster_id != original_cluster_id)

    if is_correct:
        correct_locality_routing += 1

    print(f"Original Prompt (Idx {i}): '{original_prompt}' -> Target Cluster: {original_cluster_id}")
    print(f"Locality Prompt (Idx {i}): '{locality_prompt}' -> Routed Cluster: {predicted_locality_cluster_id}, Confidence: {locality_confidence:.4f}")
    print(f"  -> Locality Routing Correct? {'Yes' if is_correct else 'No'}")
    print("-" * 20)

locality_accuracy = correct_locality_routing / total_locality if total_locality > 0 else 0


# --- 2. 测试rephrase_prompts的路由情况 ---
#  给出对应的prompt的路由目标、rephrase目标和置信度
print("\n--- Rephrase Prompts Routing Test ---")
correct_rephrase_routing = 0
total_rephrase = len(prompts)
for i in range(total_rephrase):
    original_prompt = prompts[i]
    rephrase_prompt = rephrase_prompts[i]

    # 获取原始 prompt 的目标 cluster ID (来自路由表)
    original_cluster_id = router.route_table.get(original_prompt, -99)
    if original_cluster_id == -99:
        # 前面已经检查过，理论上不会再出现，但以防万一
        continue

    # 预测 rephrase prompt 的 cluster ID 和置信度
    predicted_rephrase_cluster_id, rephrase_confidence = router.route_with_confidence(rephrase_prompt)

    # 理想情况下，rephrase prompt 应路由到与 original_prompt 相同的 cluster ID
    is_correct = (predicted_rephrase_cluster_id == original_cluster_id)

    if is_correct:
        correct_rephrase_routing += 1

    print(f"Original Prompt (Idx {i}): '{original_prompt}' -> Target Cluster: {original_cluster_id}")
    print(f"Rephrase Prompt (Idx {i}): '{rephrase_prompt}' -> Routed Cluster: {predicted_rephrase_cluster_id}, Confidence: {rephrase_confidence:.4f}")
    print(f"  -> Rephrase Routing Correct? {'Yes' if is_correct else 'No'}")
    print("-" * 20)

rephrase_accuracy = correct_rephrase_routing / total_rephrase if total_rephrase > 0 else 0

print(f"\nLocality Routing Accuracy (Routed to different cluster): {correct_locality_routing}/{total_locality} = {locality_accuracy:.4f}")
print(f"\nRephrase Routing Accuracy (Routed to same cluster): {correct_rephrase_routing}/{total_rephrase} = {rephrase_accuracy:.4f}")

print('簇的数量：', router.get_num_clusters())
print('离群点的数量', router.get_num_outlier())

# 测试4.16的路由策略能否区分相近表述
# 4.21 这个脚本需要手动改参数，
# 4.21 这个脚本实际上变为，给定一组参数，观察KnowRouter的工作效果
# 4.23 这个脚本实际上变为，给定一组参数，观察KnowRouter在counterfact上的工作效果

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
data_dir = 'EasyEdit/data/KnowEdit/benchmark_wiki_counterfact_train_cf.json'
ds_size = 300

datas = KnowEditDataset(data_dir, size=ds_size)
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

# --- 0.5 创建路由器 ---
editing_hparams = ZZZHyperParams
hparams = editing_hparams.from_hparams('EasyEdit/hparams/ZZZ/llama3.2-1b.yaml')


config = {
    "use_umap": True,
    "random_seed": 42,
    "umap_params": {
        "n_neighbors": 5,
        "min_dist": 0.1,
        "n_components": 100,
        "metric": "cosine"
    },
    "hdbscan_params": {
        "min_cluster_size": 10,
        "min_samples": 3,
        "metric": "euclidean",
        "cluster_selection_method": "eom",
        "allow_single_cluster": False
    }
}

hparams.clustering = config
hparams.embedding.model_name = './finetuned_sbert_triplet/final_model_2'


print(hparams.clustering)
router = KnowRouter(cfg=hparams)


import json
print("路由表构建完成")

router.build_route_table(prompt_list=prompts)

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

import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  # 确保 numpy 已导入

print("\n\n--- 开始生成可视化图表 ---")
print("\n\n--- 开始生成可视化图表 ---")

# --- 1. 可视化聚类结果 (使用 Plotly - 更新版，加入原始标签) ---
if router.built and hasattr(router.clustering, 'reducer') and hasattr(router.clustering.reducer, 'embedding_'):
    print("生成聚类结果散点图 (颜色=HDBSCAN簇, 形状=原始标签)...")
    try:
        # 获取 UMAP 降维后的二维坐标 (如果 n_components > 2, 取前两维)
        reduced_embeddings = router.clustering.reducer.embedding_
        if reduced_embeddings.shape[1] < 2:
             print("警告：UMAP降维结果少于2维，无法生成二维散点图。")
        else:
            # 获取聚类标签和原始提示
            cluster_labels = router.clustering.cluster.labels_
            # prompts 和 source_files 变量应该在之前的代码块中可用
            if not ('source_files' in locals() or 'source_files' in globals()):
                 print("错误：未找到 'source_files' 变量。请确保在脚本前面已正确获取 source_files。")
            elif len(prompts) != len(reduced_embeddings) or len(prompts) != len(source_files):
                 print(f"警告：数据长度不匹配。Prompts: {len(prompts)}, Embeddings: {len(reduced_embeddings)}, SourceFiles: {len(source_files)}。跳过散点图。")
            else:
                # 创建 DataFrame 以便绘图 (加入原始标签 '原始来源')
                df_cluster = pd.DataFrame({
                    'x': reduced_embeddings[:, 0],
                    'y': reduced_embeddings[:, 1],
                    'HDBSCAN簇': cluster_labels.astype(str), # 转为字符串以更好地区分颜色
                    '原始来源': source_files, # <--- 新增：添加原始文件名作为标签
                    '提示文本': prompts
                })

                # 处理离群点 (-1 标签)
                df_cluster['HDBSCAN簇'] = df_cluster['HDBSCAN簇'].replace('-1', '离群点')

                # 定义颜色映射，确保离群点是灰色
                unique_labels = df_cluster['HDBSCAN簇'].unique()
                color_map = {label: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                             for i, label in enumerate(unique_labels) if label != '离群点'}
                color_map['离群点'] = 'grey' # 将离群点明确设置为灰色

                # 创建交互式散点图 (使用 color 和 symbol)
                fig_scatter = px.scatter(
                    df_cluster,
                    x='x',
                    y='y',
                    color='HDBSCAN簇',            # <--- 颜色代表 HDBSCAN 分配的簇
                    symbol='原始来源',            # <--- 形状代表原始数据来源文件
                    hover_data=['提示文本', 'HDBSCAN簇', '原始来源'], # <--- hover 同时显示两者
                    title="提示词聚类结果 (颜色=HDBSCAN簇, 形状=原始来源)", # <--- 更新标题
                    labels={'x': 'UMAP维度1', 'y': 'UMAP维度2', 'HDBSCAN簇': 'HDBSCAN簇', '原始来源':'原始来源'}, # <--- 更新标签说明
                    color_discrete_map=color_map # 应用自定义颜色
                )

                # 更新离群点的标记样式，使其更明显（例如，更小或不同形状）
                # 注意：如果 symbol 被使用，这里的 marker 更新可能效果有限或被覆盖，但我们仍尝试应用
                # Plotly 通常会优先 symbol 映射，但对特定 trace（如离群点）的修改可能仍有效
                fig_scatter.update_traces(marker=dict(size=10, opacity=0.8), selector=dict(name='离群点')) # 稍微调大离群点默认标记并设透明度
                # 如果离群点过多或与其他标记重叠严重，可以考虑只用颜色区分，注释掉 symbol='原始来源'

                # 更新图例标题，使其更清晰
                fig_scatter.update_layout(legend_title_text='图例 (颜色: HDBSCAN簇, 形状: 原始来源)')

                # 显示图表 (会在浏览器中打开)
                fig_scatter.show()
                print("聚类散点图已生成并显示。请观察颜色和形状：理想情况下，相同形状的点应倾向于聚集，并有相同的颜色。")

    except Exception as e:
        print(f"生成聚类散点图时出错: {e}")
else:
    print("路由器未构建或缺少必要的聚类/降维数据，无法生成散点图。")
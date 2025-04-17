# 评估聚类效果，找出粗略最优解
# 调参代码

import sys
import os

sys.path.append(os.getcwd() + '/EasyEdit')
sys.path.append(os.getcwd() + '/EasyEdit/run_bishe')

from multiarea_dataset import MultiAreaDataset

try:
    from EasyEdit.easyeditor.models.zzz.router import KnowRouter
except:
    from easyeditor.models.zzz.router import KnowRouter
import json
import os
import random
import sys
from typing import List, Any, Union, Dict, Tuple

import hdbscan
from numpy import ndarray, dtype, floating
from sentence_transformers import SentenceTransformer
import transformers
import umap.umap_ as umap
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import adjusted_rand_score  # 引入ARI计算
import torch
from dataclasses import asdict, dataclass
import pickle
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import time  # 用于计时

ROOT_DIR = 'EasyEdit/data/output_meta_llama_3_8b_instruct/'
NUM_SAMPLES_PER_FILE = 60  # 每个文件采样多少条数据
NUM_DOMAINS_TO_SELECT = 5  # 每次选择多少个不同的领域
NUM_COMBINATIONS_TO_TRY = 300  # 尝试多少种不同的领域组合
MIN_SAMPLES_PER_FILE = 10  # 每个文件至少需要多少条数据才能被考虑
RANDOM_SEED = 42

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

DOMAIN_MAP = {
    "art_sculpture.json": "艺术",
    "business_brand.json": "商业", "business_corporation.json": "商业", "business_industry.json": "商业",
    "entertainment_anime.json": "娱乐", "entertainment_music_genre.json": "娱乐", "entertainment_song.json": "娱乐",
    "event_film.json": "事件", "event_history.json": "事件", "event_sport.json": "事件",
    "geography_forest.json": "地理", "geography_glacier.json": "地理", "geography_volcano.json": "地理",
    "health_disease.json": "健康", "health_medication.json": "健康", "health_symptom.json": "健康",
    "human_athlete.json": "人物", "human_entrepreneur.json": "人物", "human_scientist.json": "人物", "human_writer.json": "人物",
    "places_city.json": "地点", "places_country.json": "地点", "places_landmark.json": "地点",
    "technology_database.json": "科技", "technology_programming_language.json": "科技", "technology_software.json": "科技"
}

DEFAULT_ROUTER_CFG = OmegaConf.create({
    "embedding": {
        "random_seed": RANDOM_SEED,
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    },
    "clustering": {
        "use_umap": True,  # 是否使用UMAP降维
        "random_seed": RANDOM_SEED,
        "umap_params": {
            "n_neighbors": 5,  # UMAP参数：邻居数量 (影响局部/全局结构)
            "min_dist": 0.1,  # UMAP参数：点之间的最小距离 (影响簇的紧密度)
            "n_components": 50,  # UMAP参数：降维到的目标维度 (建议 >= 预期簇数)
            "metric": "cosine",  # UMAP参数：计算距离的度量 (cosine适合文本嵌入)
        },
        "hdbscan_params": {
            "min_cluster_size": 10,  # HDBSCAN参数：一个簇最少包含的点数
            "min_samples": 5,  # HDBSCAN参数：成为核心点的最小邻居数 (影响噪声点识别)
            "metric": "euclidean",  # HDBSCAN参数：在降维空间中使用的度量
            "cluster_selection_method": "eom",  # HDBSCAN参数：簇选择方法 ('eom' 或 'leaf')
            "allow_single_cluster": False,  # 是否允许只找到一个大簇
        },
    }
})


def get_file_sample_size(file_path):
    """获取文件中的数据量，决定可以采样多少条数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            num_items = len(data)
            return min(num_items, NUM_SAMPLES_PER_FILE) if num_items >= MIN_SAMPLES_PER_FILE else 0
    except Exception as e:
        print(f"[警告] 读取文件 {file_path} 时出错: {e}")
        return 0


def train_predict(data_configs: Dict,
                  data_dir: str,
                  random_sample: bool = False,
                  random_seed: int = 42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # --- 生成数据 ---
    dataset = MultiAreaDataset(
        root_dir=data_dir,
        dataset_configs=data_configs,
        seed=random_seed,
        random_sample=random_sample
    )

    prompts, _, _, _, _, source_files = dataset.get_data()
    true_labels = [DOMAIN_MAP[fname] for fname in source_files]

    from collections import Counter
    print("实际加载样本分布:", Counter(true_labels))

    print("初始化 KnowRouter...")
    router = KnowRouter(DEFAULT_ROUTER_CFG)
    router.build_route_table(prompts)  # 内部会调用 embedding 和 clustering

    predicted_labels = router.clustering.cluster.labels_
    num_clusters_found = router.get_num_clusters()
    num_outliers = np.sum(predicted_labels == -1)
    print(f"聚类完成。找到 {num_clusters_found} 个聚类 (不包括 {num_outliers} 个离群点)。")

    return predicted_labels, true_labels, num_clusters_found, num_outliers


def eval(predicted_labels, true_labels):
    if len(true_labels) != len(predicted_labels):
        print(f"[错误] 真实标签和预测标签长度不匹配: {len(true_labels)} vs {len(predicted_labels)}")
        return
    score = adjusted_rand_score(true_labels, predicted_labels)

    print(f"调整兰德指数 (ARI): {score:.4f}")
    return score


def search_best_combination():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    files_by_domain: Dict[str, List[str]] = {}
    valid_files = []
    print("检查数据文件并按领域分组...")
    for filename in ALL_FILES:
        file_path = os.path.join(ROOT_DIR, filename)
        if os.path.isfile(file_path):
            domain = DOMAIN_MAP.get(filename)
            if domain:
                if domain not in files_by_domain:
                    files_by_domain[domain] = []
                files_by_domain[domain].append(filename)
                valid_files.append(filename)
            else:
                print(f"[警告] 文件 {filename} 在 DOMAIN_MAP 中没有对应的领域，将被忽略。")
        else:
            print(f"[警告] 数据文件 {filename} 在路径 {ROOT_DIR} 未找到，将被忽略。")

    unique_domains = list(files_by_domain.keys())
    print(f"找到 {len(valid_files)} 个有效的数据文件，属于 {len(unique_domains)} 个不同的领域: {unique_domains}")

    best_score = -float('inf')
    best_combination = None
    results = []
    tried_combinations_set = set()

    print(f"\n开始评估聚类效果，将尝试 {NUM_COMBINATIONS_TO_TRY} 种不同的 {NUM_DOMAINS_TO_SELECT} 个领域的组合...")

    for i in range(NUM_COMBINATIONS_TO_TRY):
        print(f"\n--- 尝试组合 {i + 1}/{NUM_COMBINATIONS_TO_TRY} ---")
        start_time_comb = time.time()

        chosen_domains = random.sample(unique_domains, NUM_DOMAINS_TO_SELECT)

        selected_files = []
        valid_selection = True
        for domain in chosen_domains:
            available_files = files_by_domain[domain]
            if not available_files:
                print(f"[警告] 领域 '{domain}' 没有有效文件")
                valid_selection = False
                break
            selected_files.append(random.choice(available_files))

        if not valid_selection:
            print("跳过当前无效选择。")
            continue

        combination_tuple = tuple(sorted(selected_files))
        if combination_tuple in tried_combinations_set:
            print(f"组合 {selected_files} 已经尝试过，跳过。")
            continue
        tried_combinations_set.add(combination_tuple)

        print(f"选择的领域: {chosen_domains}")
        print(f"选择的文件: {selected_files}")

        dataset_configs = {filename: NUM_SAMPLES_PER_FILE for filename in selected_files}

        predicted_labels, true_labels, num_clusters_found, num_outliers = train_predict(data_configs=dataset_configs,
                                                                                        data_dir=ROOT_DIR,
                                                                                        random_sample=False,
                                                                                        random_seed=RANDOM_SEED)

        score = eval(predicted_labels, true_labels)

        results.append({
            "combination_files": selected_files,
            "combination_domains": chosen_domains,
            "score": score,
            "num_clusters_found": num_clusters_found,
            "num_outliers": num_outliers
        })

        # 更新最佳组合
        if score > best_score:
            best_score = score
            best_combination = {
                "files": selected_files,
                "domains": chosen_domains,
                "score": score,
                "num_clusters": num_clusters_found,
                "num_outliers": num_outliers
            }
            print(f"*** 新的最佳组合！ARI: {score:.4f} ***")

        end_time_comb = time.time()
        print(f"--- 组合 {i + 1} 处理完成，耗时: {end_time_comb - start_time_comb:.2f} 秒 ---")

    # --- 结果报告 ---
    print("\n==================== 评估结果 ====================")
    if best_combination:
        print(f"在尝试的 {len(results)} 个组合中，聚类效果最好的组合（基于ARI）是:")
        print(f"  领域: {best_combination['domains']}")
        print(f"  文件: {best_combination['files']}")
        print(f"  调整兰德指数 (ARI): {best_combination['score']:.4f}")
        print(f"  找到的聚类数 (不含离群点): {best_combination['num_clusters']}")
        print(f"  离群点数 (-1标签): {best_combination['num_outliers']}")
    else:
        print("未能找到任何有效的组合进行评估，或者所有尝试都失败了。")
        if results:
            print("所有尝试的结果:")
            for res in sorted(results, key=lambda x: x['score'], reverse=True):
                print(f"  文件: {res['combination_files']}, ARI: {res['score']:.4f}, 簇数: {res['num_clusters_found']}, 离群点: {res['num_outliers']}")

    # (可选) 打印所有尝试结果的排序列表
    print("\n--- 所有有效尝试的排序结果 (按ARI降序) ---")
    valid_results = [r for r in results if r['score'] > -float('inf')]  # 过滤掉完全失败的尝试
    if valid_results:
        for i, res in enumerate(sorted(valid_results, key=lambda x: x['score'], reverse=True)[:10]):  # 只显示前10个
            print(f"{i + 1}. ARI: {res['score']:.4f} | 文件: {res['combination_files']} | 簇数: {res['num_clusters_found']} | 离群点: {res['num_outliers']}")
    else:
        print("没有有效的评估结果。")



if __name__ == '__main__':
    search_best_combination()

# 评估聚类效果，找出粗略最优解

import sys
import os
sys.path.append(os.getcwd()+'/EasyEdit')
sys.path.append(os.getcwd()+'/EasyEdit/run_bishe')

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
from sklearn.metrics import adjusted_rand_score # 引入ARI计算
import torch
from dataclasses import asdict, dataclass
import pickle
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import time # 用于计时

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
        "use_umap": True,       # 是否使用UMAP降维
        "random_seed": RANDOM_SEED,
        "umap_params": {
            "n_neighbors": 5,      # UMAP参数：邻居数量 (影响局部/全局结构)
            "min_dist": 0.1,       # UMAP参数：点之间的最小距离 (影响簇的紧密度)
            "n_components": 50,      # UMAP参数：降维到的目标维度 (建议 >= 预期簇数)
            "metric": "cosine",     # UMAP参数：计算距离的度量 (cosine适合文本嵌入)
        },
        "hdbscan_params": {
            "min_cluster_size": 10,  # HDBSCAN参数：一个簇最少包含的点数
            "min_samples": 5,       # HDBSCAN参数：成为核心点的最小邻居数 (影响噪声点识别)
            "metric": "euclidean",  # HDBSCAN参数：在降维空间中使用的度量
            "cluster_selection_method": "eom",  # HDBSCAN参数：簇选择方法 ('eom' 或 'leaf')
            "allow_single_cluster": False,  # 是否允许只找到一个大簇
        },
    }
})


def predict(prompt: List[str],
         data_configs: Dict,
         data_dir: str,
         random_sample: bool = False,
         random_seed: int = 42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

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


    return predicted_labels, true_labels


def eval
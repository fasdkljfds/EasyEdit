# 评估聚类效果，找出粗略最优解

from .multiarea_dataset import MultiAreaDataset
from ..easyeditor.models.zzz.router import KnowRouter
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

ROOT_DIR = 'EasyEdit/data/output_meta_llama_3_8b_instruct'
NUM_SAMPLES_PER_FILE = 60  # 每个文件采样多少条数据
NUM_DOMAINS_TO_SELECT = 5  # 每次选择多少个不同的领域
NUM_COMBINATIONS_TO_TRY = 50  # 尝试多少种不同的领域组合
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
        "model_name": "all-MiniLM-L6-v2",  # 一个常用的轻量级SBERT模型
        # "model_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", # 备选，多语言效果更好但稍慢
    },
    "clustering": {
        "use_umap": True,       # 是否使用UMAP降维
        "random_seed": RANDOM_SEED,
        "umap_params": {
            "n_neighbors": 15,      # UMAP参数：邻居数量 (影响局部/全局结构)
            "min_dist": 0.1,       # UMAP参数：点之间的最小距离 (影响簇的紧密度)
            "n_components": 5,      # UMAP参数：降维到的目标维度 (建议 >= 预期簇数)
            "metric": "cosine",     # UMAP参数：计算距离的度量 (cosine适合文本嵌入)
        },
        "hdbscan_params": {
            "min_cluster_size": 10,  # HDBSCAN参数：一个簇最少包含的点数
            "min_samples": 5,       # HDBSCAN参数：成为核心点的最小邻居数 (影响噪声点识别)
            "metric": "euclidean",  # HDBSCAN参数：在降维空间中使用的度量
            "cluster_selection_method": "eom", # HDBSCAN参数：簇选择方法 ('eom' 或 'leaf')
            "allow_single_cluster": False, # 是否允许只找到一个大簇
        },
    }
})
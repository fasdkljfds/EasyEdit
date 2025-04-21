# 评估聚类效果，找出粗略最优解

import os
import sys

sys.path.append(os.getcwd()+'/EasyEdit')
sys.path.append(os.getcwd()+'/EasyEdit/run_bishe')

from multiarea_dataset import MultiAreaDataset
from typing import Dict

from sklearn.metrics import adjusted_rand_score # 引入ARI计算
import time # 用于计时
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List
import hdbscan
import numpy as np
import torch
import umap.umap_ as umap
from numpy import ndarray
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.distance import euclidean
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class EmbeddingConfig:
    random_seed: int
    model_name: str


class Embedding:
    def __init__(self, cfg: EmbeddingConfig) -> None:
        """
        嵌入模型的初始化
        Args:
            cfg (EmbeddingConfig): 嵌入模型的配置
        """
        self.cfg = cfg
        random.seed(cfg.random_seed)
        np.random.seed(cfg.random_seed)
        torch.manual_seed(cfg.random_seed)
        torch.cuda.manual_seed_all(cfg.random_seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.model = SentenceTransformer(cfg.model_name)

    def to_embeddings(self, sentences: List[str]) -> ndarray:
        """
        将句子转换为嵌入向量
        Args:
            sentences (List[str]): 需要转换的句子列表
        """
        return self.model.encode(sentences)

    def cosine_similarity_(self, sentences: List[str]) -> List:
        """
        计算句子之间的余弦相似度矩阵
        Args:
            sentences (List[str]): 需要计算相似度的句子列表
        """
        embeddings = self.model.encode(sentences)
        return cosine_similarity(embeddings)

    def euclidean_distance(self, sentences: List[str]):
        """
        计算句子之间的欧几里得距离矩阵
        Args:
            sentences (List[str]): 需要计算距离的句子列表
        """
        embeddings = self.model.encode(sentences)
        dist_matrix = np.zeros((len(sentences), len(sentences)))
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                dist_matrix[i][j] = euclidean(embeddings[i], embeddings[j])
        return dist_matrix


@dataclass
class ClusteringConfig:
    use_umap: bool
    random_seed: int
    umap_params: dict
    hdbscan_params: dict


class Clustering:
    def __init__(self, cfg: ClusteringConfig) -> None:
        """
        聚类模型的初始化
        Args:
            cfg (ClusteringConfig): 聚类模型的配置
        """
        self.cfg = cfg
        if cfg.use_umap:
            self.reducer = umap.UMAP(
                n_neighbors=cfg.umap_params.n_neighbors,
                min_dist=cfg.umap_params.min_dist,
                n_components=cfg.umap_params.n_components,
                metric=cfg.umap_params.metric,
                random_state=cfg.random_seed
            )
        else:
            raise NotImplementedError("Currently, only UMAP is supported for clustering.")

        self.cluster = hdbscan.HDBSCAN(
            min_cluster_size=cfg.hdbscan_params.min_cluster_size,
            min_samples=cfg.hdbscan_params.min_samples,
            metric=cfg.hdbscan_params.metric,
            cluster_selection_method=cfg.hdbscan_params.cluster_selection_method,
            allow_single_cluster=cfg.hdbscan_params.allow_single_cluster,
            prediction_data=True
        )

    def run_clustering(self, embeddings: ndarray):
        """
        运行聚类算法
        Args:
            embeddings (ndarray): 嵌入向量
        """
        reduced_embeddings_for_clustering = self.reducer.fit_transform(embeddings)
        input_for_hdbscan = reduced_embeddings_for_clustering

        cluster_labels = self.cluster.fit_predict(input_for_hdbscan)

        return cluster_labels

    def predict_cluster(self, new_embedding: np.ndarray) -> tuple[int, float]:
        """
        预测新嵌入的聚类标签和强度
        Args:
            new_embedding (np.ndarray): 新嵌入向量
        """
        if len(new_embedding.shape) == 1:
            new_embedding = new_embedding.reshape(1, -1)

        reduced_embedding = self.reducer.transform(new_embedding)

        label, strengths = hdbscan.approximate_predict(self.cluster, reduced_embedding)
        predicted_label = label[0]

        if predicted_label == -1:
            strength = 0.0
        else:
            try:
                strength = strengths[0][predicted_label]
            except IndexError:
                print(f"警告：尝试访问的索引 {predicted_label} 超出强度数组范围。Strengths shape: {strengths.shape}")
                strength = 0.0

        return predicted_label, strength


class KnowRouter:
    def __init__(self, cfg) -> None:
        cfg = OmegaConf.create(cfg) if not isinstance(cfg, DictConfig) else cfg

        self.cfg = cfg
        self.embedding = Embedding(cfg.embedding)
        self.clustering = Clustering(cfg.clustering)

        self.route_table = None
        self.built = False

    def build_route_table(self, prompt_list: List[str]) -> None:
        """
        在编辑之前，在编辑数据集上构建路由表
        Args:
            prompt_list (List[str]): 需要路由的提示列表
        """
        embeddings = self.embedding.to_embeddings(prompt_list)
        # 聚类
        cluster_labels = self.clustering.run_clustering(embeddings)

        self.route_table = {
            prompt: cluster_id
            for prompt, cluster_id in zip(prompt_list, cluster_labels)
        }
        self.built = True

    def route(self, prompt: str) -> int:
        """
        将输入句子路由到对应的聚类ID
        Args:
            prompt (str): 需要路由的句子
        """
        # 先确定聚类是否成功
        if not self.built:
            raise RuntimeError("Router not built. Call build_route_table() first.")
        if prompt in self.route_table:
            return self.route_table[prompt]

        # 生成嵌入
        embedding = self.embedding.to_embeddings([prompt])[0]
        # 预测cluster
        cluster_id, _ = self.clustering.predict_cluster(embedding)
        return cluster_id

    def _count_similarity(self):
        pass

    def get_num_clusters(self) -> int:
        """
        获取当前路由器的聚类数量
        Returns:
            int: 当前路由器的聚类数量
        """
        if not self.built:
            raise RuntimeError("路由器尚未构建。请先调用 build_route_table() 以执行聚类。")
        labels = self.clustering.cluster.labels_

        unique_labels = set(labels)
        num_clusters = len(unique_labels - {-1})

        return num_clusters

    def save(self, save_dir: str) -> None:
        """
        保存路由器和相关模型到本地
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # 保存路由表和相关配置
        with open(save_path / "router.pkl", "wb") as f:
            pickle.dump({
                "route_table": self.route_table,
                "built": self.built,
                "cfg": self.cfg
            }, f)

        # 保存嵌入模型
        self.embedding.model.save(str(save_path / "embedding_model"))

        with open(save_path / "clustering.pkl", "wb") as f:
            pickle.dump({
                "reducer": self.clustering.reducer,
                "cluster": self.clustering.cluster
            }, f)

    @classmethod
    def load(cls, save_dir: str) -> "KnowRouter":
        """
        从本地加载路由器
        """
        save_path = Path(save_dir)

        # 加载路由表和相关配置
        with open(save_path / "router.pkl", "rb") as f:
            router_data = pickle.load(f)

        # 创建 KnowRouter 实例
        router = cls(router_data["cfg"])
        router.route_table = router_data["route_table"]
        router.built = router_data["built"]

        # 加载嵌入模型
        router.embedding.model = SentenceTransformer(str(save_path / "embedding_model"))

        # 加载聚类模型
        with open(save_path / "clustering.pkl", "rb") as f:
            clustering_data = pickle.load(f)
        router.clustering.reducer = clustering_data["reducer"]
        router.clustering.cluster = clustering_data["cluster"]

        return router


ROOT_DIR = r'O:\bishe3\EasyEdit\data\output_meta_llama_3_8b_instruct'
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
            "n_neighbors": 5,      # UMAP参数：邻居数量 (影响局部/全局结构) 较小的值关注局部结构，可能导致更多、更小的簇；较大的值关注全局结构，可能导致更少、更大的簇。
            "min_dist": 0.1,       # UMAP参数：点之间的最小距离 (影响簇的紧密度) 较小的值使簇更紧凑，较大的值使簇更分散。
            "n_components": 50,      # UMAP参数：降维到的目标维度 (建议 >= 预期簇数)
            "metric": "cosine",     # UMAP参数：计算距离的度量 (cosine适合文本嵌入)
        },
        "hdbscan_params": {
            "min_cluster_size": 10,  # HDBSCAN参数：一个簇最少包含的点数 太小可能产生很多噪声小簇，太大可能合并不同质的簇。
            "min_samples": 5,       # HDBSCAN参数：成为核心点的最小邻居数 (影响噪声点识别) 它影响模型对噪声的敏感度。值越大，被标记为噪声的点越多
            "metric": "euclidean",  # HDBSCAN参数：在降维空间中使用的度量
            "cluster_selection_method": "eom",  # HDBSCAN参数：簇选择方法 ('eom' 或 'leaf')
            "allow_single_cluster": False,  # 是否允许只找到一个大簇
        },
    }
})


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

    prompts, _, _, _, _, source_files = dataset.to_edit_dataset()
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


def search_hparams(data_configs: dict,
                   data_dir: str,
                   param_grid: dict,
                   n_trials: int = 100,
                   ):
    """在给定的数据集上，寻找最优的超参数配置"""

    # --- 1. 设置随机种子 ---
    random_seed = RANDOM_SEED
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # --- 2. 加载数据并计算嵌入 (只做一次) ---
    dataset = MultiAreaDataset(
        root_dir=data_dir,
        dataset_configs=data_configs,
        seed=random_seed,
        random_sample=False
    )
    prompts, _, _, _, _, source_files = dataset.to_edit_dataset()

    true_labels = [DOMAIN_MAP[fname] for fname in source_files]
    print(f"数据集加载完成。样本总数: {len(prompts)}")
    from collections import Counter
    print("真实标签分布:", Counter(true_labels))

    embedding_model = Embedding(DEFAULT_ROUTER_CFG.embedding)
    embeddings = embedding_model.to_embeddings(prompts)
    print(f"嵌入向量生成完成。嵌入维度: {embeddings.shape[1]}")

    # --- 3. 随机搜索超参数 ---
    best_score = -1.0
    best_params = {}
    results = []  # 存储每次试验的结果

    print(f"开始进行 {n_trials} 次随机搜索试验...")
    for i in range(n_trials):
        trial_params = {}
        print(f"\n--- 试验 {i + 1}/{n_trials} ---")

        # --- 3.1 随机选择超参数 ---
        current_umap_params = {}
        for key, values in param_grid.items():
            chosen_value = random.choice(values)
            trial_params[key] = chosen_value
            if key.startswith("umap_"):
                current_umap_params[key.replace("umap_", "")] = chosen_value

        current_hdbscan_params = {}
        for key, values in param_grid.items():
            if key.startswith("hdbscan_"):
                current_hdbscan_params[key.replace("hdbscan_", "")] = trial_params[key]  # 使用上面选好的值

        print("当前参数:")
        print(f"  UMAP: {current_umap_params}")
        print(f"  HDBSCAN: {current_hdbscan_params}")

        try:
            # --- 3.2 UMAP 降维 ---
            umap_reducer = umap.UMAP(
                n_neighbors=current_umap_params['n_neighbors'],
                min_dist=current_umap_params['min_dist'],
                n_components=current_umap_params['n_components'],
                metric=DEFAULT_ROUTER_CFG.clustering.umap_params.metric,  # 使用默认或也加入搜索
                random_state=random_seed
            )
            reduced_embeddings = umap_reducer.fit_transform(embeddings)

            # --- 3.3 HDBSCAN 聚类 ---
            hdbscan_clusterer = hdbscan.HDBSCAN(
                min_cluster_size=current_hdbscan_params['min_cluster_size'],
                min_samples=current_hdbscan_params.get('min_samples', None),  # 允许min_samples可选
                metric=DEFAULT_ROUTER_CFG.clustering.hdbscan_params.metric,  # 使用默认或也加入搜索
                cluster_selection_method=DEFAULT_ROUTER_CFG.clustering.hdbscan_params.cluster_selection_method,  # 使用默认或也加入搜索
                allow_single_cluster=DEFAULT_ROUTER_CFG.clustering.hdbscan_params.allow_single_cluster  # 使用默认
            )
            predicted_labels = hdbscan_clusterer.fit_predict(reduced_embeddings)

            # --- 3.4 评估 ---
            num_clusters_found = len(set(predicted_labels) - {-1})
            num_outliers = np.sum(predicted_labels == -1)
            print(f"聚类完成。找到 {num_clusters_found} 个聚类，{num_outliers} 个离群点。")

            current_score = eval(predicted_labels, true_labels)  # 使用你的eval函数计算ARI
            results.append({'params': trial_params, 'score': current_score, 'num_clusters': num_clusters_found, 'num_outliers': num_outliers})

            # --- 3.5 更新最佳结果 ---
            if current_score > best_score:
                best_score = current_score
                best_params = trial_params
                print(f"*** 新的最佳分数: {best_score:.4f} ***")

        except Exception as e:
            print(f"试验 {i + 1} 遇到错误: {e}")
            results.append({'params': trial_params, 'score': -1.0, 'error': str(e)})  # 记录错误

    print(f"\n--- 超参数搜索完成 ---")
    if best_params:
        print(f"找到的最佳 ARI 分数: {best_score:.4f}")
        print("对应的最佳参数:")
        # 为了清晰，分开打印UMAP和HDBSCAN参数
        best_umap = {k.replace("umap_", ""): v for k, v in best_params.items() if k.startswith("umap_")}
        best_hdbscan = {k.replace("hdbscan_", ""): v for k, v in best_params.items() if k.startswith("hdbscan_")}
        print(f"  UMAP: {best_umap}")
        print(f"  HDBSCAN: {best_hdbscan}")
    else:
        print("未能找到任何有效的超参数组合。")

    # 可以选择性地返回所有结果，以便进一步分析
    # return best_params, best_score, results
    return best_params, best_score



if __name__ == '__main__':
    data_configs = {
        "business_industry.json": 50,
        "human_scientist.json": 50,
        "event_sport.json": 50,
        "geography_forest.json": 50,
        "places_landmark.json": 50,
    }

    param_search_grid = {
        'umap_n_neighbors': [5, 10, 15, 20, 30],  # 探索邻居数量
        'umap_min_dist': [0.0, 0.1, 0.25, 0.5],  # 探索簇的紧密度
        'umap_n_components': [5, 10, 20, 30, 50],  # 探索降维维度
        'hdbscan_min_cluster_size': [5, 10, 15, 20],  # 探索最小簇大小
        'hdbscan_min_samples': [1, 3, 5, 10]  # 探索核心点密度/噪声敏感度
    }

    num_search_trials = 100

    best_found_params, best_found_score = search_hparams(
        data_configs=data_configs,
        data_dir=ROOT_DIR,
        param_grid=param_search_grid,
        n_trials=num_search_trials,
    )

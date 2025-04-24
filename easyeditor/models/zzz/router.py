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
                strength = strengths[0]
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

        self.anchors = []
        self.route_table = None
        self.built = False

        # for update
        if self.cfg.two_stages:
            self.boundary_embedding = Embedding(
                EmbeddingConfig(
                    random_seed=cfg.embedding.random_seed,
                    model_name=cfg.boundary_model_name
                )
            )

    def build_route_table(self, prompt_list: List[str]) -> None:
        """
        在编辑之前，在编辑数据集上构建路由表
        Args:
            prompt_list (List[str]): 需要路由的提示列表
        """
        embeddings = self.embedding.to_embeddings(prompt_list)
        # 聚类
        cluster_labels = self.clustering.run_clustering(embeddings)

        self.anchors = prompt_list
        self.anchor_embeddings = self.boundary_embedding.to_embeddings(self.anchors)

        self.route_table = {
            prompt: cluster_id
            for prompt, cluster_id in zip(prompt_list, cluster_labels)
        }
        self.built = True

    def boundary_route(self, prompt: str, threshold: float) -> bool:
        prompt_embedding = self.boundary_embedding.to_embeddings([prompt])

        similarities = cosine_similarity(prompt_embedding, self.anchor_embeddings)

        max_similarity = np.max(similarities[0])

        should_route_to_original = max_similarity < threshold

        print(f"Prompt: '{prompt[:50]}...', Max Similarity with Anchors: {max_similarity:.4f}, Threshold: {threshold}, Route to Original: {should_route_to_original}")

        return should_route_to_original

        pass

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

        if self.cfg.two_stages and self.boundary_route(prompt, self.cfg.boundary_threshold):
            return -2
            
        # 生成嵌入
        embedding = self.embedding.to_embeddings([prompt])[0]
        # 预测cluster
        cluster_id, _ = self.clustering.predict_cluster(embedding)
        return cluster_id

    def route_with_confidence(self, prompt: str) -> tuple[int, float]:
        """
        将输入句子路由到对应的聚类ID，并返回置信度
        Args:
            prompt (str): 需要路由的句子
        """
        if not self.built:
            raise RuntimeError("Router not built. Call build_route_table() first.")

        embedding = self.embedding.to_embeddings([prompt])[0]
        cluster_id, confidence = self.clustering.predict_cluster(embedding)
        return cluster_id, confidence


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

    def get_num_outlier(self) -> int:
        """
        获取离群点数量

        """
        if not self.built:
            raise RuntimeError("路由器尚未构建。请先调用 build_route_table() 以执行聚类。")
        labels = self.clustering.cluster.labels_

        num_outliers = 0
        for i in labels:
            if i == -1:
                num_outliers += 1

        return num_outliers

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


class ScopeRouter:
    def __init__(self, cfg) -> None:
        cfg = OmegaConf.create(cfg) if not isinstance(cfg, DictConfig) else cfg
        self.cfg = cfg














if __name__ == '__main__':
    pass

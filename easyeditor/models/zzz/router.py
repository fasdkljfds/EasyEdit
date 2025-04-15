import os
import sys
from typing import List, Any, Union

import hdbscan
from numpy import ndarray, dtype, floating
from sentence_transformers import SentenceTransformer
import transformers
import hydra
import umap.umap_ as umap
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity
import torch
from dataclasses import asdict
import random
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf


@dataclass
class EmbeddingConfig:
    random_seed: int
    model_name: str


class Embedding:
    def __init__(self, cfg: EmbeddingConfig) -> None:
        self.cfg = cfg
        random.seed(cfg.random_seed)
        np.random.seed(cfg.random_seed)
        torch.manual_seed(cfg.random_seed)
        torch.cuda.manual_seed_all(cfg.random_seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.model = SentenceTransformer(cfg.model_name)

    def to_embeddings(self, sentences: List[str]) -> ndarray:
        return self.model.encode(sentences)

    def cosine_similarity_(self, sentences: List[str]) -> List:
        embeddings = self.model.encode(sentences)
        return cosine_similarity(embeddings)

    def euclidean_distance(self, sentences: List[str]):
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
        reduced_embeddings_for_clustering = self.reducer.fit_transform(embeddings)
        input_for_hdbscan = reduced_embeddings_for_clustering

        cluster_labels = self.cluster.fit_predict(input_for_hdbscan)

        return cluster_labels

    def predict_cluster(self, new_embedding: np.ndarray) -> tuple[int, float]:
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
        cfg = OmegaConf.create(asdict(cfg))


        self.cfg = cfg
        self.embedding = Embedding(cfg.embedding)
        self.clustering = Clustering(cfg.clustering)

        self.route_table = None
        self.built = False

    def build_route_table(self, prompt_list: List[str]) -> None:
        # 生成句子嵌入
        embeddings = self.embedding.to_embeddings(prompt_list)
        # 聚类
        cluster_labels = self.clustering.run_clustering(embeddings)

        self.route_table = {
            prompt: cluster_id
            for prompt, cluster_id in zip(prompt_list, cluster_labels)
        }
        self.built = True

    def route(self, prompt: str) -> int:
        if not self.built:
            raise RuntimeError("Router not built. Call build_route_table() first.")

        # 生成嵌入
        embedding = self.embedding.to_embeddings([prompt])[0]
        # 预测cluster
        cluster_id, _ = self.clustering.predict_cluster(embedding)
        return cluster_id

    def _count_similarity(self):
        pass

    def get_num_clusters(self) -> int:
        if not self.built:
            raise RuntimeError("路由器尚未构建。请先调用 build_route_table() 以执行聚类。")
        if not self.clustering.hdbscan_fitted:
             print("警告：HDBSCAN 未拟合。返回 0 个簇。")
             return 0
        labels = self.clustering.cluster.labels_

        unique_labels = set(labels)
        num_clusters = len(unique_labels - {-1})

        return num_clusters


if __name__ == '__main__':
    pass

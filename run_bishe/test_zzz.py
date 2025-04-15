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
import random
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.getcwd() + '/EasyEdit')
try:
    from EasyEdit.easyeditor import (
        FTHyperParams,
        IKEHyperParams,
        KNHyperParams,
        MEMITHyperParams,
        ROMEHyperParams,
        LoRAHyperParams,
        MENDHyperParams,
        SERACHparams,
        WISEHyperParams,
    )

    from EasyEdit.easyeditor import BaseEditor
    from EasyEdit.easyeditor.models.ike import encode_ike_facts
    from sentence_transformers import SentenceTransformer
    from EasyEdit.easyeditor import KnowEditDataset

except ImportError:
    from easyeditor import (
        FTHyperParams,
        IKEHyperParams,
        KNHyperParams,
        MEMITHyperParams,
        ROMEHyperParams,
        LoRAHyperParams,
        MENDHyperParams,
        SERACHparams,
        WISEHyperParams,
    )

    from easyeditor import BaseEditor
    from easyeditor.models.ike import encode_ike_facts
    from sentence_transformers import SentenceTransformer
    from easyeditor import KnowEditDataset


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


class DataHandler:
    def __init__(self):
        pass

    def load_counterfact_data(self, filepath: str, size: int) -> List[str]:
        datas = KnowEditDataset(filepath, size=size)
        prompts = [data['prompt'] for data in datas]

        return prompts

    def load_counterfact_prompt_loc(self, filepath: str, size: int) -> List:
        datas = KnowEditDataset(filepath, size=size)
        prompts = [data['prompt'] for data in datas]

        locality_rs = [data['locality_rs'] for data in datas]
        locality_f = [data['locality_f'] for data in datas]
        locality_Relation_Specificity_prompts = []
        locality_Relation_Specificity_ans = []
        locality_Forgetfulness_prompts = []
        locality_Forgetfulness_ans = []

        locality_data = [locality_rs, locality_f]
        locality_prompts = [locality_Relation_Specificity_prompts, locality_Forgetfulness_prompts]
        locality_answers = [locality_Relation_Specificity_ans, locality_Forgetfulness_ans]
        for data, local_prompts, local_answers in zip(locality_data, locality_prompts, locality_answers):
            for item in data:
                if item is None:
                    local_prompts.append(None)
                    local_answers.append(None)
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
                    local_prompts.append(temp_prompts)
                    local_answers.append(temp_answers)
        assert len(prompts) == len(locality_Relation_Specificity_prompts) == len(locality_Forgetfulness_prompts)
        locality_inputs = {}
        portability_inputs = {}

        locality_inputs = {
            'Relation_Specificity': {
                'prompt': locality_Relation_Specificity_prompts,
                'ground_truth': locality_Relation_Specificity_ans
            },
            'Forgetfulness': {
                'prompt': locality_Forgetfulness_prompts,
                'ground_truth': locality_Forgetfulness_ans
            }
        }

        relation_spec_prompts = locality_inputs['Relation_Specificity']['prompt']

        forget_prompts = locality_inputs['Forgetfulness']['prompt']

        loc_prompts = [(r, f) for r, f in zip(relation_spec_prompts, forget_prompts)]

        return [prompts, loc_prompts]

@hydra.main(config_path="config", config_name="config")
def test1(cfg: DictConfig) -> None:
    data_handler = DataHandler()

    sentences = data_handler.load_counterfact_data(r'O:\bishe3\EasyEdit\data\KnowEdit\benchmark_wiki_counterfact_train_cf.json', 100)

    new_sentence = "A kitty lying on the carpet"

    print("\nEmbedding Configuration:")
    print(OmegaConf.to_yaml(cfg.embedding))
    embedder = Embedding(cfg.embedding)

    embeddings = embedder.to_embeddings(sentences)
    print(f"\nGenerated embeddings shape: {embeddings.shape}")

    print("\nClustering Configuration:")
    print(OmegaConf.to_yaml(cfg.clustering))
    clusterer = Clustering(cfg.clustering)

    labels = clusterer.run_clustering(embeddings)
    print("\nClustering results:")
    for sent, label in zip(sentences, labels):
        print(f"{label}: {sent}")

    new_embedding = embedder.to_embeddings([new_sentence])
    label, strength = clusterer.predict_cluster(new_embedding)
    print(f"\nNew sentence prediction: '{new_sentence}'")
    print(f"Cluster: {label}, Strength: {strength:.2f}")

    print("\nCosine similarities between first sentence and others:")
    first_embedding = embeddings[0].reshape(1, -1)
    similarities = cosine_similarity(first_embedding, embeddings)[0]
    for sent, sim in zip(sentences, similarities):
        print(f"{sim:.2f}: {sent}")


@hydra.main(config_path="config", config_name="config")
def test2(cfg: DictConfig) -> None:
    data_handler = DataHandler()

    # 获取数据，结构为 [main_prompts, loc_prompts]
    main_prompts, loc_prompts = data_handler.load_counterfact_prompt_loc(
        r'O:\bishe3\EasyEdit\data\KnowEdit\benchmark_wiki_counterfact_train_cf.json',
        100
    )

    print("\nEmbedding Configuration:")
    print(OmegaConf.to_yaml(cfg.embedding))
    embedder = Embedding(cfg.embedding)

    main_embeddings = embedder.to_embeddings(main_prompts)

    # 分析每个主prompt和对应的loc_prompts的相似度
    for i, (main_prompt, (rs_prompts, f_prompts)) in enumerate(zip(main_prompts, loc_prompts)):
        print(f"\n\n=== 主prompt {i + 1} ===")
        print(f"主内容: {main_prompt}")

        # 嵌入主prompt
        main_embedding = main_embeddings[i].reshape(1, -1)

        # 处理Relation Specificity prompts
        print("\nRelation Specificity 相似度分析:")
        if rs_prompts is None:
            print("无Relation Specificity数据")
        else:
            rs_embeddings = embedder.to_embeddings(rs_prompts)
            similarities = cosine_similarity(main_embedding, rs_embeddings)[0]

            for prompt, sim in zip(rs_prompts, similarities):
                print(f"相似度: {sim:.4f} - {prompt}")

        # 处理Forgetfulness prompts
        print("\nForgetfulness 相似度分析:")
        if f_prompts is None:
            print("无Forgetfulness数据")
        else:
            f_embeddings = embedder.to_embeddings(f_prompts)
            similarities = cosine_similarity(main_embedding, f_embeddings)[0]

            for prompt, sim in zip(f_prompts, similarities):
                print(f"相似度: {sim:.4f} - {prompt}")


if __name__ == '__main__':
    test2()


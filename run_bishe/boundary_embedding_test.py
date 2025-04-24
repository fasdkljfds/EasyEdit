# 用于测试boundary_embedding对counterfact的分类能力

import sys
import os

sys.path.append(os.getcwd()+'/EasyEdit')
sys.path.append(os.getcwd()+'/EasyEdit/run_bishe')

from omegaconf import DictConfig, OmegaConf
from scipy.spatial.distance import euclidean
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
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



def prepare_counterfact_data(data_dir, ds_size):
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

    locality_prompts = [i[0] if i else ' 'for i in locality_Relation_Specificity_prompts ]
    rephrase_prompts = [i[0] if i else ' ' for i in portability_Subject_Aliasing_prompts]
    return prompts, locality_prompts, rephrase_prompts



def load_boundary_embedding(random_seed, model_path):
    boundary_embedding = Embedding(
        EmbeddingConfig(
            random_seed=42,
            model_name=model_path
        )
    )
    return boundary_embedding



if __name__ == '__main__':
    embedding = load_boundary_embedding(
        random_seed=42,
        model_path='./finetuned_sbert_triplet/final_model_1'
    )


    threshold = 0.5
    # --- 2. 准备 CounterFact 测试数据 ---
    print("\n准备 CounterFact 测试数据...")
    # 注意：你的 prepare_counterfact_data 函数内部似乎硬编码了路径和大小
    # 这里直接调用它，但请确保路径 'EasyEdit/data/KnowEdit/benchmark_wiki_counterfact_train_cf.json' 和 ds_size=300 是你想要的

    prompts, locality_prompts, rephrase_prompts = prepare_counterfact_data(
        data_dir='EasyEdit/data/KnowEdit/benchmark_wiki_counterfact_train_cf.json',
        ds_size=300
    )
    print(f"CounterFact 数据准备完毕，共 {len(prompts)} 条样本。")

    # threshold = 0.5 # 这个阈值在这里暂时用不上，我们主要比较正负例距离

    # --- 3. 计算距离并基于阈值进行分类评估 (逐条打印) ---
    print("\n开始测试模型在 CounterFact 数据上的分类能力 (基于阈值)...")
    print("评估标准：")
    print("  - Anchor 与 Rephrase (正例) 应被分类为 '相关' (距离 < 阈值)")
    print("  - Anchor 与 Locality (负例) 应被分类为 '不相关' (距离 >= 阈值)")
    print("-" * 50)  # 分隔符

    import numpy as np
    from scipy.spatial.distance import cosine as cosine_distance  # 导入余弦距离计算函数

    # 初始化分类指标计数器
    tp = 0  # 真阳性: Anchor-Rephrase 距离 < threshold (正确分类为相关)
    fn = 0  # 假阴性: Anchor-Rephrase 距离 >= threshold (错误分类为不相关)
    tn = 0  # 真阴性: Anchor-Locality 距离 >= threshold (正确分类为不相关)
    fp = 0  # 假阳性: Anchor-Locality 距离 < threshold (错误分类为相关)

    total_valid_samples = 0  # 记录处理了多少个有效的原始 prompt (三元组)
    all_dist_pos = []  # 存储所有的 anchor-positive (rephrase) 距离
    all_dist_neg = []  # 存储所有的 anchor-negative (locality) 距离

    # 遍历 CounterFact 数据集中的每一条样本
    for i in range(len(prompts)):
        anchor_text = prompts[i]
        positive_text = rephrase_prompts[i]  # Rephrase (预期相关)
        negative_text = locality_prompts[i]  # Locality (预期不相关)

        # 进行简单的有效性检查
        if not anchor_text or not positive_text or not negative_text or \
                anchor_text.strip() == '' or positive_text.strip() == '' or negative_text.strip() == '':
            # print(f"信息：跳过空/无效样本 {i}") # 可选的跳过信息
            continue  # 跳过这条数据

        # 使用加载的模型获取嵌入向量
        try:
            texts_to_encode = [anchor_text, positive_text, negative_text]
            embeddings_list = embedding.to_embeddings(texts_to_encode)
            anchor_emb = embeddings_list[0]
            positive_emb = embeddings_list[1]
            negative_emb = embeddings_list[2]
        except Exception as e:
            print(f"\n错误：在处理索引 {i} 的文本时发生嵌入错误: {e}")
            continue  # 跳过此样本

        # 计算余弦距离
        dist_pos = cosine_distance(anchor_emb, positive_emb)
        dist_neg = cosine_distance(anchor_emb, negative_emb)

        # 检查距离是否为 NaN
        if np.isnan(dist_pos) or np.isnan(dist_neg):
            print(f"\n警告：索引 {i} 计算得到 NaN 距离，跳过。dist_pos={dist_pos}, dist_neg={dist_neg}")
            continue

        total_valid_samples += 1  # 有效样本计数增加
        all_dist_pos.append(dist_pos)
        all_dist_neg.append(dist_neg)

        # --- 逐条打印分类情况 ---
        print(f"\n--- 样本 {i} ---")
        print(f"  Anchor:   '{anchor_text[:80]}...'")  # 打印部分文本以便识别

        # 评估 Anchor-Rephrase (正例) 对
        is_pos_related = dist_pos < threshold
        print(f"  Positive (Rephrase): '{positive_text[:80]}...'")
        print(f"    距离: {dist_pos:.4f}")
        print(f"    分类: {'相关' if is_pos_related else '不相关'} (预期: 相关) -> {'正确 (TP)' if is_pos_related else '错误 (FN)'}")
        if is_pos_related:
            tp += 1
        else:
            fn += 1

        # 评估 Anchor-Locality (负例) 对
        is_neg_unrelated = dist_neg >= threshold
        print(f"  Negative (Locality): '{negative_text[:80]}...'")
        print(f"    距离: {dist_neg:.4f}")
        print(f"    分类: {'不相关' if is_neg_unrelated else '相关'} (预期: 不相关) -> {'正确 (TN)' if is_neg_unrelated else '错误 (FP)'}")
        if is_neg_unrelated:
            tn += 1
        else:
            fp += 1

    # --- 4. 输出最终的汇总统计结果 (这部分保持不变) ---
    print("\n" + "=" * 50)
    print("--- 基于阈值的分类评估结果 (汇总) ---")
    print("=" * 50)

    if total_valid_samples > 0:
        total_predictions = tp + fn + tn + fp  # 总的判断次数 = 正例对数 + 负例对数 = 2 * total_valid_samples
        if total_predictions != 2 * total_valid_samples:
            print(f"警告：总预测数 ({total_predictions}) 与预期 ({2 * total_valid_samples}) 不符，请检查逻辑。")

        # 计算各项指标
        accuracy = (tp + tn) / total_predictions if total_predictions > 0 else 0.0
        # --- '相关' 类别的指标 ---
        precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # 精确率(查准率): 预测为'相关'的里面，实际有多少是'相关'(Rephrase)的
        recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # 召回率(查全率): 实际'相关'(Rephrase)的里面，有多少被预测为'相关'
        f1_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else 0.0
        # --- '不相关' 类别的指标 ---
        precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # 精确率: 预测为'不相关'的里面，实际有多少是'不相关'(Locality)的
        recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # 召回率(特异度): 实际'不相关'(Locality)的里面，有多少被预测为'不相关'
        f1_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg) if (precision_neg + recall_neg) > 0 else 0.0

        print(f"处理的有效原始 prompt (三元组) 数量: {total_valid_samples}")
        print(f"总的分类判断次数 (每个 prompt 包含正负例各一次): {total_predictions}")
        print("-" * 20)
        print("混淆矩阵计数:")
        print(f"  真阳性 (TP - Anchor-Rephrase < Threshold): {tp}")
        print(f"  假阴性 (FN - Anchor-Rephrase >= Threshold): {fn}")
        print(f"  真阴性 (TN - Anchor-Locality >= Threshold): {tn}")
        print(f"  假阳性 (FP - Anchor-Locality < Threshold): {fp}")
        print("-" * 20)
        print(f"总体准确率 (Accuracy): {accuracy:.4f}")
        print("-" * 20)
        print("针对 '相关' 类别 (Anchor-Rephrase) 的评估:")
        print(f"  精确率 (Precision): {precision_pos:.4f}")
        print(f"  召回率 (Recall): {recall_pos:.4f}")
        print(f"  F1 分数 (F1-Score): {f1_pos:.4f}")
        print("-" * 20)
        print("针对 '不相关' 类别 (Anchor-Locality) 的评估:")
        print(f"  精确率 (Precision): {precision_neg:.4f}")
        print(f"  召回率 (Recall/Specificity): {recall_neg:.4f}")
        print(f"  F1 分数 (F1-Score): {f1_neg:.4f}")
        print("-" * 20)

        # 同时打印之前的平均距离信息，作为参考
        avg_dist_pos = np.mean(all_dist_pos) if all_dist_pos else float('nan')
        avg_dist_neg = np.mean(all_dist_neg) if all_dist_neg else float('nan')
        print("\n补充信息：平均距离统计 (与阈值判断无关，仅供参考)")
        print(f"  平均 Anchor-Rephrase (正例) 余弦距离: {avg_dist_pos:.4f}")
        print(f"  平均 Anchor-Locality (负例) 余弦距离: {avg_dist_neg:.4f}")

        # 对结果进行简单解读
        print("\n--- 结果解读 ---")
        print(f"使用阈值 {threshold}：")
        if accuracy > 0.75:  # 举例阈值
            print(f"- 模型的整体分类准确率 ({accuracy:.2%}) 较高。")
        else:
            print(f"- 模型的整体分类准确率 ({accuracy:.2%}) 可能有提升空间。")

        if recall_pos > 0.75 and recall_neg > 0.75:
            print(f"- 模型在识别'相关'(召回率 {recall_pos:.2%}) 和 '不相关'(召回率 {recall_neg:.2%}) 样本方面表现均衡且较好。")
        elif recall_pos < recall_neg:
            print(f"- 模型可能更容易将实际'相关'的样本误判为'不相关' (正例召回率 {recall_pos:.2%} 较低)。")
        elif recall_neg < recall_pos:
            print(f"- 模型可能更容易将实际'不相关'的样本误判为'相关' (负例召回率/特异度 {recall_neg:.2%} 较低)。")

        if fp > fn:
            print(f"- 模型倾向于产生更多的假阳性 (FP={fp})，即将不相关的识别为相关。")
        elif fn > fp:
            print(f"- 模型倾向于产生更多的假阴性 (FN={fn})，即将相关的识别为不相关。")


    else:
        print("错误：没有找到有效的 CounterFact 样本进行测试。请检查 `prepare_counterfact_data` 函数和数据文件。")

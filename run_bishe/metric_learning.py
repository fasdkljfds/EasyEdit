# 实现对counterfact和multi-area的度量学习

import sys
import os
import torch
from tqdm import tqdm # 用于显示进度条
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

sys.path.append(os.getcwd()+'/EasyEdit')
sys.path.append(os.getcwd()+'/EasyEdit/run_bishe')

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

import sys
import os
import torch
from tqdm import tqdm # 用于显示进度条
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.getcwd()+'/EasyEdit')
sys.path.append(os.getcwd()+'/EasyEdit/run_bishe')

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



import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
from tqdm.notebook import tqdm # 或者 from tqdm import tqdm，取决于你的环境
import math
import os

# --- 1. 定义三元组数据集 ---
class TripletDataset(Dataset):
    """
    用于 Triplet Loss 训练的 PyTorch 数据集。
    接收锚点、正样本、负样本列表。
    """
    def __init__(self, anchors: List[str], positives: List[str], negatives: List[str]):
        # 确保三者长度一致
        assert len(anchors) == len(positives) == len(negatives), \
            "锚点、正样本和负样本的数量必须相同！"
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives
        print(f"TripletDataset 初始化完成，包含 {len(anchors)} 个三元组样本。")

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        # 返回一个 InputExample 对象，SentenceTransformer 的 TripletLoss 可以直接处理
        # 或者你可以只返回元组 (anchor, positive, negative)，在训练循环中处理
        # 使用 InputExample 更符合 sentence-transformers 的习惯
        return InputExample(texts=[self.anchors[idx], self.positives[idx], self.negatives[idx]])

# --- 2. DML 训练函数 ---
def train_dml_encoder(
    base_model_name: str,
    anchors: List[str],
    positives: List[str],
    negatives: List[str],
    output_path: str,
    epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    margin: float = 5.0, # TripletLoss 的边界参数，可以调整
    warmup_steps: int = 100,
    device: str = None
    ):
    """
    使用 Triplet Loss 微调 SentenceTransformer 模型。

    Args:
        base_model_name (str): 基础预训练模型的名称或路径 (例如 'sentence-transformers/all-MiniLM-L6-v2')。
        anchors (List[str]): 锚点句子列表。
        positives (List[str]): 正样本句子列表 (等价改写)。
        negatives (List[str]): 负样本句子列表 (非等价改写)。
        output_path (str): 微调后模型的保存路径。
        epochs (int): 训练轮数。
        batch_size (int): 训练批次大小。
        learning_rate (float): 学习率。
        margin (float): TripletLoss 的边界值。
        warmup_steps (int): 学习率预热步数。
        device (str): 训练设备 ('cuda' 或 'cpu')。如果为 None，则自动检测。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}进行训练")

    # 步骤 1: 加载基础模型
    print(f"加载基础模型: {base_model_name}")
    model = SentenceTransformer(base_model_name, device=device)

    # 步骤 2: 准备数据集和数据加载器
    print("准备训练数据...")
    train_dataset = TripletDataset(anchors, positives, negatives)
    # 注意：SentenceTransformer 的 TripletLoss DataLoader 不需要特殊 collate_fn
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 步骤 3: 定义损失函数
    # distance_metric 可以选择，例如 losses.SiameseDistanceMetric.COSINE_DISTANCE
    # TripletLoss 需要 (anchor, positive, negative)
    train_loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE, # 使用余弦距离
        triplet_margin=margin
    )
    print(f"使用 TripletLoss，边界 (margin) = {margin}")

    # 步骤 4: 设置训练参数并进行训练
    num_training_steps = len(train_dataloader) * epochs
    if warmup_steps == -1: # 如果没指定，默认设置为 10% 的训练步数
        warmup_steps = math.ceil(num_training_steps * 0.1)
    print(f"总训练步数: {num_training_steps}, 预热步数: {warmup_steps}")

    print("开始微调模型...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        optimizer_params={'lr': learning_rate},
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
        checkpoint_save_steps=len(train_dataloader)//2, # 每半个epoch保存一次checkpoint（可选）
        checkpoint_path=os.path.join(output_path, "checkpoints") # checkpoint保存路径（可选）
    )

    print(f"微调完成！模型已保存到: {output_path}")
    return output_path

# --- 3. DML 微调执行部分 ---
if __name__ == '__main__':
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

    # rephrease_prompts即等价改写问题，locality_prompts即非等价改写问题
    prompts, rephrase_prompts, target_new, subjects, locality_inputs, _ = multiarea_dataset.to_edit_dataset()
    locality_prompts = locality_inputs['neighborhood']['prompt']  # 这个loc数据要单独拿出来
    print("数据加载完成。")
    print(f"样本数量: prompts={len(prompts)}, rephrases={len(rephrase_prompts)}, localities={len(locality_prompts)}")

    # --- 设置 DML 微调参数 ---
    BASE_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2' #! 你可以改成你实际使用的模型
    FINETUNED_MODEL_SAVE_PATH = './finetuned_sbert_dml' # 微调后模型的保存路径
    DML_EPOCHS = 3 # 训练轮数，根据效果调整
    DML_BATCH_SIZE = 16 # 根据你的 GPU 显存调整
    DML_LEARNING_RATE = 1e-5 # 学习率，可以调小一些
    DML_MARGIN = 1.0 # Triplet Loss 的 margin，常用的值是 1.0 或 5.0，需要实验

    # --- 执行 DML 微调 ---
    print("\n--- 开始 DML 微调 ---")
    # 确保保存路径存在
    os.makedirs(FINETUNED_MODEL_SAVE_PATH, exist_ok=True)

    # 调用训练函数
    trained_model_path = train_dml_encoder(
        base_model_name=BASE_MODEL_NAME,
        anchors=prompts,
        positives=rephrase_prompts,
        negatives=locality_prompts,
        output_path=FINETUNED_MODEL_SAVE_PATH,
        epochs=DML_EPOCHS,
        batch_size=DML_BATCH_SIZE,
        learning_rate=DML_LEARNING_RATE,
        margin=DML_MARGIN,
        warmup_steps = -1, # 设置为 -1，则自动计算 10% 的预热步数
        # device='cuda' # 或者 'cpu'，或者不指定让函数自动检测
    )
    print(f"--- DML 微调结束，模型保存在: {trained_model_path} ---")

    # --- 使用微调后的模型初始化 KnowRouter ---
    print("\n--- 使用微调后的模型初始化 KnowRouter ---")
    # 假设你的配置加载逻辑在这里
    # 注意：你需要修改你的配置加载方式，或者直接创建一个新的配置字典
    # 关键是 cfg.embedding.model_name 要指向 trained_model_path
    router_cfg_dict = {
        'embedding': {
            'random_seed': 42,
            'model_name': trained_model_path #! 这里使用微调后模型的路径
        },
        'clustering': { # 这里沿用你之前的聚类配置示例
            'use_umap': True,
            'random_seed': 42,
            'umap_params': {'n_neighbors': 15, 'min_dist': 0.1, 'n_components': 5, 'metric': 'cosine'},
            'hdbscan_params': {'min_cluster_size': 5, 'min_samples': None, 'metric': 'euclidean', 'cluster_selection_method': 'eom', 'allow_single_cluster': True}
        }
    }
    # 使用 OmegaConf 创建配置对象 (如果你的 KnowRouter 需要)
    from omegaconf import OmegaConf
    router_cfg = OmegaConf.create(router_cfg_dict)

    # 初始化 KnowRouter
    know_router_dml = KnowRouter(router_cfg)
    print("KnowRouter 初始化完成，使用的是 DML 微调后的嵌入模型。")

    # --- 使用新的 Router 构建路由表 ---
    print("\n--- 使用 DML 优化的嵌入构建路由表 ---")
    know_router_dml.build_route_table(prompts) # 使用原始 prompts 列表构建路由表
    print("路由表构建完成。")
    print(f"聚类数量: {know_router_dml.get_num_clusters()}")
    print(f"离群点数量: {know_router_dml.get_num_outlier()}")

    # --- 示例：路由一个 prompt ---
    if prompts:
        example_prompt = prompts[0]
        cluster_id, confidence = know_router_dml.route_with_confidence(example_prompt)
        print(f"\n示例路由:")
        print(f"Prompt: '{example_prompt}'")
        print(f"路由到 Cluster ID: {cluster_id} (置信度: {confidence:.4f})")

        # 对比：路由其等价改写问题
        if rephrase_prompts:
            example_rephrase = rephrase_prompts[0]
            cluster_id_rephrase, confidence_rephrase = know_router_dml.route_with_confidence(example_rephrase)
            print(f"等价改写 Prompt: '{example_rephrase}'")
            print(f"路由到 Cluster ID: {cluster_id_rephrase} (置信度: {confidence_rephrase:.4f})")
            # 理想情况下，cluster_id_rephrase 应该等于 cluster_id

        # 对比：路由其非等价改写问题
        if locality_prompts:
            example_locality = locality_prompts[0]
            cluster_id_locality, confidence_locality = know_router_dml.route_with_confidence(example_locality)
            print(f"非等价改写 Prompt: '{example_locality}'")
            print(f"路由到 Cluster ID: {cluster_id_locality} (置信度: {confidence_locality:.4f})")
            # 理想情况下，cluster_id_locality 可能不等于 cluster_id，或者置信度较低

    # --- (可选) 保存使用了 DML 模型的 Router ---
    # know_router_dml.save("./router_with_dml_model")
    # print("\n已保存使用 DML 模型的 KnowRouter。")

# ============================================================================
# --- DML 微调代码结束 ---
# ============================================================================
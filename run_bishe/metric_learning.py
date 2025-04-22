# 实现对counterfact和multi-area的度量学习

import os
import sys
from typing import List
# 导入 sentence-transformers 核心库
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments # 用于更精细的训练配置 (可选)
from torch.utils.data import Dataset

sys.path.append(os.getcwd()+'/EasyEdit')
sys.path.append(os.getcwd()+'/EasyEdit/run_bishe')

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



# --- 1. 定义三元组数据集 (你已提供，这里包含进来保持完整) ---
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
        print(f"TripletDataset 初始化完成，包含 {len(self.anchors)} 个三元组样本。")

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        # 返回一个 InputExample 对象，SentenceTransformer 的 TripletLoss 可以直接处理
        return InputExample(texts=[self.anchors[idx], self.positives[idx], self.negatives[idx]])

# --- 2. 数据加载函数 ---
def load_triplet_data_from_file(file_path: str,
                                anchor_key: str = 'prompt',
                                positive_key: str = 'rephrase_prompt',
                                negative_key: str = 'locality_prompt') -> (List[str], List[str], List[str]):
    """
    从 JSON 文件加载数据，并提取用于三元组训练的 锚点、正样本、负样本。

    Args:
        file_path (str): 数据集 JSON 文件的路径。
        anchor_key (str): JSON 对象中代表锚点句子的键名。
        positive_key (str): JSON 对象中代表正样本（泛化）句子的键名。
        negative_key (str): JSON 对象中代表负样本（局部性）句子的键名。

    Returns:
        tuple[List[str], List[str], List[str]]: 包含 anchors, positives, negatives 列表的元组。
                                                如果文件不存在或格式错误，返回空列表。
                                                如果某条记录缺少必要的键，则跳过该记录。
    """
    anchors, positives, negatives = [], [], []
    if not os.path.exists(file_path):
        print(f"错误：数据文件未找到于 {file_path}")
        return anchors, positives, negatives

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"错误：无法解析 JSON 文件 {file_path}")
        return anchors, positives, negatives
    except Exception as e:
        print(f"加载文件时发生未知错误 {file_path}: {e}")
        return anchors, positives, negatives

    print(f"从 {file_path} 加载数据...")
    skipped_count = 0
    # 假设 JSON 文件是一个包含多个字典的列表
    if isinstance(data, list):
        for record in tqdm(data, desc="处理数据记录"):
            anchor = record.get(anchor_key)
            # 处理 'requested_rewrite' 嵌套的情况 (适配 CounterFact 格式)
            if anchor is None and 'requested_rewrite' in record:
                 anchor = record['requested_rewrite'].get(anchor_key)

            positive = record.get(positive_key)
            negative = record.get(negative_key)

            # 确保三个元素都存在且不为空
            if anchor and positive and negative:
                # 特别处理：如果 positive 或 negative 是列表，随机选一个或取第一个
                # 这里简化处理，假设它们都是字符串，或者取列表第一个元素（如果适用）
                if isinstance(positive, list):
                    if not positive: # 跳过空列表
                        skipped_count += 1
                        continue
                    positive = positive[0] # 或者 random.choice(positive)
                if isinstance(negative, list):
                    if not negative: # 跳过空列表
                        skipped_count += 1
                        continue
                    negative = negative[0] # 或者 random.choice(negative)

                # 再次确认取出的值是字符串
                if isinstance(anchor, str) and isinstance(positive, str) and isinstance(negative, str):
                    anchors.append(anchor)
                    positives.append(positive)
                    negatives.append(negative)
                else:
                     skipped_count += 1
            else:
                skipped_count += 1
    else:
        print(f"错误：期望 JSON 文件根元素是列表，但得到的是 {type(data)}")

    print(f"数据加载完成。成功加载 {len(anchors)} 个三元组。跳过 {skipped_count} 条记录（因缺少必要字段或格式问题）。")
    return anchors, positives, negatives

# --- 3. 主训练函数 ---
def train_metric_learning_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    dataset_file_path: str = "data/counterfact_train_triplets.json", # 假设你的训练数据文件名
    output_path: str = "output/finetuned_sbert_model",
    epochs: int = 1,
    batch_size: int = 16, # 根据你的 GPU 显存调整
    triplet_margin: float = 1.0, # Triplet Loss 的 margin
    warmup_steps_ratio: float = 0.1, # 预热步数占总步数的比例
    learning_rate: float = 2e-5, # 学习率
    seed: int = 42 # 随机种子，保证可复现性
):
    """
    使用 SentenceTransformerTrainer 和 Triplet Loss 微调 Sentence Transformer 模型。

    Args:
        model_name (str): 要加载的预训练 Sentence Transformer 模型名称或路径。
        dataset_file_path (str): 包含三元组数据的 JSON 文件路径。
        output_path (str): 微调后模型的保存路径。
        epochs (int): 训练轮数。
        batch_size (int): 训练批次大小。
        triplet_margin (float): Triplet Loss 的 margin 值。
        warmup_steps_ratio (float): 学习率预热步数占总训练步数的比例。
        learning_rate (float): 训练的学习率。
        seed (int): 用于初始化的随机种子。
    """
    print("--- 开始度量学习模型微调 ---")
    print(f"模型: {model_name}")
    print(f"数据集: {dataset_file_path}")
    print(f"输出路径: {output_path}")
    print(f"训练超参数: epochs={epochs}, batch_size={batch_size}, margin={triplet_margin}, lr={learning_rate}, seed={seed}")

    # 设置随机种子
    torch.manual_seed(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)

    # 1. 加载三元组数据
    # 注意：这里的 key 需要根据你的实际 JSON 文件结构调整
    anchors, positives, negatives = load_triplet_data_from_file(
        dataset_file_path,
        anchor_key='prompt', # 或者 'requested_rewrite.prompt' 如果是 CounterFact
        positive_key='rephrase_prompt', # 或者 'paraphrase_prompts'
        negative_key='locality_prompt' # 或者 'neighborhood_prompts'
    )

    if not anchors:
        print("错误：未能加载到任何有效的三元组数据，训练终止。")
        return

    # 2. 创建 TripletDataset 实例
    train_dataset = TripletDataset(anchors, positives, negatives)

    # 3. 加载预训练的 Sentence Transformer 模型
    model = SentenceTransformer(model_name)
    print(f"成功加载预训练模型: {model_name}")

    # 4. 定义 Triplet Loss
    # 可以选择不同的距离度量，默认是 cosine similarity，也可以是 Euclidean distance 等
    # 对于 cosine similarity, margin 通常在 0 到 1 之间
    # 对于 Euclidean distance, margin 通常更大，例如 1 或 5
    train_loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.COSINE, # 或 EUCLIDEAN
        triplet_margin=triplet_margin
    )
    print(f"使用 TripletLoss，距离度量: {train_loss.distance_metric}, margin: {triplet_margin}")

    # 5. 配置训练参数
    # 计算预热步数
    num_training_steps = math.ceil(len(train_dataset) / batch_size) * epochs
    warmup_steps = math.ceil(num_training_steps * warmup_steps_ratio)
    print(f"总训练步数: {num_training_steps}, 预热步数: {warmup_steps}")

    # (可选) 使用 SentenceTransformerTrainingArguments 进行更详细配置
    # args = SentenceTransformerTrainingArguments(
    #     output_dir=output_path,
    #     num_train_epochs=epochs,
    #     per_device_train_batch_size=batch_size,
    #     learning_rate=learning_rate,
    #     warmup_steps=warmup_steps,
    #     seed=seed,
    #     save_strategy="epoch", # 或 "steps"
    #     # save_steps=... # 如果 save_strategy="steps"
    #     logging_dir='./logs', # TensorBoard logs
    #     logging_steps=100, # 每多少步记录一次日志
    # )

    # 6. 初始化 SentenceTransformerTrainer
    # 注意：SentenceTransformerTrainer 是一个较新的 API，
    # 如果你的 sentence-transformers 版本较旧，可能需要使用旧的 model.fit() 方法。
    # 这里我们优先使用 Trainer API。
    # 旧版 model.fit() 示例:
    # train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    # model.fit(train_objectives=[(train_dataloader, train_loss)],
    #           epochs=epochs,
    #           warmup_steps=warmup_steps,
    #           output_path=output_path,
    #           show_progress_bar=True,
    #           optimizer_params={'lr': learning_rate})

    # 使用 Trainer API (推荐)
    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        loss=train_loss,
        # args=args, # 如果使用了 SentenceTransformerTrainingArguments
        # 如果不使用 args，可以直接传递参数:
        batch_size=batch_size,
        num_epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': learning_rate},
        output_path=output_path,
        show_progress_bar=True,
        save_best_model=False, # Trainer 默认不保存最佳模型，会在结束后保存最终模型
        seed=seed
    )
    print("SentenceTransformerTrainer 初始化完成，准备开始训练...")

    # 7. 开始训练
    trainer.train()

    # 8. (可选) 训练完成后，模型已保存在 output_path。也可以手动再保存一次。
    # model.save(output_path) # Trainer 的 train() 方法结束后通常会保存
    print(f"--- 训练完成 ---")
    print(f"微调后的模型已保存到: {output_path}")

# --- 4. 示例用法 ---
if __name__ == "__main__":
    # 设定你的数据集路径和输出路径
    # 重要: 请确保你的数据集文件是 JSON 格式，并且包含对应的 anchor, positive, negative 句子。
    # 你可能需要先对 CounterFact 或 MultiArea 数据集进行预处理，生成这样的 JSON 文件。
    DATASET_FILE = "path/to/your/triplet_data.json" # <--- 请修改为你的数据文件路径
    OUTPUT_MODEL_DIR = "output/my_finetuned_sbert" # <--- 请修改为你希望保存模型的路径

    # 检查数据文件是否存在
    if not os.path.exists(DATASET_FILE):
        print(f"错误：找不到数据集文件 '{DATASET_FILE}'。")
        print("请修改脚本中的 DATASET_FILE 变量，指向包含三元组数据的 JSON 文件。")
        print("该 JSON 文件应为一个列表，其中每个元素是一个字典，包含锚点、正样本和负样本的句子。")
        # 示例数据格式:
        # [
        #   { "prompt": "Anchor sentence 1", "rephrase_prompt": "Positive sentence 1", "locality_prompt": "Negative sentence 1" },
        #   { "prompt": "Anchor sentence 2", "rephrase_prompt": "Positive sentence 2", "locality_prompt": "Negative sentence 2" },
        #   ...
        # ]
        # 注意: load_triplet_data_from_file 函数中的 key 参数需要与你的 JSON 文件匹配。
        sys.exit(1) # 退出程序

    # 创建输出目录（如果不存在）
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

    # 执行训练
    train_metric_learning_model(
        model_name="sentence-transformers/all-MiniLM-L6-v2", # 可以换成其他 SBERT 模型
        dataset_file_path=DATASET_FILE,
        output_path=OUTPUT_MODEL_DIR,
        epochs=3,         # 训练轮数，根据你的数据量和效果调整
        batch_size=32,    # 批处理大小，根据显存调整
        triplet_margin=0.8, # Triplet Loss 的边距，需要调优
        learning_rate=2e-5, # 学习率
        seed=42           # 随机种子
    )

    # 训练完成后，你可以加载微调后的模型进行测试或用于你的知识编辑流程
    print("\n加载微调后的模型进行测试...")
    try:
        finetuned_model = SentenceTransformer(OUTPUT_MODEL_DIR)
        print(f"成功加载微调后的模型从: {OUTPUT_MODEL_DIR}")

        # 可以在这里添加一些简单的测试代码，比如编码几个句子看看效果
        # sentences = ["这是一个测试句子。", "这是另一个句子。", "苹果是一种水果。"]
        # embeddings = finetuned_model.encode(sentences)
        # print("测试编码完成，嵌入向量形状:", embeddings.shape)

    except Exception as e:
        print(f"加载或测试微调模型时出错: {e}")
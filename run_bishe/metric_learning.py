# 实现对counterfact和multi-area的度量学习

import os
import sys
from typing import List, Dict, Optional, Any

from sentence_transformers.evaluation import SentenceEvaluator

from multiarea_dataset import MultiAreaDataset
# 导入 sentence-transformers 核心库
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments # 用于更精细的训练配置 (可选)
import math
from torch.utils.data import Dataset
from sentence_transformers.losses.BatchHardTripletLoss import BatchHardTripletLossDistanceFunction # TripletLoss 相关

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
        if len(self.anchors) == 0:
            print("警告：创建的 TripletDataset 为空！")

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        # 返回一个 InputExample 对象，SentenceTransformer 的 TripletLoss 可以直接处理
        # InputExample 的 texts 列表顺序必须是 [anchor, positive, negative]
        return InputExample(texts=[self.anchors[idx], self.positives[idx], self.negatives[idx]])


def prepare_triplet_data(root_dir: str, dataset_configs: Dict[str, int], seed: int = 42, random_sample: bool = False) -> Optional[TripletDataset]:
    """
    准备用于三元组损失训练的数据集。

    Args:
        root_dir (str): MultiAreaDataset 的根目录。
        dataset_configs (Dict[str, int]): 数据集配置文件和样本数量。
        seed (int, optional): 随机种子 (仅当 random_sample=True 时使用). Defaults to 42.
        random_sample (bool, optional): 是否随机采样. Defaults to False.

    Returns:
        Optional[TripletDataset]: 成功则返回 TripletDataset 实例，否则返回 None。
    """
    print(f"开始准备 MultiArea 数据集，根目录: {root_dir}")
    if not os.path.isdir(root_dir):
        print(f"错误：指定的 MultiAreaDataset 根目录不存在: {root_dir}")
        return None

    try:
        multiarea_dataset = MultiAreaDataset(
            root_dir=root_dir,
            dataset_configs=dataset_configs,
            seed=seed,
            random_sample=random_sample
        )
        print("MultiAreaDataset 实例创建成功。")

        # 提取数据
        prompts, rephrase_prompts, _, _, locality_inputs, _ = multiarea_dataset.to_edit_dataset()

        # 提取 locality prompts
        if 'neighborhood' in locality_inputs and 'prompt' in locality_inputs['neighborhood']:
             locality_prompts = locality_inputs['neighborhood']['prompt']
             print(f"成功提取到 {len(prompts)} 个编辑问题 (anchors)。")
             print(f"成功提取到 {len(rephrase_prompts)} 个等价改写问题 (positives)。")
             print(f"成功提取到 {len(locality_prompts)} 个局部性问题 (negatives)。")

             # 数据量检查和对齐
             min_len = min(len(prompts), len(rephrase_prompts), len(locality_prompts))
             if not (len(prompts) == len(rephrase_prompts) == len(locality_prompts)):
                 print(f"警告：提取的 Anchors ({len(prompts)}), Positives ({len(rephrase_prompts)}), Negatives ({len(locality_prompts)}) 数量不一致！")
                 print(f"将使用前 {min_len} 条数据构建三元组。")
                 prompts = prompts[:min_len]
                 rephrase_prompts = rephrase_prompts[:min_len]
                 locality_prompts = locality_prompts[:min_len]

             if min_len == 0:
                 print("错误：没有有效的三元组数据可供使用。")
                 return None

             # 创建 TripletDataset 实例
             train_dataset = TripletDataset(anchors=prompts, positives=rephrase_prompts, negatives=locality_prompts)
             return train_dataset

        else:
            print("错误：无法从 locality_inputs 中提取 'neighborhood' -> 'prompt'。请检查 MultiAreaDataset 的输出结构。")
            print("locality_inputs 内容:", locality_inputs) # 打印内容以帮助调试
            return None

    except Exception as e:
        print(f"准备数据时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_base_model(model_name_or_path: str) -> Optional[SentenceTransformer]:
    """
    加载预训练的 SentenceTransformer 模型。

    Args:
        model_name_or_path (str): 模型的名称 (如 'all-MiniLM-L6-v2') 或本地路径。

    Returns:
        Optional[SentenceTransformer]: 成功则返回模型实例，否则返回 None。
    """
    print(f"正在加载基础模型: {model_name_or_path}...")
    try:
        model = SentenceTransformer(model_name_or_path)
        print("基础模型加载成功。")
        return model
    except Exception as e:
        print(f"加载模型 {model_name_or_path} 时出错: {e}")
        print("请确保已安装 sentence-transformers 库并且网络连接正常，或者模型路径有效。")
        return None


def define_triplet_loss(model: SentenceTransformer,
                        distance_metric: BatchHardTripletLossDistanceFunction = BatchHardTripletLossDistanceFunction.COSINE,
                        margin: float = 0.5) -> losses.TripletLoss:
    """
    定义 TripletLoss 损失函数。

    Args:
        model (SentenceTransformer): 需要使用损失函数的模型实例。
        distance_metric (BatchHardTripletLossDistanceFunction, optional): 距离度量. Defaults to BatchHardTripletLossDistanceFunction.COSINE.
        margin (float, optional): Triplet margin. Defaults to 0.5 (适合 COSINE).

    Returns:
        losses.TripletLoss: 配置好的 TripletLoss 实例。
    """
    train_loss = losses.TripletLoss(model=model, distance_metric=distance_metric, triplet_margin=margin)
    print(f"使用 TripletLoss，距离度量: {distance_metric}, Margin: {margin}")
    return train_loss


def configure_training_args(output_dir: str,
                            num_epochs: int,
                            train_batch_size: int,
                            learning_rate: float,
                            warmup_steps: int,
                            weight_decay: float = 0.01,
                            logging_steps: int = 50,
                            save_strategy: str = "epoch",
                            save_total_limit: int = 2,
                            evaluation_strategy: str = "no",
                            eval_steps: Optional[int] = None,
                            load_best_at_end: bool = False,
                            report_to: Optional[List[str]] = None) -> SentenceTransformerTrainingArguments:
    """
    配置训练参数。

    Args:
        output_dir (str): 模型 checkpoints 和最终模型的保存路径。
        num_epochs (int): 训练轮数。
        train_batch_size (int): 训练批次大小。
        learning_rate (float): 学习率。
        warmup_steps (int): 预热步数。
        weight_decay (float, optional): 权重衰减. Defaults to 0.01.
        logging_steps (int, optional): 日志记录步数. Defaults to 50.
        save_strategy (str, optional): 保存策略 ('no', 'epoch', 'steps'). Defaults to "epoch".
        save_total_limit (int, optional): 最多保留 checkpoint 数量. Defaults to 2.
        evaluation_strategy (str, optional): 评估策略 ('no', 'epoch', 'steps'). Defaults to "no".
        eval_steps (Optional[int], optional): 如果评估策略是 'steps', 指定评估步数. Defaults to None.
        load_best_at_end (bool, optional): 训练结束后是否加载最佳模型 (需要评估集). Defaults to False.
        report_to (Optional[List[str]], optional): 日志报告目标 (如 ["tensorboard"]). Defaults to None.


    Returns:
        SentenceTransformerTrainingArguments: 配置好的训练参数对象。
    """
    if report_to is None:
        report_to = [] # 默认为空列表

    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir=os.path.join(output_dir, 'logs'), # 日志放在输出目录下
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_steps=eval_steps if save_strategy == "steps" else 500, # 如果按步保存，默认500，可调整
        save_total_limit=save_total_limit,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps if evaluation_strategy in ["steps", "epoch"] else None, # 只有在需要评估时才设置eval_steps
        load_best_model_at_end=load_best_at_end and evaluation_strategy != "no", # 只有评估时才能加载最优
        report_to=report_to,
        # Dataloader 参数 (可选，按需调整)
        # dataloader_num_workers=4, # 增加数据加载进程数可能加速，但也可能引入问题
        # dataloader_pin_memory=True, # 通常建议开启
    )
    print(f"训练参数配置完成，输出目录: {output_dir}")
    return training_args


def initialize_trainer(model: SentenceTransformer,
                       args: SentenceTransformerTrainingArguments,
                       train_dataset: Dataset,
                       loss_func: Any, # losses._LossFunction 类
                       eval_dataset: Optional[Dataset] = None,
                       evaluator: Optional[SentenceEvaluator] = None) -> SentenceTransformerTrainer:
    """
    初始化 SentenceTransformerTrainer。

    Args:
        model (SentenceTransformer): 要训练的模型。
        args (SentenceTransformerTrainingArguments): 训练参数。
        train_dataset (Dataset): 训练数据集。
        loss_func (Any): 损失函数实例。
        eval_dataset (Optional[Dataset], optional): 评估数据集. Defaults to None.
        evaluator (Optional[SentenceEvaluator], optional): Sentence Transformer 评估器. Defaults to None.

    Returns:
        SentenceTransformerTrainer: 初始化后的训练器实例。
    """
    print("初始化 SentenceTransformerTrainer...")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss_func,
        eval_dataset=eval_dataset,
        evaluator=evaluator # Sentence Transformer 特有的评估器接口
    )
    print("训练器初始化完成。")
    return trainer


def run_training(trainer: SentenceTransformerTrainer) -> bool:
    """
    执行模型训练。

    Args:
        trainer (SentenceTransformerTrainer): 已初始化的训练器。

    Returns:
        bool: 训练是否成功完成。
    """
    print("开始训练...")
    try:
        trainer.train()
        print("训练成功完成。")
        return True
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_final_model(model: SentenceTransformer, save_path: str):
    """
    保存最终的模型。

    Args:
        model (SentenceTransformer): 训练完成的模型。
        save_path (str): 保存模型的路径。
    """
    print(f"正在将最终微调后的模型保存到: {save_path}")
    try:
        os.makedirs(save_path, exist_ok=True) # 确保目录存在
        model.save(save_path)
        print("最终模型保存成功。")
    except Exception as e:
        print(f"保存最终模型时出错: {e}")


# --- 主流程函数 ---
def finetune_sentence_transformer_for_knowledge_editing(
    # 数据相关参数
    data_root_dir: str,
    data_configs: Dict[str, int],
    # 模型相关参数
    base_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
    output_model_dir: str = "./output/finetuned_sbert_multi_area_triplet",
    final_model_subdir: str = "final_model", # 最终模型保存在 output_model_dir 下的子目录名
    # 损失函数相关参数
    distance_metric_name: str = "COSINE", # "COSINE" 或 "EUCLIDEAN" 或 "MANHATTAN"
    triplet_margin: float = 0.5,
    # 训练超参数
    num_train_epochs: int = 2,
    train_batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1, # 使用比例计算 warmup steps
    # 其他训练参数
    logging_steps: int = 50,
    save_strategy: str = "epoch",
    save_total_limit: int = 1, # 减少 checkpoint 数量
    use_tensorboard: bool = True # 是否启用 TensorBoard
    ) -> Optional[str]:
    """
    执行 Sentence Transformer 模型微调的完整流程，用于知识编辑任务中的度量学习。

    Args:
        data_root_dir (str): MultiArea 数据集的根目录。
        data_configs (Dict[str, int]): MultiArea 数据集配置。
        base_model_name (str): 基础预训练 Sentence Transformer 模型名称或路径。
        output_model_dir (str): 训练过程和最终模型的输出根目录。
        final_model_subdir (str): 最终模型在 output_model_dir 下的子目录名。
        distance_metric_name (str): TripletLoss 使用的距离度量名称 ("COSINE", "EUCLIDEAN", "MANHATTAN")。
        triplet_margin (float): TripletLoss 的 margin 值。
        num_train_epochs (int): 训练轮数。
        train_batch_size (int): 训练批次大小。
        learning_rate (float): 学习率。
        warmup_ratio (float): 预热步数占总训练步数的比例。
        logging_steps (int): 日志记录步频。
        save_strategy (str): Checkpoint 保存策略。
        save_total_limit (int): 最多保留的 Checkpoint 数量。
        use_tensorboard (bool): 是否将日志写入 TensorBoard。

    Returns:
        Optional[str]: 如果成功，返回最终保存的模型路径；否则返回 None。
    """

    print("="*30)
    print("开始 Sentence Transformer 微调流程")
    print("="*30)

    # 1. 准备数据
    train_dataset = prepare_triplet_data(root_dir=data_root_dir, dataset_configs=data_configs)
    if train_dataset is None or len(train_dataset) == 0:
        print("错误：数据准备失败或数据集为空，终止流程。")
        return None

    # 2. 加载基础模型
    model = load_base_model(base_model_name)
    if model is None:
        print("错误：基础模型加载失败，终止流程。")
        return None

    # 3. 定义损失函数
    try:
        if distance_metric_name.upper() == "COSINE":
            distance_metric = losses.TripletDistanceMetric.COSINE
        elif distance_metric_name.upper() == "EUCLIDEAN":
            distance_metric = losses.TripletDistanceMetric.EUCLIDEAN
        elif distance_metric_name.upper() == "MANHATTAN":
            distance_metric = losses.TripletDistanceMetric.MANHATTAN
        else:
            print(f"警告：未知的距离度量 '{distance_metric_name}'，将使用默认的 COSINE。")
            distance_metric = losses.TripletDistanceMetric.COSINE
        loss_func = define_triplet_loss(model=model, distance_metric=distance_metric, margin=triplet_margin)
    except Exception as e:
        print(f"定义损失函数时出错: {e}")
        return None

    # 4. 配置训练参数
    # 计算总训练步数和预热步数
    steps_per_epoch = math.ceil(len(train_dataset) / train_batch_size)
    total_steps = steps_per_epoch * num_train_epochs
    warmup_steps = math.ceil(total_steps * warmup_ratio)
    print(f"总训练步数: {total_steps}, 预热步数: {warmup_steps}")

    report_to = ["tensorboard"] if use_tensorboard else []

    training_args = configure_training_args(
        output_dir=output_model_dir,
        num_epochs=num_train_epochs,
        train_batch_size=train_batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        report_to=report_to,
        # 暂时不进行评估
        evaluation_strategy="no",
        load_best_at_end=False
    )

    # 5. 初始化训练器
    # 注意：TripletLoss 不需要特殊的 evaluator，除非你想在验证集上评估 triplet loss 本身或相关指标
    trainer = initialize_trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss_func=loss_func
    )

    # 6. 执行训练
    training_successful = run_training(trainer)

    # 7. 保存最终模型
    if training_successful:
        # trainer.save_model() # Trainer 也可以保存，但我们用 model.save() 更明确
        final_model_path = os.path.join(output_model_dir, final_model_subdir)
        save_final_model(model, final_model_path)
        print("="*30)
        print("微调流程成功结束。")
        print(f"最终模型已保存至: {final_model_path}")
        print("="*30)
        return final_model_path
    else:
        print("="*30)
        print("微调流程因训练失败而终止。")
        print("="*30)
        return None


# --- 主程序入口 ---
if __name__ == "__main__":
    # 定义数据集配置
    dataset_configs = {
        'business_industry.json': 50,
        'human_scientist.json': 50,
        'event_sport.json': 50,
        'geography_forest.json': 50,
        'places_landmark.json': 50
    }

    multi_area_root_dir = 'EasyEdit/data/output_meta_llama_3_8b_instruct'

    # 定义输出目录
    output_directory = "./output/finetuned_sbert_multiarea_triplet_functional"

    # 调用主流程函数
    final_model_path = finetune_sentence_transformer_for_knowledge_editing(
        data_root_dir=multi_area_root_dir,
        data_configs=dataset_configs,
        output_model_dir=output_directory,
        # 可以按需修改其他参数:
        # base_model_name='paraphrase-multilingual-MiniLM-L12-v2', # 换个模型试试？
        num_train_epochs=3, # 增加训练轮数
        train_batch_size=8,  # 减小 batch size 适应显存
        learning_rate=1e-5, # 降低学习率
        triplet_margin=0.8, # 调整 margin
        use_tensorboard=True
    )
# 主要是为了提取大模型的某一层输出

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

prompts, rephrase_prompts, target_new, subjects, locality_inputs, _ = multiarea_dataset.to_edit_dataset()

# --- 1. 加载Llama模型和分词器 ---

MODEL_NAME_OR_PATH = "meta-llama/Llama-3.2-1B-Instruct" # <--- 在这里修改为你实际的1B模型路径或ID
OUTPUT_EMBEDDING_FILE = "llama_layer_embeddings.pt" # 定义保存嵌入的文件名
TARGET_LAYER_INDEX = 6
BATCH_SIZE = 1

# 确定设备 (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
# Llama 系列通常没有专门的 pad token，但我们可以用 eos_token 作为 pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer 没有 pad_token, 使用 eos_token 作为 pad_token。")

# 加载模型
# 尝试使用 bfloat16 和 device_map='auto' 来优化加载和显存使用
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
    )
    print(f"模型 {MODEL_NAME_OR_PATH} 加载成功 ")
    print(f'模型隐藏状态维度{model.config.hidden_size}')
except Exception as e:
    print(f"尝试用 bfloat16 和 device_map='auto' 加载失败: {e}")
    print("尝试使用默认方式加载...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_OR_PATH)
    model.to(device) # 如果没有 device_map='auto', 手动移动到设备
    print(f"模型 {MODEL_NAME_OR_PATH} 加载成功 (默认方式, device: {device})。")

# 将模型设置为评估模式
model.eval()

# --- 2. 提取所有 prompt 在指定层的输出 ---

all_embeddings = [] # 用于存储所有提取到的嵌入向量
print(f"开始提取 {len(prompts)} 条 prompts 在第 {TARGET_LAYER_INDEX} 层的隐藏状态...")

# 使用 torch.no_grad() 避免计算梯度，节省显存和计算资源
with torch.no_grad():
    # 分批处理 prompts
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="提取嵌入中"):
        # 获取当前批次的 prompts
        batch_prompts = prompts[i : i + BATCH_SIZE]

        # 对当前批次进行分词
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,        # 对批次内的序列进行填充，使其长度一致
            truncation=True,     # 如果序列超过模型最大长度，则截断
            max_length=512       # 可以根据需要调整最大长度
        ).to(model.device if hasattr(model, 'device') else device) # 将输入移动到模型所在的设备

        # 模型推理，获取隐藏状态
        outputs = model(**inputs, output_hidden_states=True)

        # 提取目标层的隐藏状态
        # outputs.hidden_states 是一个元组，包含了 embedding 层 + 所有 transformer 层的输出
        # Llama3-8B 有 32 层 transformer layer, hidden_states 长度为 33 (1 embedding + 32 layers)
        # TARGET_LAYER_INDEX = -1 指的是最后一层 transformer layer, 对应元组索引 -1
        # 如果 TARGET_LAYER_INDEX = 15 (第16层), 对应元组索引 16 (假设第一项是embedding)
        # **注意**: 索引方式可能随 transformers 版本或模型结构变化, TARGET_LAYER_INDEX=-1 通常最稳妥代表最后一层
        # 如果你需要精确的第k层，请检查 hidden_states 的长度和内容确认
        target_hidden_states = outputs.hidden_states[TARGET_LAYER_INDEX]
        # 形状: (batch_size, sequence_length, hidden_dim)

        # --- 平均池化 (Mean Pooling) ---
        # 获取 attention mask, 形状: (batch_size, sequence_length)
        attention_mask = inputs['attention_mask']
        # 将 attention_mask 扩展维度以匹配 hidden_states: (batch_size, sequence_length, 1)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(target_hidden_states.size()).float()
        # 计算每个序列的有效 token 数量 (避免除以0)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        # 将隐藏状态中 padding 部分置零，然后求和
        sum_embeddings = torch.sum(target_hidden_states * input_mask_expanded, 1)
        # 计算平均值
        pooled_embeddings = sum_embeddings / sum_mask
        # 形状: (batch_size, hidden_dim)

        # 收集结果 (移动到 CPU 并 detach)
        all_embeddings.append(pooled_embeddings.cpu().detach())

# 将所有批次的嵌入向量拼接成一个大张量
stacked_embeddings = torch.cat(all_embeddings, dim=0)
print(f"嵌入提取完成，最终张量形状: {stacked_embeddings.shape}") # 应为 (num_prompts, hidden_dim)

# --- 3. 将输出存储到本地文件 ---

try:
    torch.save(stacked_embeddings, OUTPUT_EMBEDDING_FILE)
    print(f"嵌入向量已成功保存到: {OUTPUT_EMBEDDING_FILE}")
except Exception as e:
    print(f"保存嵌入向量时出错: {e}")

# --- 4. 完成从本地文件读取的逻辑 ---

def load_llama_embeddings(filepath=OUTPUT_EMBEDDING_FILE):
    """
    从本地文件加载预先计算好的 Llama 层嵌入向量。

    Args:
        filepath (str): 存储嵌入向量的文件路径。

    Returns:
        torch.Tensor or None: 加载的嵌入向量张量，如果文件不存在或加载失败则返回 None。
    """
    if not os.path.exists(filepath):
        print(f"错误：嵌入文件 {filepath} 不存在。")
        return None
    try:
        embeddings = torch.load(filepath)
        print(f"从 {filepath} 加载嵌入向量成功，形状: {embeddings.shape}")
        return embeddings
    except Exception as e:
        print(f"加载嵌入向量时出错: {e}")
        return None

# --- 示例：加载刚才保存的嵌入 ---
loaded_embeddings = load_llama_embeddings()

# 你现在可以在后续的代码中使用 loaded_embeddings (或 stacked_embeddings)
# 例如，用于你的聚类算法输入
if loaded_embeddings is not None:
    print("成功加载嵌入，可以用于后续步骤。")
    # 示例：可以将其转换为 NumPy 数组用于 scikit-learn 等库
    # import numpy as np
    # embeddings_np = loaded_embeddings.numpy()
    # print(f"转换为 NumPy 数组形状: {embeddings_np.shape}")

    # 接下来你可以用 loaded_embeddings (torch.Tensor) 或 embeddings_np (numpy.ndarray)
    # 替换掉原来使用 SBERT 嵌入的地方，进行聚类等操作。
    # 例如： run_clustering(loaded_embeddings)
else:
    print("加载嵌入失败，请检查之前的步骤。")
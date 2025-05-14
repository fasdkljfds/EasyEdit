# 用于测试语义路由 (boundary_embedding) 对不同类型提示进行分类的能力
# 特别是针对 CounterFact, ZsRE, MultiArea 数据集

import sys
import os

# 调整这些路径，如果您的项目结构不同
# 确保 'EasyEdit' 和 'EasyEdit/run_bishe' 在 PYTHONPATH 中
# 或者您从 'EasyEdit' 的父目录运行此脚本
# 为简单起见，假设 EasyEdit 是一个顶级可导入模块，或者当前工作目录已正确设置。
# 如果脚本与 EasyEdit 目录在同一级，可能需要 sys.path.append(os.path.join(os.getcwd(), 'EasyEdit'))
# 假设脚本在项目根目录，EasyEdit 是子目录:
current_dir = os.getcwd()
project_root = current_dir  # 或者根据您的项目结构调整
easyedit_path = os.path.join(project_root, "EasyEdit")
if easyedit_path not in sys.path:
    sys.path.append(easyedit_path)
# 如果 run_bishe 在 EasyEdit 下，可能不需要额外添加
# run_bishe_path = os.path.join(easyedit_path, "run_bishe")
# if run_bishe_path not in sys.path:
#     sys.path.append(run_bishe_path)


from omegaconf import DictConfig, OmegaConf
from scipy.spatial.distance import euclidean
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import hdbscan  # 虽然这里不直接用HDBSCAN，但您的项目依赖它
import numpy as np
import torch
import umap.umap_ as umap  # 同上
from numpy import ndarray
from tqdm import tqdm  # 用于显示进度条


@dataclass
class EmbeddingConfig:
    random_seed: int
    model_name: str  # 可以是HuggingFace模型名称或本地模型路径


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
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.random_seed)

        # 对于SBERT推理，通常不需要下面这些，且可能降低速度
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

        self.model = SentenceTransformer(cfg.model_name)
        if torch.cuda.is_available():
            self.model.to('cuda')
        print(f"SBERT 模型 ({cfg.model_name}) 已加载到: {self.model.device}")

    def to_embeddings(self, sentences: List[str], batch_size=32) -> ndarray:
        """
        将句子列表转换为嵌入向量。
        处理 None 或空字符串，为其返回 NaN 向量。
        Args:
            sentences (List[str]): 需要转换的句子列表
            batch_size (int): 编码时使用的批处理大小
        """
        # 在编码前过滤掉 None 或空字符串，因为SBERT可能会出错或给出不良嵌入
        valid_sentences_with_indices = [(i, s) for i, s in enumerate(sentences) if s and s.strip()]
        original_indices = [item[0] for item in valid_sentences_with_indices]
        sentences_to_encode = [item[1] for item in valid_sentences_with_indices]

        if not sentences_to_encode:
            # 如果所有输入都无效，则返回形状正确的零向量或NaN向量
            try:
                emb_dim = self.model.get_sentence_embedding_dimension()
            except:  # 备用方案
                # print("警告: 无法获取嵌入维度，默认为384。如果此维度错误，请注意处理NaN。") # 减少重复打印
                emb_dim = self.model.tokenizer.model_max_length if hasattr(self.model, 'tokenizer') and hasattr(self.model.tokenizer, 'model_max_length') else 384
                if emb_dim is None: emb_dim = 384  # 再次检查
            return np.full((len(sentences), emb_dim), np.nan)

        embeddings_encoded = self.model.encode(sentences_to_encode,
                                               batch_size=batch_size,
                                               show_progress_bar=False,  # 长列表时可设为True
                                               convert_to_numpy=True)

        # 重建完整的嵌入列表，为无效输入填充NaN
        if embeddings_encoded.ndim == 1 and len(sentences_to_encode) == 1:  # 单个句子编码可能返回1D数组
            embeddings_encoded = embeddings_encoded.reshape(1, -1)

        if embeddings_encoded.shape[0] == 0:  # 如果编码后结果为空 (所有输入都无效过滤后)
            emb_dim = self.model.get_sentence_embedding_dimension() if hasattr(self.model, 'get_sentence_embedding_dimension') else 384
            return np.full((len(sentences), emb_dim), np.nan)

        full_embeddings = np.full((len(sentences), embeddings_encoded.shape[1]), np.nan, dtype=float)
        for i, original_idx in enumerate(original_indices):
            full_embeddings[original_idx] = embeddings_encoded[i]

        return full_embeddings

    def cosine_similarity_pair(self, emb1: ndarray, emb2: ndarray) -> float:
        """计算两个单个嵌入向量之间的余弦相似度。"""
        if np.all(np.isnan(emb1)) or np.all(np.isnan(emb2)):
            return 0.0  # 或者 np.nan，取决于您希望如何处理
        # 确保它们是2D的以便cosine_similarity函数使用
        return cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0, 0]


def prepare_multiarea_data(base_data_dir_for_multiarea: str, ds_size_per_file: int = 50):
    # 假设 multiarea_dataset.py 在 EasyEdit/run_bishe/
    # 如有必要，调整导入路径
    try:
        from run_bishe.multiarea_dataset import MultiAreaDataset  # 假设运行目录或PYTHONPATH设置正确
    except ImportError:
        print("错误：无法导入 MultiAreaDataset。请确保路径正确或脚本从项目根目录运行。")
        print(f"当前Python搜索路径: {sys.path}")
        raise

    ds_size_per_file = int(ds_size_per_file / 5)
    dataset_configs = {
        'business_industry.json': ds_size_per_file,
        'human_scientist.json': ds_size_per_file,
        'event_sport.json': ds_size_per_file,
        'geography_forest.json': ds_size_per_file,
        'places_landmark.json': ds_size_per_file
    }

    multiarea_dataset = MultiAreaDataset(
        root_dir=base_data_dir_for_multiarea,
        dataset_configs=dataset_configs,
        seed=42,
        random_sample=False  # 为了获得一致的数据，直到ds_size_per_file
    )

    prompts, rephrase_prompts, _, _, locality_inputs, _ = multiarea_dataset.to_edit_dataset()
    locality_prompts = locality_inputs['neighborhood']['prompt']
    # print(f"MultiArea: 加载了 {len(prompts)} 条编辑目标问题。")
    return prompts, locality_prompts, rephrase_prompts


def prepare_zsre_data(data_filepath: str, ds_size: int = None):
    import json
    with open(data_filepath, 'r', encoding='utf-8') as f:
        edit_data_full = json.load(f)

    if ds_size is not None and ds_size < len(edit_data_full):
        edit_data = edit_data_full[:ds_size]
    else:
        edit_data = edit_data_full

    prompts = [data.get('src') for data in edit_data]
    rephrase_prompts = [data.get('rephrase') for data in edit_data]
    locality_prompts = [data.get('loc') for data in edit_data]

    # print(f"ZsRE: 加载了 {len(prompts)} 条编辑目标问题。")
    return prompts, locality_prompts, rephrase_prompts


def prepare_counterfact_data(data_filepath: str, ds_size: int = None):
    # 简化版CounterFact加载器，专注于路由测试所需的提示
    try:
        from easyeditor import KnowEditDataset  # EasyEdit可能需要添加到PYTHONPATH
    except ImportError:
        print("错误：无法导入 KnowEditDataset。请确保EasyEdit在PYTHONPATH中。")
        raise

    # 先加载全部数据，KnowEditDataset内部如果有size参数会打乱
    datas_full = KnowEditDataset(data_filepath, size=None)
    if ds_size is not None and ds_size < len(datas_full):
        # 为了复现性，我们取前ds_size个
        datas = [datas_full[i] for i in range(ds_size)]
    else:
        datas = datas_full

    prompts = [data['prompt'] for data in datas]

    # 对于CounterFact:
    # x_gen (泛化问题) 通常来自 'portability_s' (Subject Aliasing)
    # x_loc (局部性问题) 通常来自 'locality_rs' (Relation Specificity)
    rephrase_prompts = []
    for data in datas:
        sa_prompts_list = data.get('portability_s', [])  # portability_s 是个列表
        if sa_prompts_list and len(sa_prompts_list) > 0 and sa_prompts_list[0].get("prompt"):
            rephrase_prompts.append(sa_prompts_list[0]["prompt"])  # 取第一个作为代表
        else:
            rephrase_prompts.append(None)  # 或者一个占位符如 ""

    locality_prompts = []
    for data in datas:
        rs_prompts_list = data.get('locality_rs', [])  # locality_rs 是个列表
        if rs_prompts_list and len(rs_prompts_list) > 0 and rs_prompts_list[0].get("prompt"):
            locality_prompts.append(rs_prompts_list[0]["prompt"])  # 取第一个作为代表
        else:
            locality_prompts.append(None)

    # print(f"CounterFact: 加载了 {len(prompts)} 条编辑目标问题。")
    return prompts, locality_prompts, rephrase_prompts


def load_embedding_model(random_seed: int, model_path: str) -> Embedding:
    """加载SBERT嵌入模型。"""
    embedding_model_instance = Embedding(
        EmbeddingConfig(
            random_seed=random_seed,
            model_name=model_path
        )
    )
    return embedding_model_instance


def test_semantic_router(
        embedding_model_instance: Embedding,
        edited_prompts_as_reference: List[str],  # 定义“已编辑知识边界”的 x_e 集合
        prompts_to_test_target: List[str],  # 应该被分类为True的原始 x_e
        rephrases_to_test_generalization: List[str],  # 应该被分类为True的 x_gen
        localities_to_test_locality: List[str],  # 应该被分类为False的 x_loc
        similarity_threshold: float
) -> Dict[str, float]:
    """
    测试语义路由。
    """
    # print(f"正在为 {len(edited_prompts_as_reference)} 条“边界参考问题(edited_prompts)”计算嵌入向量...")
    # 为所有已知的“边界参考问题”预计算嵌入，以提高效率
    # 过滤掉None或空字符串，因为它们在to_embeddings中会变成NaN，后续比较中应跳过
    valid_edited_prompts = [p for p in edited_prompts_as_reference if p and p.strip()]
    if not valid_edited_prompts:  # 如果所有参考问题都无效
        print("警告: 所有 'edited_prompts_as_reference' 均无效，无法进行测试。")
        return {
            "编辑目标问题准确率(x_e vs E)": 0,
            "泛化问题准确率(x_gen vs E)": 0,
            "局部性问题准确率(x_loc vs E)": 0,
            "总体准确率": 0,
        }
    edited_embeddings_reference = embedding_model_instance.to_embeddings(valid_edited_prompts)

    results = {
        "prompt_target_correct": 0, "prompt_target_total": 0,
        "rephrase_correct": 0, "rephrase_total": 0,
        "locality_correct": 0, "locality_total": 0
    }

    # 测试 1: 原始编辑目标问题 (x_e) 应该被识别为“边界内”
    # print(f"正在测试 {len(prompts_to_test_target)} 条原始编辑目标问题 (x_e)...")
    target_embeddings = embedding_model_instance.to_embeddings(prompts_to_test_target)
    for i in tqdm(range(len(prompts_to_test_target)), desc="测试x_e", leave=False):
        current_prompt_text = prompts_to_test_target[i]
        if not current_prompt_text or not current_prompt_text.strip() or np.all(np.isnan(target_embeddings[i])):
            continue  # 跳过空或无效的提示

        emb_current = target_embeddings[i]
        decision = False
        for emb_edited_ref in edited_embeddings_reference:
            if np.all(np.isnan(emb_edited_ref)): continue
            similarity = embedding_model_instance.cosine_similarity_pair(emb_current, emb_edited_ref)
            if similarity > similarity_threshold:
                decision = True
                break

        if decision:  # 期望为 True
            results["prompt_target_correct"] += 1
        results["prompt_target_total"] += 1

    # 测试 2: 泛化问题 (x_gen) 应该被识别为“边界内”
    # print(f"正在测试 {len(rephrases_to_test_generalization)} 条泛化问题 (x_gen)...")
    rephrase_embeddings = embedding_model_instance.to_embeddings(rephrases_to_test_generalization)
    for i in tqdm(range(len(rephrases_to_test_generalization)), desc="测试x_gen", leave=False):
        rephrase_prompt_text = rephrases_to_test_generalization[i]
        if not rephrase_prompt_text or not rephrase_prompt_text.strip() or np.all(np.isnan(rephrase_embeddings[i])):
            continue

        emb_rephrase = rephrase_embeddings[i]
        decision = False
        for emb_edited_ref in edited_embeddings_reference:
            if np.all(np.isnan(emb_edited_ref)): continue
            similarity = embedding_model_instance.cosine_similarity_pair(emb_rephrase, emb_edited_ref)
            if similarity > similarity_threshold:
                decision = True
                break

        if decision:  # 期望为 True
            results["rephrase_correct"] += 1
        results["rephrase_total"] += 1

    # 测试 3: 局部性问题 (x_loc) 应该被识别为“边界外”
    # print(f"正在测试 {len(localities_to_test_locality)} 条局部性问题 (x_loc)...")
    locality_embeddings = embedding_model_instance.to_embeddings(localities_to_test_locality)
    for i in tqdm(range(len(localities_to_test_locality)), desc="测试x_loc", leave=False):
        locality_prompt_text = localities_to_test_locality[i]
        if not locality_prompt_text or not locality_prompt_text.strip() or np.all(np.isnan(locality_embeddings[i])):
            continue

        emb_locality = locality_embeddings[i]
        decision = False
        for emb_edited_ref in edited_embeddings_reference:
            if np.all(np.isnan(emb_edited_ref)): continue
            similarity = embedding_model_instance.cosine_similarity_pair(emb_locality, emb_edited_ref)
            if similarity > similarity_threshold:
                decision = True
                break

        if not decision:  # 期望为 False
            results["locality_correct"] += 1
        results["locality_total"] += 1

    accuracies = {
        "编辑目标问题准确率(x_e vs E)": (results["prompt_target_correct"] / results["prompt_target_total"]) if results["prompt_target_total"] > 0 else 0,
        "泛化问题准确率(x_gen vs E)": (results["rephrase_correct"] / results["rephrase_total"]) if results["rephrase_total"] > 0 else 0,
        "局部性问题准确率(x_loc vs E)": (results["locality_correct"] / results["locality_total"]) if results["locality_total"] > 0 else 0,
    }

    total_correct = results["prompt_target_correct"] + results["rephrase_correct"] + results["locality_correct"]
    total_overall = results["prompt_target_total"] + results["rephrase_total"] + results["locality_total"]
    accuracies["总体准确率"] = (total_correct / total_overall) if total_overall > 0 else 0

    return accuracies


# --- 预定义要测试的模型和数据集列表 ---
# !!! 重要: 如果您的微调模型不是HuggingFace上的标准名称，请确保路径正确 !!!
# 例如: 'path/to/your/fine_tuned_model_variant_1'
MODEL_LIST = [
    'BAAI/bge-m3',
    'sentence-transformers/all-MiniLM-L6-v2',
    'sentence-transformers/all-mpnet-base-v2',
    './finetuned_sbert_triplet/final_model_0.9',
]

DATASET_LIST = [
    'counterfact',
    'zsre',
    'multiarea'
]

if __name__ == '__main__':
    # --- 全局配置 ---
    RANDOM_SEED = 42
    SIMILARITY_THRESHOLD = 0.8  # 您论文中设定的 tau_pos 相似度阈值
    DATA_SIZE = 200  # 每个数据集测试的样本数量 (对于multiarea是每个文件)。设为 None 使用全部数据。
    SHOW_MISCLASSIFIED_EXAMPLES_MAIN = False  # 是否显示错误分类示例，循环中可能信息过多

    # --- 数据路径配置 (如果需要请调整) ---
    if project_root == "":  # 如果未自动获取项目根目录
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # 尝试从脚本位置推断
        print(f"自动推断项目根目录为: {project_root}")
        if "EasyEdit" not in os.listdir(project_root):  # 简单检查
            project_root = input("请输入项目根目录 (例如 /path/to/your_project): ")

    base_data_dir = os.path.join(project_root, "EasyEdit", "data")
    if not os.path.isdir(base_data_dir):
        print(f"错误: 基础数据目录 {base_data_dir} 不存在。请检查路径。")
        sys.exit(1)

    multiarea_data_dir_for_script = os.path.join(base_data_dir, 'output_meta_llama_3_8b_instruct')
    zsre_data_filepath = os.path.join(base_data_dir, 'wise', 'ZsRE', 'zsre_mend_edit.json')
    # CounterFact 我们通常用训练集来模拟“已知编辑”，或用验证集独立评估。
    # KnowEditDataset 加载器默认会用 _train_cf.json
    counterfact_data_filepath = os.path.join(base_data_dir, 'KnowEdit', 'benchmark_wiki_counterfact_train_cf.json')
    # 如果要用验证集评估，可以改成：
    # counterfact_data_filepath = os.path.join(base_data_dir, 'counterfact', 'counterfact-val.json') # 需要确保 prepare_counterfact_data 能处理此格式

    # --- 主测试循环 ---
    all_results_summary = []

    for sbert_model_path_iter in MODEL_LIST:
        print(f"\n\n{'=' * 20} 测试 SBERT 模型: {sbert_model_path_iter} {'=' * 20}")
        try:
            embedding_model_main = load_embedding_model(
                random_seed=RANDOM_SEED,
                model_path=sbert_model_path_iter
            )
        except Exception as e:
            print(f"错误: 无法加载模型 {sbert_model_path_iter}。跳过此模型。错误信息: {e}")
            continue

        for dataset_to_test_iter in DATASET_LIST:
            print(f"\n--- 测试数据集: {dataset_to_test_iter} (模型: {sbert_model_path_iter.split('/')[-1]}) ---")

            prompts_main, locality_prompts_main, rephrase_prompts_main = [], [], []
            try:
                if dataset_to_test_iter == "counterfact":
                    if not os.path.exists(counterfact_data_filepath):
                        print(f"警告: CounterFact 数据文件 {counterfact_data_filepath} 未找到。跳过此数据集。")
                        continue
                    prompts_main, locality_prompts_main, rephrase_prompts_main = prepare_counterfact_data(counterfact_data_filepath, ds_size=DATA_SIZE)
                elif dataset_to_test_iter == "zsre":
                    if not os.path.exists(zsre_data_filepath):
                        print(f"警告: ZsRE 数据文件 {zsre_data_filepath} 未找到。跳过此数据集。")
                        continue
                    prompts_main, locality_prompts_main, rephrase_prompts_main = prepare_zsre_data(zsre_data_filepath, ds_size=DATA_SIZE)
                elif dataset_to_test_iter == "multiarea":
                    if not os.path.exists(multiarea_data_dir_for_script):
                        print(f"警告: MultiArea 数据目录 {multiarea_data_dir_for_script} 未找到。跳过此数据集。")
                        continue
                    prompts_main, locality_prompts_main, rephrase_prompts_main = prepare_multiarea_data(multiarea_data_dir_for_script, ds_size_per_file=DATA_SIZE if DATA_SIZE else 50)
                else:
                    print(f"警告: 未知的数据集 {dataset_to_test_iter}。跳过。")
                    continue

                print(f"为 {dataset_to_test_iter} 加载了 {len(prompts_main)} 条编辑目标问题。")
                if not prompts_main:
                    print(f"警告: 数据集 {dataset_to_test_iter} 加载后编辑目标问题列表为空。跳过测试。")
                    continue

            except Exception as e:
                print(f"错误: 为数据集 {dataset_to_test_iter} 准备数据时发生错误: {e}。跳过此数据集。")
                continue

            edited_knowledge_prompts_for_test = prompts_main

            final_accuracies = test_semantic_router(
                embedding_model_instance=embedding_model_main,
                edited_prompts_as_reference=edited_knowledge_prompts_for_test,
                prompts_to_test_target=prompts_main,
                rephrases_to_test_generalization=rephrase_prompts_main,
                localities_to_test_locality=locality_prompts_main,
                similarity_threshold=SIMILARITY_THRESHOLD
            )

            current_result = {
                "model": sbert_model_path_iter.split('/')[-1],  # 取模型名称的最后一部分
                "dataset": dataset_to_test_iter,
                "threshold": SIMILARITY_THRESHOLD,
                "num_reference_prompts": len(edited_knowledge_prompts_for_test),
                "accuracy_target_xe": final_accuracies['编辑目标问题准确率(x_e vs E)'],
                "accuracy_generalization_xgen": final_accuracies['泛化问题准确率(x_gen vs E)'],
                "accuracy_locality_xloc": final_accuracies['局部性问题准确率(x_loc vs E)'],
                "overall_accuracy": final_accuracies['总体准确率']
            }
            all_results_summary.append(current_result)

            # --- 打印当前组合的结果 ---
            print("\n--- 单次语义路由准确率测试结果 ---")
            print(f"测试数据集: {dataset_to_test_iter}")
            print(f"SBERT 模型: {sbert_model_path_iter}")
            print(f"相似度阈值 (tau_pos): {SIMILARITY_THRESHOLD}")
            print(f"作为边界参考的编辑目标问题数量 (E 中的 |x_e|): {len(edited_knowledge_prompts_for_test)}")
            print(f"  编辑目标问题准确率 (x_e vs E): {final_accuracies['编辑目标问题准确率(x_e vs E)']:.4f}")
            print(f"  泛化问题准确率 (x_gen vs E): {final_accuracies['泛化问题准确率(x_gen vs E)']:.4f}")
            print(f"  局部性问题准确率 (x_loc vs E): {final_accuracies['局部性问题准确率(x_loc vs E)']:.4f}")
            print(f"  总体准确率: {final_accuracies['总体准确率']:.4f}")
            print("--------------------------------------")

            if SHOW_MISCLASSIFIED_EXAMPLES_MAIN:
                # 这里可以按需调用显示错误分类的逻辑，但要注意信息量
                pass

        # 清理模型以释放GPU内存（如果模型在GPU上）
        if hasattr(embedding_model_main, 'model') and hasattr(embedding_model_main.model, 'cpu'):
            embedding_model_main.model.cpu()
        del embedding_model_main
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- 打印所有测试的总结 ---
    print(f"\n\n{'='*30} 所有测试总结 {'='*30}")
    if all_results_summary:
        # 打印表头
        header = "| {:<30} | {:<15} | {:<8} | {:<10} | {:<10} | {:<10} | {:<10} |"
        actual_separator_line = "|{s1}|{s2}|{s3}|{s4}|{s5}|{s6}|{s7}|".format(
            s1='-' * 32, s2='-' * 17, s3='-' * 10, s4='-' * 12, s5='-' * 12, s6='-' * 12, s7='-' * 12
        )
        print(actual_separator_line)

        print(header.format("SBERT 模型", "数据集", "阈值", "Acc(x_e)", "Acc(x_gen)", "Acc(x_loc)", "总体Acc"))
        print(actual_separator_line)

        # 打印每一行结果
        for res in all_results_summary:
            print(header.format(
                res["model"],
                res["dataset"],
                f"{res['threshold']:.2f}",
                f"{res['accuracy_target_xe']:.4f}",
                f"{res['accuracy_generalization_xgen']:.4f}",
                f"{res['accuracy_locality_xloc']:.4f}",
                f"{res['overall_accuracy']:.4f}"
            ))
        print(actual_separator_line)
    else:
        print("没有执行任何测试，或所有测试均失败。")
    print(f"{'=' * 30} 测试结束 {'=' * 30}")
import json
import os
# from collections.abc import Set # Set from typing is fine
from typing import List, Tuple, Dict, Optional, Set as TypingSet # Use TypingSet for type hints
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import umap.umap_ as umap # 确保 umap 已安装
from datasets import load_from_disk, Dataset
from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity # get_embeddings + np.sum 更高效
from tqdm import tqdm

# --- Matplotlib and Seaborn Styling ---
plt.style.use('seaborn-whitegrid') # 使用兼容的样式
try:
    plt.rcParams['font.sans-serif'] = ['SimHei'] # FONT_CHINESE
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    print("SimHei font not found, Chinese characters might not display correctly.")

# --- Helper Functions (保持不变) ---
def get_embeddings(model: SentenceTransformer, sentences: List[str], batch_size: int = 64) -> np.ndarray:
    """Encodes sentences into embeddings."""
    if not sentences:
        return np.array([])
    return model.encode(sentences, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)

def calculate_similarities_for_eval(
    sbert_model: SentenceTransformer,
    test_data: Dataset
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Calculates cosine similarities for anchor-positive and anchor-negative pairs."""
    if not test_data or len(test_data) == 0:
        # print("警告: calculate_similarities_for_eval 接收到空的 test_data。")
        return None, None
    valid_indices = [
        i for i, item in enumerate(test_data)
        if item and isinstance(item, dict) and \
           item.get('anchor') and item.get('positive') and item.get('negative')
    ]
    if not valid_indices:
        # print("警告: 测试数据中没有有效的 (anchor, positive, negative) 样本。")
        return None, None
    if len(valid_indices) < len(test_data):
        # print(f"警告: 过滤掉 {len(test_data) - len(valid_indices)} 条因文本缺失或格式不正确的样本。")
        try:
            if isinstance(test_data, Dataset):
                 test_data_filtered = test_data.select(valid_indices)
            else:
                 test_data_filtered = [test_data[i] for i in valid_indices]
        except Exception as e:
            # print(f"选择有效索引时出错: {e}. 将使用列表推导式。")
            test_data_filtered = [test_data[i] for i in valid_indices]
        if not test_data_filtered :
             # print("警告: 过滤后测试数据为空。")
             return None,None
    else:
        test_data_filtered = test_data
    anchors = [item['anchor'] for item in test_data_filtered]
    positives_gen = [item['positive'] for item in test_data_filtered]
    negatives_loc = [item['negative'] for item in test_data_filtered]
    if not anchors or not positives_gen or not negatives_loc:
        # print("警告: 提取的 anchor, positive, 或 negative 列表为空。")
        return None, None
    emb_anchor = get_embeddings(sbert_model, anchors)
    emb_gen = get_embeddings(sbert_model, positives_gen)
    emb_loc = get_embeddings(sbert_model, negatives_loc)
    if emb_anchor.size == 0 or emb_gen.size == 0 or emb_loc.size == 0:
        # print("警告: 生成的嵌入向量为空。")
        return None, None
    norm_anchor = np.linalg.norm(emb_anchor, axis=1, keepdims=True)
    norm_gen = np.linalg.norm(emb_gen, axis=1, keepdims=True)
    norm_loc = np.linalg.norm(emb_loc, axis=1, keepdims=True)
    # Avoid division by zero
    sim_e_gen = np.sum(emb_anchor * emb_gen, axis=1) / (norm_anchor.flatten() * norm_gen.flatten() + 1e-9)
    sim_e_loc = np.sum(emb_anchor * emb_loc, axis=1) / (norm_anchor.flatten() * norm_loc.flatten() + 1e-9)
    return sim_e_gen, sim_e_loc

# --- 3.A. Embedding Space Metrics & Distribution Plot (修改后) ---
def plot_kde_on_ax(
    ax: plt.Axes, # 传入子图对象
    model_name_label: str,
    sim_e_gen: Optional[np.ndarray],
    sim_e_loc: Optional[np.ndarray],
    show_legend: bool = True
) -> Tuple[Optional[float], Optional[float], Optional[float]]: # 返回指标
    # print(f"\n--- A. 嵌入空间度量 (on ax): {model_name_label} ---")
    if sim_e_gen is None or sim_e_loc is None or len(sim_e_gen) == 0 or len(sim_e_loc) == 0:
        # print(f"数据不足，无法评估 {model_name_label} 的嵌入空间。")
        ax.text(0.5, 0.5, "无相似度数据", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(f'{model_name_label}\n(无数据)', fontsize=10)
        return None, None, None

    avg_intra_sim = np.mean(sim_e_gen)
    avg_inter_sim = np.mean(sim_e_loc)
    sim_gap = avg_intra_sim - avg_inter_sim

    # print(f"  {model_name_label}: Avg Intra: {avg_intra_sim:.4f}, Avg Inter: {avg_inter_sim:.4f}, Gap: {sim_gap:.4f}")

    sns.kdeplot(sim_e_gen, fill=True, label=r'$sim(x_e, x_{gen})$' if show_legend else None, alpha=0.7, warn_singular=False, ax=ax)
    sns.kdeplot(sim_e_loc, fill=True, label=r'$sim(x_e, x_{loc})$' if show_legend else None, alpha=0.7, warn_singular=False, ax=ax)

    ax.set_title(model_name_label, fontsize=11)
    ax.set_xlabel('余弦相似度', fontsize=9)
    ax.set_ylabel('密度', fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)


    # Add vertical lines for means and reference tau_pos
    current_handles, current_labels = [], [] # Store handles/labels for this subplot if show_legend is True

    if len(sim_e_gen) > 1:
        mean_gen_line = ax.axvline(avg_intra_sim, color='blue', linestyle='--', alpha=0.6, linewidth=1)
        if show_legend:
            current_handles.append(mean_gen_line)
            current_labels.append(f'均值 $Xgen$: {avg_intra_sim:.2f}')
    if len(sim_e_loc) > 1:
        mean_loc_line = ax.axvline(avg_inter_sim, color='darkorange', linestyle='--', alpha=0.6, linewidth=1)
        if show_legend:
            current_handles.append(mean_loc_line)
            current_labels.append(f'均值 $Xloc$: {avg_inter_sim:.2f}')

    tau_line = ax.axvline(0.85, color='red', linestyle=':', alpha=0.8, linewidth=1)
    if show_legend:
        current_handles.append(tau_line)
        current_labels.append(r'$\tau_{pos}=0.85$')

    # Add existing KDE plot handles/labels to current_handles/labels for the legend
    kde_handles, kde_labels = ax.get_legend_handles_labels()
    if show_legend:
        all_handles = kde_handles + current_handles
        all_labels = kde_labels + current_labels
        ax.legend(all_handles, all_labels, fontsize=8, loc='upper left')
    else: # If not showing legend on this subplot, clear any auto-generated ones
        ax.legend_ = None


    return avg_intra_sim, avg_inter_sim, sim_gap


# --- 3.B. Semantic Routing Threshold τ_pos Sensitivity Analysis (保持不变) ---
def analyze_tau_pos_sensitivity(
    model_name_label: str,
    sim_e_gen: Optional[np.ndarray],
    sim_e_loc: Optional[np.ndarray]
) -> Optional[Dict]:
    # ... (代码与之前相同)
    if sim_e_gen is None or sim_e_loc is None or len(sim_e_gen) == 0 or len(sim_e_loc) == 0:
        return None
    tau_pos_values = np.arange(0.5, 0.96, 0.02)
    recalls_gen = []
    specificities_loc = []
    for tau in tau_pos_values:
        recall_gen = np.sum(sim_e_gen > tau) / len(sim_e_gen) if len(sim_e_gen) > 0 else 0.0
        specificity_loc = np.sum(sim_e_loc <= tau) / len(sim_e_loc) if len(sim_e_loc) > 0 else 0.0
        recalls_gen.append(recall_gen)
        specificities_loc.append(specificity_loc)
    one_minus_specificity = [1 - spec for spec in specificities_loc]
    return {"model_name": model_name_label, "one_minus_specificity": one_minus_specificity,
            "recalls_gen": recalls_gen, "tau_pos_values": tau_pos_values.tolist()}


# --- 3.C. “混淆点”/错误案例分析 (保持不变) ---
def analyze_focused_confusion_points(
    current_model_name_label: str,
    test_data: Dataset,
    sim_e_gen_current: Optional[np.ndarray],
    sim_e_loc_current: Optional[np.ndarray],
    baseline_fp_loc_indices: TypingSet[int],
    baseline_fn_gen_indices: TypingSet[int],
    tau_pos_fixed: float,
    output_dir: str,
    num_samples_to_show: int = 3
):
    # ... (代码与之前相同, 仅调整打印和文件写入的简洁性)
    print(f"\n--- C. 聚焦混淆点分析: {current_model_name_label} (τ_pos = {tau_pos_fixed}) ---")
    if sim_e_gen_current is None or sim_e_loc_current is None or \
       len(sim_e_gen_current) == 0 or len(sim_e_loc_current) == 0:
        print(f"数据不足 (C)。") # 简化
        return None, None, None, None, None, None
    corrected_fp_loc_indices = set()
    still_fp_loc_indices = set()
    if baseline_fp_loc_indices:
        for idx_orig in baseline_fp_loc_indices:
            idx = int(idx_orig)
            if idx < len(sim_e_loc_current):
                if sim_e_loc_current[idx] <= tau_pos_fixed: corrected_fp_loc_indices.add(idx)
                else: still_fp_loc_indices.add(idx)
    corrected_fn_gen_indices = set()
    still_fn_gen_indices = set()
    if baseline_fn_gen_indices:
        for idx_orig in baseline_fn_gen_indices:
            idx = int(idx_orig)
            if idx < len(sim_e_gen_current):
                if sim_e_gen_current[idx] > tau_pos_fixed: corrected_fn_gen_indices.add(idx)
                else: still_fn_gen_indices.add(idx)
    current_model_fp_loc_indices_np = np.where(sim_e_loc_current > tau_pos_fixed)[0]
    new_fp_loc_indices = set(map(int, current_model_fp_loc_indices_np)) - baseline_fp_loc_indices
    current_model_fn_gen_indices_np = np.where(sim_e_gen_current <= tau_pos_fixed)[0]
    new_fn_gen_indices = set(map(int, current_model_fn_gen_indices_np)) - baseline_fn_gen_indices
    # print(f"  Baseline FP_loc: {len(baseline_fp_loc_indices)}, Corrected: {len(corrected_fp_loc_indices)}")
    # print(f"  Baseline FN_gen: {len(baseline_fn_gen_indices)}, Corrected: {len(corrected_fn_gen_indices)}")
    # print(f"  New FP_loc: {len(new_fp_loc_indices)}, New FN_gen: {len(new_fn_gen_indices)}")
    # 文件写入部分可以保持，但这里省略以聚焦主要逻辑
    return len(corrected_fp_loc_indices), len(still_fp_loc_indices), \
           len(corrected_fn_gen_indices), len(still_fn_gen_indices), \
           len(new_fp_loc_indices), len(new_fn_gen_indices)


# --- 4.A. 二维嵌入空间投影 (UMAP) (保持不变) ---
def plot_umap_on_ax(
    ax: plt.Axes,
    model_name_label: str,
    sbert_model: SentenceTransformer,
    test_data: Dataset,
    seed: int = 42,
    show_legend: bool = True
):
    # ... (代码与之前相同)
    if not test_data or len(test_data) == 0:
        ax.text(0.5, 0.5, "无数据", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{model_name_label}\n(无数据)', fontsize=10)
        return
    umap_data = test_data
    if len(umap_data) < 5:
        ax.text(0.5, 0.5, "UMAP样本不足", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{model_name_label}\n(样本不足)', fontsize=10)
        return
    valid_umap_items = [item for item in umap_data if item and isinstance(item, dict) and item.get('anchor') and item.get('positive') and item.get('negative')]
    if len(valid_umap_items) < 5 :
        ax.text(0.5, 0.5, "有效UMAP数据不足", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{model_name_label}\n(UMAP数据不足)', fontsize=10)
        return
    anchors = [item['anchor'] for item in valid_umap_items]
    positives_gen = [item['positive'] for item in valid_umap_items]
    negatives_loc = [item['negative'] for item in valid_umap_items]
    try:
        emb_anchor = get_embeddings(sbert_model, anchors)
        emb_gen = get_embeddings(sbert_model, positives_gen)
        emb_loc = get_embeddings(sbert_model, negatives_loc)
    except Exception as e:
        # print(f"  UMAP embed error for {model_name_label}: {e}")
        ax.text(0.5, 0.5, "嵌入失败", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{model_name_label}\n(嵌入失败)', fontsize=10)
        return
    if emb_anchor.size == 0 or emb_gen.size == 0 or emb_loc.size == 0 :
        ax.text(0.5, 0.5, "空嵌入", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{model_name_label}\n(空嵌入)', fontsize=10)
        return
    all_embeddings = np.vstack([emb_anchor, emb_gen, emb_loc])
    labels_np = np.array([0] * len(emb_anchor) + [1] * len(emb_gen) + [2] * len(emb_loc))
    n_neighbors_umap = min(15, len(all_embeddings) - 1) if len(all_embeddings) > 1 else 1
    if n_neighbors_umap <= 1 and len(all_embeddings) > 1: n_neighbors_umap = max(1, len(all_embeddings) // 2)
    if len(all_embeddings) <=1 or n_neighbors_umap < 1 :
        ax.text(0.5, 0.5, "UMAP总样本不足", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{model_name_label}\n(UMAP样本不足)', fontsize=10)
        return
    reducer = umap.UMAP(n_neighbors=n_neighbors_umap, min_dist=0.1, n_components=2, metric='cosine', random_state=seed, low_memory=True, transform_seed=seed)
    try:
        reduced_embeddings = reducer.fit_transform(all_embeddings)
    except Exception:
        try:
            reducer = umap.UMAP(n_neighbors=n_neighbors_umap, min_dist=0.1, n_components=2, metric='euclidean', random_state=seed, low_memory=True, transform_seed=seed)
            reduced_embeddings = reducer.fit_transform(all_embeddings)
        except Exception as e_alt:
            # print(f"  UMAP transform error for {model_name_label}: {e_alt}")
            ax.text(0.5, 0.5, "UMAP转换失败", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{model_name_label}\n(UMAP失败)', fontsize=10)
            return
    palette = sns.color_palette("husl", 3)
    category_names = [r'$x_e$', r'$x_{gen}$', r'$x_{loc}$']
    current_handles, current_labels = [], []
    for i, (cat_name, color) in enumerate(zip(category_names, palette)):
        indices = np.where(labels_np == i)[0]
        if len(indices) > 0:
            scatter = ax.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], s=20, alpha=0.6, color=color, label=cat_name)
            if show_legend:
                current_handles.append(scatter)
                current_labels.append(cat_name)
    ax.set_title(model_name_label, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    if show_legend and current_handles:
        ax.legend(current_handles, current_labels, fontsize=8, loc='best') # Or loc='upper left'
    elif not show_legend:
        ax.legend_ = None


# --- Main Evaluation Script ---
def main():
    # ============================== 用户配置 ==============================
    exp_output_base_dir = './exp_sbert_finetuning_output'
    n_samples_for_all_umaps = 300
    fixed_tau_for_confusion = 0.85
    ncols_display = 3 # 组合图中每行显示的子图数量 (KDE 和 UMAP 都用这个)
    # =======================================================================

    exp_summary_path = os.path.join(exp_output_base_dir, 'experiment_summary.json')
    # ... (文件和路径检查，与之前相同) ...
    if not os.path.exists(exp_summary_path):
        print(f"错误: 实验摘要文件未找到于 {exp_summary_path}")
        return
    with open(exp_summary_path, 'r', encoding='utf-8') as f:
        exp_summary = json.load(f)
    trained_model_paths = exp_summary.get('trained_model_paths', {})
    global_test_dataset_path = exp_summary.get('global_test_dataset_path')
    base_model_name_from_summary = exp_summary.get('base_model_name')
    seed = exp_summary.get('seed', 42)
    if not base_model_name_from_summary:
        print("错误：实验摘要中未找到 base_model_name。")
        return
    eval_results_output_dir = os.path.join(exp_output_base_dir, "evaluation_and_visualization_results_focused")
    os.makedirs(eval_results_output_dir, exist_ok=True)
    print(f"评估结果将保存到: {eval_results_output_dir}")
    if not global_test_dataset_path or not os.path.exists(global_test_dataset_path):
        print(f"错误: 全局测试集路径 ({global_test_dataset_path}) 无效。")
        return
    try:
        global_test_dataset = load_from_disk(global_test_dataset_path)
    except Exception as e:
        print(f"加载全局测试集失败: {e}")
        return
    if len(global_test_dataset) == 0:
        print("错误: 全局测试集为空。")
        return
    print(f"全局测试集加载成功，包含 {len(global_test_dataset)} 条样本。")

    actual_samples_for_umap = min(n_samples_for_all_umaps, len(global_test_dataset))
    umap_shared_test_data = None
    if actual_samples_for_umap > 0 :
        np.random.seed(seed)
        umap_sample_indices = np.random.choice(len(global_test_dataset), actual_samples_for_umap, replace=False)
        umap_shared_test_data = global_test_dataset.select(umap_sample_indices)
        print(f"为所有UMAP图统一采样 {len(umap_shared_test_data)} 条数据。")
    else:
        print("警告：用于UMAP的采样数据量为0。")

    models_to_eval_meta: Dict[str, Dict] = {}
    models_to_eval_meta['Baseline SBERT'] = {'path': base_model_name_from_summary, 'is_finetuned': False}
    desired_order = ['Baseline SBERT', 'Small-Data', 'Medium-Data', 'Large-Data',
                     'Full-Data']
    # Populate models_to_eval_meta according to desired_order first
    temp_models_meta = {'Baseline SBERT': models_to_eval_meta['Baseline SBERT']}
    for label in desired_order:
        if label in trained_model_paths and label != 'Baseline SBERT':
            model_path = trained_model_paths[label]
            if model_path and os.path.exists(model_path):
                temp_models_meta[label] = {'path': model_path, 'is_finetuned': True}
    # Add any other models not in desired_order
    for label, path in trained_model_paths.items():
        if label not in temp_models_meta:
            if path and os.path.exists(path):
                temp_models_meta[label] = {'path': path, 'is_finetuned': True}
    models_to_eval_meta = temp_models_meta


    all_metrics_summary = []
    all_tau_analysis_results = []
    all_model_similarities: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}
    loaded_sbert_models: Dict[str, SentenceTransformer] = {} # Renamed for clarity
    focused_confusion_data_summary = []

    print("\n步骤1&2: 加载模型并计算所有模型的相似度...")
    for model_name_label_iter, meta_iter in tqdm(models_to_eval_meta.items(), desc="加载与计算相似度"):
        # print(f"  处理模型: {model_name_label_iter} from {meta_iter['path']}")
        try:
            sbert_model_instance = SentenceTransformer(meta_iter['path'])
            loaded_sbert_models[model_name_label_iter] = sbert_model_instance # Store for KDE and UMAP
            sim_e_gen_val, sim_e_loc_val = calculate_similarities_for_eval(sbert_model_instance, global_test_dataset)
            all_model_similarities[model_name_label_iter] = (sim_e_gen_val, sim_e_loc_val)
        except Exception as e:
            print(f"  加载/计算相似度出错 ({model_name_label_iter}): {e}")
            all_model_similarities[model_name_label_iter] = (None, None)
            if model_name_label_iter in loaded_sbert_models:
                del loaded_sbert_models[model_name_label_iter]

    baseline_model_key = 'Baseline SBERT'
    baseline_fp_loc_indices_set: TypingSet[int] = set()
    baseline_fn_gen_indices_set: TypingSet[int] = set()
    sim_e_gen_baseline_val, sim_e_loc_baseline_val = all_model_similarities.get(baseline_model_key, (None, None))
    if sim_e_gen_baseline_val is not None and sim_e_loc_baseline_val is not None:
        baseline_fp_loc_indices_set = set(map(int, np.where(sim_e_loc_baseline_val > fixed_tau_for_confusion)[0]))
        baseline_fn_gen_indices_set = set(map(int, np.where(sim_e_gen_baseline_val <= fixed_tau_for_confusion)[0]))
        # print(f"\n  Baseline ({baseline_model_key}): {len(baseline_fp_loc_indices_set)} FP_loc, {len(baseline_fn_gen_indices_set)} FN_gen")
    else:
        print(f"  无法计算 Baseline ({baseline_model_key}) 相似度。")

    print("\n步骤3: 进行其他评估...")

    # --- 准备组合KDE图 ---
    num_models_for_plot = len(loaded_sbert_models) # Models that were successfully loaded
    fig_kde_combined, axes_kde_combined_flat = None, []
    if num_models_for_plot > 0:
        ncols_kde = min(num_models_for_plot, ncols_display)
        nrows_kde = (num_models_for_plot + ncols_kde - 1) // ncols_kde
        fig_kde_combined, axes_kde_arr = plt.subplots(
            nrows_kde, ncols_kde,
            figsize=(ncols_kde * 4.5, nrows_kde * 3.8), # Adjusted figsize
            squeeze=False
        )
        axes_kde_combined_flat = axes_kde_arr.flatten()
        kde_plot_idx_counter = 0
    else:
        print("没有成功加载的模型可用于绘制KDE图。")


    # --- 准备组合UMAP图 ---
    fig_umap_combined, axes_umap_combined_flat = None, []
    if num_models_for_plot > 0 and umap_shared_test_data is not None:
        ncols_umap = min(num_models_for_plot, ncols_display)
        nrows_umap = (num_models_for_plot + ncols_umap - 1) // ncols_umap
        fig_umap_combined, axes_umap_arr = plt.subplots(
            nrows_umap, ncols_umap,
            figsize=(ncols_umap * 3.8, nrows_umap * 3.8), # Adjusted figsize
            squeeze=False
        )
        axes_umap_combined_flat = axes_umap_arr.flatten()
        umap_plot_idx_counter = 0
    else:
        print("没有UMAP数据或没有成功加载的模型可用于绘制UMAP图。")


    for model_name_label_eval, meta_eval in models_to_eval_meta.items(): # Iterate through all defined models
        print(f"\n{'=' * 20} 正在评估: {model_name_label_eval} {'=' * 20}")
        sim_e_gen_curr_val, sim_e_loc_curr_val = all_model_similarities.get(model_name_label_eval, (None, None))

        # A. Embedding Space Metrics - Plotted on combined KDE figure
        if model_name_label_eval in loaded_sbert_models and kde_plot_idx_counter < len(axes_kde_combined_flat):
            ax_kde_to_plot_on = axes_kde_combined_flat[kde_plot_idx_counter]
            show_kde_legend = (kde_plot_idx_counter == 0) # Legend on first KDE subplot

            print(f"  绘制 KDE for {model_name_label_eval} on ax {kde_plot_idx_counter}")
            avg_intra_val, avg_inter_val, gap_val = plot_kde_on_ax(
                ax=ax_kde_to_plot_on,
                model_name_label=model_name_label_eval,
                sim_e_gen=sim_e_gen_curr_val,
                sim_e_loc=sim_e_loc_curr_val,
                show_legend=show_kde_legend
            )
            if avg_intra_val is not None: # Metrics were successfully calculated
                all_metrics_summary.append({
                    "model": model_name_label_eval,
                    "avg_intra_sim (x_e, x_gen)": f"{avg_intra_val:.4f}",
                    "avg_inter_sim (x_e, x_loc)": f"{avg_inter_val:.4f}",
                    "similarity_gap": f"{gap_val:.4f}"
                })
            kde_plot_idx_counter += 1
        elif sim_e_gen_curr_val is not None and sim_e_loc_curr_val is not None : # Model not loaded, but has similarities, calc metrics
            avg_intra_val = np.mean(sim_e_gen_curr_val)
            avg_inter_val = np.mean(sim_e_loc_curr_val)
            gap_val = avg_intra_val - avg_inter_val
            all_metrics_summary.append({
                "model": model_name_label_eval,
                "avg_intra_sim (x_e, x_gen)": f"{avg_intra_val:.4f}",
                "avg_inter_sim (x_e, x_loc)": f"{avg_inter_val:.4f}",
                "similarity_gap": f"{gap_val:.4f}"
            })
            print(f"  计算了 {model_name_label_eval} 的指标，但未绘制KDE (模型未加载或子图不足)。")
        else:
            print(f"  跳过 {model_name_label_eval} 的KDE图和指标 (无相似度数据)。")


        # B. τ_pos Sensitivity Analysis
        tau_result_val = analyze_tau_pos_sensitivity(model_name_label_eval, sim_e_gen_curr_val, sim_e_loc_curr_val)
        if tau_result_val:
            all_tau_analysis_results.append(tau_result_val)

        # C. Focused Confusion Points Analysis
        # ... (logic for focused confusion, same as before) ...
        if model_name_label_eval == baseline_model_key:
            if sim_e_gen_baseline_val is not None : # Check if baseline data exists
                focused_confusion_data_summary.append({
                     "model": baseline_model_key,
                     "baseline_fp_loc_total": len(baseline_fp_loc_indices_set),
                     "corrected_fp_loc": 0, "corrected_fp_loc_pct": "0.00%",
                     "still_fp_loc": len(baseline_fp_loc_indices_set), "new_fp_loc": 0,
                     "baseline_fn_gen_total": len(baseline_fn_gen_indices_set),
                     "corrected_fn_gen": 0, "corrected_fn_gen_pct": "0.00%",
                     "still_fn_gen": len(baseline_fn_gen_indices_set), "new_fn_gen": 0,
                 })
        elif sim_e_gen_curr_val is not None and sim_e_loc_curr_val is not None:
            analysis_results = analyze_focused_confusion_points(
                model_name_label_eval, global_test_dataset,
                sim_e_gen_curr_val, sim_e_loc_curr_val,
                baseline_fp_loc_indices_set, baseline_fn_gen_indices_set,
                fixed_tau_for_confusion, eval_results_output_dir
            )
            if analysis_results is not None:
                corr_fp, still_fp, corr_fn, still_fn, new_fp, new_fn = analysis_results
                fp_total = len(baseline_fp_loc_indices_set)
                fn_total = len(baseline_fn_gen_indices_set)
                focused_confusion_data_summary.append({
                    "model": model_name_label_eval,
                    "baseline_fp_loc_total": fp_total,
                    "corrected_fp_loc": corr_fp, "corrected_fp_loc_pct": f"{(corr_fp/fp_total*100 if fp_total > 0 else 0):.2f}%",
                    "still_fp_loc": still_fp, "new_fp_loc": new_fp,
                    "baseline_fn_gen_total": fn_total,
                    "corrected_fn_gen": corr_fn, "corrected_fn_gen_pct": f"{(corr_fn/fn_total*100 if fn_total > 0 else 0):.2f}%",
                    "still_fn_gen": still_fn, "new_fn_gen": new_fn,
                })


        # 4.A. UMAP Visualization on subplot
        if model_name_label_eval in loaded_sbert_models and \
           umap_shared_test_data is not None and \
           umap_plot_idx_counter < len(axes_umap_combined_flat):
            current_sbert_model_for_umap = loaded_sbert_models[model_name_label_eval]
            ax_to_plot_on_val = axes_umap_combined_flat[umap_plot_idx_counter]
            show_umap_legend = (umap_plot_idx_counter == 0)

            print(f"  绘制 UMAP for {model_name_label_eval} on ax {umap_plot_idx_counter}")
            plot_umap_on_ax(
                ax=ax_to_plot_on_val,
                model_name_label=model_name_label_eval,
                sbert_model=current_sbert_model_for_umap,
                test_data=umap_shared_test_data,
                seed=seed,
                show_legend=show_umap_legend
            )
            umap_plot_idx_counter += 1
        # else:
            # print(f"  跳过 {model_name_label_eval} 的UMAP图 (模型未加载, 无UMAP数据或子图不足)。")


    # --- 完成组合KDE图 ---
    if fig_kde_combined is not None and kde_plot_idx_counter > 0 :
        for i in range(kde_plot_idx_counter, len(axes_kde_combined_flat)): # Hide unused subplots
            fig_kde_combined.delaxes(axes_kde_combined_flat[i])
        # Optional: Add common legend for KDE if not shown on first subplot or if desired
        if kde_plot_idx_counter > 0 and not show_kde_legend: # If no subplot showed legend
            # Create a dummy legend or get from one of the axes if they were drawn
            pass # Logic for common legend can be complex if individual legends are off

        # fig_kde_combined.suptitle('不同SBERT模型相似度分布比较', fontsize=16) # Optional
        plt.figure(fig_kde_combined.number) # Set current figure to fig_kde_combined
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
        combined_kde_plot_path = os.path.join(eval_results_output_dir, "Vis_A_combined_kde_distributions.png")
        fig_kde_combined.savefig(combined_kde_plot_path, dpi=300, bbox_inches='tight')
        print(f"\n组合 KDE 分布图已保存: {combined_kde_plot_path}")
        plt.close(fig_kde_combined)


    # --- 完成组合UMAP图 ---
    if fig_umap_combined is not None and umap_plot_idx_counter > 0:
        for i in range(umap_plot_idx_counter, len(axes_umap_combined_flat)): # Hide unused
            fig_umap_combined.delaxes(axes_umap_combined_flat[i])
        # Optional: Add common legend for UMAP
        if umap_plot_idx_counter > 0 and axes_umap_combined_flat[0].get_legend() is not None:
            handles_umap, labels_umap = axes_umap_combined_flat[0].get_legend_handles_labels()
            if handles_umap and labels_umap:
                 # Adjust y offset based on number of rows for UMAP legend
                 num_rows_umap_actual = (umap_plot_idx_counter + ncols_display -1) // ncols_display
                 legend_y_offset_umap = -0.02 - (0.05 * max(0, (num_rows_umap_actual -1)) / 2)
                 fig_umap_combined.legend(handles_umap, labels_umap, loc='lower center', ncol=3,
                                          bbox_to_anchor=(0.5, legend_y_offset_umap))

        # fig_umap_combined.suptitle('不同SBERT模型嵌入空间UMAP可视化', fontsize=16) # Optional
        plt.figure(fig_umap_combined.number) # Set current figure
        plt.tight_layout(rect=[0, abs(legend_y_offset_umap) if 'legend_y_offset_umap' in locals() and umap_plot_idx_counter > 0 else 0.03, 1, 0.96]) # Adjust rect for legend
        combined_umap_plot_path = os.path.join(eval_results_output_dir, "Vis_A_combined_umap_projection.png")
        fig_umap_combined.savefig(combined_umap_plot_path, dpi=300, bbox_inches='tight')
        print(f"\n组合 UMAP 投影图已保存: {combined_umap_plot_path}")
        plt.close(fig_umap_combined)

    # 清理加载的模型
    print("清理已加载的SBERT模型...")
    for model_instance_del in loaded_sbert_models.values():
        del model_instance_del
    loaded_sbert_models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("已清理。")

    # --- 生成其他组合图表和摘要 (ROC等，与之前相同) ---
    if all_tau_analysis_results:
        plt.figure(figsize=(10, 7))
        for result_item in all_tau_analysis_results:
            plt.plot(result_item["one_minus_specificity"], result_item["recalls_gen"], marker='.', markersize=7, linestyle='-', label=result_item["model_name"])
        # ... (ROC plot details) ...
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='随机')
        plt.xlabel(r'1 - Specificity$_{loc}$', fontsize=11)
        plt.ylabel(r'Recall$_{gen}$', fontsize=11)
        plt.title(r'不同模型下 $\tau_{pos}$ 敏感性分析', fontsize=15)
        plt.legend(fontsize=10, loc='lower right')
        plt.grid(True); plt.xlim([-0.02, 1.02]); plt.ylim([-0.02, 1.02])
        combined_roc_plot_path = os.path.join(eval_results_output_dir, "Vis_B_combined_tau_sensitivity.png")
        plt.savefig(combined_roc_plot_path, dpi=300)
        print(f"\n组合 τ_pos 敏感性图已保存: {combined_roc_plot_path}")
        plt.close()


    # --- 保存摘要JSON文件并打印 ---
    if all_metrics_summary:
        summary_path = os.path.join(eval_results_output_dir, "evaluation_metrics_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics_summary, f, indent=4, ensure_ascii=False)
        print(f"评估指标摘要已保存: {summary_path}")
        # ... (打印指标摘要)

    if focused_confusion_data_summary:
        focused_summary_path = os.path.join(eval_results_output_dir, "focused_confusion_summary.json")
        with open(focused_summary_path, "w", encoding="utf-8") as f:
            json.dump(focused_confusion_data_summary, f, indent=4, ensure_ascii=False)
        print(f"聚焦混淆点摘要已保存: {focused_summary_path}")
        # ... (打印聚焦混淆点摘要)

    print("\n\n评估和可视化流程完成。")

if __name__ == "__main__":
    main()
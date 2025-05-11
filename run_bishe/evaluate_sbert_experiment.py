import json
import os
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import umap.umap_ as umap
from datasets import load_from_disk, Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# from scipy.stats import gaussian_kde # Not strictly needed if using seaborn.kdeplot
from tqdm import tqdm

# --- Matplotlib and Seaborn Styling ---
plt.style.use('seaborn-whitegrid')
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    print("SimHei font not found, Chinese characters might not display correctly.")


# --- Helper Functions ---
def get_embeddings(model: SentenceTransformer, sentences: List[str], batch_size: int = 64) -> np.ndarray:
    return model.encode(sentences, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)


def calculate_similarities_for_eval(
        sbert_model: SentenceTransformer,
        test_data: Dataset
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    # print(f"Calculating embeddings for {len(test_data)} test samples...")

    valid_indices = [
        i for i, item in enumerate(test_data)
        if item['anchor'] and item['positive'] and item['negative']
    ]
    if len(valid_indices) < len(test_data):
        print(f"警告: 过滤掉 {len(test_data) - len(valid_indices)} 条因文本缺失的样本。")
        test_data = test_data.select(valid_indices)

    if len(test_data) == 0:
        print("警告: 过滤后测试数据为空。")
        return None, None

    anchors = [item['anchor'] for item in test_data]
    positives_gen = [item['positive'] for item in test_data]
    negatives_loc = [item['negative'] for item in test_data]

    emb_anchor = get_embeddings(sbert_model, anchors)
    emb_gen = get_embeddings(sbert_model, positives_gen)
    emb_loc = get_embeddings(sbert_model, negatives_loc)

    # Batch cosine similarity calculation
    sim_e_gen = np.sum(emb_anchor * emb_gen, axis=1) / (np.linalg.norm(emb_anchor, axis=1) * np.linalg.norm(emb_gen, axis=1))
    sim_e_loc = np.sum(emb_anchor * emb_loc, axis=1) / (np.linalg.norm(emb_anchor, axis=1) * np.linalg.norm(emb_loc, axis=1))

    return sim_e_gen, sim_e_loc


# --- 3.A. Embedding Space Metrics & Distribution Plot ---
def evaluate_embedding_space(
        model_name_label: str,
        sim_e_gen: np.ndarray,
        sim_e_loc: np.ndarray,
        output_dir: str
):
    print(f"\n--- A. 嵌入空间度量: {model_name_label} ---")
    if sim_e_gen is None or sim_e_loc is None or len(sim_e_gen) == 0 or len(sim_e_loc) == 0:
        print(f"数据不足，无法评估 {model_name_label} 的嵌入空间。")
        return None, None, None

    avg_intra_sim = np.mean(sim_e_gen)
    avg_inter_sim = np.mean(sim_e_loc)
    sim_gap = avg_intra_sim - avg_inter_sim

    print(f"  平均类内相似度 (x_e, x_gen): {avg_intra_sim:.4f}")
    print(f"  平均类间相似度 (x_e, x_loc): {avg_inter_sim:.4f}")
    print(f"  相似度差距: {sim_gap:.4f}")

    plt.figure(figsize=(12, 7))
    sns.kdeplot(sim_e_gen, fill=True, label=r'$sim(x_e, x_{gen})$ (目标区域)', alpha=0.7, warn_singular=False)
    sns.kdeplot(sim_e_loc, fill=True, label=r'$sim(x_e, x_{loc})$ (近邻非等价)', alpha=0.7, warn_singular=False)

    plt.title(f'相似度分布 ({model_name_label})', fontsize=16)
    plt.xlabel('余弦相似度', fontsize=12)
    plt.ylabel('密度', fontsize=12)

    # Add vertical lines for means and reference tau_pos
    handles, labels = plt.gca().get_legend_handles_labels()  # Get existing handles/labels
    if len(sim_e_gen) > 1:  # Mean is well-defined
        mean_gen_line = plt.axvline(avg_intra_sim, color='blue', linestyle='--', alpha=0.6)
        handles.append(mean_gen_line)
        labels.append(f'均值 sim(x_e, x_gen): {avg_intra_sim:.2f}')
    if len(sim_e_loc) > 1:
        mean_loc_line = plt.axvline(avg_inter_sim, color='darkorange', linestyle='--', alpha=0.6)  # darkorange for visibility
        handles.append(mean_loc_line)
        labels.append(f'均值 sim(x_e, x_loc): {avg_inter_sim:.2f}')

    tau_line = plt.axvline(0.85, color='red', linestyle=':', alpha=0.8)
    handles.append(tau_line)
    labels.append(r'$\tau_{pos}=0.85$ (参考)')

    plt.legend(handles=handles, labels=labels, fontsize=11, loc='upper left')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"A_similarity_dist_{model_name_label.replace(' ', '_').replace('/', '-')}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"  相似度分布图已保存: {plot_path}")
    plt.close()

    return avg_intra_sim, avg_inter_sim, sim_gap


# --- 3.B. Semantic Routing Threshold τ_pos Sensitivity Analysis ---
def analyze_tau_pos_sensitivity(
        model_name_label: str,
        sim_e_gen: np.ndarray,
        sim_e_loc: np.ndarray
) -> Optional[Dict]:
    # print(f"\n--- B. τ_pos 敏感性分析: {model_name_label} ---") # Called per model, plotting is combined
    if sim_e_gen is None or sim_e_loc is None or len(sim_e_gen) == 0 or len(sim_e_loc) == 0:
        # print(f"数据不足，无法进行 {model_name_label} 的 τ_pos 敏感性分析。")
        return None

    tau_pos_values = np.arange(0.5, 0.96, 0.02)
    recalls_gen = []
    specificities_loc = []

    for tau in tau_pos_values:
        recall_gen = np.sum(sim_e_gen > tau) / len(sim_e_gen)
        specificity_loc = np.sum(sim_e_loc <= tau) / len(sim_e_loc)
        recalls_gen.append(recall_gen)
        specificities_loc.append(specificity_loc)

    one_minus_specificity = [1 - spec for spec in specificities_loc]
    return {"model_name": model_name_label, "one_minus_specificity": one_minus_specificity,
            "recalls_gen": recalls_gen, "tau_pos_values": tau_pos_values.tolist()}


# --- 3.C. “混淆点”/错误案例分析 ---
def analyze_confusion_points(
        model_name_label: str,
        test_data: Dataset,
        sim_e_gen: np.ndarray,
        sim_e_loc: np.ndarray,
        tau_pos_fixed: float,
        output_dir: str,
        num_samples_to_show: int = 5
):
    print(f"\n--- C. “混淆点”分析: {model_name_label} (τ_pos = {tau_pos_fixed}) ---")
    if sim_e_gen is None or sim_e_loc is None or len(sim_e_gen) == 0 or len(sim_e_loc) == 0:
        print(f"数据不足，无法进行 {model_name_label} 的混淆点分析。")
        return

    fp_loc_indices = np.where(sim_e_loc > tau_pos_fixed)[0]
    fn_gen_indices = np.where(sim_e_gen <= tau_pos_fixed)[0]

    print(f"  测试样本总数: {len(test_data)}")
    print(f"  FP_loc (x_loc 被误认为相关): {len(fp_loc_indices)}")
    print(f"  FN_gen (x_gen 被误认为不相关): {len(fn_gen_indices)}")

    error_analysis_path = os.path.join(output_dir, f"C_confusion_analysis_{model_name_label.replace(' ', '_').replace('/', '-')}_tau{tau_pos_fixed:.2f}.txt")
    with open(error_analysis_path, "w", encoding="utf-8") as f:
        f.write(f"模型 {model_name_label} 在 τ_pos = {tau_pos_fixed} 下的混淆点分析\n\n")

        f.write(f"--- FP_loc (False Positives for x_loc) - 共 {len(fp_loc_indices)} 例 ---\n")
        f.write("含义: x_loc (本应不相似) 被错误地判断为与 x_e 相似 (sim > τ_pos)\n")
        for i, idx_orig in enumerate(fp_loc_indices[:num_samples_to_show]):
            idx = int(idx_orig)
            item = test_data[idx]
            f.write(f"  示例 {i + 1}:\n")
            f.write(f"    x_e (锚点): {item['anchor']}\n")
            f.write(f"    x_loc (负例-误判): {item['negative']}\n")
            f.write(f"    sim(x_e, x_loc): {sim_e_loc[idx]:.4f} > {tau_pos_fixed}\n\n")

        f.write(f"\n--- FN_gen (False Negatives for x_gen) - 共 {len(fn_gen_indices)} 例 ---\n")
        f.write("含义: x_gen (本应相似) 被错误地判断为与 x_e 不相似 (sim <= τ_pos)\n")
        for i, idx_orig in enumerate(fn_gen_indices[:num_samples_to_show]):
            idx = int(idx_orig)
            item = test_data[idx]
            f.write(f"  示例 {i + 1}:\n")
            f.write(f"    x_e (锚点): {item['anchor']}\n")
            f.write(f"    x_gen (正例-误判): {item['positive']}\n")
            f.write(f"    sim(x_e, x_gen): {sim_e_gen[idx]:.4f} <= {tau_pos_fixed}\n\n")
    print(f"  混淆点分析报告已保存: {error_analysis_path}")


# --- 4.A. 二维嵌入空间投影 (UMAP) ---
def visualize_umap_projection(
        model_name_label: str,
        sbert_model: SentenceTransformer,
        test_data: Dataset,
        output_dir: str,
        n_samples_for_umap: int = 500,
        seed: int = 42
):
    print(f"\n--- 4.A. UMAP 可视化: {model_name_label} ---")
    if len(test_data) == 0: return

    # Sample data for UMAP
    if len(test_data) > n_samples_for_umap:
        np.random.seed(seed)
        sample_indices = np.random.choice(len(test_data), n_samples_for_umap, replace=False)
        umap_data = test_data.select(sample_indices)
    else:
        umap_data = test_data

    if len(umap_data) < 5:
        print("  UMAP样本不足 (<5)，跳过可视化。")
        return

    anchors = [item['anchor'] for item in umap_data]
    positives_gen = [item['positive'] for item in umap_data]
    negatives_loc = [item['negative'] for item in umap_data]

    emb_anchor = get_embeddings(sbert_model, anchors)
    emb_gen = get_embeddings(sbert_model, positives_gen)
    emb_loc = get_embeddings(sbert_model, negatives_loc)

    all_embeddings = np.vstack([emb_anchor, emb_gen, emb_loc])
    labels_np = np.array([0] * len(emb_anchor) + [1] * len(emb_gen) + [2] * len(emb_loc))

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine', random_state=seed, low_memory=True)
    try:
        reduced_embeddings = reducer.fit_transform(all_embeddings)
    except Exception as e:
        print(f"  UMAP转换失败 for {model_name_label}: {e}. 尝试欧氏距离。")
        try:
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', random_state=seed, low_memory=True)
            reduced_embeddings = reducer.fit_transform(all_embeddings)
        except Exception as e_alt:
            print(f"  UMAP再次失败 (欧氏距离): {e_alt}. 跳过此可视化。")
            return

    plt.figure(figsize=(14, 10))
    palette = sns.color_palette("husl", 3)
    category_names = [r'$x_e$ (锚点)', r'$x_{gen}$ (正例-泛化)', r'$x_{loc}$ (负例-近邻)']

    for i, (cat_name, color) in enumerate(zip(category_names, palette)):
        indices = np.where(labels_np == i)[0]
        plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1],
                    s=50, alpha=0.7, color=color, label=cat_name)

    plt.title(f'二维嵌入空间投影 (UMAP) - {model_name_label}', fontsize=16)
    plt.xlabel('UMAP Dimension 1 (可视化)', fontsize=12)
    plt.ylabel('UMAP Dimension 2 (可视化)', fontsize=12)

    plt.legend(title='样本类型', title_fontsize='13', fontsize='11',
               bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0.)

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plot_path = os.path.join(output_dir, f"Vis_A_umap_{model_name_label.replace(' ', '_').replace('/', '-')}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"  UMAP投影图已保存: {plot_path}")
    plt.close()


# --- Main Evaluation Script ---
def main():
    # 此路径应与 train_sbert_experiment.py 中的 OUTPUT_BASE_DIR 一致
    exp_output_base_dir = './exp_sbert_finetuning_output'
    exp_summary_path = os.path.join(exp_output_base_dir, 'experiment_summary.json')

    if not os.path.exists(exp_summary_path):
        print(f"错误: 实验摘要文件未找到于 {exp_summary_path}")
        print("请先运行 train_sbert_experiment.py 脚本。")
        return

    with open(exp_summary_path, 'r') as f:
        exp_summary = json.load(f)

    trained_model_paths = exp_summary['trained_model_paths']
    global_test_dataset_path = exp_summary['global_test_dataset_path']
    base_model_name = exp_summary['base_model_name']
    seed = exp_summary.get('seed', 42)

    eval_results_output_dir = os.path.join(exp_output_base_dir, "evaluation_and_visualization_results")
    os.makedirs(eval_results_output_dir, exist_ok=True)
    print(f"评估结果将保存到: {eval_results_output_dir}")

    if not global_test_dataset_path or not os.path.exists(global_test_dataset_path):
        print(f"错误: 全局测试集路径 ({global_test_dataset_path}) 无效或文件不存在。")
        return

    print(f"加载全局测试集从: {global_test_dataset_path}")
    global_test_dataset = load_from_disk(global_test_dataset_path)
    print(f"全局测试集加载成功，包含 {len(global_test_dataset)} 条样本。")

    if len(global_test_dataset) == 0:
        print("错误: 全局测试集为空。评估无法继续。")
        return

    models_to_eval_meta = {}
    # 1. Baseline SBERT Model
    models_to_eval_meta['Baseline SBERT'] = {'path': base_model_name, 'is_finetuned': False}
    # 2. Finetuned SBERT Models
    for model_label, model_path in trained_model_paths.items():
        if model_path and os.path.exists(model_path):
            models_to_eval_meta[model_label] = {'path': model_path, 'is_finetuned': True}
        else:
            print(f"警告: 模型路径 {model_path} for {model_label} 无效，跳过。")

    if not models_to_eval_meta:
        print("没有模型可供评估。")
        return

    all_metrics_summary = []
    all_tau_analysis_results = []
    all_model_similarities = {}  # Store similarities to avoid recomputing

    # --- Pre-calculate similarities for all models ---
    print("\n预计算所有模型的相似度分数...")
    for model_name, meta in tqdm(models_to_eval_meta.items(), desc="计算相似度"):
        print(f"  加载模型: {model_name} from {meta['path']}")
        try:
            sbert_model = SentenceTransformer(meta['path'])
            sim_e_gen, sim_e_loc = calculate_similarities_for_eval(sbert_model, global_test_dataset)
            all_model_similarities[model_name] = (sim_e_gen, sim_e_loc)
            if torch.cuda.is_available():
                del sbert_model
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  加载模型 {model_name} 或计算相似度时出错: {e}")
            all_model_similarities[model_name] = (None, None)

    # --- Perform evaluations using pre-calculated similarities ---
    for model_name_label, meta in tqdm(models_to_eval_meta.items(), desc="评估模型"):
        print(f"\n{'=' * 20} 正在评估: {model_name_label} {'=' * 20}")

        sim_e_gen, sim_e_loc = all_model_similarities.get(model_name_label, (None, None))

        # A. Embedding Space Metrics
        avg_intra, avg_inter, gap = evaluate_embedding_space(
            model_name_label, sim_e_gen, sim_e_loc, eval_results_output_dir
        )
        if avg_intra is not None:
            all_metrics_summary.append({
                "model": model_name_label,
                "avg_intra_sim (x_e, x_gen)": f"{avg_intra:.4f}",
                "avg_inter_sim (x_e, x_loc)": f"{avg_inter:.4f}",
                "similarity_gap": f"{gap:.4f}"
            })

        # B. τ_pos Sensitivity Analysis data collection
        tau_result = analyze_tau_pos_sensitivity(model_name_label, sim_e_gen, sim_e_loc)
        if tau_result: all_tau_analysis_results.append(tau_result)

        # C. Confusion Points Analysis
        fixed_tau_for_confusion = 0.85
        analyze_confusion_points(model_name_label, global_test_dataset, sim_e_gen, sim_e_loc,
                                 fixed_tau_for_confusion, eval_results_output_dir)

        # 4.A. UMAP Visualization (needs to load model again)
        try:
            sbert_model_for_umap = SentenceTransformer(meta['path'])
            visualize_umap_projection(model_name_label, sbert_model_for_umap, global_test_dataset,
                                      eval_results_output_dir, seed=seed)
            if torch.cuda.is_available():
                del sbert_model_for_umap
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  为 {model_name_label} 进行 UMAP 可视化时出错: {e}")

    # --- Combined Plots & Summaries ---
    if all_tau_analysis_results:
        plt.figure(figsize=(12, 9))
        for result in all_tau_analysis_results:
            plt.plot(result["one_minus_specificity"], result["recalls_gen"], marker='.', markersize=8, linestyle='-', label=result["model_name"])

        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='随机猜测')
        plt.xlabel(r'1 - Specificity$_{loc}$ (FP Rate for $x_{loc}$)', fontsize=12)
        plt.ylabel(r'Recall$_{gen}$ (TP Rate for $x_{gen}$)', fontsize=12)
        plt.title(r'不同模型下 $\tau_{pos}$ 敏感性分析 (ROC式曲线)', fontsize=16)
        plt.legend(fontsize=11, loc='lower right')
        plt.grid(True)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        combined_roc_plot_path = os.path.join(eval_results_output_dir, "Vis_B_combined_tau_sensitivity.png")
        plt.savefig(combined_roc_plot_path, dpi=300)
        print(f"\n组合 τ_pos 敏感性图已保存: {combined_roc_plot_path}")
        plt.close()

    if all_metrics_summary:
        summary_path = os.path.join(eval_results_output_dir, "evaluation_metrics_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics_summary, f, indent=4, ensure_ascii=False)
        print(f"评估指标摘要已保存: {summary_path}")

        print("\n\n{'='*20} 指标摘要 {'='*20}")
        for item in all_metrics_summary:
            print(f"模型: {item['model']}")
            print(f"  平均类内相似度 (x_e, x_gen): {item['avg_intra_sim (x_e, x_gen)']}")
            print(f"  平均类间相似度 (x_e, x_loc): {item['avg_inter_sim (x_e, x_loc)']}")
            print(f"  相似度差距: {item['similarity_gap']}")
            print("-" * 10)

    print("\n\n评估和可视化流程完成。")


if __name__ == "__main__":
    main()
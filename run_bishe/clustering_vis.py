# 论文的进一步实验
# 聚类可视化 5.11

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
import hdbscan
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


# 假设 multiarea_dataset.py 等已在PYTHONPATH或同目录
# from multiarea_dataset import MultiAreaDataset # 假设已正确导入或定义
# from metric_learning import finetune_sentence_transformer # 如有需要

# --- MultiAreaDataset Class (Copied from your provided code if not imported) ---
class MultiAreaDataset:
    def __init__(self, root_dir, dataset_configs, random_sample=True, seed=42):
        self.prompts = []
        self.subjects = []
        self.target_news = []
        self.locality_prompts = []
        self.rephrase_prompts = []
        self.source_files = []  # This will be our ground truth domain label source

        self.all_locality_prompts = []
        self.all_locality_targets = []

        random.seed(seed)
        np.random.seed(seed)

        for filename, K_samples in dataset_configs.items():
            file_path = os.path.join(root_dir, filename)
            if not os.path.isfile(file_path):
                print(f"[⚠️ 警告] 文件 {filename} 不存在，跳过它！")
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if K_samples > len(data) or K_samples is None:
                print(f"[ℹ️ 信息] 采样数 {K_samples} 大于或等于数据集长度 {len(data)} (或未指定)，将使用文件 {filename} 中的全部 {len(data)} 条数据。")
                K_samples = len(data)

            if K_samples == 0:
                print(f"[⚠️ 警告] 文件 {filename} 的采样数为 0，跳过。")
                continue

            if random_sample:
                sampled_data = random.sample(data, K_samples)
            else:
                sampled_data = data[:K_samples]

            print(f'从文件 {file_path} 采样数据：{filename}, 采样数：{K_samples}, 实际获取: {len(sampled_data)}')

            self.prompts.extend([item['prompt'] for item in sampled_data])
            self.subjects.extend([item['subject'] for item in sampled_data])
            self.target_news.extend([item['target_new'] for item in sampled_data])
            self.locality_prompts.extend([item['locality']['prompt'] for item in sampled_data])

            self.all_locality_prompts.extend([item['locality']['prompt'] for item in sampled_data])
            self.all_locality_targets.extend([item['target_new'] for item in sampled_data])

            for item in sampled_data:
                rephrase_list = item.get('generalization', {}).get('rephrase', [])
                if rephrase_list:
                    self.rephrase_prompts.append(rephrase_list[0]['prompt'])
                else:
                    self.rephrase_prompts.append("")

            self.source_files.extend([filename.replace(".json", "")] * len(sampled_data))

        self.locality_inputs = {
            'neighborhood': {
                'prompt': self.all_locality_prompts,
                'ground_truth': self.all_locality_targets
            }
        }

    def __len__(self):
        return len(self.prompts)

    def to_edit_dataset(self):
        return self.prompts, self.rephrase_prompts, self.target_news, self.subjects, self.locality_inputs, self.source_files


# --- End of MultiAreaDataset Class ---


# --- 1. 实验目的 (已在问题中详细描述) ---

# --- 2. 实验数据选取与准备 ---
print("--- 2. 实验数据选取与准备 ---")
dataset_configs_exp = {
    'business_industry.json': 50,
    'human_scientist.json': 50,
    'event_sport.json': 50,
    'geography_forest.json': 50,
    'places_landmark.json': 50
}
multi_area_root_dir = 'EasyEdit/data/output_meta_llama_3_8b_instruct'
if not os.path.exists(multi_area_root_dir):
    print(f"错误：数据集根目录 {multi_area_root_dir} 不存在。请检查路径。")
    exit()

print(f"从以下路径加载数据: {multi_area_root_dir}")
print(f"选定的领域和样本数: {dataset_configs_exp}")

dataset = MultiAreaDataset(
    root_dir=multi_area_root_dir,
    dataset_configs=dataset_configs_exp,
    random_sample=True,
    seed=42
)

prompts, _, _, _, _, ground_truth_labels_str = dataset.to_edit_dataset()

if not prompts:
    print("错误：未能从数据集中加载任何 prompts。")
    exit()

print(f"成功加载 {len(prompts)} 个编辑问题用于实验。")

label_encoder = LabelEncoder()
ground_truth_labels_int = label_encoder.fit_transform(ground_truth_labels_str)
domain_names = label_encoder.classes_
print(f"领域名称: {domain_names}")

# 数据处理流程
print("\n--- 数据处理流程 ---")
sbert_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
print(f"加载原始SBERT模型: {sbert_model_name}")
sbert_model = SentenceTransformer(sbert_model_name)
emb_e = sbert_model.encode(prompts, show_progress_bar=True)
print(f"生成SBERT嵌入向量，形状: {emb_e.shape}")

# --- 场景1: SBERT -> UMAP(50D) -> HDBSCAN (论文流程) ---
print("\n--- 场景1: SBERT -> UMAP(50D) -> HDBSCAN (论文流程) ---")
umap_params_paper = {
    'n_neighbors': 15, 'min_dist': 0.1, 'n_components': 50,
    'metric': 'cosine', 'random_state': 42
}
print(f"UMAP降维 (论文设定) 至 {umap_params_paper['n_components']}D，参数: {umap_params_paper}")
reducer_paper_50d = umap.UMAP(**umap_params_paper)
emb_umap_50d = reducer_paper_50d.fit_transform(emb_e)
print(f"UMAP(50D)降维后向量形状: {emb_umap_50d.shape}")

hdbscan_params_paper = {
    'min_cluster_size': 10, 'min_samples': 5, 'metric': 'euclidean',
    'cluster_selection_method': 'eom', 'allow_single_cluster': False,
    'prediction_data': True  # Important for later prediction/visualization consistency
}
print(f"HDBSCAN聚类 (基于UMAP 50D)，参数: {hdbscan_params_paper}")
clusterer_hdbscan_on_umap50d = hdbscan.HDBSCAN(**hdbscan_params_paper)
hdbscan_ids_on_umap50d = clusterer_hdbscan_on_umap50d.fit_predict(emb_umap_50d)
n_clusters_scene1 = len(set(hdbscan_ids_on_umap50d)) - (1 if -1 in hdbscan_ids_on_umap50d else 0)
n_noise_scene1 = np.sum(hdbscan_ids_on_umap50d == -1)
print(f"场景1 HDBSCAN完成。发现簇数量: {n_clusters_scene1}, 噪声点数量: {n_noise_scene1}")

# --- 场景2: SBERT -> HDBSCAN (直接在原始SBERT嵌入上聚类) ---
print("\n--- 场景2: SBERT -> HDBSCAN (直接在原始SBERT嵌入上聚类) ---")
# 注意：HDBSCAN的metric需要与SBERT嵌入的特性匹配，通常是'cosine'或'euclidean'
# 如果SBERT输出的是归一化向量，'euclidean'和'angular'/'cosine'距离有一定关系
# SBERT的all-MiniLM-L6-v2输出的是归一化向量，所以'euclidean'也可以尝试
hdbscan_params_on_sbert = {
    'min_cluster_size': 10, 'min_samples': 5, 'metric': 'euclidean',  # 使用cosine因为SBERT嵌入常用
    'cluster_selection_method': 'eom', 'allow_single_cluster': False,
    'prediction_data': True
}
print(f"HDBSCAN聚类 (直接基于原始SBERT嵌入)，参数: {hdbscan_params_on_sbert}")
clusterer_hdbscan_on_sbert = hdbscan.HDBSCAN(**hdbscan_params_on_sbert)
# emb_e 已经是 (n_samples, n_features)
hdbscan_ids_on_sbert = clusterer_hdbscan_on_sbert.fit_predict(emb_e)
n_clusters_scene2 = len(set(hdbscan_ids_on_sbert)) - (1 if -1 in hdbscan_ids_on_sbert else 0)
n_noise_scene2 = np.sum(hdbscan_ids_on_sbert == -1)
print(f"场景2 HDBSCAN完成。发现簇数量: {n_clusters_scene2}, 噪声点数量: {n_noise_scene2}")

# --- 3. 可视化方案设计 ---
print("\n--- 3. 可视化方案设计 ---")
umap_params_viz = {
    'n_neighbors': 15, 'min_dist': 0.25, 'n_components': 2,
    'metric': 'cosine', 'random_state': 42
}
print(f"UMAP降维 (可视化目的) 至 {umap_params_viz['n_components']}D，参数: {umap_params_viz}")
reducer_viz_2d = umap.UMAP(**umap_params_viz)

# 可视化坐标：
# 1. 基于 emb_umap_50d (论文流程中的50D中间态) 降维到2D
vis_coords_from_umap50d = reducer_viz_2d.fit_transform(emb_umap_50d)
# 2. 基于原始 emb_e 直接降维到2D
vis_coords_from_sbert = reducer_viz_2d.fit_transform(emb_e)  # Re-fit for fairness if needed, or use a new reducer

df_vis = pd.DataFrame({
    'prompt': prompts,
    'true_domain_str': ground_truth_labels_str,
    'true_domain_id': ground_truth_labels_int,
    # 场景1 (论文流程) 的结果
    'vis_x_scene1': vis_coords_from_umap50d[:, 0],
    'vis_y_scene1': vis_coords_from_umap50d[:, 1],
    'hdbscan_id_scene1': hdbscan_ids_on_umap50d,
    # 场景2 (直接SBERT) 的结果 (聚类ID)
    'hdbscan_id_scene2': hdbscan_ids_on_sbert,
    # 场景2的可视化坐标 (从原始SBERT降维)
    'vis_x_scene2': vis_coords_from_sbert[:, 0],
    'vis_y_scene2': vis_coords_from_sbert[:, 1],
})
plt.style.use('seaborn-whitegrid')  # 更新为新版seaborn的style名
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except Exception as e:
    print(f"设置中文字体失败: {e}. 图表中的中文可能无法正确显示。")

num_true_domains = len(domain_names)
true_palette = sns.color_palette("husl", num_true_domains)


def create_hdbscan_palette(cluster_ids, base_palette_name="Paired"):
    unique_ids = sorted(list(set(cluster_ids)))
    n_clusters = len(unique_ids) - (1 if -1 in unique_ids else 0)
    # 确保即使只有一个非噪声簇，也能从调色板获取颜色
    palette = sns.color_palette(base_palette_name, n_clusters if n_clusters > 0 else 1)

    hdbscan_palette_map = {}
    color_idx = 0
    for cid in unique_ids:
        if cid == -1:
            hdbscan_palette_map[cid] = (0.5, 0.5, 0.5, 0.7)  # Noise: semi-transparent grey
        else:
            if color_idx < len(palette):
                hdbscan_palette_map[cid] = palette[color_idx]
            else:  # Fallback, 理论上 Paired 对于合理数量的簇是够用的
                hdbscan_palette_map[cid] = (random.random(), random.random(), random.random(), 0.7)
            color_idx += 1
    return hdbscan_palette_map


def plot_clusters(df, x_col, y_col, hue_col, title, filename, palette, legend_title, is_hdbscan_hue=False):
    # --- 修改点：调整图形大小和字体，以适应论文 ---
    plt.figure(figsize=(9, 5.5))  # 原为 (14, 10)，调整为更适合论文的尺寸

    hue_order = None
    current_palette = palette

    if is_hdbscan_hue:
        # 为HDBScan的hue和图例准备标签
        df[f'{hue_col}_str'] = df[hue_col].astype(str).replace('-1', 'Noise/-1')
        hue_order = sorted(df[f'{hue_col}_str'].unique(), key=lambda x: int(x.split('/')[0]) if x != "Noise/-1" else -1)
        # 使用预计算的调色板映射
        current_palette = {str(k).replace('-1', 'Noise/-1'): v for k, v in palette.items()}  # 确保键与字符串化标签匹配

        sns.scatterplot(
            data=df, x=x_col, y=y_col, hue=f'{hue_col}_str',
            palette=current_palette, hue_order=hue_order,
            s=35, alpha=0.75, legend='full'  # s 原为 50, alpha 原为 0.8
        )
    else:  # 针对 true_domain_str
        sns.scatterplot(
            data=df, x=x_col, y=y_col, hue=hue_col,
            palette=current_palette,  # 这将是 true_palette 的颜色列表
            s=35, alpha=0.75, legend='full'  # s 原为 50, alpha 原为 0.8
        )

    plt.title(title, fontsize=13)  # 原为 16
    plt.xlabel('UMAP Dimension 1', fontsize=10)  # 原为 12
    plt.ylabel('UMAP Dimension 2', fontsize=10)  # 原为 12

    # 调整图例位置和字体大小，使其更适合较小的图
    # bbox_to_anchor 将图例放在图形外部的右侧。loc='center left' 表示图例的左中点对齐到 bbox_to_anchor 指定的点。
    # borderaxespad 控制图例与 bbox_to_anchor 指定点之间的间距。
    plt.legend(title=legend_title, bbox_to_anchor=(1.02, 0.5), loc='center left',
               borderaxespad=0., fontsize=8, title_fontsize=10)  # fontsize 原为 11, title_fontsize 原为 13

    # rect=[left, bottom, right, top] 调整子图布局以适应图例
    # 这里的 0.78 或 0.80 是为了给右边的图例留出空间
    plt.tight_layout(rect=[0, 0, 0.78, 1])  # rect中right值原为 0.82，可能需根据图例宽度微调

    plt.savefig(filename, dpi=300, bbox_inches='tight')  # bbox_inches='tight' 尝试裁剪掉空白边缘
    print(f"图表已保存为 {filename}")
    # plt.show() # 在脚本中批量生成时，可以注释掉show，避免弹出太多窗口
    plt.close()  # 关闭图像，释放内存，特别是在循环生成多个图时


# --- 可视化 ---
# (确保 df_vis 已经按之前的代码正确生成)

print("\n--- 开始生成图表 (已调整尺寸和样式以适应论文) ---")

# 图表1: SBERT -> UMAP(50D) -> UMAP(2D), 按真实领域着色 (原图表1)
plot_clusters(df_vis, 'vis_x_scene1', 'vis_y_scene1', 'true_domain_str',
              '图1: SBERT->UMAP(50D)->UMAP(2D) (真实领域)',  # 标题可以更简洁
              'plot1_sbert_umap50_umap2_vs_truth_paper.png',  # 文件名区分
              true_palette, '真实领域')

# 图表2: SBERT -> UMAP(50D) -> UMAP(2D), 按场景1的HDBSCAN结果着色 (原图表2)
hdbscan_palette_scene1 = create_hdbscan_palette(hdbscan_ids_on_umap50d)
plot_clusters(df_vis, 'vis_x_scene1', 'vis_y_scene1', 'hdbscan_id_scene1',
              '图2: SBERT->UMAP(50D)->UMAP(2D) (HDBSCAN on UMAP50D)',
              'plot2_sbert_umap50_umap2_vs_hdbscan_on_umap50_paper.png',
              hdbscan_palette_scene1, 'HDBSCAN (on UMAP50D)', is_hdbscan_hue=True)

# 图表3: SBERT -> UMAP(2D), 按真实领域着色 (原图表3a)
plot_clusters(df_vis, 'vis_x_scene2', 'vis_y_scene2', 'true_domain_str',
              '图3: SBERT->UMAP(2D) (真实领域)',
              'plot3_sbert_umap2_vs_truth_paper.png',
              true_palette, '真实领域')

# 新增图表4: SBERT -> UMAP(2D), 按场景2的HDBSCAN结果着色
hdbscan_palette_scene2 = create_hdbscan_palette(hdbscan_ids_on_sbert)
plot_clusters(df_vis, 'vis_x_scene2', 'vis_y_scene2', 'hdbscan_id_scene2',
              '图4: SBERT->UMAP(2D) (HDBSCAN on SBERT)',
              'plot4_sbert_umap2_vs_hdbscan_on_sbert_paper.png',
              hdbscan_palette_scene2, 'HDBSCAN (on SBERT)', is_hdbscan_hue=True)
# --- 4. 聚类评估指标 ---
print("\n--- 4. 聚类评估指标 ---")


def evaluate_clustering(embeddings, labels_true, cluster_labels, method_name):
    print(f"\n评估方法: {method_name}")
    # 过滤掉噪声点进行评估，因为某些指标（如轮廓系数）不接受-1标签
    valid_indices = cluster_labels != -1
    if np.sum(valid_indices) < 2 or len(set(cluster_labels[valid_indices])) < 2:  # silhouette needs at least 2 samples in 2 clusters
        print("  噪声点过多或簇数量不足，无法计算轮廓系数。")
        sil_score = "N/A"
    else:
        # 确保传递给 silhouette_score 的 embeddings 和 labels 维度匹配
        sil_score = silhouette_score(embeddings[valid_indices], cluster_labels[valid_indices])
        print(f"  轮廓系数 (Silhouette Score, 过滤噪声点): {sil_score:.4f}")

    ari = adjusted_rand_score(labels_true, cluster_labels)
    nmi = normalized_mutual_info_score(labels_true, cluster_labels)
    print(f"  调整兰德指数 (ARI): {ari:.4f}")
    print(f"  标准化互信息 (NMI): {nmi:.4f}")
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    num_noise = np.sum(cluster_labels == -1)
    print(f"  发现簇数量: {num_clusters}, 噪声点数量: {num_noise}")
    return {"silhouette": sil_score, "ari": ari, "nmi": nmi, "clusters": num_clusters, "noise": num_noise}


results_metrics = {}

# 评估场景1: HDBSCAN on UMAP(50D) embeddings
# 对于轮廓系数，使用其聚类所基于的嵌入 (emb_umap_50d)
results_metrics['HDBSCAN_on_UMAP50D'] = evaluate_clustering(
    emb_umap_50d, ground_truth_labels_int, hdbscan_ids_on_umap50d, "HDBSCAN on UMAP(50D) Embeddings"
)

# 评估场景2: HDBSCAN on original SBERT embeddings
# 对于轮廓系数，使用其聚类所基于的嵌入 (emb_e)
results_metrics['HDBSCAN_on_SBERT'] = evaluate_clustering(
    emb_e, ground_truth_labels_int, hdbscan_ids_on_sbert, "HDBSCAN on Original SBERT Embeddings"
)

print("\n--- 指标汇总 ---")
df_metrics = pd.DataFrame(results_metrics).T
print(df_metrics)

# --- 5. 预期结果与分析角度 (更新) ---
print("\n--- 5. 预期结果与分析角度 (更新) ---")
print("请根据生成的图表 (图表1-4) 和聚类评估指标进行分析：")
print("1. UMAP(50D)对HDBSCAN聚类的影响 (核心对比):")
print("   - 对比图表2 (HDBSCAN on UMAP50D) 和 图表4 (HDBSCAN on SBERT，投影到SBERT的2D UMAP空间)。")
print("     哪个场景下的HDBSCAN簇在对应的2D投影中看起来更合理、边界更清晰、与真实领域（图表1和图表3）的对应更好？")
print("   - 对比聚类评估指标 (轮廓系数, ARI, NMI, 簇数量, 噪声点数量) for 'HDBSCAN_on_UMAP50D' vs 'HDBSCAN_on_SBERT'.")
print("     如果UMAP(50D)有效，则基于它的HDBSCAN结果的ARI和NMI应该更高（更接近真实领域划分），")
print("     轮廓系数也可能更高（簇内更紧密，簇间更分离），噪声点数量可能更少或更合理。")
print("     簇数量是否更接近真实领域数量？")
print("2. SBERT嵌入本身的领域区分能力:")
print("   - 观察图表1和图表3 (按真实领域着色)。原始SBERT嵌入 (图表3的2D投影) 是否已经能大致区分不同领域？")
print("     UMAP(50D)处理后 (图表1的2D投影)，这种区分度是增强了还是减弱了？")
print("3. HDBSCAN的特性:")
print("   - 观察HDBSCAN识别的噪声点（灰色）。它们在真实领域分布中处于什么位置？")
print("   - 观察HDBSCAN形成的簇的形状和数量。")
print("4. 对领域路由的启示:")
print("   - 结合聚类质量指标和可视化结果，哪种预处理方式（直接SBERT vs SBERT+UMAP50D）产生的聚类结果更适合作为领域路由的依据？")
print("     目标是找到能够准确反映真实知识领域边界的簇。")

print("\n实验完成。请查看生成的PNG图片和控制台输出的评估指标进行分析。")
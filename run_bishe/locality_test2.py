# 直接暴力超参数搜索，看看单路由策略的上限 4.21

import sys
import os

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

import optuna
import numpy as np
from omegaconf import DictConfig, OmegaConf
import copy

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

editing_hparams = ZZZHyperParams
hparams = editing_hparams.from_hparams('EasyEdit/hparams/ZZZ/llama3.2-1b.yaml')
router = KnowRouter(cfg=hparams)

prompts, rephrase_prompts, target_new, subjects, locality_inputs, _ = multiarea_dataset.to_edit_dataset()
locality_prompts = locality_inputs['neighborhood']['prompt']  # 这个loc数据要单独拿出来

# --- 1. 定义 Optuna 的目标函数 ---
# 这个函数会接收一个 trial 对象，用它来建议超参数，
# 然后用这些参数训练（构建路由表）并评估模型，最后返回评估分数。

def objective(trial: optuna.trial.Trial) -> float:
    """
    Optuna 目标函数：尝试一组超参数，返回 Loc 和 Gen 的平均准确率。
    """
    # --- 1.1 定义并建议超参数搜索范围 ---
    # UMAP 参数
    umap_n_neighbors = trial.suggest_int('umap_n_neighbors', 5, 50)  # UMAP 邻居数，影响局部/全局平衡
    umap_min_dist = trial.suggest_float('umap_min_dist', 0.0, 0.5) # UMAP 最小距离，控制嵌入紧密度
    umap_n_components = trial.suggest_int('umap_n_components', 30, 128) # UMAP 降维后的维度

    # HDBSCAN 参数
    hdbscan_min_cluster_size = trial.suggest_int('hdbscan_min_cluster_size', 2, 20) # HDBSCAN 最小簇大小
    # min_samples 通常 <= min_cluster_size，这里先建议一个范围，后面再约束
    hdbscan_min_samples = trial.suggest_int('hdbscan_min_samples', 1, 15)
    # 确保 min_samples <= min_cluster_size
    hdbscan_min_samples = min(hdbscan_min_samples, hdbscan_min_cluster_size)

    # 可以考虑加入 cluster_selection_method，但 'eom' 通常比较鲁棒
    hdbscan_cluster_selection_method = trial.suggest_categorical('hdbscan_cluster_selection_method', ['eom'])

    print(f"\n--- 开始 Trial {trial.number} ---")
    print(f"  参数: umap_n_neighbors={umap_n_neighbors}, umap_min_dist={umap_min_dist:.3f}, "
          f"umap_n_components={umap_n_components}, hdbscan_min_cluster_size={hdbscan_min_cluster_size}, "
          f"hdbscan_min_samples={hdbscan_min_samples}, hdbscan_cluster_selection_method='{hdbscan_cluster_selection_method}'")


    # --- 1.2 使用建议的参数创建配置 ---
    # 深拷贝基础配置，避免修改原始 hparams
    current_hparams = copy.deepcopy(hparams)

    # 更新配置中的参数
    current_hparams.clustering.umap_params.n_neighbors = umap_n_neighbors
    current_hparams.clustering.umap_params.min_dist = umap_min_dist
    current_hparams.clustering.umap_params.n_components = umap_n_components
    # UMAP metric 保持 cosine

    current_hparams.clustering.hdbscan_params.min_cluster_size = hdbscan_min_cluster_size
    current_hparams.clustering.hdbscan_params.min_samples = hdbscan_min_samples
    current_hparams.clustering.hdbscan_params.cluster_selection_method = hdbscan_cluster_selection_method
    # HDBSCAN metric 保持 euclidean, allow_single_cluster 保持 false

    # --- 1.3 实例化和构建路由器 ---
    try:
        router = KnowRouter(cfg=current_hparams) # 使用更新后的配置
        router.build_route_table(prompt_list=prompts)
        print(f"  路由表构建完成. 簇数量: {router.get_num_clusters()}, 离群点数量: {router.get_num_outlier()}") # 假设 get_num_outlier 可以工作
    except Exception as e:
        print(f"  构建路由表时出错: {e}. Pruning trial.")
        # 如果构建失败（例如参数组合无效导致HDBSCAN错误），剪枝该试验
        raise optuna.TrialPruned()

    # --- 1.4 评估 Locality 准确率 ---
    correct_locality_routing = 0
    total_locality = len(prompts)
    if total_locality == 0: return 0.0 # 防止除零错误

    for i in range(total_locality):
        original_prompt = prompts[i]
        # 确保 locality_prompts 列表在此范围内有效
        if i >= len(locality_prompts):
             print(f"警告：索引 {i} 超出 locality_prompts 范围 ({len(locality_prompts)})。")
             continue # 跳过这个样本
        locality_prompt = locality_prompts[i]

        original_cluster_id = router.route_table.get(original_prompt, -99)
        if original_cluster_id == -99: continue # 跳过不在表中的

        try:
            predicted_locality_cluster_id, _ = router.route_with_confidence(locality_prompt)
             # Locality 正确定义：路由到 *不同* 的簇
             # 或者，如果原始是噪声(-1)，局部性也是噪声(-1)也算对。（此处采用之前的严格定义，仅不同就算对）
            # is_correct = (predicted_locality_cluster_id != original_cluster_id) or \
            #              (original_cluster_id == -1 and predicted_locality_cluster_id == -1)
            is_correct = (predicted_locality_cluster_id != original_cluster_id)

            if is_correct:
                correct_locality_routing += 1
        except Exception as e:
            print(f"  评估 Locality 时出错 (Idx {i}): {e}")
            # 如果评估单个样本出错，可以跳过这个样本或者剪枝试验，这里选择跳过
            continue

    locality_accuracy = correct_locality_routing / total_locality if total_locality > 0 else 0.0

    # --- 1.5 评估 Rephrase 准确率 ---
    correct_rephrase_routing = 0
    total_rephrase = len(prompts)
    if total_rephrase == 0: return 0.0 # 防止除零错误

    for i in range(total_rephrase):
        original_prompt = prompts[i]
        # 确保 rephrase_prompts 列表在此范围内有效
        if i >= len(rephrase_prompts):
            print(f"警告：索引 {i} 超出 rephrase_prompts 范围 ({len(rephrase_prompts)})。")
            continue # 跳过这个样本
        rephrase_prompt = rephrase_prompts[i]

        original_cluster_id = router.route_table.get(original_prompt, -99)
        if original_cluster_id == -99: continue # 跳过不在表中的

        try:
            predicted_rephrase_cluster_id, _ = router.route_with_confidence(rephrase_prompt)
            # Rephrase 正确定义：路由到 *相同* 的簇
            is_correct = (predicted_rephrase_cluster_id == original_cluster_id)

            if is_correct:
                correct_rephrase_routing += 1
        except Exception as e:
            print(f"  评估 Rephrase 时出错 (Idx {i}): {e}")
            continue # 跳过这个样本

    rephrase_accuracy = correct_rephrase_routing / total_rephrase if total_rephrase > 0 else 0.0

    # --- 1.6 计算平均准确率并返回 ---
    average_accuracy = (locality_accuracy + rephrase_accuracy) / 2.0

    abs_difference = abs(locality_accuracy - rephrase_accuracy)
    penalty_weight = 0.25

    objective_value = average_accuracy - penalty_weight * abs_difference

    print(f"  Trial {trial.number} 完成. Loc Acc: {locality_accuracy:.4f}, Gen Acc: {rephrase_accuracy:.4f}, "
          f"平均 Acc: {average_accuracy:.4f}", f'目标值: {objective_value:.4f}')

    return objective_value  # 返回目标值，Optuna 会尝试最大化这个值
    
# --- 2. 执行 Optuna 搜索 ---
# 创建一个 study 对象，指定优化方向为最大化
study = optuna.create_study(direction='maximize')

# 运行优化，n_trials 是尝试的总次数，可以根据算力调整
n_trials = 100 # 例如，先尝试 100 次
print(f"开始超参数搜索，将进行 {n_trials} 次试验...")
study.optimize(objective, n_trials=n_trials)

# --- 3. 输出最佳结果 ---
print("\n--- 超参数搜索完成 ---")
print(f"最佳试验序号: {study.best_trial.number}")
print(f"最佳平均准确率 (Loc + Gen / 2): {study.best_value:.4f}")
print("最佳参数组合:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# (可选) 可以用最佳参数重新构建一次路由器并进行详细评估
print("\n使用最佳参数重新评估...")
best_hparams = copy.deepcopy(hparams)
best_hparams.clustering.umap_params.n_neighbors = study.best_params['umap_n_neighbors']
best_hparams.clustering.umap_params.min_dist = study.best_params['umap_min_dist']
best_hparams.clustering.umap_params.n_components = study.best_params['umap_n_components']
best_hparams.clustering.hdbscan_params.min_cluster_size = study.best_params['hdbscan_min_cluster_size']
best_hparams.clustering.hdbscan_params.min_samples = min(study.best_params['hdbscan_min_samples'], study.best_params['hdbscan_min_cluster_size']) # 再次确保约束
best_hparams.clustering.hdbscan_params.cluster_selection_method = study.best_params['hdbscan_cluster_selection_method']

# 重新构建和评估
final_router = KnowRouter(cfg=best_hparams)
final_router.build_route_table(prompt_list=prompts)

# 重新计算最终的 Loc 和 Gen 准确率 (代码与 objective 函数中类似，这里省略重复，只打印最终结果)
# ... (此处省略详细的评估循环代码，可以直接复用 objective 函数中的逻辑，或重新实现) ...
# 假设我们已经重新运行了评估逻辑并得到了 best_loc_acc 和 best_gen_acc
# 这里我们直接使用 best_value 来示意，实际应用中应重新计算以确认
best_avg_acc = study.best_value
# (需要重新运行评估循环来获取准确的 best_loc_acc 和 best_gen_acc)
# best_loc_acc = ...
# best_gen_acc = ...
# print(f"最终 Loc Acc (最佳参数): {best_loc_acc:.4f}")
# print(f"最终 Gen Acc (最佳参数): {best_gen_acc:.4f}")
print(f"最终 平均 Acc (最佳参数，来自Optuna): {best_avg_acc:.4f}")
print(f"最终 簇数量 (最佳参数): {final_router.get_num_clusters()}")
# print(f"最终 离群点数量 (最佳参数): {final_router.get_num_outlier(prompts)}") # 需要实现 get_num_outlier
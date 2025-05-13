import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 数据整理 (与之前相同)
data = {
    ('TSR (τ=0.85)', 'T=60'): {'rewrite_acc': 0.83535, 'Forgetfulness_acc': 0.24468, 'Relation_Specificity_acc': 0.14052, 'Logical_Generalization_acc': 0.30122, 'Subject_Aliasing_acc': 0.52072, 'Reasoning_acc': 0.13406},
    ('TSR (τ=0.85)', 'T=100'): {'rewrite_acc': 0.77372, 'Forgetfulness_acc': 0.31724, 'Relation_Specificity_acc': 0.24183, 'Logical_Generalization_acc': 0.27567, 'Subject_Aliasing_acc': 0.72432, 'Reasoning_acc': 0.28906},
    ('TSR (τ=0.85)', 'T=200'): {'rewrite_acc': 0.80836, 'Forgetfulness_acc': 0.23702, 'Relation_Specificity_acc': 0.15248, 'Logical_Generalization_acc': 0.21588, 'Subject_Aliasing_acc': 0.68397, 'Reasoning_acc': 0.15360},
    ('TSR (τ=0.95)', 'T=60'): {'rewrite_acc': 0.83535, 'Forgetfulness_acc': 0.74008, 'Relation_Specificity_acc': 0.16284, 'Logical_Generalization_acc': 0.30122, 'Subject_Aliasing_acc': 0.49810, 'Reasoning_acc': 0.02536},
    ('TSR (τ=0.95)', 'T=100'): {'rewrite_acc': 0.77372, 'Forgetfulness_acc': 0.70028, 'Relation_Specificity_acc': 0.25617, 'Logical_Generalization_acc': 0.27567, 'Subject_Aliasing_acc': 0.69199, 'Reasoning_acc': 0.10677},
    ('TSR (τ=0.95)', 'T=200'): {'rewrite_acc': 0.80836, 'Forgetfulness_acc': 0.62915, 'Relation_Specificity_acc': 0.19285, 'Logical_Generalization_acc': 0.21913, 'Subject_Aliasing_acc': 0.66278, 'Reasoning_acc': 0.13509},
    ('WISE', 'T=60'): {'rewrite_acc': 0.67735, 'Forgetfulness_acc': 0.07212, 'Relation_Specificity_acc': 0.10982, 'Logical_Generalization_acc': 0.13030, 'Subject_Aliasing_acc': 0.64525, 'Reasoning_acc': 0.25362},
    ('WISE', 'T=100'): {'rewrite_acc': 0.62973, 'Forgetfulness_acc': 0.14075, 'Relation_Specificity_acc': 0.22157, 'Logical_Generalization_acc': 0.19562, 'Subject_Aliasing_acc': 0.59481, 'Reasoning_acc': 0.29427},
    ('WISE', 'T=200'): {'rewrite_acc': 0.62693, 'Forgetfulness_acc': 0.16045, 'Relation_Specificity_acc': 0.22656, 'Logical_Generalization_acc': 0.25065, 'Subject_Aliasing_acc': 0.56422, 'Reasoning_acc': 0.17047}
}

df = pd.DataFrame.from_dict(data, orient='index')
df.index = pd.MultiIndex.from_tuples(df.index, names=['Method', 'T'])
df = df.reset_index()

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

metrics_to_plot = {
    'rewrite_acc': '编辑重写准确率 (Rewrite Accuracy)',
    'Forgetfulness_acc': '遗忘准确率 (Forgetfulness Accuracy - Locality)',
    'Relation_Specificity_acc': '关系特异性准确率 (Relation Specificity Accuracy - Specificity)',
    'Logical_Generalization_acc': '逻辑泛化准确率 (Logical Generalization Accuracy - Generality)',
    'Subject_Aliasing_acc': '主题混淆准确率 (Subject Aliasing Accuracy - Specificity)',
    'Reasoning_acc': '推理准确率 (Reasoning Accuracy - Portability)'
}

T_values_sorted = ['T=60', 'T=100', 'T=200']
methods_sorted = ['TSR (τ=0.85)', 'TSR (τ=0.95)', 'WISE']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

for metric_key, metric_title in metrics_to_plot.items():
    plt.figure(figsize=(10, 6))

    # 为每个方法绘制条形
    bar_width = 0.10  # <<----- 修改条形宽度
    index = np.arange(len(T_values_sorted))

    for i, method in enumerate(methods_sorted):
        method_data = df[df['Method'] == method].set_index('T').reindex(T_values_sorted)[metric_key].values
        # 调整条形位置以适应新的宽度并保持分组
        # (index + i * bar_width) 会让同组的条形中心对齐在 i*bar_width 的偏移处
        # 为了让一组条形图的中心在 index 处，整体向左偏移 (num_methods-1)*bar_width/2
        # 每个条形的中心位置是：index - (len(methods_sorted)-1)*bar_width/2 + i*bar_width
        position_offset = (len(methods_sorted) - 1) * bar_width / 2.0
        positions = index - position_offset + i * bar_width

        plt.bar(positions, method_data, bar_width, label=method, color=colors[i])

    plt.xlabel('编辑数量 (T)', fontsize=14)
    plt.ylabel('准确率', fontsize=14)
    plt.title(f'{metric_title} \n(越高越好)', fontsize=16, pad=20)
    plt.xticks(index, T_values_sorted, fontsize=12)  # X轴刻度仍然在组的中心
    plt.yticks(fontsize=12)
    plt.legend(fontsize=11, title='方法', title_fontsize='12')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05 if metric_key == 'rewrite_acc' or metric_key == 'Forgetfulness_acc' else 0.45)  # 调整Y轴上限

    # 在条形上显示数值
    for i, method in enumerate(methods_sorted):
        method_data = df[df['Method'] == method].set_index('T').reindex(T_values_sorted)[metric_key].values
        position_offset = (len(methods_sorted) - 1) * bar_width / 2.0
        positions = index - position_offset + i * bar_width

        bars = plt.bar(positions, method_data, bar_width, label=method, color=colors[i], alpha=0)  # 透明绘制用于获取位置
        for bar_idx, bar_obj in enumerate(bars):
            yval = method_data[bar_idx]
            plt.text(bar_obj.get_x() + bar_obj.get_width() / 2.0, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom', fontsize=8)  # 减小了字体

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局防止标题重叠
    plt.show()
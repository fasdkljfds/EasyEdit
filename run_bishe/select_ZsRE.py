import json
import random
import os

# 假设您的运行脚本在 EasyEdit 的父目录，或者您可以根据实际情况调整路径
# 例如，如果您的脚本在 EasyEdit 目录内，路径可能是 'data/wise/ZsRE/zsre_mend_train.json'
# 如果您的脚本在 EasyEdit 的父目录，路径可能是 'EasyEdit/data/wise/ZsRE/zsre_mend_train.json'
# 请根据您的实际文件结构修改此处路径
data_path = 'EasyEdit/data/wise/ZsRE/zsre_mend_train.json' # <-- 请检查此路径是否正确

try:
    with open(data_path, 'r', encoding='utf-8') as f:
        zsre_data = json.load(f)

    if len(zsre_data) < 5:
        print(f"警告: 数据集只有 {len(zsre_data)} 条数据，不足 5 条。")
        num_examples = len(zsre_data)
    else:
        num_examples = 5

    # 随机选择 num_examples 条数据
    selected_examples = random.sample(zsre_data, num_examples)

    # 按照图示格式打印
    for i, example in enumerate(selected_examples):
        x_e = example['src']
        y_e = example['alt'] # 在ZsRE数据集中，'alt'通常是编辑后的新答案
        x_loc = example['loc']
        loc_ans = example['loc_ans'] # 局部性问题的原始正确答案
        x_prime_e = example['rephrase'] # 改述的问题

        print(f"x_e, y_e   {x_e} **{y_e}**")
        # 图示中x_loc只显示了prompt和bold的answer，我们这里也遵循这个格式
        print(f"x_loc      {x_loc} **{loc_ans}**")
        print(f"x'_e, y_e  {x_prime_e} **{y_e}**")

        if i < num_examples - 1:
            print("-----------") # 分隔线

except FileNotFoundError:
    print(f"错误: 未找到数据集文件，请检查路径: {data_path}")
except json.JSONDecodeError:
    print(f"错误: 数据集文件 {data_path} 不是有效的 JSON 格式。")
except Exception as e:
    print(f"发生其他错误: {e}")
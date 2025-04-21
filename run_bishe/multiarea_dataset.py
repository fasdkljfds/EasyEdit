# 处理多领域数据集，合并成编辑数据集
# 4.15


import json
import os
import random


class MultiAreaDataset:
    def __init__(self, root_dir, dataset_configs, random_sample=True, seed=42):
        """
        参数：
        - root_dir: 数据文件根目录
        - dataset_configs: dict，格式为 {'文件名.json': 采样数K (或None表示全量)}
        - seed: 随机种子
        """
        self.prompts = []
        self.subjects = []
        self.target_news = []
        self.locality_prompts = []
        self.rephrase_prompts = []
        self.source_files = []

        self.all_locality_prompts = []
        self.all_locality_targets = []

        random.seed(seed)

        for filename, K in dataset_configs.items():
            sample_algo = '随机' if random_sample else '顺序'
            file_path = os.path.join(root_dir, filename)
            print(f'从文件中{file_path}采样数据：{filename}, 采样数：{K}')
            if not os.path.isfile(file_path):
                print(f"[⚠️ 警告] 文件 {filename} 不存在，跳过它！")
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # --- 采样 ---
            if K > len(data):
                print(f"[⚠️ 警告] 采样数 {K} 大于数据集长度 {len(data)}，使用全量数据")
                K = len(data)

            if random_sample:
                data = random.sample(data, min(K, len(data)))  # 随机采样K条
            else:
                data = data[:K]
            # ------------

            self.prompts.extend([item['prompt'] for item in data])
            self.subjects.extend([item['subject'] for item in data])
            self.target_news.extend([item['target_new'] for item in data])
            self.locality_prompts.extend([item['locality']['prompt'] for item in data])

            self.all_locality_prompts.extend([item['locality']['prompt'] for item in data])
            self.all_locality_targets.extend([item['target_new'] for item in data])  # 形式主义罢了

            for item in data:
                rephrase_list = item.get('generalization', {}).get('rephrase', [])
                if rephrase_list:
                    self.rephrase_prompts.append(rephrase_list[0]['prompt'])
                else:
                    self.rephrase_prompts.append("")
                    print(f"[😅 提醒] rephrase 不存在于 {filename} 的某条数据中，哥只能补个空字符串啦")

            self.source_files.extend([filename] * K)

        # 合并后的 locality_inputs 统一成一个入口
        self.locality_inputs = {
            'neighborhood': {
                'prompt': self.all_locality_prompts,
                'ground_truth': self.all_locality_targets
            }
        }

    def __len__(self):
        return len(self.prompts)

    def to_classification_dataest(self):
        return self.prompts, self.all_locality_prompts

    def to_edit_dataset(self):
        return self.prompts, self.rephrase_prompts, self.target_news, self.subjects, self.locality_inputs, self.source_files


if __name__ == '__main__':
    configs = {
        'health_symptom.json': 5,
        'technology_database.json': 5,
        'places_city.json': 5
    }

    dataset = MultiAreaDataset(r'O:\bishe3\EasyEdit\data\output_llama_2_7b_chat_hf', configs, seed=42, random_sample=False)

    # 看一眼 locality_inputs 是否合并得漂漂亮亮
    print("locality 总条数：", len(dataset.locality_inputs['neighborhood']['prompt']))
    print('prompt')
    print(dataset.prompts)
    print('target new')
    print(dataset.target_news)
    print('rephrase')
    print(dataset.rephrase_prompts)
    print('locality')
    print(dataset.locality_inputs)
    print('source')
    print(dataset.source_files)
    print(len(dataset.source_files))



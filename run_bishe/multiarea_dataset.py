# 处理多领域数据集，合并成编辑数据集

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

        all_locality_prompts = []
        all_locality_targets = []

        random.seed(seed)

        for filename, K in dataset_configs.items():
            print('从文件中读取数据：', filename, '采样数：', K)
            file_path = os.path.join(root_dir, filename)
            if not os.path.isfile(file_path):
                print(f"[⚠️ 警告] 文件 {filename} 不存在，跳过它！")
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if K is not None:
                if random_sample:
                    data = random.sample(data, min(K, len(data)))  # 随机采样K条
                else:
                    data = data[:K]

            self.prompts.extend([item['prompt'] for item in data])
            self.subjects.extend([item['subject'] for item in data])
            self.target_news.extend([item['target_new'] for item in data])
            self.locality_prompts.extend([item['locality']['prompt'] for item in data])

            all_locality_prompts.extend([item['locality']['prompt'] for item in data])
            all_locality_targets.extend([item['target_new'] for item in data])  # 形式主义罢了

            for item in data:
                rephrase_list = item.get('generalization', {}).get('rephrase', [])
                if rephrase_list:
                    self.rephrase_prompts.append(rephrase_list[0]['prompt'])
                else:
                    self.rephrase_prompts.append("")
                    print(f"[😅 提醒] rephrase 不存在于 {filename} 的某条数据中，哥只能补个空字符串啦")

        # 合并后的 locality_inputs 统一成一个入口
        self.locality_inputs = {
            'neighborhood': {
                'prompt': all_locality_prompts,
                'ground_truth': all_locality_targets
            }
        }

    def __len__(self):
        return len(self.prompts)

    def get_data(self):
        return self.prompts, self.rephrase_prompts, self.target_news, self.subjects, self.locality_inputs


if __name__ == '__main__':
    configs = {
        'health_symptom.json': 20,
        'technology_database.json': 0,
        'places_city.json': 0
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

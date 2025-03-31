import json
data_filepath = r'O:\bishe3\EasyEdit\data\KnowEdit\benchmark_wiki_counterfact_test_cf.json'

with open(data_filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)  # 假设data是一个包含多个对象的列表

for item in data:
    if isinstance(item, dict):  # 确保item是字典类型
        item.pop('portability', None)  # 使用pop方法安全删除键，如果键不存在也不会报错

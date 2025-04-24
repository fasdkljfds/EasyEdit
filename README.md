# README

### 一、关键启动文件

#### 1. 知识编辑

* **EasyEdit/run_bishe/run_zzz_multiarea.py**

测试ZZZ在multiarea上的表现。适配了两阶段路由

* **EasyEdit/run_bishe/run_zzz.py**

测试ZZZ在counterfact上的表现。未适配两阶段路由

* **EasyEdit/run_bishe/run_wiser.py**

测试WISE在coutnerfact上的表现。

* **EasyEdit/run_bishe/run_counterfact.py**

比run_wiser更通用的counterfact测试脚本，适配了WISE、GRACE和MEMIT

* **EasyEdit/run_bishe/run_baseline_zzz.py**

测试WISE、GRACE在multiarea上的表现。

#### 2. 训练和测试等

* **EasyEdit/run_bishe/metric_learning.py**

在Multiarea上微调SBERT模型

* **EasyEdit/run_bishe/metric_learning_zsre.py**

在ZsRE上微调SBERT模型。但实际上，SBERT未经微调就能区分locality_inputs和prompt。

* **EasyEdit/run_bishe/multiarea_dataset.py**

用于加载multiarea数据集。

* **EasyEdit/run_bishe/semantic_alignment.py**

从Llama3.2中提取中间层输出

‍

### 二、运行ZZZ

#### 0. 配置运行环境

‍

```python
使用环境：
python3.9
```

在ipynb文件中配置环境代码：

```python
!git clone https://github.com/fasdkljfds/EasyEdit.git
%cd EasyEdit
!pip install -r requirements311.txt
from huggingface_hub import login
login(token='your_huggingface_token')
%cd ..

```

‍

#### 1. 微调SBERT

在指定数据集上微调SBERT。运行metric_learning，超参数硬编码于main代码中的train_hparams字典内，

其中triplet_margin是关键参数，较大的triplet_margin值会导致微调后的SBERT对语义变化极度敏感，甚至丧失部分原本能力，这可能是它在聚类中表现不佳的原因。

微调后，应当把超参数中output_model_dir/final_model_subdir写入hparams/zzz/llama3.2-1b.yaml的boundary_model_name中，以便于程序加载模型。

![image](assets/image-20250424151139-ikhpnba.png)​

#### 2. 执行知识编辑

运行**run_zzz_multiarea.py文件。**

启动命令：

```python
!python 'EasyEdit/run_bishe/run_zzz_multiarea.py' --editing_method=ZZZ --hparams_dir=EasyEdit/hparams/ZZZ/llama3.2-1b.yaml --data_dir=EasyEdit/data/output_meta_llama_3_8b_instruct --data_configs=business_industry:10,human_scientist:11,event_sport:11,geography_forest:10,places_landmark:10 --retrain=True --two_stages=True --boundary_threshold=0.5 --boundary_model_name=/kaggle/input/finetuned1/pytorch/default/1/finetuned_sbert_triplet/final_model_1
```

参数说明：

```python
--data_configs 选择加载multiarea数据集中的数据
--retrain 设置为True即可
--two_stages 使用两阶段路由，设置为True即可，会覆盖hparams下的超参数配置
--boundary_threshold 会覆盖超参数配置。对于locality_input和rephrase_prompts，和prompts的余弦相似度超过0.5的认为是不相关问题，低于0.5的认为是相关问题
--boundary_model_name 微调后的SBRET路径，会覆盖hparams下的超参数配置
```

‍

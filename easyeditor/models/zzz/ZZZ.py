import copy
import gc

import numpy as np
import torch
import transformers
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from .utils import EarlyStopMeter, EditingMeanAct


def brackets_to_periods(name):
    return name.replace("[", ".").replace("]", "")


def parent_module(model, pname):
    """
    小蝌蚪找妈妈
    """
    components = pname.split('.')
    parent = model

    for component in components[:-1]:
        if hasattr(parent, component):
            parent = getattr(parent, component)
        elif component.isdigit():
            parent = parent[int(component)]
        else:
            raise RuntimeError(f"Couldn't find child module {component}")

    if not hasattr(parent, components[-1]):
        raise RuntimeError(f"Couldn't find child module {components[-1]}")

    return parent


def euc(query, key, config, act_mask=None, infer=False):
    # Euclidean distance
    return 0
    #
    # act_fn =
    # l2_norm = torch.norm(act_fn(key) - act_fn(query), dim=-1)
    # if infer and l2_norm.size(1) > 100:
    #     topk = torch.topk(l2_norm, k=1, largest=True)
    #     return topk.values.mean()
    #
    # if act_mask is not None:
    #     return torch.sum(l2_norm * act_mask, dim=1) / torch.sum(act_mask, dim=1)
    # else:
    #     return torch.mean(l2_norm, dim=-1)
    #


class ZZZ(torch.nn.Module):
    def __init__(self, config, model, device, num_ffn, ffn_id):
        """
        ZZZ模型的初始化函数
        Args:
            config: 配置参数
            model: 预训练模型
            device: GPU/CPU设备
            num_ffn: 总FFN数量
            ffn_id: 当前编辑的FFN ID
        """
         
        super(ZZZ, self).__init__()
        self.config = config
        self.model = model
        self.device = device
        self.num_ffn = num_ffn
        self.ffn_id = ffn_id
        
        if hasattr(self.model.config, 'hidden_act'):
            self.config.hidden_act = self.model.config.hidden_act
        elif hasattr(self.model.config, 'activation_function'):
            self.config.hidden_act = self.model.config.activation_function

        # --- 定位编辑层 ---
        layer = config.inner_params[0]  # model.layers[12].mlp.down_proj.weight
        self.adapter_layer = None
        self.original_layer = None

        suffixes = [".weight", ".bias"]
        self.layer = layer.rsplit(".", 1)[0] if any(layer.endswith(x) for x in suffixes) else layer  # model.layers[12].mlp.down_proj
        # ----------------

        # 冻结参数
        for n, p in self.model.named_parameters():
            p.requires_grad = False

        # 这里总是 True
        if isinstance(self.model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
            transpose = False
        else:
            transpose = True

        self.edit_module = parent_module(self.model, brackets_to_periods(self.layer))  # model.layers[12].mlp

        self.layer_name = self.layer.rsplit(".", 1)[-1]
        adapter_layer = getattr(self.edit_module, self.layer_name)
        # ----------------

        if type(adapter_layer) is not ZZZAdapter:
            setattr(self.edit_module, self.layer_name, ZZZAdapter(config, adapter_layer, transpose=transpose, n_area=self.num_ffn))
            self.original_layer = copy.deepcopy(adapter_layer)
            print(f"New weights successfully inserted into {layer}")

        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    def get_adapter_layer(self):
        # 中间变量信息：
        # edit_module = model.layers[12].mlp
        # layer_name = down_proj
        adapter_layer = getattr(self.edit_module, self.layer_name)
        assert type(adapter_layer) is ZZZAdapter, print('Adapter Layer is not added correctly....')
        return adapter_layer.to(self.model.device)

    def __call__(self, **kwargs):
        return self.model(**kwargs)

    # 编辑单个样本的函数。因为不用batch
    def edit(self, config, tokens, act_mask=None, deact_mask=None):
        last_prompt_token_loc = (tokens["labels"] == -100).sum(dim=-1) - 1
        setattr(eval(f"self.model.{self.layer}"), "training", True)
        setattr(eval(f"self.model.{self.layer}"), "editing", True)

        self.get_adapter_layer().set_parameter_tunable()

        # --- train ZZZAdapter value ---
        loss_meter = EarlyStopMeter()
        for i in range(config.n_iter):
            if i == 0:
                # --- we only need to create an optimizer for the first iteration (but forward pass instantiates the key, so optimzer is passed after first inference) ---
                optimizer = torch.optim.SGD([self.get_adapter_layer().new_weight], config.edit_lr, weight_decay=1e-5)

            ft_loss = self._cal_ft_loss(tokens, last_prompt_token_loc)

            loss = ft_loss

            if loss_meter.stop():
                self.get_adapter_layer().save_editing_activation()  # add last gradient
                break
            if i == config.n_iter - 1:
                self.get_adapter_layer().save_editing_activation()  # add last gradient

            optimizer.zero_grad()
            loss.backward()

            self.get_adapter_layer().mask_new_weight_gradient()
            print(
                f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)}"
            )

            optimizer.step()
            loss_meter.update(loss.item())

            if type(self.config.norm_constraint) is float:
                self._norm_constraint(self.config.norm_constraint)

        setattr(eval(f"self.model.{self.layer}"), "editing", False)
        setattr(eval(f"self.model.{self.layer}"), "training", False)

        editing_total_cnt = getattr(eval(f"self.model.{self.layer}"), "editing_total_cnt") + 1
        setattr(eval(f"self.model.{self.layer}"), "editing_total_cnt", editing_total_cnt)

    def _norm_constraint(self, norm_constraint):
        """正则化避免过拟合"""
        new_weight = self.get_adapter_layer().new_weight
        original_weight = self.get_adapter_layer().weight
        with torch.no_grad():
            new_weight[...] = torch.clamp(
                new_weight, min=original_weight - norm_constraint, max=original_weight + norm_constraint
            )

    def _cal_ft_loss(self, tokens, last_prompt_token_loc):
        """计算微调损失"""
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        bs = tokens["input_ids"].shape[0] - k
        logits = self.model(**tokens).logits
        shift_logits = logits[:-k, :-1, :].contiguous()
        shift_labels = tokens['labels'][:-k, 1:].contiguous()

        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(bs, -1)

        label_mask = torch.zeros_like(loss, dtype=torch.bool)

        for i, col_index in enumerate(last_prompt_token_loc[:-k]):
            label_mask[i, col_index - 1:] = True

        ft_loss = ((loss * label_mask).sum(1) / label_mask.sum(1)).mean()
        return ft_loss


class ZZZAdapter(torch.nn.Module):
    def __init__(self, config, layer, transpose, n_area):
        # layer = model.layers[12].mlp.down_proj
        super(ZZZAdapter, self).__init__()

        self.layer = layer
        self.weight = self.layer.weight   # model.layers[12].mlp.down_proj.weight
        self.device = layer.weight.device
        self.config = config
        self.ffn_id = 0

        # --- 创建专家层 ---
        self.expert_layers = torch.nn.ModuleList([
            copy.deepcopy(layer) for _ in range(n_area)  # 深拷贝原始层
        ])
        for expert in self.expert_layers:
            expert.weight.data.copy_(layer.weight.data)
            if hasattr(layer, 'bias') and layer.bias is not None:
                expert.bias.data.copy_(layer.bias.data)
        # ----------------

        # self.original_layer = copy.deepcopy(self.layer)
        self.memory_weight = []
        self.memory_mean_act = []

        if 'gpt2' in self.config.model_name:
            self.bias = self.layer.bias  # For Conv1D
        else:
            self.bias = None

        self.merge_cnt = 0  # only for retrieve
        assert not self.weight.requires_grad, print('Original Layer can not be tunable....')

        self.used_mask = None

        if transpose:
            self.key_shape = layer.weight.shape[1]
            self.value_shape = layer.weight.shape[0]
        else:
            self.key_shape = layer.weight.shape[0]
            self.value_shape = layer.weight.shape[1]
        self.training = False
        self.editing = False

        self.editing_mean_act = EditingMeanAct()
        self.editing_total_cnt = 0

    def set_ffn(self, ffn_id):
        """
        设置当前编辑的FFN ID
        """
        self.ffn_id = ffn_id

    def set_parameter_tunable(self):
        """
        设置专家层的参数为可训练状态
        """
        for expert in self.expert_layers:
            expert.weight.requires_grad = True
            if hasattr(expert, 'bias') and expert.bias is not None:
                expert.bias.requires_grad = True

    def save_editing_activation(self):
        in_scope_dist = euc(self.original_layer_output[:-1, ...], self.new_weight_layer_output[:-1, ...], self.config)
        self.editing_mean_act.update(in_scope_dist.mean().item())

    def forward(self, *args):
        return self.expert_forward(*args)

    def expert_forward(self, _input: Tensor) -> Tensor:
        new_weight = self.expert_layers[self.ffn_id].weight
        return F.linear(_input, new_weight) if self.bias is None else torch.addmm(self.bias, _input.view(-1, _input.size(-1)), new_weight).view(_input.size()[:-1] + (self.layer.nf,))

    def generate_activation_mask(self, mask_ratio):
        p_grad = self.new_weight.reshape(-1)
        p_mask = np.random.choice([1, 0], size=p_grad.size()[0], p=[mask_ratio, 1 - mask_ratio])
        p_mask = torch.from_numpy(p_mask).to(p_grad.device)
        self.weight_mask = p_mask

    def generate_non_overlapping_mask(self, mask_ratio):
        p_grad = self.new_weight.reshape(-1)
        mask_size = int(mask_ratio * p_grad.size()[0])
        if self.used_mask is None:
            self.used_mask = np.zeros(p_grad.size()[0], dtype=bool)
        available_indices = np.where(~self.used_mask)[0]  # 获取未被遮罩的元素索引
        if len(available_indices) < mask_size:
            raise ValueError("Not enough unused elements to generate a new mask.")
        chosen_indices = np.random.choice(available_indices, size=mask_size, replace=False)
        mask_array = np.zeros(p_grad.size()[0], dtype=int)
        mask_array[chosen_indices] = 1
        self.used_mask[chosen_indices] = True  # 更新遮罩状态
        self.weight_mask = torch.from_numpy(mask_array).to(p_grad.device)

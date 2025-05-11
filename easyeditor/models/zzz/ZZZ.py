import copy
import random

import torch
from torch.nn import functional as F
from .utils import EarlyStopMeter, EditingMeanAct
import transformers
import numpy as np
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
import torch.nn as nn
from .router import KnowRouter
import gc


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


class ZZZ(torch.nn.Module):
    def __init__(self, config, model, device, router):
        super(ZZZ, self).__init__()
        self.config = config
        self.model = model

        if hasattr(self.model.config, 'hidden_act'):
            self.config.hidden_act = self.model.config.hidden_act
        elif hasattr(self.model.config, 'activation_function'):
            self.config.hidden_act = self.model.config.activation_function
        # self.tokenizer = model.tokenizer

        layer = config.inner_params[0]
        self.device = device
        self.adapter_layer = None
        self.original_layer = None

        # --- ensure proper formatting (WISE edits weights matrices) ---
        suffixes = [".weight", ".bias"]
        # 这个layer 就是目标编辑层
        self.layer = layer.rsplit(".", 1)[0] if any(layer.endswith(x) for x in suffixes) else layer

        for n, p in self.model.named_parameters():
            p.requires_grad = False

        if isinstance(self.model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
            transpose = False
        else:
            transpose = True

        # --- Add WISE to chosen layers ---
        self.edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        self.layer_name = self.layer.rsplit(".", 1)[-1]
        adapter_layer = getattr(self.edit_module, self.layer_name)

        # WISEAdapter就是那个新的层，WISE算法的核心结构都实现在这个类里面
        if type(adapter_layer) is not ZZZAdapter:

            setattr(self.edit_module, self.layer_name, ZZZAdapter(config, adapter_layer, router=router, transpose=transpose))
            self.original_layer = copy.deepcopy(adapter_layer)
            print(f"New weights successfully inserted into {layer}")
        self.get_adapter_layer().generate_activation_mask(self.config.mask_ratio)

        self.representation_layer_index = self.config.representation_layer_index
        self._hidden_state_index_to_access = self.representation_layer_index + 1


        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    # Forward
    # !! 评估的时候，也就是输出的时候，调用的就是这个函数
    def __call__(self, **kwargs):
        return self.model(**kwargs)

    def _get_routing_representation(self, prompt_tokens):
        pass

    # 重置，应该是适配单次编辑的
    def reset_layer(self):
        layer = getattr(self.edit_module, self.layer_name)
        del layer
        setattr(self.edit_module, self.layer_name, self.get_adapter_layer().original_layer)
    
    # 正如名字
    def get_adapter_layer(self):
        adapter_layer = getattr(self.edit_module, self.layer_name)
        assert type(adapter_layer) is ZZZAdapter, print('Adapter Layer is not added correctly....')
        return adapter_layer.to(self.model.device)

    # TODO: generation
    def generate(self, *args, **kwargs):
        setattr(eval(f"self.model.{self.layer}"), "key_id", -1)
        return self.model.generate(*args, **kwargs)

    def route(self, prompt):
        self.get_adapter_layer().route(prompt)

    def edit(self, config, tokens, act_mask=None, deact_mask=None, prompt=None):
        assert prompt is not None
        last_prompt_token_loc = (tokens["labels"] == -100).sum(dim=-1) - 1

        setattr(eval(f"self.model.{self.layer}"), "training", True)
        setattr(eval(f"self.model.{self.layer}"), "editing", True)
        self.get_adapter_layer().set_parameter_tunable()
        self.get_adapter_layer().route(prompt)

        # --- train Wise value ---
        loss_meter = EarlyStopMeter()
        for i in range(config.n_iter):
            if i == 0:
                # --- we only need to create an optimizer for the first iteration (but forward pass instantiates the key, so optimzer is passed after first inference) ---
                optimizer = torch.optim.SGD([self.get_adapter_layer().get_expert_weight()], config.edit_lr, weight_decay=1e-5)

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


            # print(f"loss {np.round(loss.item(), 3)}")

            optimizer.step()
            loss_meter.update(loss.item())

            if type(self.config.norm_constraint) is float:
                self._norm_constraint(self.config.norm_constraint)

        # --- pull out info we want to log from the Wise layer ---
        setattr(eval(f"self.model.{self.layer}"), "editing", False)
        setattr(eval(f"self.model.{self.layer}"), "training", False)


    def _norm_constraint(self, norm_constraint):
        new_weight = self.get_adapter_layer().new_weight
        original_weight = self.get_adapter_layer().weight
        with torch.no_grad():
            new_weight[...] = torch.clamp(
                new_weight, min=original_weight - norm_constraint, max=original_weight + norm_constraint
            )

    def _cal_ft_loss(self, tokens, last_prompt_token_loc):
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
    def __init__(self, config, layer, transpose, router: KnowRouter):
        super(ZZZAdapter, self).__init__()

        self.layer = layer
        self.weight = self.layer.weight
        self.device = layer.weight.device
        self.config = config
        self.ffn_id = 0
        self.router = router

        self.new_weight = copy.deepcopy(self.weight)
        self.original_layer = copy.deepcopy(self.layer)

        # --- 创建专家层 ---
        n_area = router.get_num_clusters()
        self.expert_layers = torch.nn.ModuleList([
            copy.deepcopy(layer) for _ in range(n_area+2)  # 深拷贝原始层
        ])

        # expert_layers[0]是原始层
        for expert in self.expert_layers:
            expert.weight.data.copy_(layer.weight.data)
            if hasattr(layer, 'bias') and layer.bias is not None:
                expert.bias.data.copy_(layer.bias.data)
        # ----------------
        self.weight_mask = [None]*(n_area+1)


        if 'gpt2' in self.config.model_name:
            self.bias = self.layer.bias # For Conv1D
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

    #       
    def set_parameter_tunable(self):
        for expert in self.expert_layers:
            expert.weight.requires_grad = True
            if hasattr(expert, 'bias') and expert.bias is not None:
                expert.bias.requires_grad = True

    def set_ffn(self, ffn_id):
        """
        设置当前编辑的FFN ID
        """
        self.ffn_id = ffn_id
    
    def route(self, prompt):
        if type(prompt) is not str:
            prompt = prompt[0]


        ffn_id = self.router.route(prompt)+1
        print(f'[router] {prompt} ==> {ffn_id-1}')

        self.set_ffn(ffn_id)

    def get_expert_weight(self) -> Tensor:
        """
        获取当前专家层的权重
        """
        return self.expert_layers[self.ffn_id].weight

    def save_editing_activation(self):
        pass

    def generate_activation_mask(self, mask_ratio):
        p_grad = self.get_expert_weight().reshape(-1)
        p_mask = np.random.choice([1, 0], size=p_grad.size()[0], p=[mask_ratio, 1 - mask_ratio])
        p_mask = torch.from_numpy(p_mask).to(p_grad.device)
        self.weight_mask = p_mask


    def expert_forward(self, input: Tensor) -> Tensor:
        current_expert = self.expert_layers[self.ffn_id]
        bias = current_expert.bias if hasattr(current_expert, 'bias') else None
        return F.linear(input, self.get_expert_weight()) if bias is None else torch.addmm(bias, input.view(-1, input.size(-1)), self.get_expert_weight()).view(input.size()[:-1] + (self.layer.nf,))


    def mask_new_weight_gradient(self):
        assert self.get_expert_weight().grad is not None, print('Gradient Collection for New Weight error, gradient not found')
        p_size = self.get_expert_weight().grad.size()
        p_grad = self.get_expert_weight().grad.reshape(-1)

        p_grad = p_grad * self.weight_mask
        self.get_expert_weight().grad = p_grad.view(p_size).to(self.get_expert_weight().grad.dtype)

    def forward(self, *args):
        layer_out = self.expert_forward(*args)
        return layer_out

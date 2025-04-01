import copy
import random

import torch
from torch.nn import functional as F
from .utils import parent_module, brackets_to_periods, EarlyStopMeter, EditingMeanAct
import transformers
import numpy as np
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from .merge import slerp, GTA, linear
import torch.nn as nn
import gc

# 新增: 导入语义解耦所需的模块
from sklearn.decomposition import PCA
from torch.linalg import svd

merge_dict = {
    'slerp': slerp(),
    'ties': GTA('magnitude', 'sum', normalize=True),
    'magnitude_norm': GTA('magnitude', None, normalize=True),
    'magnitude': GTA('magnitude', None, normalize=False),
    'sign': GTA(None, 'sum', normalize=True),
    'dare_ties': GTA('rescaled_random', 'sum'),
    'dare_linear': GTA('random', None),
    'linear': linear()
}

edit_history = []
merge_group_edit_history = []

# 新增: 语义因果解耦模块
class SemanticCausalDisentanglement:
    """实现语义因果解耦的类，包含因果干预和概念空间投影"""
    
    def __init__(self, config):
        self.config = config
        self.concept_basis = None
        self.concept_dim = config.concept_dim if hasattr(config, 'concept_dim') else 32
        
    def compute_concept_basis(self, activations):
        """计算概念空间的基向量"""
        # 使用SVD分解获取概念空间
        if activations.dim() > 2:
            activations = activations.reshape(-1, activations.size(-1))
        
        # 添加随机噪声以增强稳定性
        noise = torch.randn_like(activations) * 1e-6
        activations = activations + noise
        
        # 使用SVD分解
        U, S, V = svd(activations)
        
        # 取前concept_dim个维度作为概念空间
        self.concept_basis = V[:, :self.concept_dim]
        return self.concept_basis
    
    def project_to_concept_space(self, activations):
        """将激活投影到概念空间"""
        if self.concept_basis is None:
            self.compute_concept_basis(activations)
            
        if activations.dim() > 2:
            orig_shape = activations.shape
            activations = activations.reshape(-1, activations.size(-1))
            
        # 投影到概念空间
        projected = torch.matmul(activations, self.concept_basis)
        
        # 如果需要恢复原始维度
        if activations.dim() > 2:
            projected = projected.reshape(*orig_shape[:-1], self.concept_dim)
            
        return projected
    
    def causal_intervention(self, original_act, edited_act, act_mask=None):
        """
        执行因果干预，将目标区域的激活替换为编辑后的激活，
        同时保持非目标区域不变
        """
        if act_mask is None:
            return edited_act
            
        # 投影到概念空间
        orig_proj = self.project_to_concept_space(original_act)
        edited_proj = self.project_to_concept_space(edited_act)
        
        # 创建干预结果
        intervention = torch.zeros_like(orig_proj)
        
        # 根据mask执行干预
        if act_mask.dim() < orig_proj.dim():
            # 扩展维度以匹配
            expanded_mask = act_mask.unsqueeze(-1).expand_as(orig_proj)
        else:
            expanded_mask = act_mask
            
        # 在目标区域使用编辑后的值，在非目标区域保持原值
        intervention = edited_proj * expanded_mask + orig_proj * (1 - expanded_mask)
        
        # 投影回原始空间
        intervention_orig_space = torch.matmul(intervention, self.concept_basis.transpose(0, 1))
        
        return intervention_orig_space
    
    def compute_orthogonality_loss(self, act1, act2):
        """
        计算两组激活之间的正交性损失
        """
        # 投影到概念空间
        proj1 = self.project_to_concept_space(act1)
        proj2 = self.project_to_concept_space(act2)
        
        # 归一化
        norm1 = F.normalize(proj1, p=2, dim=-1)
        norm2 = F.normalize(proj2, p=2, dim=-1)
        
        # 计算余弦相似度
        cos_sim = torch.abs(torch.sum(norm1 * norm2, dim=-1))
        
        # 正交性损失 (鼓励正交: 余弦相似度为0)
        orthogonality_loss = cos_sim.mean()
        
        return orthogonality_loss

def euc(query, key, config, act_mask=None, infer=False):
    # Euclidean distance

    act_fn = ACT2FN[config.hidden_act]
    l2_norm = torch.norm(act_fn(key) - act_fn(query), dim=-1)
    if infer and l2_norm.size(1) > 100:
        topk = torch.topk(l2_norm, k=1, largest=True)
        return topk.values.mean()

    if act_mask is not None:
        return torch.sum(l2_norm * act_mask, dim=1) / torch.sum(act_mask, dim=1)
    else:
        return torch.mean(l2_norm, dim=-1)

# 修改WISE类以加入语义因果解耦支持
class WISE(torch.nn.Module):
    def __init__(self, config, model, device):
        super(WISE, self).__init__()
        self.config = config
        self.model = model
        self.config = config
        if hasattr(self.model.config, 'hidden_act'):
            self.config.hidden_act = self.model.config.hidden_act
        elif hasattr(self.model.config, 'activation_function'):
            self.config.hidden_act = self.model.config.activation_function
        # self.tokenizer = model.tokenizer
        layer = config.inner_params[0]
        self.device = device
        self.adapter_layer = None
        self.original_layer = None
        
        # 新增: 初始化语义因果解耦模块
        self.semantic_disentanglement = SemanticCausalDisentanglement(config)
        
        # 新增: 冲突度量追踪
        self.conflict_meter = ConflictMeter(config.conflict_threshold if hasattr(config, 'conflict_threshold') else 0.6)

        # --- ensure proper formatting (WISE edits weights matrices) ---
        suffixes = [".weight", ".bias"]
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

        if type(adapter_layer) is not WISEAdapter:
            setattr(self.edit_module, self.layer_name, WISEAdapter(config, adapter_layer, transpose=transpose))
            self.original_layer = copy.deepcopy(adapter_layer)
            print(f"New weights successfully inserted into {layer}")
        
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    # Forward
    def __call__(self, **kwargs):
        if not self.config.retrieve:
            if hasattr(self.get_adapter_layer(), 'editing') and not self.get_adapter_layer().editing:
                # final merge
                if not self.get_adapter_layer().original_layer.weight.equal(self.get_adapter_layer().new_weight) and self.get_adapter_layer().editing_total_cnt >= self.config.save_freq:
                    self.get_adapter_layer().memory_weight.append(self.get_adapter_layer().new_weight)
                if len(self.get_adapter_layer().memory_weight) > 0 and self.get_adapter_layer().editing_total_cnt >= self.config.save_freq:
                    print('length of memory is ', len(self.get_adapter_layer().memory_weight), '!!!!!!')
                    self.get_adapter_layer().merge_weight()
        return self.model(**kwargs)

    def reset_layer(self):
        layer = getattr(self.edit_module, self.layer_name)
        del layer
        setattr(self.edit_module, self.layer_name, self.get_adapter_layer().original_layer)

    def get_adapter_layer(self):
        adapter_layer = getattr(self.edit_module, self.layer_name)
        assert type(adapter_layer) is WISEAdapter, print('Adapter Layer is not added correctly....')
        return adapter_layer

    # TODO: generation
    def generate(self, *args, **kwargs):
        setattr(eval(f"self.model.{self.layer}"), "key_id", -1)
        return self.model.generate(*args, **kwargs)
                         
    # 修改edit方法以加入语义因果解耦与正交性约束
    def edit(self, config, tokens, act_mask=None, deact_mask=None):
        # for retrieve ##
        global edit_history
        global merge_group_edit_history
        edit_history.append([{f"{k1}" : v1.to('cpu') for k1, v1 in tokens.items()}, False])
        # for retrieve ##
        last_prompt_token_loc = (tokens["labels"] == -100).sum(dim=-1) - 1

        setattr(eval(f"self.model.{self.layer}"), "training", True)
        setattr(eval(f"self.model.{self.layer}"), "editing", True)
        self.get_adapter_layer().set_parameter_tunable()
        
        # 修改: 使用自适应稀疏率而非固定mask_ratio
        if getattr(eval(f"self.model.{self.layer}"), "editing_total_cnt") % self.config.save_freq == 0:
            if hasattr(self.config, 'use_adaptive_sparsification') and self.config.use_adaptive_sparsification:
                # 获取当前冲突程度
                conflict_level = self.conflict_meter.get_conflict_level()
                # 根据冲突程度自适应调整稀疏率
                adaptive_mask_ratio = min(0.9, max(0.1, self.config.mask_ratio * (1 + conflict_level)))
                print(f"Using adaptive mask ratio: {adaptive_mask_ratio} (conflict level: {conflict_level})")
                self.get_adapter_layer().generate_activation_mask(adaptive_mask_ratio)
            else:
                self.get_adapter_layer().generate_activation_mask(self.config.mask_ratio)

        # --- train Wise value ---
        loss_meter = EarlyStopMeter()
        for i in range(config.n_iter):

            if i == 0:
                # --- we only need to create an optimizer for the first iteration (but forward pass instantiates the key, so optimzer is passed after first inference) ---
                optimizer = torch.optim.SGD([self.get_adapter_layer().new_weight], config.edit_lr, weight_decay=1e-5)

            ft_loss = self.__cal_ft_loss(tokens, last_prompt_token_loc)

            act_loss = self.__cal_activation_loss(self.get_adapter_layer().original_layer_output, self.get_adapter_layer().new_weight_layer_output,
                                                  config=config, act_mask=act_mask, deact_mask=deact_mask)
            
            # 新增: 计算正交性损失
            orthogonality_loss = 0
            if hasattr(self.config, 'use_orthogonality') and self.config.use_orthogonality:
                print('into orthogonality loss')
                # 对于每个激活掩码
                for j, a_mask in enumerate(act_mask):
                    if a_mask is not None:
                        # 计算目标知识与非目标知识之间的正交性
                        target_act = self.get_adapter_layer().original_layer_output * a_mask.unsqueeze(-1)
                        nontarget_act = self.get_adapter_layer().original_layer_output * deact_mask[j].unsqueeze(-1)
                        
                        # 只有当两者都有有效值时才计算
                        if target_act.abs().sum() > 0 and nontarget_act.abs().sum() > 0:
                            print('values valid')
                            orth_loss = self.semantic_disentanglement.compute_orthogonality_loss(
                                target_act, nontarget_act
                            )
                            orthogonality_loss += orth_loss
                
                # 平均所有掩码的正交性损失
                if len(act_mask) > 0:
                    orthogonality_loss /= len(act_mask)
                
                # 乘以权重因子
                orthogonality_loss *= self.config.orthogonality_weight if hasattr(self.config, 'orthogonality_weight') else 0.1
                print('Orthogonality Loss:', orthogonality_loss)

            # 新增正交性损失到总损失
            loss = ft_loss + act_loss.to(ft_loss.device) + orthogonality_loss
            
            # 更新冲突度量
            self.conflict_meter.update(act_loss.item())

            if loss_meter.stop():
                self.get_adapter_layer().save_editing_activation()  # add last gradient
                break
            if i == config.n_iter - 1:
                self.get_adapter_layer().save_editing_activation()  # add last gradient

            if self.config.retrieve and self.get_adapter_layer().merge_cnt > 0 and self.config.replay:
                memory_loss = []
                for _ in merge_group_edit_history:
                    idx = 0
                    while True:
                        memo_input, is_used = _[idx]
                        if not is_used:
                            _[idx][1] = True
                            break
                        idx += 1
                        if idx == len(_): ## re Assign
                            for m in range(len(_)):
                                _[m][1] = False
                            idx = 0

                    memo_input = {f"{k1}" : v1.to(self.config.device) for k1, v1 in memo_input.items()}
                    self.model(**memo_input)

                    memory_act_loss = self.__cal_memory_neg_activation_loss(self.get_adapter_layer().original_layer_output,
                                                    self.get_adapter_layer().new_weight_layer_output, config=config,
                                                    act_mask=act_mask, deact_mask=deact_mask)
                    memory_loss.append(memory_act_loss.to(ft_loss.device))
                    del memo_input
                neg_memo_loss = torch.stack(memory_loss).mean()
                loss += neg_memo_loss
                if len(edit_history) > 0:
                    memo_input = random.choice(edit_history)[0]
                    memo_input = {f"{k1}" : v1.to(self.config.device) for k1, v1 in memo_input.items()}
                    self.model(**memo_input)

                    pos_memo_loss = self.__cal_memory_pos_activation_loss(self.get_adapter_layer().original_layer_output,
                                                    self.get_adapter_layer().new_weight_layer_output, config=config,
                                                    act_mask=act_mask, deact_mask=deact_mask)
                    del memo_input
                    loss += pos_memo_loss.to(ft_loss.device)
            # for replay Appendix B.3

            optimizer.zero_grad()

            loss.backward()
            self.get_adapter_layer().mask_new_weight_gradient()

            # 更新日志输出以包括正交性损失
            if hasattr(self.config, 'use_orthogonality') and self.config.use_orthogonality:
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)} + {np.round(act_loss.item(), 3)} + {np.round(orthogonality_loss, 3)}"
                )
            else:
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)} + {np.round(act_loss.item(), 3)}"
                )

            optimizer.step()
            loss_meter.update(loss.item())

            if type(self.config.norm_constraint) is float:
                self.__norm_constraint(self.config.norm_constraint)

        # --- pull out info we want to log from the Wise layer ---
        setattr(eval(f"self.model.{self.layer}"), "editing", False)
        setattr(eval(f"self.model.{self.layer}"), "training", False)

        editing_total_cnt = getattr(eval(f"self.model.{self.layer}"), "editing_total_cnt") + 1
        setattr(eval(f"self.model.{self.layer}"), "editing_total_cnt", editing_total_cnt)
        #
        if self.config.save_freq is not None and editing_total_cnt % self.config.save_freq == 0:
            self.get_adapter_layer().save_weight()
            print(f'Add New Weight to Memory...')
        if editing_total_cnt % self.config.merge_freq == 0:
            # for retrieve ##
            merge_group_edit_history.append(edit_history)
            edit_history = []
            # for retrieve ##

            self.get_adapter_layer().merge_weight()
            print(f'Merge Weight of (New, Original) Matrix... with {self.config.merge_alg}')

    def __norm_constraint(self, norm_constraint):
        new_weight = self.get_adapter_layer().new_weight
        original_weight = self.get_adapter_layer().weight
        with torch.no_grad():
            new_weight[...] = torch.clamp(
                new_weight, min=original_weight - norm_constraint, max=original_weight + norm_constraint
            )

    def __cal_ft_loss(self, tokens, last_prompt_token_loc):
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

    def __cal_activation_loss(self, original_layer_output, new_weight_layer_output, config=None, act_mask=None,
                            deact_mask=None):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        total_loss = []
        len_temp = original_layer_output.shape[0] / k - 1
          
        for i, act_mk in enumerate(act_mask):
            if act_mk is not None:
                # 新增: 使用语义因果解耦进行干预
                if hasattr(self.config, 'use_causal_intervention') and self.config.use_causal_intervention:
                    # 获取当前批次的激活
                    cur_orig_act = original_layer_output[int(i*len_temp):int((i+1)*len_temp), ...]
                    cur_new_act = new_weight_layer_output[int(i*len_temp):int((i+1)*len_temp), ...]
                    
                    # 执行因果干预
                    intervened_act = self.semantic_disentanglement.causal_intervention(
                        cur_orig_act, cur_new_act, act_mk
                    )
                    
                    # 使用干预后的激活计算距离
                    in_scope_dist = euc(cur_orig_act, intervened_act, config, act_mask=act_mk)
                    out_scope_dist = euc(cur_orig_act, intervened_act, config, act_mask=deact_mask[i])
                else:
                    # 原始距离计算方法
                    in_scope_dist = euc(original_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], 
                                     new_weight_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], 
                                     config, act_mask=act_mk)
                    out_scope_dist = euc(original_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], 
                                      new_weight_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], 
                                      config, act_mask=deact_mask[i])
            else:
                in_scope_dist = euc(original_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], 
                                 new_weight_layer_output[int(i*len_temp):int((i+1)*len_temp), ...], config)
                if (i==k-1):
                    out_scope_dist = euc(original_layer_output[int(i-k):, ...], 
                                      new_weight_layer_output[int(i-k):, ...], config)
                else:
                    out_scope_dist = euc(original_layer_output[int(i-k):int(i+1-k), ...], 
                                      new_weight_layer_output[int(i-k):int(i+1-k), ...], config)

            loss = out_scope_dist.view(-1,1) - in_scope_dist + config.gamma
            loss2 = out_scope_dist - config.alpha
            loss3 = config.beta - in_scope_dist
            loss3 = torch.mean(loss3[loss3 > 0]) if min(loss3[loss3 > 0].size()) > 0 else torch.tensor(0.).to(original_layer_output.device)
            loss2 = torch.mean(loss2[loss2 > 0]) if min(loss2[loss2 > 0].size()) > 0 else torch.tensor(0.).to(original_layer_output.device)
            loss = torch.mean(loss[loss > 0]) if min(loss[loss > 0].size()) > 0 else torch.tensor(0.).to(original_layer_output.device)
            total_loss.append(loss + loss2 + loss3)
            
        return sum(total_loss) / len(total_loss)

    def __cal_memory_pos_activation_loss(self, original_layer_output, new_weight_layer_output, config=None, act_mask=None,
                              deact_mask=None):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        in_scope_dist = euc(original_layer_output[:-k, ...], new_weight_layer_output[:-k, ...], config)
        loss4 = 20 - in_scope_dist

        return torch.mean(loss4[loss4 > 0]) if min(loss4[loss4 > 0].size()) > 0 else torch.tensor(0.)

    def __cal_memory_neg_activation_loss(self, original_layer_output, new_weight_layer_output, config=None, act_mask=None,
                              deact_mask=None):
        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        in_scope_dist = euc(original_layer_output[:-k, ...], new_weight_layer_output[:-k, ...], config)
        loss4 = in_scope_dist - 5

        return torch.mean(loss4[loss4 > 0]) if min(loss4[loss4 > 0].size()) > 0 else torch.tensor(0.)

    def save(self, save_path):
        import os
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)  # Create the directory if it doesn't exist

        # Save additional information, such as memory_weight, memory_mean_act, etc.
        additional_info = {
            'memory_weight': self.get_adapter_layer().memory_weight,
            'memory_mean_act': self.get_adapter_layer().memory_mean_act,
            'merge_cnt': self.get_adapter_layer().merge_cnt,
            'editing_mean_act': self.get_adapter_layer().editing_mean_act,
            'editing_total_cnt': self.get_adapter_layer().editing_total_cnt,
            'weight_mask': self.get_adapter_layer().weight_mask,
            # Add other variables that need to be saved
        }
        if hasattr(self.get_adapter_layer(), 'key_id') and self.get_adapter_layer().key_id is not None:
            additional_info['key_id'] = self.get_adapter_layer().key_id
        # Save all information to the file
        torch.save({
            'adapter_state_dict': self.get_adapter_layer().state_dict(),
            'config': self.config,
            'additional_info': additional_info,
            'edit_history': edit_history,
            'merge_group_edit_history': merge_group_edit_history
        }, save_path)

    def load(self, load_path):
        import os
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Checkpoint file not found: {load_path}")

        # Load all previously saved information
        saved_data = torch.load(load_path)
        if hasattr(self.model.config, 'hidden_act'):
            saved_data['config'].hidden_act = self.model.config.hidden_act
        elif hasattr(self.model.config, 'activation_function'):
            saved_data['config'].hidden_act = self.model.config.activation_function
        if saved_data['config'] != self.config:
            print("Warning: The loaded WISE config is different from the original config")

        # Restore the state dictionary of the WISE Adapter instance
        self.get_adapter_layer().load_state_dict(saved_data['adapter_state_dict'])
        # Restore additional information
        adapter_layer = self.get_adapter_layer()
        for key, value in saved_data['additional_info'].items():
            setattr(adapter_layer, key, value)
        
        # Restore editing history
        global edit_history, merge_group_edit_history
        edit_history = saved_data['edit_history']
        merge_group_edit_history = saved_data['merge_group_edit_history']
        print(f"Model configuration and WISE state loaded from {load_path}")

class WISEAdapter(torch.nn.Module):
    def __init__(self, config, layer, transpose):
        super(WISEAdapter, self).__init__()

        self.layer = layer
        self.weight = self.layer.weight
        self.device = layer.weight.device
        self.config = config
        self.new_weight = copy.deepcopy(self.weight)
        self.original_layer = copy.deepcopy(self.layer)
        self.memory_weight = []
        self.memory_mean_act = []
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

        self.editing_mean_act = EditingMeanAct()
        self.editing_total_cnt = 0

    def set_parameter_tunable(self):
        self.new_weight.requires_grad = True

    def save_weight(self):
        self.memory_weight.append(copy.deepcopy(self.new_weight))
        self.new_weight = copy.deepcopy(self.original_layer.weight)
        if self.config.retrieve:
            self.memory_mean_act.append(copy.deepcopy(self.editing_mean_act))
            self.editing_mean_act = EditingMeanAct()

    def merge_weight(self):
        if self.config.save_freq is not None:  # for ties dare dare_ties
            if not self.config.retrieve:
                merge_alg = merge_dict[self.config.merge_alg]
                if self.original_layer.weight.equal(self.layer.weight):
                    cur_new_weight = merge_alg.execute([self.config.weights / len(self.memory_weight) for _ in range(len(self.memory_weight))], self.original_layer.weight, self.memory_weight, densities=self.config.densities)
                else:
                    cur_new_weight = merge_alg.execute([0.4 / len(self.memory_weight) for _ in range(len(self.memory_weight))] + [0.6], self.original_layer.weight, self.memory_weight + [self.layer.weight], densities=self.config.densities)
                self.layer.weight = torch.nn.Parameter(cur_new_weight.to(self.layer.weight.device), requires_grad=False)
                self.new_weight = copy.deepcopy(self.original_layer.weight)
                del self.memory_weight
                self.memory_weight = []
            else:
                merge_alg = merge_dict[self.config.merge_alg]
                merge_num = self.config.merge_freq // self.config.save_freq
                assert len(self.memory_weight) >= merge_num
                new_merge_weight = merge_alg.execute([self.config.weights / merge_num for _ in range(merge_num)], self.original_layer.weight, self.memory_weight[-merge_num:], densities=self.config.densities)
                min_a = 1e9
                for _ in range(merge_num):
                    self.memory_weight.pop()
                    edit_act = self.memory_mean_act.pop()
                    min_a = min(min_a, edit_act.min_act())
                self.new_weight = copy.deepcopy(self.original_layer.weight)
                self.memory_weight.append(new_merge_weight)
                self.memory_mean_act.append(EditingMeanAct(min_a=min_a))
                print(len(self.memory_weight))
                assert len(self.memory_mean_act) == len(self.memory_weight)
                self.merge_cnt += 1
        else:
            merge_alg = merge_dict[self.config.merge_alg]
            cur_new_weight = merge_alg.execute(0.5, self.layer.weight, [self.new_weight],
                                               densities=self.config.densities)
            self.layer.weight = torch.nn.Parameter(cur_new_weight.to(self.layer.weight.device), requires_grad=False)
            self.new_weight = copy.deepcopy(self.original_layer.weight)

    def save_editing_activation(self):
        in_scope_dist = euc(self.original_layer_output[:-1, ...], self.new_weight_layer_output[:-1, ...], self.config)
        self.editing_mean_act.update(in_scope_dist.mean().item())

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

    def new_weight_forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.new_weight) if self.bias is None else torch.addmm(self.bias, input.view(-1, input.size(-1)), self.new_weight).view(input.size()[:-1] + (self.layer.nf,))

    def mask_new_weight_gradient(self):
        assert self.new_weight.grad is not None, print('Gradient Collection for New Weight error, gradient not found')
        # Add gradient mask after the loss updates
        p_size = self.new_weight.grad.size()
        p_grad = self.new_weight.grad.reshape(-1)

        # mask = torch.from_numpy(np.random.choice([0, 1], size=p_grad.size()[0], p=[.1, .9])).cuda()
        p_grad = p_grad * self.weight_mask
        self.new_weight.grad = p_grad.view(p_size).to(self.new_weight.grad.dtype)

    def forward(self, *args):
        if self.editing:
            layer_out = self.new_weight_forward(*args)
            self.new_weight_layer_output = layer_out
            self.original_layer_output = self.original_layer(*args)
        else:
            if not self.config.retrieve:
                original_layer_output = self.original_layer(*args)
                layer_output = self.layer(*args)
                new_weight_layer_output = self.new_weight_forward(*args)
                dist2 = euc(original_layer_output, new_weight_layer_output, self.config, infer=True)
                dist1 = euc(original_layer_output, layer_output, self.config, infer=True)
                threshold = self.editing_mean_act.min_act() * self.config.act_ratio

                if dist1.item() < threshold and dist2.item() < threshold:
                    layer_out = original_layer_output
                elif dist1.item() > dist2.item():
                    layer_out = layer_output
                else:
                    layer_out = new_weight_layer_output
            else:
                original_layer_output = self.original_layer(*args)
                new_weight_layer_output = self.new_weight_forward(*args)
                dist1 = euc(original_layer_output, new_weight_layer_output, self.config, infer=True)
                threshold = self.editing_mean_act.min_act() * self.config.act_ratio
                min_dist = dist1
                if min_dist.dim() > 0:  
                    min_dist = min_dist.mean()
                if min_dist.item() < threshold:
                    layer_out = original_layer_output
                else:
                    layer_out = new_weight_layer_output

                for i in range(len(self.memory_weight)):
                    memory_retrieve_weight = self.memory_weight[i]
                    memory_weight_layer_output = F.linear(*args, memory_retrieve_weight)
                    dist = euc(original_layer_output, memory_weight_layer_output, self.config, infer=True)
                    if dist > min_dist and dist > self.memory_mean_act[i].min_act() * self.config.act_ratio:
                        layer_out = memory_weight_layer_output
                        min_dist = dist
        return layer_out

# 新增: 冲突度量追踪类
class ConflictMeter:
    """跟踪和测量编辑过程中的知识冲突程度"""
    
    def __init__(self, threshold=0.6, window_size=10):
        self.threshold = threshold
        self.window_size = window_size
        self.loss_history = []
        self.conflict_detected = False
        
    def update(self, loss_value):
        """更新冲突度量"""
        self.loss_history.append(loss_value)
        
        # 保持固定窗口大小
        if len(self.loss_history) > self.window_size:
            self.loss_history.pop(0)
        
        # 如果最近的损失值超过阈值，标记为冲突
        if len(self.loss_history) >= 3:  # 至少需要几个样本
            recent_avg = sum(self.loss_history[-3:]) / 3
            if recent_avg > self.threshold:
                self.conflict_detected = True
            else:
                self.conflict_detected = False
    
    def get_conflict_level(self):
        """获取当前冲突程度 (0-1范围)"""
        if not self.loss_history:
            return 0.0
            
        # 计算归一化的冲突程度
        recent_losses = self.loss_history[-min(3, len(self.loss_history)):]
        avg_loss = sum(recent_losses) / len(recent_losses)
        
        # 将损失值映射到0-1范围的冲突程度
        conflict_level = min(1.0, avg_loss / (self.threshold * 2))
        return conflict_level
    
    def is_conflict_detected(self):
        """检查是否检测到冲突"""
        return self.conflict_detected

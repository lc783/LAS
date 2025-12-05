import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor,AutoModelForVision2Seq,PretrainedConfig
from qwen_vl_utils import process_vision_info
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel,Qwen2VLConfig,repeat_kv,Qwen2VLRotaryEmbedding,apply_multimodal_rotary_pos_emb
import torch.nn as nn
import torch
from typing import List, Optional, Tuple, Union
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    logging,
    find_adapter_config_file,
    is_peft_available,
    replace_return_docstrings,
)
from transformers.cache_utils import Cache
import math
logger = logging.get_logger(__name__)
from transformers.activations import ACT2FN
import numpy as np
from transformers.dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code


from util import get_exp3, invert_tensor2, SNNMACOperater,get_squre_post,get_sqr_acc,get_squre_post2,get_sqr_acc2,get_sliu2


class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_scaled):
        z = (v_scaled > 0).float()
        ctx.save_for_backward(v_scaled)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        v_scaled, = ctx.saved_tensors
        dz_dv_scaled = torch.clamp(1 - torch.abs(v_scaled), min=0)
        grad_input = grad_output * dz_dv_scaled
        return grad_input


def fs_coding(x, h, d, T, K):
    v = x.clone()
    z = torch.zeros_like(x)
    out = []
    v_reg = 0.0
    z_reg = 0.0
    for t in range(K):
        v_scaled = (v - T[t]) / (torch.abs(v) + 1)  #
        z = SpikeFunction.apply(v_scaled)  # 
        out.append(z * d[t])  
        v = v - z * h[t]  
    return torch.stack(out)


def fs_relu(x: torch.Tensor, n_neurons=16, v_max=1, return_reg=False, fast=True):
    def generate_params(relu_K=n_neurons, alpha=v_max):
        relu_h = alpha * 2 ** (-relu_K) * np.array([float(2 ** (relu_K - i)) for i in range(1,
                                                                                            relu_K + 1)])
        return (
            torch.from_numpy(relu_h).to(x.device),  
            torch.from_numpy(relu_h).to(x.device),  
            torch.from_numpy(relu_h).to(x.device)  
        )

    if fast:
        x = torch.maximum(x, torch.tensor(0.0, device=x.device)) 
        x /= v_max

        x *= 2 ** n_neurons
        # i_out = torch.floor(x).to(torch.float32) 
        i_out = torch.floor(x).to(x.dtype)  
        i_out /= 2 ** n_neurons
        i_out *= v_max
        i_out = torch.minimum(i_out, torch.tensor(v_max * (1 - 2 ** (-n_neurons)), device=x.device))

        if return_reg:
            return i_out.detach(), torch.tensor(1., device=x.device)
        return i_out.detach()
    else:
        relu_h, relu_d, relu_T = generate_params()
        out = fs_coding(
            x,
            h=relu_h,
            d=relu_d,
            T=relu_T,
            K=len(relu_h)
        )
        return out


class FsCoding10(nn.Module):
    def __init__(self):
        super(FsCoding10, self).__init__()
        sigmoid_h = np.array([-0.11408895, 1.1758447, 1.0135335, 1.2213913, 1.3241712, 0.8555309,
                              1.4268297, 1.1778928, 0.80728793, 0.43790972, 0.2571573, 0.13942857,
                              -0.3750314, 0.08095932, 1.354895, 5.0583024])
        sigmoid_d = np.array([1.1759424, 1.0134453, 1.221452, 1.3243698, 0.85503244, 1.4270093,
                              1.1782365, 0.8076809, 0.43830687, 0.2571347, 0.13936771, -0.04276754,
                              0.07785574, 0.05046117, 0.10240205, 0.02330057])
        sigmoid_T = np.array([1.3393638, 1.3052133, 1.3697962, 1.3982388, 1.264561, 1.3956275,
                              1.1693456, 0.79711413, 0.42847168, 0.24717589, 0.13019533, 0.52669114,
                              0.06617433, 0.0366448, 1.0000271, 0.01022558])
        self.h = torch.tensor(sigmoid_h, dtype=torch.float32)
        self.d = torch.tensor(sigmoid_d, dtype=torch.float32)
        self.T = torch.tensor(sigmoid_T, dtype=torch.float32)
        self.K = self.h.shape[0]

    def forward(self, x, fast=True):
        if fast:
            or_shape = x.shape
            signs = torch.sign(x).detach()
            x_abs = x.abs()
            x_abs = x_abs.view(-1)

            v = x_abs.clone()
            z = torch.zeros_like(x_abs)
            out = torch.zeros_like(x_abs)
            for t in range(self.K):
                v = v - z * self.h[t] 
                v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)  
                z = SpikeFunction.apply(v_scaled) 
                out = out + z * self.d[t] 

            out = out.view(or_shape) * signs
        else:
            signs = torch.sign(x).detach()
            x_abs = x.abs()

            v = x_abs.clone()
            z = torch.zeros_like(x_abs)
            out = []
            # out2 = torch.zeros_like(x_abs)
            for t in range(self.K):
                v = v - z * self.h[t]     
                v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)   
                z = SpikeFunction.apply(v_scaled)    
                out.append((z * self.d[t]))     
                # out2 = out2 + z * self.d[t]
            out = torch.stack(out) * signs


        return out


class FsCoding50(nn.Module):
    def __init__(self):
        super(FsCoding50, self).__init__()
         
        sigmoid_h = np.array([-0.60999554, 6.6207733, 4.812895, 3.968032, 4.161328, 4.4064946,
                              4.173108, 3.6446128, 3.8171945, 3.6436832, 3.2056203, 3.0562265,
                              3.2158725, 1.9074137, 0.9322021, -1.661519])
        sigmoid_d = np.array([4.2054086, 4.8060417, 3.9730065, 4.1607947, 4.4024496, 4.1759186,
                              3.6378167, 3.8240826, 3.6393092, 3.2110925, 3.0602808, 3.2107215,
                              1.9014466, 0.9346704, 0.26098925, 0.34727606])
        sigmoid_T = np.array([-1.0000008, 6.4771256, 6.091373, 5.6789684, 5.31082, 2.2500517,
                              1.1327136, 1.1378073, 1.1088642, 1.0150509, 1.026008, 0.6764903,
                              -0.66469634, -1.6324021, -2.27651, -0.353094])
         
        self.h = torch.tensor(sigmoid_h, dtype=torch.float32)
        self.d = torch.tensor(sigmoid_d, dtype=torch.float32)
        self.T = torch.tensor(sigmoid_T, dtype=torch.float32)
        self.K = self.h.shape[0]

    def forward(self, x, fast=True):
        if fast:
            or_shape = x.shape
            signs = torch.sign(x).detach()
            x_abs = x.abs()
            x_abs = x_abs.view(-1)

            v = x_abs.clone()
            z = torch.zeros_like(x_abs)
            out = torch.zeros_like(x_abs)
            for t in range(self.K):
                v = v - z * self.h[t]     
                v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)   
                z = SpikeFunction.apply(v_scaled)    
                out = out + z * self.d[t]     

            out = out.view(or_shape) * signs
        else:
            signs = torch.sign(x).detach()
            x_abs = x.abs()

            v = x_abs.clone()
            z = torch.zeros_like(x_abs)
            out = []
            for t in range(self.K):
                v = v - z * self.h[t]     
                v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)   
                z = SpikeFunction.apply(v_scaled)    
                out.append((z * self.d[t]))     
            out = torch.stack(out) * signs
        return out


class FsCoding1(nn.Module):
    def __init__(self):
        super(FsCoding1, self).__init__()
         
        sigmoid_h = np.array([-0.65453446, 0.4741694, 1.5345732, 0.34063083, 0.10334169,
                              1.5662742, 1.5825152, 0.08264282, 1.271305, 0.05652184,
                              0.8087557, 0.48658752, -0.98152035, 5.504204, 3.9964867,
                              -19.865507])
        sigmoid_d = np.array([0.47450694, 0.61293966, 0.34035066, -0.06458966, 0.37327227, 0.26768795,
                              0.08609799, 0.21590698, 0.06298151, 0.02074145, 0.07180741, -0.1188115,
                              0.08973021, 0.06638855, 0.02828525, 0.00312297])
        sigmoid_T = np.array([0.46731842, 0.8967325, 0.32910728, -1.0000211, 0.18526751, 1.000173,
                              0.01329661, 0.06418973, -0.01542763, 1.0000025, -0.08678632, 0.9953476,
                              0.70950633, -0.5301457, -1.300978, -1.0005718])
         
        self.h = torch.tensor(sigmoid_h, dtype=torch.float32)
        self.d = torch.tensor(sigmoid_d, dtype=torch.float32)
        self.T = torch.tensor(sigmoid_T, dtype=torch.float32)
        self.K = self.h.shape[0]

    def forward(self, x):
        or_shape = x.shape
        signs = torch.sign(x).detach()
        x_abs = x.abs()
        x_abs = x_abs.view(-1)

        v = x_abs.clone()
        z = torch.zeros_like(x_abs)
        out = torch.zeros_like(x_abs)
        for t in range(self.K):
            v = v - z * self.h[t]     
            v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)   
            z = SpikeFunction.apply(v_scaled)    
            out = out + z * self.d[t]     

        out = out.view(or_shape) * signs
        return out


toSpilke10 = FsCoding10()

toSpilke50 = FsCoding50()
toSpilke1 = FsCoding1()


def toSpike(ox, fast=True, hight_acc=False, hight_acc_fast=True, thres1=0, thres2 = 0):
    x = ox.to(torch.float32)
    with torch.cuda.amp.autocast(enabled=False):
     
        if fast:   
            if not hight_acc:
                hidden_states = x.abs()
                mask_0_10 = hidden_states < 10
                mask_10_50 = hidden_states >= 10
                hidden_states_0_10 = x[mask_0_10]
                hidden_states_10_50 = x[mask_10_50]

                output_0_10 = toSpilke10(hidden_states_0_10).to(x.dtype)

                output_10_50 = toSpilke50(hidden_states_10_50).to(x.dtype)

                result = torch.empty_like(hidden_states)
                result[mask_0_10] = output_0_10
                result[mask_10_50] = output_10_50
                return result.to(ox.dtype)
            else:    
                if thres1!= 0:
                    x = torch.clamp(x, min=-thres2, max=thres2)

                    hidden_states = x.abs()
                    signs = torch.sign(x).detach()
                    mask_pos = hidden_states <= thres1
                    mask_neg = hidden_states > thres1
                    stata_min = hidden_states[mask_pos]
                    stata_max = hidden_states[mask_neg]

                    stata_min = fs_relu(stata_min, v_max=thres1, fast=hight_acc_fast)
                    stata_max = fs_relu(stata_max, v_max=thres2, fast=hight_acc_fast)

                    result = torch.empty_like(x)
                   
                    result[mask_pos] = stata_min
                    result[mask_neg] = stata_max
                    result = result * signs
                    return result.to(ox.dtype)
                x = torch.clamp(x, min=-500, max=500)
                signs = torch.sign(x).detach()
                hidden_states = x.abs()
                mask_0_10 = hidden_states < 10
                mask_10_50 = hidden_states >= 10
                hidden_states_0_10 = hidden_states[mask_0_10]
                hidden_states_10_50 = hidden_states[mask_10_50]

                hidden_states_0_10 = fs_relu(hidden_states_0_10, v_max=10, fast=hight_acc_fast)
                hidden_states_10_50 = fs_relu(hidden_states_10_50, v_max=500, fast=hight_acc_fast)
                result = torch.empty_like(x)
                result[mask_0_10] = hidden_states_0_10
                result[mask_10_50] = hidden_states_10_50
                result = result * signs
                return result.to(ox.dtype)
        else:
            if not hight_acc:    
                if thres2!= 0:
                    x = torch.clamp(x, min=-thres2, max=thres2)

                    hidden_states = x.abs()
                    signs = torch.sign(x).detach()
                    mask_pos = hidden_states < thres1
                    mask_neg = hidden_states >= thres1
                    stata_min = hidden_states[mask_pos]
                    stata_max = hidden_states[mask_neg]

                    stata_min = fs_relu(stata_min, v_max=thres1, fast=False)
                    stata_max = fs_relu(stata_max, v_max=thres2, fast=False)

                    result = torch.empty_like(x)

                    K = stata_min.shape[0]
                    result = result.unsqueeze(0).repeat(K, *([1] * result.dim()))

                    result[:, mask_pos] = stata_min
                    result[:, mask_neg] = stata_max
                    result = result * signs
                    # result = result.sum(0)
                    return result.to(ox.dtype)
                else:
                    x = torch.clamp(x, min=-20, max=20)

                    hidden_states = x.abs()
                    signs = torch.sign(x).detach()
                    mask_pos = hidden_states < 5
                    mask_neg = hidden_states >= 5
                    stata_min = hidden_states[mask_pos]
                    stata_max = hidden_states[mask_neg]

                    stata_min = fs_relu(stata_min, v_max=5, fast=False)
                    stata_max = fs_relu(stata_max, v_max=20, fast=False)

                    result = torch.empty_like(x)

                    K = stata_min.shape[0]
                    result = result.unsqueeze(0).repeat(K, *([1] * result.dim()))

                    result[:, mask_pos] = stata_min
                    result[:, mask_neg] = stata_max
                    result = result * signs
                    # result = result.sum(0)
                    return result.to(ox.dtype)
            else:

                x = torch.clamp(x, min=-200, max=2000)  

                hidden_states = x.abs()
                signs = torch.sign(x).detach()
                mask_pos = x >= 0
                mask_neg = x < 0
                stata_pos = hidden_states[mask_pos]
                stata_neg = hidden_states[mask_neg]


                Spike_pos = fs_relu(stata_pos, v_max=2000, fast=hight_acc_fast)  # 13B
                Spike_neg = fs_relu(stata_neg, v_max=200, fast=hight_acc_fast)


                result = torch.empty_like(x)

                K = Spike_pos.shape[0]
                result = result.unsqueeze(0).repeat(K, *([1] * result.dim()))

                result[:, mask_pos] = Spike_pos
                result[:, mask_neg] = Spike_neg
                result = result * signs
                return result.to(ox.dtype)


def toSpike2(ox, fast=True, hight_acc=False, hight_acc_fast=True):
    x = ox.to(torch.float32)
    with torch.cuda.amp.autocast(enabled=False):
     
        if fast:   
            if not hight_acc:
                hidden_states = x.abs()
                mask_0_10 = hidden_states < 10
                mask_10_50 = hidden_states >= 10
                hidden_states_0_10 = x[mask_0_10]
                hidden_states_10_50 = x[mask_10_50]

                output_0_10 = toSpilke10(hidden_states_0_10).to(x.dtype)

                output_10_50 = toSpilke50(hidden_states_10_50).to(x.dtype)

                result = torch.empty_like(hidden_states)
                result[mask_0_10] = output_0_10
                result[mask_10_50] = output_10_50
                return result.to(ox.dtype)
            else:    
                x = torch.clamp(x, min=-500, max=500)
                signs = torch.sign(x).detach()
                hidden_states = x.abs()
                mask_0_10 = hidden_states < 10
                mask_10_50 = hidden_states >= 10
                hidden_states_0_10 = hidden_states[mask_0_10]
                hidden_states_10_50 = hidden_states[mask_10_50]

                hidden_states_0_10 = fs_relu(hidden_states_0_10, v_max=10, fast=hight_acc_fast)
                hidden_states_10_50 = fs_relu(hidden_states_10_50, v_max=500, fast=hight_acc_fast)
                result = torch.empty_like(x)
                result[mask_0_10] = hidden_states_0_10
                result[mask_10_50] = hidden_states_10_50
                result = result * signs
                return result.to(ox.dtype)
        else:
            if not hight_acc:    
                x = torch.clamp(x, min=-500, max=500)
                hidden_states = x.abs()
                signs = torch.sign(x).detach()
                mask_pos = hidden_states < 10
                mask_neg = hidden_states >= 10
                stata_min = hidden_states[mask_pos]
                stata_max = hidden_states[mask_neg]

                stata_min = fs_relu(stata_min, v_max=10, fast=False)
                stata_max = fs_relu(stata_max, v_max=500, fast=False)

                result = torch.empty_like(x)

                K = stata_min.shape[0]
                result = result.unsqueeze(0).repeat(K, *([1] * result.dim()))

                result[:, mask_pos] = stata_min
                result[:, mask_neg] = stata_max
                result = result * signs
                return result.to(ox.dtype)
            else:

                x = torch.clamp(x, min=-200, max=700)  

                hidden_states = x.abs()
                signs = torch.sign(x).detach()
                mask_pos = x >= 0
                mask_neg = x < 0
                stata_pos = hidden_states[mask_pos]
                stata_neg = hidden_states[mask_neg]



                Spike_pos = fs_relu(stata_pos, v_max=700, fast=hight_acc_fast)  # 13B
                Spike_neg = fs_relu(stata_neg, v_max=200, fast=hight_acc_fast)



                result = torch.empty_like(x)

                K = Spike_pos.shape[0]
                result = result.unsqueeze(0).repeat(K, *([1] * result.dim()))

                result[:, mask_pos] = Spike_pos
                result[:, mask_neg] = Spike_neg
                result = result * signs
                return result.to(ox.dtype)

class MyAt(nn.Module):
    def __init__(self):
        super(MyAt, self).__init__()

    def forward(self, x, y):
        return x @ y


class TestNeuron(nn.Module):
    def __init__(self, place=None, percent=None):
        super(TestNeuron, self).__init__()
        self.place = place
        self.percent = percent
        self.num = 0

    def forward(self, x, times=2, gap=3, show=0, tmptime=0, scaletimes=1):


        return x

    def reset(self):
        pass


def fit_softmax(X, dim=-1, analsis=None, run_number=None):
    X = X.to(dtype=torch.float32)
    with torch.cuda.amp.autocast(enabled=False):

        # tmax2 = X[:,:, :, 0].unsqueeze(-1)
        # tmax = tmax2
        # tp = X - (tmax - 1)  # equal to:x1+x2+...+xt - (x1[-1]+x2[-1]+...+xt[-1])=x-x[-1] linear

        # tp = X

        tmax = X.max(dim=dim, keepdim=True)[0]
        tp = X - tmax

        tp[tp > 80] = 80
        index = tp > -100

        # exp = get_exp(tp[index], fast=False).to(dtype=X.dtype)  
        exp = get_exp3(tp[index], fast=False).to(dtype=X.dtype)  
        X_up = torch.zeros_like(tp)
        K = exp.shape[0]
        X_up = X_up.unsqueeze(0).repeat(K, *([1] * X_up.dim()))
        X_up[:, index] = exp
        # X_up = X_up.sum(0)

        partition = X_up.sum(0).sum(dim=dim, keepdim=True)  

        p_inv = invert_tensor2(partition,fast=False)  
        out = SNNMACOperater(X_up, p_inv)

      
    return out


import time
class Qwen2VLAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2VLConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        
       
        hidden_states = toSpike(hidden_states, hight_acc=True,thres1=10,thres2=50)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += cache_position[0] + 1

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        
        

        query_states = toSpike(query_states, fast=False,thres1=10,thres2=100).sum(0)       
        key_states = toSpike(key_states, fast=False,thres1=20,thres2=500).sum(0)      
        value_states = toSpike(value_states, fast=False,thres1=5,thres2=50).sum(0)   

        # Here, formula (13) can be used to perform the multiplication, as demonstrated in the open-source BERT implementation. 
        # Since it does not affect the result, the implementation is omitted here for simplicity.
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Fix precision issues in Qwen2-VL float16 inference
        # Replace inf values with zeros in attention weights to prevent NaN propagation
        if query_states.dtype == torch.float16:
            attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        # upcast attention to fp32

        # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = fit_softmax(attn_weights, dim=-1).to(query_states.dtype)     


        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_weights = toSpike(attn_weights, fast=False,thres1=0.001,thres2=1).to(value_states.dtype).sum(0)       

        # Here, formula (13) can be used to perform the multiplication, as demonstrated in the open-source BERT implementation. 
        # Since it does not affect the result, the implementation is omitted here for simplicity.
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)


        attn_output = toSpike(attn_output, hight_acc=True,thres1=2,thres2=15)  
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

QWEN2_VL_ATTENTION_CLASSES = {
    "eager": Qwen2VLAttention,
    "flash_attention_2": Qwen2VLAttention,
    "sdpa": Qwen2VLAttention,
}

class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
    
def SpikeRMSNorm2(hidden_states: torch.Tensor, weight, eps=1e-6, 
                device="cuda", fast=False, hight_acc=True):
    with torch.cuda.amp.autocast(enabled=False):
        input_dtype = hidden_states.dtype
        inputs = hidden_states.to(torch.float32)
        n = inputs.shape[-1]
        
        squared = get_squre_post(inputs)    
        
        W_mean = torch.full((n, 1), 1/n).to(device=squared.device)
        variance = squared@W_mean
        
        sqrt_var = get_sqr_acc(variance, fast=True)   
        # inv_sqrt = nolinearOperaterInv(sqrt_var)
        inv_sqrt = 1/(sqrt_var)

        # inputs = toSpike(inputs,fast=False,thres1=20,thres2=3000) 
        # norm_states = SNNMACOperater(inputs, inv_sqrt) 
    
        norm_states = inputs * inv_sqrt.sum(0)
    return norm_states.to(input_dtype)*weight

def SpikeRMSNorm3(hidden_states: torch.Tensor, weight, eps=1e-6, 
                device="cuda", fast=False, hight_acc=True):
    with torch.cuda.amp.autocast(enabled=False):
          
        input_dtype = hidden_states.dtype
        inputs = hidden_states.to(torch.float32)
        n = inputs.shape[-1]
        
        squared = get_squre_post2(inputs)   
        
        W_mean = torch.full((n, 1), 1/n).to(device=squared.device)
        variance = squared@W_mean
        
        # variance = torch.clamp(variance, min=0, max=30000)
        # sqrt_var = torch.sqrt(variance+1e-6)
        # inv_sqrt = 1/(sqrt_var)

        # if variance.min() <= 0.0001:
            # print(variance.min())
        sqrt_var = get_sqr_acc2(variance, fast=True)    

        inv_sqrt = 1/(sqrt_var)

        # inputs = toSpike(inputs,fast=False,thres1=20,thres2=3000) 
        # norm_states = SNNMACOperater(inputs, inv_sqrt) 
    
        # inputs = torch.clamp(inputs, min=-3000, max=3000) 
        norm_states = inputs * inv_sqrt
        return norm_states.to(input_dtype)*weight


# Copied from transformers.models.qwen2.modeling_qwen2.Qwen2MLP
class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

        x = toSpike(x, hight_acc=True, thres1=5, thres2=100)   

        gate_proj = self.gate_proj(x)   
        up_proj = self.up_proj(x)


        up_proj = toSpike(up_proj, hight_acc=True, thres1=10, thres2=100)  

        gate_proj = torch.clamp(gate_proj,min=-20,max=50)  
        # act = self.act_fn(gate_proj)       
        act = get_sliu2(gate_proj)

        all = act * up_proj

        all = toSpike(all, hight_acc=True, thres1=5, thres2=100)   
        down_proj = self.down_proj(all)  
        return down_proj
    
class Qwen2VLDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2VLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = QWEN2_VL_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        # hidden_states = self.input_layernorm(hidden_states)
        hidden_states = SpikeRMSNorm3(hidden_states,weight=self.input_layernorm.weight, eps=self.input_layernorm.variance_epsilon,)


        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        # hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = SpikeRMSNorm3(hidden_states,weight=self.post_attention_layernorm.weight)  


        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class SNNQwen2VLModel(Qwen2VLModel):
    def __init__(self, config):
        super().__init__(config)
        print(config._attn_implementation)
        self.layers = nn.ModuleList(
            [Qwen2VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
       
class SNNQwen2VLForConditionalGeneration(Qwen2VLForConditionalGeneration):
    """
    SNNQwen2VLForConditionalGeneration is a subclass of Qwen2VLForConditionalGeneration that adds support for SNN (Spiking Neural Networks) specific features.
    """
    def __init__(self, config):
        super().__init__(config)
        config._attn_implementation = "eager"
        self.model = SNNQwen2VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)



class SNNAutoModelForVision2Seq(AutoModelForVision2Seq):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        kwargs["_from_auto"] = True
        hub_kwargs_names = [
            "cache_dir",
            "force_download",
            "local_files_only",
            "proxies",
            "resume_download",
            "revision",
            "subfolder",
            "use_auth_token",
            "token",
        ]
        hub_kwargs = {name: kwargs.pop(name) for name in hub_kwargs_names if name in kwargs}
        code_revision = kwargs.pop("code_revision", None)
        commit_hash = kwargs.pop("_commit_hash", None)
        adapter_kwargs = kwargs.pop("adapter_kwargs", None)

        token = hub_kwargs.pop("token", None)
        use_auth_token = hub_kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None:
            hub_kwargs["token"] = token

        if commit_hash is None:
            if not isinstance(config, PretrainedConfig):
                # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
                resolved_config_file = cached_file(
                    pretrained_model_name_or_path,
                    CONFIG_NAME,
                    _raise_exceptions_for_gated_repo=False,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                    **hub_kwargs,
                )
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            else:
                commit_hash = getattr(config, "_commit_hash", None)

        if is_peft_available():
            if adapter_kwargs is None:
                adapter_kwargs = {}
                if token is not None:
                    adapter_kwargs["token"] = token

            maybe_adapter_path = find_adapter_config_file(
                pretrained_model_name_or_path, _commit_hash=commit_hash, **adapter_kwargs
            )

            if maybe_adapter_path is not None:
                with open(maybe_adapter_path, "r", encoding="utf-8") as f:
                    adapter_config = json.load(f)

                    adapter_kwargs["_adapter_model_path"] = pretrained_model_name_or_path
                    pretrained_model_name_or_path = adapter_config["base_model_name_or_path"]

        if not isinstance(config, PretrainedConfig):
            kwargs_orig = copy.deepcopy(kwargs)
            # ensure not to pollute the config object with torch_dtype="auto" - since it's
            # meaningless in the context of the config object - torch.dtype values are acceptable
            if kwargs.get("torch_dtype", None) == "auto":
                _ = kwargs.pop("torch_dtype")
            # to not overwrite the quantization_config if config has a quantization_config
            if kwargs.get("quantization_config", None) is not None:
                _ = kwargs.pop("quantization_config")

            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                return_unused_kwargs=True,
                trust_remote_code=trust_remote_code,
                code_revision=code_revision,
                _commit_hash=commit_hash,
                **hub_kwargs,
                **kwargs,
            )

            # if torch_dtype=auto was passed here, ensure to pass it on
            if kwargs_orig.get("torch_dtype", None) == "auto":
                kwargs["torch_dtype"] = "auto"
            if kwargs_orig.get("quantization_config", None) is not None:
                kwargs["quantization_config"] = kwargs_orig["quantization_config"]

        has_remote_code = hasattr(config, "auto_map") and cls.__name__ in config.auto_map
        has_local_code = type(config) in cls._model_mapping.keys()
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code
        )

        # Set the adapter kwargs
        kwargs["adapter_kwargs"] = adapter_kwargs

        if has_remote_code and trust_remote_code:
            class_ref = config.auto_map[cls.__name__]
            model_class = get_class_from_dynamic_module(
                class_ref, pretrained_model_name_or_path, code_revision=code_revision, **hub_kwargs, **kwargs
            )
            _ = hub_kwargs.pop("code_revision", None)
            if os.path.isdir(pretrained_model_name_or_path):
                model_class.register_for_auto_class(cls.__name__)
            else:
                cls.register(config.__class__, model_class, exist_ok=True)
            return model_class.from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
            )
        elif type(config) in cls._model_mapping.keys():
            # model_class = _get_model_class(config, cls._model_mapping)
            return SNNQwen2VLForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
            )
        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )




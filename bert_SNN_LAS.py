import math
import numpy as np
from transformers import BertModel, PretrainedConfig, apply_chunking_to_forward
import torch.nn as nn
import torch
from transformers.activations import ACT2FN
from typing import List, Optional, Tuple, Union
import json
import os
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions, \
    BaseModelOutputWithPastAndCrossAttentions
from transformers import AutoModelForSequenceClassification
from transformers.utils import (
    CONFIG_NAME,
    cached_file,
    copy_func,
    extract_commit_hash,
    find_adapter_config_file,
    is_peft_available,
    logging,
    requires_backends, add_start_docstrings_to_model_forward,
)
logger = logging.get_logger(__name__)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
import warnings
import copy
from util import get_exp, get_sqr, SNNMatrixOperater, SNNMACOperater,  \
    get_gelu, get_gelu2, get_squre, invert_tensor_precise

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
    mt =1  # 1,2....
    all_spike = 0
    for t in range(K):
        not_fire = torch.ones_like(x)
        temp_out = 0
        for j in range(mt):
            V = T[t]*(1+(mt-1-j)/mt)
            D = d[t]*(1+(mt-1-j)/mt)
            H = h[t]*(1+(mt-1-j)/mt)
            v_scaled = (v - V) / (torch.abs(v) + 1)
            z = SpikeFunction.apply(v_scaled)
            z = z*not_fire
            all_spike += z.sum()
            mask = z!=0
            not_fire[mask] = 0
            temp_out = temp_out + z * D
            v = v - z * H
        out.append(temp_out)
    neuron_number = torch.ones_like(x).sum()
    fr = all_spike / neuron_number
    return torch.stack(out)

def mtn(x: torch.Tensor, n_neurons=16, v_max=1, return_reg=False, fast=True):

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
            for t in range(self.K):
                v = v - z * self.h[t]
                v_scaled = (v - self.T[t]) / (torch.abs(v) + 1)
                z = SpikeFunction.apply(v_scaled)
                out.append((z * self.d[t]))
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


def OATN(ox, fast=True, hight_acc=False, hight_acc_fast=True):
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

                hidden_states_0_10 = mtn(hidden_states_0_10, v_max=10, fast=hight_acc_fast)
                hidden_states_10_50 = mtn(hidden_states_10_50, v_max=500, fast=hight_acc_fast)
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

                stata_min = mtn(stata_min, v_max=10, fast=False)
                stata_max = mtn(stata_max, v_max=500, fast=False)

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

                Spike_pos = mtn(stata_pos, v_max=700, fast=hight_acc_fast)  # 13B
                Spike_neg = mtn(stata_neg, v_max=200, fast=hight_acc_fast)

                result = torch.empty_like(x)

                K = Spike_pos.shape[0]
                result = result.unsqueeze(0).repeat(K, *([1] * result.dim()))

                result[:, mask_pos] = Spike_pos
                result[:, mask_neg] = Spike_neg
                result = result * signs
                return result.to(ox.dtype)


def OATN2(ox, fast=True, hight_acc=False, hight_acc_fast=True, thres1=0, thres2=0):
    x = ox.to(torch.float32)
    with torch.cuda.amp.autocast(enabled=False):
        if fast:
            if thres1 != 0:
                x = torch.clamp(x, min=-thres2, max=thres2)

                hidden_states = x.abs()
                signs = torch.sign(x).detach()
                mask_pos = hidden_states <= thres1
                mask_neg = hidden_states > thres1
                stata_min = hidden_states[mask_pos]
                stata_max = hidden_states[mask_neg]

                stata_min = mtn(stata_min, v_max=thres1, fast=hight_acc_fast)
                stata_max = mtn(stata_max, v_max=thres2, fast=hight_acc_fast)

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

            hidden_states_0_10 = mtn(hidden_states_0_10, v_max=10, fast=hight_acc_fast)
            hidden_states_10_50 = mtn(hidden_states_10_50, v_max=500, fast=hight_acc_fast)
            result = torch.empty_like(x)
            result[mask_0_10] = hidden_states_0_10
            result[mask_10_50] = hidden_states_10_50
            result = result * signs
            return result.to(ox.dtype)
        else:
            if not hight_acc:
                if thres2 != 0:
                    x = torch.clamp(x, min=-thres2, max=thres2)

                    hidden_states = x.abs()
                    signs = torch.sign(x).detach()
                    mask_pos = hidden_states < thres1
                    mask_neg = hidden_states >= thres1
                    stata_min = hidden_states[mask_pos]
                    stata_max = hidden_states[mask_neg]

                    stata_min = mtn(stata_min, v_max=thres1, fast=False)
                    stata_max = mtn(stata_max, v_max=thres2, fast=False)

                    result = torch.empty_like(x)

                    K = stata_min.shape[0]
                    result = result.unsqueeze(0).repeat(K, *([1] * result.dim()))

                    result[:, mask_pos] = stata_min
                    result[:, mask_neg] = stata_max
                    result = result * signs
                    return result.to(ox.dtype)
                else:
                    x = torch.clamp(x, min=-500, max=500)

                    hidden_states = x.abs()
                    signs = torch.sign(x).detach()
                    mask_pos = hidden_states < 10
                    mask_neg = hidden_states >= 10
                    stata_min = hidden_states[mask_pos]
                    stata_max = hidden_states[mask_neg]

                    stata_min = mtn(stata_min, v_max=10, fast=False)
                    stata_max = mtn(stata_max, v_max=500, fast=False)

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

                Spike_pos = mtn(stata_pos, v_max=700, fast=hight_acc_fast)
                Spike_neg = mtn(stata_neg, v_max=200, fast=hight_acc_fast)



                result = torch.empty_like(x)

                K = Spike_pos.shape[0]
                result = result.unsqueeze(0).repeat(K, *([1] * result.dim()))

                result[:, mask_pos] = Spike_pos
                result[:, mask_neg] = Spike_neg
                result = result * signs
                return result.to(ox.dtype)



class spike_softmax(nn.Module):
    def __init__(self, ):
        super(spike_softmax, self).__init__()

    def forward(self, X, dim=-1):
        X = X.to(dtype=torch.float32)
        with torch.cuda.amp.autocast(enabled=False):
            ## data translation ##
            # tmax2 = X[:,:, :, 0].unsqueeze(-1)
            # tmax = tmax2
            # tp = X - (tmax - 1)  # equal to:x1+x2+...+xt - (x1[-1]+x2[-1]+...+xt[-1])=x-x[-1] linear

            tmax = X.max(dim=dim, keepdim=True)[0]  # refer to Equation (16) in paper
            tp = X - tmax+0.1

            tp[tp > 10] = 10
            index = tp > -20

            exp = get_exp(tp[index], fast=False).to(dtype=X.dtype)

            X_up = torch.zeros_like(tp)
            K = exp.shape[0]
            X_up = X_up.unsqueeze(0).repeat(K, *([1] * X_up.dim()))
            X_up[:, index] = exp

            partition = X_up.sum(0).sum(dim=dim, keepdim=True)  # Linear function
            p_inv = invert_tensor_precise(partition,
                                  fast=False)
            out = SNNMACOperater(X_up, p_inv)
        return out


class spikeLN(nn.Module):
    def __init__(self, ):
        super(spikeLN, self).__init__()

    def forward(self, input: torch.Tensor, gamma=None, beta=None):
        inputs = input.to(torch.float32)
        with torch.cuda.amp.autocast(enabled=False):
            n = input.shape[-1]
            W_rmvmean = torch.full((n, n), -1 / n).to(device=input.device)
            W_rmvmean.fill_diagonal_(1 - 1 / n)
            W_var = torch.full((n, 1), 1 / n).to(device=input.device)
            rmvmean = (inputs @ W_rmvmean)

            rmvmean = OATN(rmvmean, fast=False, hight_acc=True, hight_acc_fast=False)

            vars = get_squre(rmvmean.sum(0))

            var = (vars @ W_var)

            sqr = get_sqr(var, fast=False)
            sqrinv = invert_tensor_precise(sqr.sum(0),fast=False)
            prod = SNNMACOperater(rmvmean, sqrinv)
            prod = OATN2(prod, fast=False, thres1=2, thres2=50)
            prod = prod.sum(0)

        norm = prod * gamma + beta
        return norm




class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:   #
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = OATN2(hidden_states, fast=False)
        hidden_states = hidden_states.sum(0)

        hidden_states = self.dense(hidden_states)

        hidden_states = torch.clamp(hidden_states,min=-6, max=30)
        hidden_states = get_gelu2(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.spike_ln = spikeLN()

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = OATN(hidden_states, fast=False)
        hidden_states = hidden_states.sum(0)

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # hidden_states = SpikeLN(hidden_states + input_tensor, self.LayerNorm.weight, self.LayerNorm.bias)
        hidden_states = self.spike_ln(hidden_states + input_tensor, self.LayerNorm.weight, self.LayerNorm.bias)

        return hidden_states


class BertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

        self.softmax = nn.Softmax(dim=-1)
        self.spike_softmax = spike_softmax()

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        hidden_states = OATN2(hidden_states,fast=False)

        hidden_states = hidden_states.sum(0)

        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        query_layer = OATN(query_layer, fast=False)
        key_layer = OATN(key_layer, fast=False)
        value_layer = OATN(value_layer, fast=False)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # attention_scores = self.at1(query_layer, key_layer.transpose(-1, -2))
        # self.enMatrix1(query_layer, key_layer.transpose(-1, -2))
        attention_scores = SNNMatrixOperater(query_layer, key_layer.transpose(-1, -2)).sum(0)


        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # attention_probs = self.softmax(attention_scores)

        # attention_probs = fit_softmax(attention_scores, dim=-1)
        attention_probs = self.spike_softmax(attention_scores, dim=-1)




        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        attention_probs = mtn(attention_probs, fast=False)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = SNNMatrixOperater(attention_probs, value_layer).sum(0)


        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.spike_ln = spikeLN()

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = OATN2(hidden_states, fast=False)
        hidden_states = hidden_states.sum(0)

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # hidden_states = SpikeLN(hidden_states + input_tensor, self.LayerNorm.weight, self.LayerNorm.bias)
        hidden_states = self.spike_ln(hidden_states + input_tensor, self.LayerNorm.weight, self.LayerNorm.bias)

        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output





class CustomBertModel(BertModel):
    def __init__(self, config, T=0):
        super(CustomBertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.T = T
        self.merge = MergeTemporalDim(0)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )



from transformers import BertForSequenceClassification, BertModel, BertConfig, AutoConfig

class CustomBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, T=0):
        super(CustomBertForSequenceClassification, self).__init__(config)
        self.bert = CustomBertModel(config, T=T)
        self.T = T
        self.expand = ExpandTemporalDim(T)
        self.count = 0

    def set_T(self, T):
        self.T = T
        self.bert.T = T
        self.expand.T = T
        return


    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def _get_model_class(config, model_mapping):
    supported_models = model_mapping[type(config)]
    if not isinstance(supported_models, (list, tuple)):
        return supported_models

    name_to_model = {model.__name__: model for model in supported_models}
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in name_to_model:
            return name_to_model[arch]
        elif f"TF{arch}" in name_to_model:
            return name_to_model[f"TF{arch}"]
        elif f"Flax{arch}" in name_to_model:
            return name_to_model[f"Flax{arch}"]

    # If not architecture is set in the config or match the supported models, the first element of the tuple is the
    # defaults.
    return supported_models[0]

class SNNAutoModelForSequenceClassification(AutoModelForSequenceClassification):
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
            model_class = _get_model_class(config, cls._model_mapping)
            return CustomBertForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **hub_kwargs, **kwargs
            )
        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )

class MergeTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        return x_seq.flatten(0, 1).contiguous()

class ExpandTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        y_shape = [self.T, int(x_seq.shape[0]/self.T)]
        y_shape.extend(x_seq.shape[1:])
        return x_seq.view(y_shape)

def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(T, 1, 1, 1)
    return x

def add_dimention_mask(x, T):
    x.unsqueeze_(1)
    x = x.repeat(T, 1, 1, 1, 1)
    return x

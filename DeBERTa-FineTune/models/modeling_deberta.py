# coding=utf-8
# Copyright 2020 Microsoft and the Hugging Face Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" PyTorch DeBERTa model. """

import math
import torch

from collections.abc import Sequence

from torch import _softmax_backward_data, nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

from .configuration_deberta import DebertaConfig


_CONFIG_FOR_DOC = "DebertaConfig"
_TOKENIZER_FOR_DOC = "DebertaTokenizer"
_CHECKPOINT_FOR_DOC = "microsoft/deberta-base"

DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/deberta-base",
    "microsoft/deberta-large",
    "microsoft/deberta-xlarge",
    "microsoft/deberta-base-mnli",
    "microsoft/deberta-large-mnli",
    "microsoft/deberta-xlarge-mnli",
]


class ContextPooler(nn.Module):
    """这里实质上并没有做池化，仅仅是取出第1个 token 然后经过 Dropout、FC 以及 激活函数"""
    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        self.dropout = StableDropout(config.pooler_dropout)

        self.config = config

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding to the first token.

        # hidden_states 是 (B,L,C) 这里取出第一个 token 则 context_token 是 (B,C)
        context_token = hidden_states[:, 0]
        pooled_output = self.dense(self.dropout(context_token))

        return ACT2FN[self.config.pooler_hidden_act](pooled_output)

    @property
    def output_dim(self):
        return self.config.hidden_size


class XSoftmax(torch.autograd.Function):
    """
    Masked Softmax which is optimized for saving memory

    Args:
        input (:obj:`torch.tensor`): The input tensor that will apply softmax.
        mask (:obj:`torch.IntTensor`): The mask matrix where 0 indicate that element will be ignored in the softmax calculation.
        dim (int): The dimension that will apply softmax

    Example::

          >>> import torch
          >>> from transformers.models.deberta.modeling_deberta import XSoftmax

          >>> # Make a tensor
          >>> x = torch.randn([4,20,100])

          >>> # Create a mask
          >>> mask = (x > 0).int()

          >>> y = XSoftmax.apply(x, mask, dim=-1)
    """

    @staticmethod
    def forward(self, inp, mask, dim):
        self.dim = dim
        reverse_mask = ~(mask.bool())

        output = inp.masked_fill(reverse_mask, float("-inf"))
        output = torch.softmax(output, self.dim)
        output.masked_fill_(reverse_mask, 0)

        self.save_for_backward(output)

        return output

    @staticmethod
    def backward(self, grad_output):
        (output,) = self.saved_tensors
        inputGrad = _softmax_backward_data(grad_output, output, self.dim, output)

        return inputGrad, None, None

    @staticmethod
    def symbolic(g, self, mask, dim):
        import torch.onnx.symbolic_helper as sym_help
        from torch.onnx.symbolic_opset9 import masked_fill, softmax

        mask_cast_value = g.op("Cast", mask, to_i=sym_help.cast_pytorch_to_onnx["Long"])
        r_mask = g.op(
            "Cast",
            g.op("Sub", g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64)), mask_cast_value),
            to_i=sym_help.cast_pytorch_to_onnx["Byte"],
        )
        output = masked_fill(g, self, r_mask, g.op("Constant", value_t=torch.tensor(float("-inf"))))
        output = softmax(g, output, dim)

        return masked_fill(g, output, r_mask, g.op("Constant", value_t=torch.tensor(0, dtype=torch.uint8)))


class DropoutContext(object):
    def __init__(self):
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True


def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None

    if dropout > 0 and mask is None:
        mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).bool()

    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask

    return mask, dropout


class XDropout(torch.autograd.Function):
    """Optimized dropout function to save computation and memory by using mask operation instead of multiplication."""

    @staticmethod
    def forward(ctx, input, local_ctx):
        mask, dropout = get_mask(input, local_ctx)
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            ctx.save_for_backward(mask)
            return input.masked_fill(mask, 0) * ctx.scale
        else:
            return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale > 1:
            (mask,) = ctx.saved_tensors
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        else:
            return grad_output, None


class StableDropout(nn.Module):
    """
    Optimized dropout module for stabilizing the training

    Args:
        drop_prob (float): the dropout probabilities
    """

    def __init__(self, drop_prob):
        super().__init__()

        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def forward(self, x):
        """
        Call the module

        Args:
            x (:obj:`torch.tensor`): The input tensor to apply dropout
        """

        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())

        return x

    def clear_context(self):
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []
            
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())

            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1

            return ctx
        else:
            return self.drop_prob


class DebertaLayerNorm(nn.Module):
    """LayerNorm module in the TF style (epsilon inside the square root)."""

    def __init__(self, size, eps=1e-12):
        super().__init__()

        # size 对应 hidden dim
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))

        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # 记录下原始的数据类型
        input_type = hidden_states.dtype

        # 转为浮点类型以便计算
        hidden_states = hidden_states.float()
        mean = hidden_states.mean(-1, keepdim=True)
        # 计算方差 注意是在 hidden dim 计算
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        # 标准化
        hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)

        # 恢复原始数据类型
        hidden_states = hidden_states.to(input_type)

        # 仿射变换
        y = self.weight * hidden_states + self.bias
        return y


class DebertaSelfOutput(nn.Module):
    """FC->Dropout->LN"""
    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class DebertaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.self = DisentangledSelfAttention(config)
        # FC->Dropout->LN
        self.output = DebertaSelfOutput(config)

        self.config = config

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        self_output = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if output_attentions:
            self_output, att_matrix = self_output

        if query_states is None:
            query_states = hidden_states
        # 将 self_output 经过 FC & Dropout 然后将输出与 query_states 相加后送入 LN
        attention_output = self.output(self_output, query_states)

        return attention_output, att_matrix if output_attentions else attention_output


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->Deberta
class DebertaIntermediate(nn.Module):
    """FC->ACT"""
    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class DebertaOutput(nn.Module):
    """FC->Dropout->LN"""
    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)

        self.config = config

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # 残差连接
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class DebertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attention = DebertaAttention(config)
        # FFN
        self.intermediate = DebertaIntermediate(config)
        # FC->Dropout->LN
        self.output = DebertaOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        output_attentions=False,
    ):
        '''i. 解耦注意力计算'''
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if output_attentions:
            attention_output, att_matrix = attention_output

        '''ii.'''
        # 全连接(将 hidden_size 映射到 intermediate_size) + 激活函数
        intermediate_output = self.intermediate(attention_output)
        '''iii.'''
        # 将 intermediate_output 先经过 全连接(将 intermediate_size 映射回 hidden_size) & Dropout，
        # 然后与 attention_output 相加(残差连接)后输入到 LN
        layer_output = self.output(intermediate_output, attention_output)

        return (layer_output, att_matrix) if output_attentions else layer_output


class DebertaEncoder(nn.Module):
    """Modified BertEncoder with relative position bias support"""

    def __init__(self, config):
        super().__init__()

        self.layer = nn.ModuleList([DebertaLayer(config) for _ in range(config.num_hidden_layers)])

        self.relative_attention = getattr(config, "relative_attention", False)
        if self.relative_attention:
            # 相对位置的最大数值
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                # 'max_position_embeddings' 代表最大序列长度
                self.max_relative_positions = config.max_position_embeddings

            self.rel_embeddings = nn.Embedding(self.max_relative_positions * 2, config.hidden_size)

        # 不使用 checkpoint 模块节省运行时内存
        self.gradient_checkpointing = False

    def get_rel_embedding(self):
        return self.rel_embeddings.weight if self.relative_attention else None

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() <= 2:
            # (B,L)->(B,1,1,L)
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # (B,1,1,L) * (B,1,L,1) = (B,1,L,L)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
            attention_mask = attention_mask.byte()
        elif attention_mask.dim() == 3:
            # (B,L,L)->(B,1,L,L)
            attention_mask = attention_mask.unsqueeze(1)

        # (B,1,L,L) dim1 对应注意力头部
        return attention_mask

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
            q_size = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
            relative_pos = build_relative_position(q_size, hidden_states.size(-2), hidden_states.device)

        return relative_pos

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_hidden_states=True,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        return_dict=True,
    ):
        '''i. 准备 attention mask & 计算 query 和 key 的相对位置值'''
        # 对于第1层 Transformer 来说，输入的 hidden_states 是 patch embedding
        # (B,1,L,L) 对于第一层 Transformer 来说，(B,L)->(B,1,L,L)
        attention_mask = self.get_attention_mask(attention_mask)
        # (1,query_size or hidden_size,hidden_size) 
        # 对于第一层来说，就是 (1,L,L) 其中每个值的范围是 [-(L-1),L-1]
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

        '''ii. 决定是否要输出所有层的隐层状态以及注意力矩阵、取出相对位置 embedding 矩阵权重，从隐层中拿出下一层要计算的 key, value'''
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states

        # 相对位置 Embedding 矩阵的 weight: (2 * max_relative_positions,hidden_size)
        rel_embeddings = self.get_rel_embedding()

        '''iii. 每一层交互计算注意力，收集每层输出的隐层状态和注意力矩阵'''
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                # checkpoint 是以计算时间换取内存节省的一种方式
                # 在前向过程中不在计算图中保留中间变量，
                # 而是在反向传播时重新计算这些中间变量的前向过程然后获取相应的梯度
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    next_kv,
                    attention_mask,
                    query_states,
                    relative_pos,
                    rel_embeddings,
                )
            else:
                hidden_states = layer_module(
                    next_kv,
                    attention_mask,
                    query_states=query_states,
                    relative_pos=relative_pos,
                    rel_embeddings=rel_embeddings,
                    output_attentions=output_attentions,
                )

            if output_attentions:
                # att_m 代表 attention matrix
                hidden_states, att_m = hidden_states

            if query_states is not None:
                query_states = hidden_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = hidden_states

            if output_attentions:
                all_attentions = all_attentions + (att_m,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


def build_relative_position(query_size, key_size, device):
    """
    Build relative position according to the query and key

    We assume the absolute position of query :math:`P_q` is range from (0, query_size) and the absolute position of key
    :math:`P_k` is range from (0, key_size), The relative positions from query to key is :math:`R_{q \\rightarrow k} =
    P_q - P_k`

    Args:
        query_size (int): the length of query
        key_size (int): the length of key

    Return:
        :obj:`torch.LongTensor`: A tensor with shape [1, query_size, key_size]

    """

    q_ids = torch.arange(query_size, dtype=torch.long, device=device)
    k_ids = torch.arange(key_size, dtype=torch.long, device=device)

    # (query_size,key_size)
    rel_pos_ids = q_ids[:, None] - k_ids.view(1, -1).repeat(query_size, 1)
    # TODO: why?
    rel_pos_ids = rel_pos_ids[:query_size, :]
    # (1,query_size,key_size)
    rel_pos_ids = rel_pos_ids.unsqueeze(0)

    return rel_pos_ids


@torch.jit.script
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)])


@torch.jit.script
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), key_layer.size(-2), key_layer.size(-2)])


@torch.jit.script
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    return pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2)))


class DisentangledSelfAttention(nn.Module):
    """
    Disentangled self-attention module

    Parameters:
        config (:obj:`str`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            `BertConfig`, for more details, please refer :class:`~transformers.DebertaConfig`
    """

    def __init__(self, config):
        super().__init__()

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = config.hidden_size

        # QKV 矩阵
        self.in_proj = nn.Linear(config.hidden_size, self.all_head_size * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros((self.all_head_size), dtype=torch.float))
        self.v_bias = nn.Parameter(torch.zeros((self.all_head_size), dtype=torch.float))

        # 位置注意力类型 i.e. ['c2p', 'p2c']
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []

        self.talking_head = getattr(config, "talking_head", False)
        if self.talking_head:
            # 对原始的注意力系数(logits)做线性映射
            self.head_logits_proj = nn.Linear(config.num_attention_heads, config.num_attention_heads, bias=False)
            # 对 Softmax 后的注意力系数做线性映射
            self.head_weights_proj = nn.Linear(config.num_attention_heads, config.num_attention_heads, bias=False)

        self.relative_attention = getattr(config, "relative_attention", False)
        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                # 最大序列长度
                self.max_relative_positions = config.max_position_embeddings

            self.pos_dropout = StableDropout(config.hidden_dropout_prob)

            # TODO: why don't need 'bias' for 'c2p' but need for 'p2c'?
            if "c2p" in self.pos_att_type or "p2p" in self.pos_att_type:
                self.pos_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
            if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
                self.pos_q_proj = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = StableDropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # (B,L,C)->(B,L,num_heads,C//num_heads)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, -1)
        x = x.view(*new_x_shape)

        # (B,L,num_heads,C//num_heads)->(B,num_heads,L,C//num_heads)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        """
        Call the module

        Args:
            hidden_states (:obj:`torch.FloatTensor`):
                Input states to the module usually the output from previous layer, it will be the Q,K and V in
                `Attention(Q,K,V)`

            attention_mask (:obj:`torch.ByteTensor`):
                An attention mask matrix of shape [`B`, `N`, `N`] where `B` is the batch size, `N` is the maximum
                sequence length in which element [i,j] = `1` means the `i` th token in the input can attend to the `j`
                th token.

            output_attentions (:obj:`bool`, optional):
                Whether return the attention matrix.

            query_states (:obj:`torch.FloatTensor`, optional):
                The `Q` state in `Attention(Q,K,V)`.

            relative_pos (:obj:`torch.LongTensor`):
                The relative position encoding between the tokens in the sequence. It's of shape [`B`, `N`, `N`] with
                values ranging in [`-max_relative_positions`, `max_relative_positions`].

            rel_embeddings (:obj:`torch.FloatTensor`):
                The embedding of relative distances. It's a tensor of shape [:math:`2 \\times
                \\text{max_relative_positions}`, `hidden_size`].


        """
        '''i. 生成 query, key, value'''
        # 若输入中没有 query，则将 hidden states 输入到一个共享矩阵生成 qkv
        if query_states is None:
            # (B,L,C)->(B,L,3C)
            qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
            # (B,num_heads,L,C//num_heads) x 3
            query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
        # 否则，将共享矩阵的权重拆成3部分，分别用于 qkv 的生成：qv 用之前的 query 生成，k 用 hidden states 生成
        else:
            def linear(w, b, x):
                out = torch.matmul(x, w.t())
                if b is not None:
                    out += b.t()
                
                return out

            # in_proj.weight 的 dim0 共有 3 * num_heads * hidden_dim_per_head
            # 3 x num_heads x (hidden_dim_per_head,*)
            ws = self.in_proj.weight.chunk(self.num_attention_heads * 3, dim=0)
            # QKV 矩阵的 weight 3 x (hidden_dim_per_head x num_heads,*)
            qkvw = [torch.cat([ws[i * 3 + k] for i in range(self.num_attention_heads)], dim=0) for k in range(3)]
            # QKV 矩阵的 bias
            qkvb = [None] * 3

            q = linear(qkvw[0], qkvb[0], query_states)
            k, v = [linear(qkvw[i], qkvb[i], hidden_states) for i in range(1, 3)]
            # 每个都经过变换：(B,L,hidden_dim_per_head x num_heads)->(B,num_heads,L,hidden_dim_per_head)
            query_layer, key_layer, value_layer = [self.transpose_for_scores(x) for x in [q, k, v]]

        # (B,num_heads,L,hidden_dim_per_head) + (1,1,num_heads x hidden_dim_per_head)->(1,num_heads,1,hidden_dim_per_head)
        query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
        value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])

        '''ii. 内容与内容之间的注意力(c2c)的计算'''
        # Take the dot product between "query" and "key" to get the raw attention scores.
        # 通常 pos_att_type 是 ['c2p', 'p2c']，于是 scale_factor = 3，
        # 从而 scale = sqrt(3 x hidden_dim_per_head)，与 paper 中的公式对应
        scale_factor = 1 + len(self.pos_att_type)
        scale = math.sqrt(query_layer.size(-1) * scale_factor)
        query_layer = query_layer / scale
        # (B,num_heads,L,L) 这部分是 c2c(内容到内容) 的注意力计算
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        '''iii. 内容与位置之间的注意力计算(c2p,p2c)'''
        rel_att = None
        if self.relative_attention:
            # (2 x max_relative_positions,hidden_dim_all_heads)
            rel_embeddings = self.pos_dropout(rel_embeddings)
            # 解耦的相对注意力计算 i.e c2p, p2c
            rel_att = self.disentangled_att_bias(query_layer, key_layer, relative_pos, rel_embeddings, scale_factor)
        
        '''iv. 注意力的后处理：内容与位置之间的注意力相加、线性映射、Softmax 映射到 (0,1) 区间'''
        if rel_att is not None:
            # c2c + (c2p + p2c) (B,num_heads,L,L)
            attention_scores = attention_scores + rel_att
        if self.talking_head:
            # 因为 head_logits_proj(实质是一个 FC) 的输入维度是 num_heads，所以需要先排列维度再恢复回来
            # (B,num_heads,L,L)->(B,L,L,num_heads)->(B,num_heads,L,L)
            attention_scores = self.head_logits_proj(attention_scores.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # Softmax 将注意力分数映射到 (0,1) 区间，mask 用于忽略掉某些位置
        # 将 mask == 0 的位置用 -inf 填充输入到 softmax，然后将输出结果在这些位置上再用0填充
        attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
        attention_probs = self.dropout(attention_probs)
        if self.talking_head:
            # 因为 head_weights_proj(实质是一个 FC) 的输入维度是 num_heads，所以需要先排列维度再恢复回来
            # (B,num_heads,L,L)->(B,L,L,num_heads)->(B,num_heads,L,L)
            attention_probs = self.head_weights_proj(attention_probs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        '''v. 将注意力施加在 value 上并恢复原来的维度'''
        context_layer = torch.matmul(attention_probs, value_layer)
        # (B,num_heads,L,hidden_dim_per_head)->(B,L,num_heads,hidden_dim_per_head)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # (B,L,-1)
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        # (B,L,num_heads,hidden_dim_per_head)->(B,L,hidden_dim = num_heads x hidden_dim_per_head)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_probs if output_attentions else context_layer

    def disentangled_att_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        '''i. 计算(确定) query 和 key 的相对位置值'''
        if relative_pos is None:
            # q_L(query length)
            q = query_layer.size(-2)
            # (1,q_L,k_L)
            relative_pos = build_relative_position(q, key_layer.size(-2), query_layer.device)
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        # 必须是对应 (B,num_heads,q_L,k_L) 实际应该是 (1,1,q_L,k_L)
        elif relative_pos.dim() != 4:
            raise ValueError(f"Relative position ids must be of dim 2 or 3 or 4. {relative_pos.dim()}")

        '''ii. 根据相对位置在 embedding 矩阵中取出对应的 embedding 部分'''
        # 以下注释记 L'=max(q_L,k_L)
        att_span = min(max(query_layer.size(-2), key_layer.size(-2)), self.max_relative_positions)
        # (1,2L',hidden_dim)
        rel_embeddings = rel_embeddings[
            (self.max_relative_positions - att_span):(self.max_relative_positions + att_span), :
        ].unsqueeze(0)

        '''iii. 根据相对位置 embedding 分别计算出 query 和 key 的位置 embedding，分别用于 c2p(& p2p) 和 p2c(& p2p)'''
        # 计算 key 的位置 embedding，用于 c2p 和 p2p
        if "c2p" in self.pos_att_type or "p2p" in self.pos_att_type:
            # (1,2L',hidden_dim)
            pos_key_layer = self.pos_proj(rel_embeddings)
            # (1,num_heads,2L',hidden_dim_per_head)
            pos_key_layer = self.transpose_for_scores(pos_key_layer)
        # 计算 query 的位置 embedding，用于 p2c 和 p2p
        if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
            # (1,2L',hidden_dim)
            pos_query_layer = self.pos_q_proj(rel_embeddings)
            # (1,num_heads,2L',hidden_dim_per_head)
            pos_query_layer = self.transpose_for_scores(pos_query_layer)

        score = 0
        relative_pos = relative_pos.long().to(query_layer.device)

        '''iv. query 内容 -> key 位置 的注意力计算'''
        # content->position
        if "c2p" in self.pos_att_type:
            # query 内容 与 所有 key 位置(2L' 个位置，maybe 当前序列并没有那么长) embedding 计算注意力
            # (B,num_heads,q_L,hidden_dim_per_head) dot (1,num_heads,hidden_dim_per_head,2L')
            # (B,num_heads,q_L,2L')
            c2p_att = torch.matmul(query_layer, pos_key_layer.transpose(-1, -2))
            # relative_pos 的值域是 [min(-k_L,-q_L), max(k_L,q_L)]，这里加上 attn_span(L') 并裁剪至 2L'-1 以便用作索引
            # (1,1,q_L,k_L)
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            # 根据 query 和 key 的相对位置在以上计算出的注意力矩阵中取出实际的 c2p 注意力值
            # (B,num_heads,q_L,k_L) 'c2p_dynamic_expand' 使得 c2p_pos 的前3个 dim 与 query 一致，最后1个与 relative_pos 一致
            c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_dynamic_expand(c2p_pos, query_layer, relative_pos))
            
            score += c2p_att

        '''iv. query 位置 -> key 内容 的注意力计算'''
        # position->content
        if "p2c" in self.pos_att_type or "p2p" in self.pos_att_type:
            # 当 query 与 key 的长度不一致的情况下，以 key 长度做相对位置计算，以兼容 p2p
            if query_layer.size(-2) != key_layer.size(-2):
                # (1,k_L,k_L)
                r_pos = build_relative_position(key_layer.size(-2), key_layer.size(-2), query_layer.device)
            else:
                # (1,1,q_L,k_L) 这种情况下 q_L=k_L
                r_pos = relative_pos

            # 加上偏移量以便用作注意力矩阵的索引
            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)

        if "p2c" in self.pos_att_type:
            # 除以 sqrt(3 x hidden_dim_per_head) 缩放
            pos_query_layer /= math.sqrt(pos_query_layer.size(-1) * scale_factor)
            # key 的内容 与 所有 query 的位置(2L' 个位置，maybe 当前序列并没有那么长) 计算注意力
            # (B,num_heads,k_L,hidden_dim_per_head) dot (1,num_heads,hidden_dim_per_head,2L')
            # (B,num_heads,k_L,2L')
            p2c_att = torch.matmul(key_layer, pos_query_layer.transpose(-1, -2))
            # 根据 query 和 key 的相对位置在以上计算出的注意力矩阵中取出实际的 p2c 注意力值
            # (B,num_heads,k_L,k_L) 这里有可能 k_L=q_L
            p2c_att = torch.gather(
                # 'p2c_dynamic_expand' 将 p2c_pos 的前两个 dim 变成与 query_layer 一致，后两个变成与 key_layer 一致
                p2c_att, dim=-1, index=p2c_dynamic_expand(p2c_pos, query_layer, key_layer)
            ).transpose(-1, -2)  # dim -1 对应 query 位置，dim -2 对应 key 内容

            # 当 query 和 key 长度不等时，进一步将对应 query 位置的注意力取出来
            # (这种情况下 query 的长度必定要比 key 小，否则会越界)
            if query_layer.size(-2) != key_layer.size(-2):
                # (1,1,q_L,1)
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)
                # 'pos_dynamic_expand' 将 pos_index 前2个 dim 变成与 p2c_att 一致，最后1个 dim 变成与 key_layer 一致
                # 注意这里是在 dim -2 做 gather，因为前面已经将 query 位置对应的 dim 置换到 -2 dim 了
                p2c_att = torch.gather(p2c_att, dim=-2, index=pos_dynamic_expand(pos_index, p2c_att, key_layer))
            
            score += p2c_att

        return score


class DebertaEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()

        # Token embedding
        # 填充部分(非真实的输入文本)的 token id
        pad_token_id = getattr(config, "pad_token_id", 0)
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        # 指定了 padding_idx 后，在 embedding table 里对应这个位置的 embedding 在训练过程中就不会更新
        # 也就是 token id 等于 pad token id 的 token 对应的 embedding 将不会被学习
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx=pad_token_id)

        # 绝对位置编码 Position embedding
        self.position_biased_input = getattr(config, "position_biased_input", True)
        if not self.position_biased_input:
            self.position_embeddings = None
        else:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)

        # 是否区分不同句子的 token
        if config.type_vocab_size > 0:
            # Segment embedding 区分不同句子的 token
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.embedding_size)

        # 若 embedding size 和 Transformers 的 hidden_size 不一致，
        # 则对输入文本进行 embedding 后要额外做一个 projection 以适配到 Transformers 输入所需的维度
        if self.embedding_size != config.hidden_size:
            self.embed_proj = nn.Linear(self.embedding_size, config.hidden_size, bias=False)

        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)

        self.config = config

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        # 标识每个 token 在序列中的绝对位置
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, mask=None, inputs_embeds=None):
        # (B,L)
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        '''i. Word embedding'''
        if inputs_embeds is None:
            # 对输入文本做词嵌入 (B,L,C)
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = inputs_embeds

        '''ii. Absolute position embedding'''
        # token 在一个句子中的绝对位置信息
        if self.position_biased_input:
            # 基于每个 token 在序列中的绝对位置做位置嵌入 (B,L,C)
            if self.position_embeddings is not None:
                # 输入文本的序列长度
                seq_length = input_shape[1]
                if position_ids is None:
                    # (1,L)
                    position_ids = self.position_ids[:, :seq_length]

                position_embeddings = self.position_embeddings(position_ids.long())
            else:
                position_embeddings = torch.zeros_like(inputs_embeds)

            embeddings += position_embeddings

        '''iii. Segment embedding'''
        # 指示 token 在输入序列的哪个句子
        # 不同的句子中的 token 用不同的 embedding 空间以区分
        if self.config.type_vocab_size > 0:
            if token_type_ids is None:
                # (B,L)
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        '''iv. Embedding projection'''
        # 对输入 embedding 做 projection 以便后续适配 Transformers 编码
        if self.embedding_size != self.config.hidden_size:
            # (B,L,C')
            embeddings = self.embed_proj(embeddings)

        '''v. Post-process: LN->Mask(optional)->Dropout'''
        embeddings = self.LayerNorm(embeddings)

        # attention mask，避免后续 Transformer 对 padding 的部分计算 attention
        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            mask = mask.to(embeddings.dtype)

            embeddings = embeddings * mask

        embeddings = self.dropout(embeddings)

        return embeddings


class DebertaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization & 
    a simple interface for downloading and loading pretrained models.
    """

    config_class = DebertaConfig
    base_model_prefix = "deberta"
    _keys_to_ignore_on_load_missing = ["position_ids"]
    _keys_to_ignore_on_load_unexpected = ["position_embeddings"]
    # 是否 Pytorch 的 checkpoint 模块以节省运行时内存
    # 父类 'PretrainedModel' 会检查 config_class 实例中是否有设置 'gradient_checkpointing' 属性
    # 若 设置为 True，则会调用子类(也就是这里的 DebertaPretrainedModel)的 _set_gradient_checkpointing() 方法
    # 可以这样设置：
    # i. cfg = AutoConfig.from_pretrained('microsoft/deberta', gradient_checkpointing=True)
    # ii. model = AutoModel.from_pretrained('microsoft/deberta', config=cfg)
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        # 若设置了使用 checkpoint 模块节省内存，则仅对模型的 Encoder 部分设置
        if isinstance(module, DebertaEncoder):
            module.gradient_checkpointing = value


DEBERTA_START_DOCSTRING = r"""
    The DeBERTa model was proposed in `DeBERTa: Decoding-enhanced BERT with Disentangled Attention
    <https://arxiv.org/abs/2006.03654>`_ by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen. It's build on top of
    BERT/RoBERTa with two improvements, i.e. disentangled attention and enhanced mask decoder. With those two
    improvements, it out perform BERT/RoBERTa on a majority of tasks with 80GB pretraining data.

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.```


    Parameters:
        config (:class:`~transformers.DebertaConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

DEBERTA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.DebertaTokenizer`. See
            :func:`transformers.PreTrainedTokenizer.encode` and :func:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare DeBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    DEBERTA_START_DOCSTRING,
)
class DebertaModel(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = DebertaEmbeddings(config)
        self.encoder = DebertaEncoder(config)
        # 用于在 Encoder 输出后，利用最后一层输出的隐层状态作为 query，倒数第二层输出 作为 key 和 value，
        # 然后继续用最后一层 Transformer layer 去编码，每次输出的状态都作为新的 query 与原本倒数第二层的隐层状态去编码
        self.z_steps = 0
        self.config = config

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} 
        See base class PreTrainedModel
        """
        raise NotImplementedError("The prune function is not implemented in DeBERTa model.")

    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # 是否输出 Transformer 所有层的 attention 权重矩阵
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 是否输出 Transformer 所有层的 hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # True 则返回的是一个包含输出信息的结构体，否则是元组
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            # token 个数
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            # 最后一个维度是嵌入维度，因此省去得到的是 token 个数
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        # 指示每个 token 属于那个句子
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        # 一个 OrderDict 子类实例，支持索引和关键字查找
        # [0] 是最后一层的输出，[1] 是所有层的输出，[2] 是注意力矩阵
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        # 取出 Encoder 所有层的输出状态
        encoded_layers = encoder_outputs[1]

        if self.z_steps > 1:
            # Encoder 倒数第二层的输出
            hidden_states = encoded_layers[-2]
            # list 中每个是 Transformer 最后一层，是 DeBERTalayer 实例
            layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
            # Encoder 最后一层的输出
            query_states = encoded_layers[-1]

            # 相对位置 embedding 矩阵
            # (2 x max_L, hidden_dim)
            rel_embeddings = self.encoder.get_rel_embedding()
            # (B,1,q_L,k_L)
            attention_mask = self.encoder.get_attention_mask(attention_mask)
            # (1,q_L,k_L)
            rel_pos = self.encoder.get_rel_pos(embedding_output)

            for layer in layers[1:]:
                query_states = layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=False,
                    query_states=query_states,
                    relative_pos=rel_pos,
                    rel_embeddings=rel_embeddings,
                )
                encoded_layers.append(query_states)

        sequence_output = encoded_layers[-1]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[(1 if output_hidden_states else 2) :]

        # 这是一个 OrderDict 子类
        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions,
        )


@add_start_docstrings("""DeBERTa Model with a `language modeling` head on top. """, DEBERTA_START_DOCSTRING)
class DebertaForMaskedLM(DebertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.deberta = DebertaModel(config)
        self.cls = DebertaOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            # Pytorch 的 CE loss 中，默认忽略的 label index 就是-100，通过 'ignore_index' 参数指定
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# copied from transformers.models.bert.BertPredictionHeadTransform with bert -> deberta
class DebertaPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


# copied from transformers.models.bert.BertLMPredictionHead with bert -> deberta
class DebertaLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = DebertaPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # TODO: what do this mean?
        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)

        return hidden_states


# copied from transformers.models.bert.BertOnlyMLMHead with bert -> deberta
class DebertaOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = DebertaLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


@add_start_docstrings(
    """
    DeBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    DEBERTA_START_DOCSTRING,
)
class DebertaForSequenceClassification(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.deberta = DebertaModel(config)
        # 并非真正意义上的 pooling，仅仅是取出第一个 token 然后经过 Dropout、FC 和 激活函数
        self.pooler = ContextPooler(config)

        self.num_labels = getattr(config, "num_labels", 2)
        self.classifier = nn.Linear(self.pooler.output_dim, self.num_labels)

        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        self.init_weights()

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 一个 OrderDict 子类的实例，支持索引和关键字查找
        # 若 output_attentions 和 output_hidden_states 都设置为 True，
        # 则会返回所有层的注意力矩阵 和 隐层状态，即：
        # outputs[0]: 最后一层隐层状态、outputs[1]: 所有层隐层状态、outputs[2]: 所有层注意力矩阵
        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # (B,L,C)
        encoder_layer = outputs[0]
        # 取出第1个 token 送入 Dropout->FC->ACT
        # (B,C)
        pooled_output = self.pooler(encoder_layer)
        # (B,num_labels)
        logits = self.classifier(self.dropout(pooled_output))

        loss = None
        if labels is not None:
            # 回归任务，用 MSE 计算 loss
            if self.num_labels == 1:
                # regression task
                loss_fn = nn.MSELoss()
                logits = logits.view(-1).to(labels.dtype)
                loss = loss_fn(logits, labels.view(-1))
            # 分类任务，用 CE 计算 loss
            elif labels.dim() == 1 or labels.size(-1) == 1:
                # 仅对 label 不小于 0 的样本(有效样本)算 loss
                label_index = (labels >= 0).nonzero()
                labels = labels.long()

                if label_index.size(0) > 0:
                    # 取出有效样本对应的 logits 和 labels
                    labeled_logits = torch.gather(logits, 0, label_index.expand(label_index.size(0), logits.size(1)))
                    labels = torch.gather(labels, 0, label_index.view(-1))

                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(labeled_logits.view(-1, self.num_labels).float(), labels.view(-1))
                # 若所有样本的标签都小于0，则不算 loss
                else:
                    loss = torch.tensor(0).to(logits)
            # 多标签的分类任务(此时 labels 和 logits 的 shape 一致)
            else:
                log_softmax = nn.LogSoftmax(-1)
                loss = -((log_softmax(logits) * labels).sum(-1)).mean()

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        else:
            # OrderDict 子类实例，支持索引、关键字查找 以及 使用 .attribute 取出对应属性
            return SequenceClassifierOutput(
                loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
            )


@add_start_docstrings(
    """
    DeBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    DEBERTA_START_DOCSTRING,
)
class DebertaForTokenClassification(DebertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.deberta = DebertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


@add_start_docstrings(
    """
    DeBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    DEBERTA_START_DOCSTRING,
)
class DebertaForQuestionAnswering(DebertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.deberta = DebertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

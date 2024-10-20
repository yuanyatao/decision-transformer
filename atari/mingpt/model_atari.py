
"""
- 初始词干由 token 编码和位置编码的组合组成 
- 其核心是 Transformer 块的统一序列
- 每个 Transformer 都是 1 个隐藏层 MLP 块和自注意力块的顺序组合 
- 所有块都输入到类似于 resnets 的中央残差通路 
- 最终解码器是到 vanilla Softmax 分类器的线性投影
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

import numpy as np
# 高斯误差线性单元激活函数
class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input) 

class GPTConfig:
    """基础 GPT 配置类，包含所有 GPT 版本通用的参数。"""

    # 嵌入层的 dropout 概率，用于防止过拟合
    embd_pdrop = 0.1 
    # 残差连接的 dropout 概率
    resid_pdrop = 0.1  
    # 注意力机制的 dropout 概率
    attn_pdrop = 0.1  

    def __init__(self, vocab_size, block_size, **kwargs):
        """
        初始化 GPT 配置。

        参数：
        - vocab_size: 词汇表的大小，即模型可以处理的不同词（或动作）的数量。
        - block_size: 上下文长度，即输入序列的最大长度（例如，一个时间步中的状态数量）。
        - kwargs: 其他配置参数，可以通过关键字参数传递，支持动态配置。
        """
        self.vocab_size = vocab_size  # 记录词汇表大小
        self.block_size = block_size    # 记录上下文长度

        # 遍历所有额外的关键字参数并将其设置为实例属性
        for k, v in kwargs.items():
            setattr(self, k, v)  # 动态添加其他属性，方便灵活配置


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module): #因果自注意力机制
    """
    一个原始的多头蒙版自注意力层，末尾有一个投影。这里可以使用 torch.nn.MultiheadAttention，
    但我在这里包含了一个明确的实现，以表明这里没有什么太可怕的。
    """

    def __init__(self, config):
        super().__init__()
        #语句确保嵌入维度 n_embd 可以被头数 n_head 整除，以便将嵌入均匀地划分到每个头。
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        #使用 tril 函数生成一个下三角矩阵作为因果掩码，确保每个位置只能关注自己及之前的位置。
        # register_buffer 确保掩码在模型保存和加载时仍然可用，但不会作为模型参数进行更新。
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
                                     .view(1, 1, config.block_size + 1, config.block_size + 1))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size() #B：批次大小，T：时间步，C：嵌入维度

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    """GPT 模型，具有指定上下文长度 block_size，用于强化学习中的序列建模。"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_type = config.model_type  # 模型类型，如 'reward_conditioned' 或 'naive'。

        # 输入的嵌入层：用于将离散的词（动作）转换为嵌入向量。
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # 位置嵌入：用于为每个时间步添加位置信息，提升模型对顺序数据的感知能力。
        # 包括局部位置嵌入和全局位置嵌入（用于时间步的编码）。
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep + 1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)  # 进行 Dropout 以防止过拟合。

        # Transformer 的主体部分，由多个 Transformer 块构成。
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # 输出层：将 Transformer 的输出转换为预测的类别（动作）。
        self.ln_f = nn.LayerNorm(config.n_embd)  # 层归一化
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # 输出层

        self.block_size = config.block_size  # 上下文长度（block_size）。
        self.apply(self._init_weights)  # 初始化模型权重。

        # 打印模型参数数量。
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        # 状态编码器：将 4 帧 84x84 的图像状态嵌入为向量，用于强化学习中的视觉输入处理。
        # 状态编码 似乎仅仅是几个卷积，是否可以考虑其他形式？
        self.state_encoder = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4, padding=0), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
            nn.Flatten(), nn.Linear(3136, config.n_embd), nn.Tanh()
        )

        # 回报的嵌入层：用于将回报信息嵌入为向量。 把1维的东西变成这么多维度，是否有些不合理呢？
        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())

        # 动作的嵌入层：将离散的动作转换为嵌入向量，并初始化嵌入权重。
        self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

    def get_block_size(self):
        # 返回模型的上下文长度（block_size）。
        return self.block_size

    def _init_weights(self, module):
        # 初始化模型的权重，使用正态分布或零初始化。
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        配置优化器，将模型的参数分为两类：需要权重衰减的参数和不需要权重衰减的参数（如偏置和嵌入层）。
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # 生成参数的完整名称

                if pn.endswith('bias'):
                    # 偏置项不进行权重衰减
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # 线性层和卷积层的权重需要进行权重衰减
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # 层归一化和嵌入层的权重不进行权重衰减
                    no_decay.add(fpn)

        # 特殊处理位置嵌入参数，不进行衰减
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # 检查是否所有参数都已分类到 decay 或 no_decay 中
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s 属于 decay 和 no_decay!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "参数 %s 未被分类为 decay 或 no_decay!" \
            % (str(param_dict.keys() - union_params), )

        # 创建 PyTorch 的 AdamW 优化器对象，分别对 decay 和 no_decay 的参数进行优化。
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    # state, action, and return
    def forward(self, states, actions, targets=None, rtgs=None, timesteps=None):
        """
        前向传播，处理状态、动作和回报，并生成预测结果。

        参数：
        - states: 状态输入 (batch, block_size, 4*84*84)
        - actions: 动作输入 (batch, block_size, 1)
        - targets: 目标动作标签 (batch, block_size, 1)
        - rtgs: 回报-to-go (batch, block_size, 1)
        - timesteps: 时间步 (batch, 1, 1)
        """
        # 将状态通过状态编码器，提取嵌入特征
        state_embeddings = self.state_encoder(states.reshape(-1, 4, 84, 84).type(torch.float32).contiguous()) 
        state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1], self.config.n_embd)

        # 根据不同情况处理动作和回报的嵌入，构造输入 token 的嵌入
        if actions is not None and self.model_type == 'reward_conditioned': 
            # 这通常在训练阶段或验证阶段的非初始时间步中使用，因为此时有历史的状态、动作以及对应的回报-to-go 信息。
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*3 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::3,:] = rtg_embeddings
            token_embeddings[:,1::3,:] = state_embeddings
            token_embeddings[:,2::3,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
        elif actions is None and self.model_type == 'reward_conditioned': # only happens at very first timestep of evaluation
            # actions 为空，说明当前模型还没有动作输入（通常是评估阶段的第一步）。
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2, self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = rtg_embeddings # really just [:,0,:]
            token_embeddings[:,1::2,:] = state_embeddings # really just [:,1,:]
        elif actions is not None and self.model_type == 'naive':
            # self.model_type 是 'naive'，表示模型是基于状态和动作的，没有显式的回报-to-go 条件。
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = state_embeddings
            token_embeddings[:,1::2,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
        elif actions is None and self.model_type == 'naive': # only happens at very first timestep of evaluation
            # 仅发生在评估的第一个时间步
            token_embeddings = state_embeddings
        else:
            raise NotImplementedError()

        # 全局位置嵌入，用于时间步信息编码
        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0) # batch_size, traj_length, n_embd

        # 获取位置嵌入，并加到 token 的嵌入中
        position_embeddings = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1)) + self.pos_emb[:, :token_embeddings.shape[1], :]

        # 将嵌入经过 Dropout、Transformer 块和 LayerNorm，得到模型输出
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # 根据模型类型调整预测结果的形状
        if actions is not None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1::3, :]  # 仅保留状态嵌入的预测
        elif actions is None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1:, :]
        elif actions is not None and self.model_type == 'naive':
            logits = logits[:, ::2, :]  # 仅保留状态嵌入的预测
        elif actions is None and self.model_type == 'naive':
            logits = logits
        else:
            raise NotImplementedError()

        # 如果提供了目标动作标签，计算交叉熵损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss
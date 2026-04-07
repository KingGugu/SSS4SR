import torch
import torch.nn as nn
import copy
from model._abstract_model import SequentialRecModel
from model._modules import LayerNorm, FeedForward

"""
[Paper]
Author: Kun Zhou et al. 
Title: "Filter-enhanced MLP is All You Need for Sequential Recommendation"
Conference: WWW 2022

[Code Reference]
https://github.com/Woeee/FMLP-Rec
"""
    
class FMLPRecModel(SequentialRecModel):
    def __init__(self, args):
        super(FMLPRecModel, self).__init__(args)

        self.args = args
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.item_encoder = FMLPRecEncoder(args)
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        sequence_emb = self.add_position_embedding(input_ids)
        item_encoded_layers = self.item_encoder(sequence_emb,
                                                output_all_encoded_layers=True,
                                                )
        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]

        return sequence_output

    # def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
    #
    #     seq_out = self.forward(input_ids)
    #     seq_out = seq_out[:, -1, :]
    #     pos_ids, neg_ids = answers, neg_answers
    #
    #     # [batch seq_len hidden_size]
    #     pos_emb = self.item_embeddings(pos_ids)
    #     neg_emb = self.item_embeddings(neg_ids)
    #
    #     # [batch hidden_size]
    #     seq_emb = seq_out # [batch*seq_len hidden_size]
    #     pos_logits = torch.sum(pos_emb * seq_emb, -1) # [batch*seq_len]
    #     neg_logits = torch.sum(neg_emb * seq_emb, -1)
    #
    #     loss = torch.mean(
    #         - torch.log(torch.sigmoid(pos_logits) + 1e-24) -
    #         torch.log(1 - torch.sigmoid(neg_logits) + 1e-24)
    #     )
    #
    #     return loss
    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        # 1. 前向传播获取序列表示 (Batch_Size, Hidden_Size)
        seq_out = self.forward(input_ids)
        seq_out = seq_out[:, -1, :]  # 只取最后一个时间步用于预测

        # -------------------------------------------------------
        # 分支 A: Cross Entropy (CE) - 全量物品 Softmax
        # -------------------------------------------------------
        if self.args.loss_type == 'CE':
            # 获取所有物品的 Embedding 权重: [item_num, hidden_size]
            # self.item_embeddings 定义在父类 SequentialRecModel 中
            item_emb = self.item_embeddings.weight

            # 计算全量 Logits: [batch, hidden] @ [hidden, n_items] -> [batch, n_items]
            logits = torch.matmul(seq_out, item_emb.transpose(0, 1))

            # 计算交叉熵 (自动忽略 padding_idx=0)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, answers)
            return loss

        # -------------------------------------------------------
        # 分支 B: Binary Cross Entropy (BCE) - 负采样
        # -------------------------------------------------------
        elif self.args.loss_type == 'BCE':
            # 获取正样本和负样本的 Embedding
            pos_emb = self.item_embeddings(answers)  # [batch, hidden]
            neg_emb = self.item_embeddings(neg_answers)  # [batch, hidden]

            # 计算点积得分 (Logits)
            # seq_out: [batch, hidden], pos_emb: [batch, hidden] -> element-wise mul -> sum -> [batch]
            pos_logits = torch.sum(pos_emb * seq_out, -1)
            neg_logits = torch.sum(neg_emb * seq_out, -1)

            # 构造标签: 正样本为1，负样本为0
            pos_labels = torch.ones_like(pos_logits)
            neg_labels = torch.zeros_like(neg_logits)

            # 使用 BCEWithLogitsLoss (自带 Sigmoid，数值更稳定)
            loss_fct = nn.BCEWithLogitsLoss()

            # 可选：如果你需要像原代码那样严格屏蔽 padding (id=0)，可以加 mask 逻辑
            # 但通常在 batch 构造时 answers 已经是有效 item 了
            loss = loss_fct(pos_logits, pos_labels) + loss_fct(neg_logits, neg_labels)

            return loss

        else:
            raise ValueError(f"Invalid loss_type: {self.args.loss_type}. Choose 'CE' or 'BCE'.")

class FMLPRecEncoder(nn.Module):
    def __init__(self, args):
        super(FMLPRecEncoder, self).__init__()
        self.args = args
        block = FMLPRecBlock(args)

        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, output_all_encoded_layers=False):

        all_encoder_layers = [ hidden_states ]

        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states,)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states) # hidden_states => torch.Size([256, 50, 64])

        return all_encoder_layers

class FMLPRecBlock(nn.Module):
    def __init__(self, args):
        super(FMLPRecBlock, self).__init__()
        self.layer = FMLPRecLayer(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states):
        layer_output = self.layer(hidden_states)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output

class FMLPRecLayer(nn.Module):
    def __init__(self, args):
        super(FMLPRecLayer, self).__init__()
        self.complex_weight = nn.Parameter(torch.randn(1, args.max_seq_length//2 + 1, args.hidden_size, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')

        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = hidden_states + input_tensor

        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states

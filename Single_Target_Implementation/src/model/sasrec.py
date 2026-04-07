import torch
import torch.nn as nn
import copy
from model._abstract_model import SequentialRecModel
from model._modules import TransformerEncoder, LayerNorm

"""
[Paper]
Author: Wang-Cheng Kang et al. 
Title: "Self-Attentive Sequential Recommendation."
Conference: ICDM 2018

[Code Reference]
https://github.com/kang205/SASRec
https://github.com/Woeee/FMLP-Rec
"""

class SASRecModel(SequentialRecModel):
    def __init__(self, args):
        super(SASRecModel, self).__init__(args)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.item_encoder = TransformerEncoder(args)
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        extended_attention_mask = self.get_attention_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids)
        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
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
    #     pos_labels, neg_labels = torch.ones(pos_logits.shape, device=seq_out.device), torch.zeros(neg_logits.shape, device=seq_out.device)
    #     indices = (pos_ids != 0).nonzero().reshape(-1)
    #     bce_criterion = torch.nn.BCEWithLogitsLoss()
    #     loss = bce_criterion(pos_logits[indices], pos_labels[indices])
    #     loss += bce_criterion(neg_logits[indices], neg_labels[indices])
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

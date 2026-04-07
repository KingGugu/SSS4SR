import torch
from torch import nn
from model._abstract_model import SequentialRecModel

"""
[Paper]
Author: Yong Kiam Tan et al.
Title: "Improved Recurrent Neural Networks for Session-based Recommendations."
Conference: DLRS 2016

[Code Reference]
https://github.com/RUCAIBox/RecBole
"""

class GRU4RecModel(SequentialRecModel):

    def __init__(self, args):
        super(GRU4RecModel, self).__init__(args)

        # load parameters info
        self.args = args
        self.embedding_size = args.hidden_size
        self.hidden_size = args.gru_hidden_size
        self.num_layers = args.num_hidden_layers
        self.dropout_prob = args.hidden_dropout_prob

        # define layers and loss
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)

        # parameters initialization
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        item_seq_emb = self.item_embeddings(input_ids)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        return gru_output

    # #BPR
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
    #     gamma = 1e-10
    #     loss =  -torch.log( gamma + torch.sigmoid( pos_logits - neg_logits ) ).mean()
    #
    #     return loss

    # # BCE
    # def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
    #     # 1. 获取最后一个时间步的输出
    #     seq_out = self.forward(input_ids)
    #     seq_out = seq_out[:, -1, :]  # [batch_size, hidden_size]
    #
    #     # 2. 获取正负样本 Embedding
    #     pos_ids, neg_ids = answers, neg_answers
    #     pos_emb = self.item_embeddings(pos_ids)
    #     neg_emb = self.item_embeddings(neg_ids)
    #
    #     # 3. 计算 Logits (点积)
    #     pos_logits = torch.sum(pos_emb * seq_out, -1)
    #     neg_logits = torch.sum(neg_emb * seq_out, -1)
    #
    #     # 4. === 核心修改部分：计算 BCE Loss ===
    #     # 方式 A：使用 PyTorch 内置函数 (推荐，数值更稳定)
    #     # 将正样本的标签设为 1，负样本的标签设为 0
    #     loss_fct = nn.BCEWithLogitsLoss()
    #     pos_labels = torch.ones_like(pos_logits)
    #     neg_labels = torch.zeros_like(neg_logits)
    #
    #     # 同时优化正样本和负样本
    #     loss = loss_fct(pos_logits, pos_labels) + loss_fct(neg_logits, neg_labels)
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
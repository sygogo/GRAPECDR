import torch.nn as nn
import torch
import torch.nn.functional as F
from src.model.layers.ExternalMemory import ExternalMemoryNetwork


class PersonalizedTransferNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(PersonalizedTransferNet, self).__init__()
        self.meta_net1 = nn.Linear(in_features, out_features)
        self.meta_net2 = nn.Linear(out_features, in_features * out_features)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, personal_features_src, embeddings):
        att_score = torch.mm(personal_features_src, embeddings.weight.permute(1, 0))
        attn_weights = F.softmax(att_score, dim=1)
        retrieved_values = torch.matmul(attn_weights, embeddings.weight)
        meta_feat = self.meta_net2(torch.relu(self.meta_net1(retrieved_values)))
        bz = personal_features_src.size(0)
        meta_feat = meta_feat.view(bz, self.in_features, self.out_features)
        personal_features_tgt = torch.bmm(meta_feat, personal_features_src.unsqueeze(-1)).squeeze(-1)
        return personal_features_tgt


class GroupPersonalizedTransferNet(nn.Module):
    def __init__(self, in_features, out_features, num_common_user):
        super(GroupPersonalizedTransferNet, self).__init__()
        self.external_memory = ExternalMemoryNetwork(in_features, num_common_user, out_features)
        self.meta_net1 = nn.Linear(in_features, out_features)
        self.meta_net2 = nn.Linear(out_features, in_features * out_features)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, personal_features_src, embeddings, memory_ids=None):
        group_personal_features_src = self.external_memory.forward(memory_ids, personal_features_src)

        att_score = torch.mm(personal_features_src, embeddings.weight.permute(1, 0))
        attn_weights = F.softmax(att_score, dim=1)
        retrieved_values = torch.matmul(attn_weights, embeddings.weight)

        meta_feat = self.meta_net2(torch.relu(self.meta_net1(retrieved_values)))
        bz = personal_features_src.size(0)
        meta_feat = meta_feat.view(bz, self.in_features, self.out_features)
        personal_features_tgt = torch.bmm(meta_feat, group_personal_features_src.unsqueeze(-1)).squeeze(-1)
        return personal_features_tgt


class FeatureGenerator(nn.Module):
    def __init__(self, in_features, out_features, transfer_types, num_common_user):
        super().__init__()
        if transfer_types == 'personal':
            self.transfer_net = PersonalizedTransferNet(in_features, out_features)
        elif transfer_types == 'group':
            self.transfer_net = GroupPersonalizedTransferNet(in_features, out_features, num_common_user)
        self.transfer_types = transfer_types

    def forward(self, personal_features_src, embeddings, common_user_ids):
        if self.transfer_types == 'group':
            features = self.transfer_net(personal_features_src, embeddings, common_user_ids)
        else:
            features = self.transfer_net(personal_features_src, embeddings)
        return features

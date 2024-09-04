import torch.nn as nn
import torch


class FeatureExtractor(nn.Module):
    def __init__(self, num_features, size_features):
        super().__init__()
        self.feature_map = nn.Embedding(num_features, size_features)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, ids, weights):
        embeddings = self.feature_map(ids)
        mask = (ids == 0)
        mask_weights = self.softmax(torch.where(mask, torch.ones_like(weights) * -1e9, weights))
        features = torch.bmm(mask_weights.unsqueeze(1), embeddings).squeeze(1)
        return features

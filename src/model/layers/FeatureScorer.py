import torch
import torch.nn as nn

from src.model.layers.Attention import Attention


class FeatureScorer(nn.Module):
    def __init__(self, args):
        super(FeatureScorer, self).__init__()
        self.args = args
        self.__init_rating_network(args.feature_types)

    def __init_rating_network(self, feature_types):
        if 'category' in feature_types:
            self.category_feature_rating = nn.Sequential(nn.Linear(self.args.size_features * 2, self.args.size_features),
                                                         nn.Linear(self.args.size_features, 1))
            self.category_attention = Attention(self.args.size_features)
        if 'brand' in feature_types:
            self.brand_feature_rating = nn.Sequential(nn.Linear(self.args.size_features * 2, self.args.size_features),
                                                      nn.Linear(self.args.size_features, 1))
        if 'aspect' in feature_types:
            self.aspect_attention = Attention(self.args.size_features)
        if 'avg' in feature_types:
            self.weight_linear1 = nn.Linear((len(feature_types) - 1) * self.args.size_features, self.args.size_features)
            self.weight_linear2 = nn.Linear(self.args.size_features, len(feature_types))
        else:
            self.weight_linear1 = nn.Linear(len(feature_types) * self.args.size_features, self.args.size_features)
            self.weight_linear2 = nn.Linear(self.args.size_features, len(feature_types))
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)

    def compute_category_rating(self, user2categories_feature_tgt, item2categories_feature_tgt, item2categories_ids_tgt):
        att, _ = self.category_attention(user2categories_feature_tgt, item2categories_feature_tgt, mask=(item2categories_ids_tgt == 0))
        item2categories_feature_tgt = torch.bmm(att.unsqueeze(1), item2categories_feature_tgt).squeeze(1)
        rating = self.category_feature_rating(self.dropout(torch.cat([user2categories_feature_tgt, item2categories_feature_tgt], dim=-1)))
        return rating

    def compute_brand_rating(self, user2brands_feature_tgt, item2brands_feature_tgt):
        rating = self.brand_feature_rating(self.dropout(torch.cat([user2brands_feature_tgt, item2brands_feature_tgt], dim=-1)))
        return rating


    def compute_aspect_rating(self, user2aspects_feature_tgt, item2aspects_polarity_tgt, item2aspects_feature_tgt, item2aspects_ids_tgt):
        att, _ = self.aspect_attention(user2aspects_feature_tgt, item2aspects_feature_tgt, mask=(item2aspects_ids_tgt == 0))
        rating = torch.tanh(torch.sum(att * item2aspects_polarity_tgt, dim=-1).unsqueeze(-1))
        return rating

    def compute_weight(self, unify_features):
        weight = torch.sigmoid(self.weight_linear2(self.dropout(self.weight_linear1(unify_features))))
        return weight

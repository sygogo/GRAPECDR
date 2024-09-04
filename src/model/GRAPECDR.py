import torch
import torch.nn as nn

from src.model.layers.FeatureExtractor import FeatureExtractor
from src.model.layers.FeatureGenerator import FeatureGenerator
from src.model.layers.FeatureScorer import FeatureScorer


def initialize_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)


class GRAPECDR(nn.Module):

    def __init__(self, args):
        super(GRAPECDR, self).__init__()
        self.args = args
        self.__init_feature_extractor_generator__(args.feature_types)
        self.__init_embeddings__(args.feature_types)
        self.feature_scorer = FeatureScorer(args)
        self.loss = nn.SmoothL1Loss()

    def __init_embeddings__(self, feature_types):
        if 'category' in feature_types:
            num_categories_tgt = len(self.args.category_vocab_tgt)
            self.category_embedding_tgt = nn.Embedding(num_categories_tgt, self.args.size_features)
        if 'aspect' in feature_types:
            num_aspects_tgt = len(self.args.aspect_vocab_tgt)
            self.aspect_embedding_tgt = nn.Embedding(num_aspects_tgt, self.args.size_features)
        if 'brand' in feature_types:
            num_brand_tgt = len(self.args.brand_vocab_tgt)
            self.brand_embedding_tgt = nn.Embedding(num_brand_tgt, self.args.size_features)

    def __init_feature_extractor_generator__(self, feature_types):

        num_categories_src = len(self.args.category_vocab_src)
        self.category_feature_extractor = FeatureExtractor(num_categories_src, self.args.size_features)

        num_brand_src = len(self.args.brand_vocab_src)
        self.brand_feature_extractor = FeatureExtractor(num_brand_src, self.args.size_features)

        num_aspects_src = len(self.args.aspect_vocab_src)
        self.aspect_feature_extractor = FeatureExtractor(num_aspects_src, self.args.size_features)

        if 'category' in feature_types:
            self.category_feature_generator = FeatureGenerator(self.args.size_features, self.args.size_features, self.args.transfer_types, len(self.args.common_user2id))
        if 'aspect' in feature_types:
            self.aspect_feature_generator = FeatureGenerator(self.args.size_features, self.args.size_features, self.args.transfer_types, len(self.args.common_user2id))
        if 'brand' in feature_types:
            self.brand_feature_generator = FeatureGenerator(self.args.size_features, self.args.size_features, self.args.transfer_types, len(self.args.common_user2id))

    def feature_extraction(self, batch, feature_types):
        user2categories_ids_src, user2categories_count_src, user2brands_ids_src, user2brands_count_src, user2aspects_ids_src, user2aspects_count_src = batch
        user2features_src = {}
        if 'category' in feature_types:
            user2features_src['category'] = self.category_feature_extractor(user2categories_ids_src, user2categories_count_src)
            user2features_src['category_embeddings'] = self.category_feature_extractor.feature_map
        if 'aspect' in feature_types:
            user2features_src['aspect'] = self.aspect_feature_extractor(user2aspects_ids_src, user2aspects_count_src)
            user2features_src['aspect_embeddings'] = self.aspect_feature_extractor.feature_map
        if 'brand' in feature_types:
            user2features_src['brand'] = self.brand_feature_extractor(user2brands_ids_src, user2brands_count_src)
            user2features_src['brand_embeddings'] = self.brand_feature_extractor.feature_map
        return user2features_src

    def feature_generation(self, user2features_src, common_user_id):
        user2features_tgt = {}
        if 'category' in user2features_src:
            user2features_tgt['category'] = self.category_feature_generator(user2features_src['category'], user2features_src['category_embeddings'], common_user_id)
        if 'aspect' in user2features_src:
            user2features_tgt['aspect'] = self.aspect_feature_generator(user2features_src['aspect'], user2features_src['aspect_embeddings'], common_user_id)
        if 'brand' in user2features_src:
            user2features_tgt['brand'] = self.brand_feature_generator(user2features_src['brand'], user2features_src['brand_embeddings'], common_user_id)

        return user2features_tgt

    def feature_rating(self, user2features_tgt, feature_types, item2categories_ids_tgt, item2brands_ids_tgt, item2aspects_ids_tgt, item2aspects_polarity_tgt):
        rating_list = []
        if 'brand' in feature_types:
            brand_embeddings_tgt = self.brand_embedding_tgt(item2brands_ids_tgt)
            brand_rating = self.feature_scorer.compute_brand_rating(user2features_tgt['brand'], brand_embeddings_tgt)
            rating_list.append(brand_rating)
        if 'category' in feature_types:
            item2categories_feature_tgt = self.category_embedding_tgt(item2categories_ids_tgt)
            category_rating = self.feature_scorer.compute_category_rating(user2features_tgt['category'], item2categories_feature_tgt, item2categories_ids_tgt)
            rating_list.append(category_rating)
        if 'aspect' in feature_types:
            item2aspects_feature_tgt = self.aspect_embedding_tgt(item2aspects_ids_tgt)
            aspect_rating = self.feature_scorer.compute_aspect_rating(user2features_tgt['aspect'], item2aspects_polarity_tgt, item2aspects_feature_tgt, item2aspects_ids_tgt)
            rating_list.append(aspect_rating)
        unify_features = torch.cat([v for k, v in user2features_tgt.items()], dim=-1)
        rating_list = torch.cat(rating_list, dim=-1)
        rating_weights = self.feature_scorer.compute_weight(unify_features)
        rating = torch.sum(rating_weights * rating_list, dim=-1).unsqueeze(-1)
        return rating

    def compute_rating(self, batch):
        (common_user_id, item, rating,
         user2categories_ids_src, user2categories_count_src, user2brands_ids_src, user2brands_count_src, user2aspects_ids_src, user2aspects_count_src,
         item2categories_ids_tgt, item2brands_ids_tgt, item2aspects_ids_tgt, item2aspects_polarity_tgt) = [i.cuda() for i in batch]
        user2features_src = self.feature_extraction((user2categories_ids_src, user2categories_count_src, user2brands_ids_src, user2brands_count_src, user2aspects_ids_src, user2aspects_count_src), self.args.feature_types)
        user2features_tgt = self.feature_generation(user2features_src, common_user_id)
        pred_rating = self.feature_rating(user2features_tgt, self.args.feature_types, item2categories_ids_tgt, item2brands_ids_tgt, item2aspects_ids_tgt, item2aspects_polarity_tgt)
        return pred_rating, rating

    def forward(self, batch):
        pred_rating, rating = self.compute_rating(batch)
        loss = self.compute_loss(pred_rating, rating)
        return loss

    def compute_loss(self, pred, target):
        pred_rating = pred
        rating = target
        loss = self.loss(pred_rating, rating.unsqueeze(1))
        return loss

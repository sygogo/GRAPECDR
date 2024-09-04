import gzip
import json
import os

from tqdm import tqdm

# 最小的评论字数
MIN_REVIEWS_LEN = 50
MAX_REVIEWS_LEN = 500

# 每个用户或商品的最多评论数
MAX_REVIEWS_USER_OR_ITEM = 5


class ReviewHandler(object):
    def __init__(self, args):
        super(ReviewHandler, self).__init__()
        args.src_meta_reviews_path = os.path.join(args.meta_data_path, 'reviews_{}_5.json.gz'.format(args.src_category))
        args.tgt_meta_reviews_path = os.path.join(args.meta_data_path, 'reviews_{}_5.json.gz'.format(args.tgt_category))
        args.src_meta_path = os.path.join(args.meta_data_path, 'meta_{}.json.gz'.format(args.src_category))
        args.tgt_meta_path = os.path.join(args.meta_data_path, 'meta_{}.json.gz'.format(args.tgt_category))
        self.args = args

    def get_target_item_reviews(self, idict_t, coldstart_user_set):
        """
        get item's reviews in target domain
        :param idict_t: item list in target
        :param coldstart_user_set: need to excluding coldstart user's reviews
        :return:
        """

        item2reviews = {}
        for line in tqdm(gzip.open(self.args.tgt_meta_reviews_path, 'rb'), ncols=100, desc='process item review from {}'.format(self.args.tgt_meta_reviews_path)):
            obj = json.loads(line)
            itemID = obj.get('asin')
            reviewerID = obj.get('reviewerID')
            if (itemID in idict_t) and (reviewerID not in coldstart_user_set):
                reviewText = obj.get('reviewText')
                # overall = obj.get('overall')
                if reviewText is None: continue
                len_reviews = len(reviewText.split())
                if len_reviews > MIN_REVIEWS_LEN:
                    reviewText = reviewText[:MAX_REVIEWS_LEN].lower()
                    if itemID not in item2reviews:
                        item2reviews[itemID] = {reviewerID: reviewText}
                    else:
                        if len(item2reviews[itemID]) < MAX_REVIEWS_USER_OR_ITEM:
                            obj = item2reviews[itemID]
                            obj[reviewerID] = reviewText

        return item2reviews

    def get_user_reviews(self, user_dict, meta_reviews_path):
        user2reviews = {}
        user2items = {}
        for line in tqdm(gzip.open(meta_reviews_path, 'rb'), ncols=100, desc='process user review from {}'.format(meta_reviews_path)):
            obj = json.loads(line)
            itemID = obj.get('asin')
            reviewerID = obj.get('reviewerID')
            if reviewerID not in user_dict:
                continue
            reviewText = obj.get('reviewText', '')
            overall = float(obj.get('overall'))
            if reviewerID not in user2items:
                user2items[reviewerID] = {itemID: overall}
            else:
                user2items[reviewerID][itemID] = overall
            if reviewText is None: continue
            len_reviews = len(reviewText.split())
            if len_reviews > MIN_REVIEWS_LEN:
                reviewText = reviewText[:MAX_REVIEWS_LEN].lower()
                if reviewerID not in user2reviews:
                    user2reviews[reviewerID] = {itemID: reviewText}
                else:
                    if len(user2reviews[reviewerID]) < MAX_REVIEWS_USER_OR_ITEM:
                        user2reviews[reviewerID][itemID] = reviewText
        return user2reviews, user2items

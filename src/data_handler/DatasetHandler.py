import gzip
import json

from numpy import mean
from tqdm import tqdm


class DatasetHandler(object):

    def get_user2items(self, user2items, users, idict):
        new_user2items = {}
        for user in users:
            items = user2items[user]
            items_by_ids, items_by_ratings = [i for i in zip(*[[k, v] for k, v in items.items() if k in idict])]
            new_user2items[user] = (items_by_ids, items_by_ratings)
        return new_user2items

    def get_item2ratings(self, meta_reviews_path, idict, coldstart_users):
        item2ratings = {}
        for line in tqdm(gzip.open(meta_reviews_path, 'rb'), ncols=100, desc='process user review from {}'.format(meta_reviews_path)):
            obj = json.loads(line)
            item = obj.get('asin')
            overall = float(obj.get('overall'))
            reviewerID = obj.get('reviewerID')
            if item not in idict: continue
            if reviewerID in coldstart_users: continue
            if item not in item2ratings:
                item2ratings[item] = [overall]
            else:
                item2ratings[item].append(overall)
        item2ratings = {k: mean(v) for k, v in item2ratings.items()}
        return item2ratings

    def get_item2categories_brands(self, idict, meta_path, category_vocab, brand_vocab):
        item2categories = {}
        item2brands = {}
        for line in tqdm(gzip.open(meta_path, 'rb'), ncols=100, desc='process item category and brand'):
            obj = eval(line)
            asin = obj['asin']
            if asin in idict:
                if 'category' in obj and len(obj['category']) > 0:
                    cat_list = obj['category'][1:]
                if 'brand' in obj and len(obj['brand'].split(',')) > 0:
                    brand_list = []
                    for brand in obj['brand'].split(','):
                        if len(brand) > 0:
                            brand_list.append(brand)
                item2categories[asin] = [i for i in cat_list if i in category_vocab]
                item2brands[asin] = [i for i in brand_list if i in brand_vocab]
        return item2categories, item2brands

    def get_user2categories_brands(self, user2items, users, meta_path, idict):
        item2category = {}
        item2brand = {}
        for line in tqdm(gzip.open(meta_path, 'rb'), ncols=100, desc='process item category'):
            obj = eval(line)
            asin = obj['asin']
            if asin in idict:
                if 'category' in obj and len(obj['category']) > 0:
                    cat_list = obj['category'][1:]
                else:
                    cat_list = []
                if 'brand' in obj and len(obj['brand'].split(',')) > 0:
                    brand_list = obj['brand'].split(',')
                else:
                    brand_list = []
                item2category[asin] = cat_list
                item2brand[asin] = brand_list

        user2categories, user2brands = {}, {}

        for user in users:
            items = user2items[user]
            items_by_ids, items_by_ratings = [i for i in zip(*[[k, v] for k, v in items.items()])]
            current_user2categories, current_user2brand = {}, {}
            for item in items_by_ids:
                cate_list = item2category.get(item, [])
                brand_list = item2brand.get(item, [])
                for cat in cate_list:
                    if cat in current_user2categories:
                        current_user2categories[cat] += 1
                    else:
                        current_user2categories[cat] = 1
                for brand in brand_list:
                    if len(brand) > 0:
                        if brand in current_user2brand:
                            current_user2brand[brand] += 1
                        else:
                            current_user2brand[brand] = 1

            if len(current_user2categories) > 1:
                current_user2categories_pairs = [i for i in zip(*sorted([(k, v) for k, v in current_user2categories.items() if v > 1], key=lambda x: x[1], reverse=True)[1:])]
                if len(current_user2categories_pairs) == 0:
                    current_user2categories_pairs = ([], [])

            if len(current_user2brand) > 1:
                current_user2brand_pairs = [i for i in zip(*sorted([(k, v) for k, v in current_user2brand.items() if v > 1], key=lambda x: x[1], reverse=True)[1:])]
                if len(current_user2brand_pairs) == 0:
                    current_user2brand_pairs = ([], [])

            user2categories[user] = current_user2categories_pairs
            user2brands[user] = current_user2brand_pairs

        return user2categories, user2brands

    def get_item2aspects(self, item2aspects, aspect_vocab):
        new_item2aspects = {}

        def add_aspect(aspect, polarity, aspect_vocab, current_item2aspects):
            if aspect not in aspect_vocab:
                return

            if polarity == 'positive':
                polarity_value = 1
            elif polarity == 'negative':
                polarity_value = -1
            else:
                polarity_value = 0

            if aspect not in current_item2aspects:
                current_item2aspects[aspect] = [polarity_value]
            else:
                current_item2aspects[aspect].append(polarity_value)

        for item in item2aspects:
            item_aspects = item2aspects[item]
            current_item2aspects = {}
            for review_aspect in item_aspects:
                for aspect in review_aspect:
                    if not isinstance(aspect, tuple):
                        if 'aspect' in aspect:
                            try:
                                aspect_name = aspect['aspect']
                                if aspect_name in aspect_vocab:
                                    polarity = aspect['polarity']
                                    add_aspect(aspect_name, polarity, aspect_vocab, current_item2aspects)
                            except Exception:
                                continue
                        else:  # aspect name as key
                            for key in aspect:
                                if key in aspect_vocab:
                                    polarity = aspect[key]
                                    add_aspect(key, polarity, aspect_vocab, current_item2aspects)
                    else:
                        for inner_aspect in aspect:
                            aspect_name = inner_aspect['aspect']
                            if aspect_name in aspect_vocab:
                                polarity = inner_aspect['polarity']
                                add_aspect(aspect_name, polarity, aspect_vocab, current_item2aspects)
            current_item2aspects = [(k, mean(v)) for k, v in current_item2aspects.items()]
            current_item2aspects_pairs = [i for i in zip(*current_item2aspects)]
            new_item2aspects[item] = current_item2aspects_pairs
        return new_item2aspects

    def get_user2aspects(self, user2aspects, aspect_vocab, users):

        def add_aspect(aspect, aspect_vocab, current_user2aspects):
            if aspect not in aspect_vocab:
                return
            if aspect not in current_user2aspects:
                current_user2aspects[aspect] = 1
            else:
                current_user2aspects[aspect] += 1

        aspect_vacab = {'<PAD>': 0, '<UNK>': 1}

        new_user2aspects = {}
        have_aspect_total = 0
        for user in users:
            current_user2aspects = {}
            if user in user2aspects:
                have_aspect_total += 1
                user_aspects = user2aspects[user]
                for review_aspect in user_aspects:
                    for aspect in review_aspect:
                        if not isinstance(aspect, tuple):
                            if 'aspect' in aspect:
                                aspect = aspect['aspect']
                                add_aspect(aspect, aspect_vocab, current_user2aspects)
                            else:  # aspect name as key
                                for key in aspect:
                                    add_aspect(key, aspect_vocab, current_user2aspects)
                        else:
                            for inner_aspect in aspect:
                                aspect = inner_aspect['aspect']
                                add_aspect(aspect, aspect_vocab, current_user2aspects)
                if len(current_user2aspects) > 1:
                    current_user2aspects_pair = [i for i in zip(*sorted([(k, v) for k, v in current_user2aspects.items() if v > 1], key=lambda x: x[1], reverse=True)[1:])]
                    if len(current_user2aspects_pair) == 0:
                        current_user2aspects_pair = ([], [])
            else:
                current_user2aspects_pair = ([], [])
            new_user2aspects[user] = current_user2aspects_pair

        return new_user2aspects

    def construct_aspect_vocab(self, aspects_pair):
        aspect_vacab = {'<PAD>': 0, '<UNK>': 1}
        for name, aspects in aspects_pair.items():
            aspect_count = {}
            for aspect_list in aspects:
                for aspect in aspect_list:
                    aspect_name = aspect['aspect']
                    if aspect_name not in aspect_count:
                        aspect_count[aspect_name] = 1
                    else:
                        aspect_count[aspect_name] += 1
            for aspect_name, count in aspect_count.items():
                if count > 1:
                    if aspect_name not in aspect_vacab:
                        aspect_vacab[aspect_name] = len(aspect_vacab)
        return aspect_vacab

    def construct_brand_and_category_vocab(self, meta_path, idict):
        category_vocab = {'<PAD>': 0, '<UNK>': 1}
        brand_vocab = {'<PAD>': 0, '<UNK>': 1}

        for line in tqdm(gzip.open(meta_path, 'rb'), ncols=100, desc='process item category'):
            obj = eval(line)
            asin = obj['asin']
            if asin in idict:
                if 'category' in obj and len(obj['category']) > 0:
                    cat_list = obj['category'][1:]
                else:
                    cat_list = []
                if 'brand' in obj and len(obj['brand'].split(',')) > 0:
                    brand_list = obj['brand'].split(',')
                else:
                    brand_list = []
                for cat in cat_list:
                    if cat not in category_vocab:
                        category_vocab[cat] = len(category_vocab)
                for brd in brand_list:
                    if brd not in brand_vocab:
                        brand_vocab[brd] = len(brand_vocab)
        return category_vocab, brand_vocab

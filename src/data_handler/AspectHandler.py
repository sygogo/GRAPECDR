import json

import requests
from tqdm import tqdm

from src.data_handler.online_extract_aspect_polarity import extract


class AspectHandler(object):
    def __init__(self, args):
        self.prompt = ("Now you are an aspect category and sentiment polarity extractor. "
                       "Your work is to extract aspect category and sentiment polarity pairs from the following sentences:\n {}.\n"
                       "If you could not detect any aspect category and sentiment polarity information from the provided sentences, please just return 'nothing'. "
                       "Remember the polarity should be 'positive', 'negative' and 'neutral'. Note that if the aspect category and sentiment polarity pairs exist, your answer should be a json,"
                       "such as {{'aspect':'price','polarity':'negative'}}. And please removing repeated aspect category.")
        self.args = args

    def extract_aspects(self, sentences):
        contents = extract(sentences, self.args.llm_url)
        return contents

    def process(self, object2reviews):
        object2aspects = {}
        reviews_cat = []
        reviews_len = []
        reviews_ids = []
        max = len(object2reviews)
        for obj_index, id in tqdm(enumerate(object2reviews), ncols=100):
            reviews = object2reviews[id]
            reviews_cat.extend([v for k, v in reviews.items()])
            reviews_len.append(len(reviews))
            reviews_ids.append(id)
            if len(reviews_len) == 20 or obj_index == (max - 1):
                extracted_pairs = self.extract_aspects(reviews_cat)
                result = []
                start = 0
                for count in reviews_len:
                    result.append(extracted_pairs[start:start + count])
                    start += count
                for index, obj in enumerate(result):
                    id = reviews_ids[index]
                    object2aspects[id] = obj
                reviews_cat = []
                reviews_len = []
                reviews_ids = []

        return object2aspects

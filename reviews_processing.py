import argparse
import os
import pickle
import logging
import numpy.random
from ordered_set import OrderedSet

from src.data_handler.AspectHandler import AspectHandler
from src.data_handler.ReviewHandler import ReviewHandler
from src.misc import init_logging

cwd = os.getcwd()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tgt_category', type=str, default='CDs_and_Vinyl')
    parser.add_argument('--src_category', type=str, default='Movies_and_TV')
    parser.add_argument('--raw_data_path', type=str, default='data/raw')
    parser.add_argument('--meta_data_path', type=str, default='data/meta')
    parser.add_argument('--processed_data_path', type=str, default='data/processed')
    parser.add_argument('--llm_url', type=str, default='http://192.168.60.175:8000/v1')
    parser.add_argument('--test_proportions', type=float, default=0.2)
    args = parser.parse_args()

    meta_data_path = os.path.join(cwd, args.meta_data_path)

    # processed path
    processed_data_path = os.path.join(args.processed_data_path, 'tgt_{}_src_{}'.format(args.tgt_category, args.src_category))
    processed_data_path = os.path.join(cwd, processed_data_path)
    user_proportions_data_path = os.path.join(processed_data_path, str(int(args.test_proportions * 100)))

    aspectHandler = AspectHandler(args)

    raw_data_path = os.path.join(args.raw_data_path, '{}-{}-{}.pkl'.format(args.src_category, args.tgt_category, args.test_proportions))
    raw_data_path = os.path.join(cwd, raw_data_path)

    if not os.path.exists(user_proportions_data_path):
        os.makedirs(user_proportions_data_path)
    logging = init_logging(log_file=user_proportions_data_path + '/data.log', stdout=True, loglevel=logging.WARN)

    udict, idict_s, idict_t, coldstart_user_set, common_user_set, _, _, _, train_common_s, train_common_t, coldstart_vali, coldstart_test, coldstart_s = pickle.load(open(raw_data_path, 'rb'))
    common_user_set = OrderedSet([k for k, v in udict.items() if v in common_user_set])
    coldstart_user_set = OrderedSet([k for k, v in udict.items() if v in coldstart_user_set])

    logging.warn('step1:构建商品评论数据集，target域，挑选最具代表性的评论，排除测试用户')
    reviewHandler = ReviewHandler(args)
    item2reviews_tgt = reviewHandler.get_target_item_reviews(idict_t, coldstart_user_set)
    pickle.dump(item2reviews_tgt, open(os.path.join(user_proportions_data_path, 'item2reviews_tgt.pkl'), 'wb'))

    logging.warn('step2:构建用户评论及用户商品的数据集，source域')
    user2reviews_src, user2items_src = reviewHandler.get_user_reviews(common_user_set.union(coldstart_user_set), args.src_meta_reviews_path)

    pickle.dump(user2reviews_src, open(os.path.join(user_proportions_data_path, 'user2reviews_src.pkl'), 'wb'))
    pickle.dump(user2items_src, open(os.path.join(user_proportions_data_path, 'user2items_src.pkl'), 'wb'))

    user2reviews_tgt, user2items_tgt = reviewHandler.get_user_reviews(common_user_set.union(coldstart_user_set), args.tgt_meta_reviews_path)

    pickle.dump(user2reviews_tgt, open(os.path.join(user_proportions_data_path, 'user2reviews_tgt.pkl'), 'wb'))
    pickle.dump(user2items_tgt, open(os.path.join(user_proportions_data_path, 'user2items_tgt.pkl'), 'wb'))
    #
    user2reviews_src = pickle.load(open(os.path.join(user_proportions_data_path, 'user2reviews_src.pkl'), 'rb'))

    logging.warn('step3:抽取source域用户评论aspects')
    user2aspect_src = aspectHandler.process(user2reviews_src)
    pickle.dump(user2aspect_src, open(os.path.join(user_proportions_data_path, 'user2aspect_src.pkl'), 'wb'))
    logging.warn('step4:抽取target域用户评论aspects')
    user2aspect_tgt = aspectHandler.process(user2reviews_tgt)
    pickle.dump(user2aspect_tgt, open(os.path.join(user_proportions_data_path, 'user2aspect_tgt.pkl'), 'wb'))
    logging.warn('step5:抽取target域商品评论aspects')
    item2aspects_tgt = aspectHandler.process(item2reviews_tgt)
    pickle.dump(item2aspects_tgt, open(os.path.join(user_proportions_data_path, 'item2aspects_tgt.pkl'), 'wb'))

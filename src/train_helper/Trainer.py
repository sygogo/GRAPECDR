import os
import pickle
from copy import deepcopy

import numpy as np
import sklearn.metrics
import torch
from ordered_set import OrderedSet
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import AdamW, Adam

from src.data_set.TrainDataset import TrainDataset
from src.misc import init_logging, worker_init_fn
from src.model.GRAPECDR import GRAPECDR, initialize_weights
import matplotlib.pyplot as plt


class Trainer(object):

    def __init__(self, args):
        self.args = args

        args.raw_data_path = os.path.join(args.raw_data_path, '{}-{}-{}-{}.pkl'.format(args.src_category, args.tgt_category, args.user_proportions, args.seed))
        processed_data_path = os.path.join(args.processed_data_path, 'tgt_{}_src_{}'.format(args.tgt_category, args.src_category))
        user_proportions_data_path = os.path.join(processed_data_path, str(int(args.user_proportions * 100)))
        user_proportions_data_model_path = os.path.join(user_proportions_data_path, 'models')

        self.args.user_proportions_data_model_path = user_proportions_data_model_path
        self.args.user_proportions_data_path = user_proportions_data_path

        self.train_logging = init_logging(log_file=user_proportions_data_model_path + '/training.log', stdout=True)

        self.args_info = deepcopy(args)

        # load data
        udict, idict_s, idict_t, coldstart_user_set, common_user_set, _, _, _, _, train_common_t, coldstart_vali, coldstart_test, _ = pickle.load(open(args.raw_data_path, 'rb'))
        common_user_set = OrderedSet([k for k, v in udict.items() if v in common_user_set])
        self.args.train_data = train_common_t
        self.args.valid_data = coldstart_vali
        self.args.test_data = coldstart_test
        # only contain common users
        self.args.common_user2id = {k: i for i, k in enumerate(common_user_set)}

        self.args.user_dict = udict
        self.args.idict_s = idict_s
        self.args.idict_t = idict_t
        self.args.processed_user2aspects_src = pickle.load(open(os.path.join(user_proportions_data_path, 'processed_user2aspects_src.pkl'), 'rb'))
        self.args.processed_item2aspects_tgt = pickle.load(open(os.path.join(user_proportions_data_path, 'processed_item2aspects_tgt.pkl'), 'rb'))
        self.args.aspect_vocab_src = pickle.load(open(os.path.join(user_proportions_data_path, 'aspect_vocab_src.pkl'), 'rb'))
        self.args.aspect_vocab_tgt = pickle.load(open(os.path.join(user_proportions_data_path, 'aspect_vocab_tgt.pkl'), 'rb'))
        self.args.processed_user2categories_src = pickle.load(open(os.path.join(user_proportions_data_path, 'processed_user2categories_src.pkl'), 'rb'))
        self.args.processed_user2brands_src = pickle.load(open(os.path.join(user_proportions_data_path, 'processed_user2brand_src.pkl'), 'rb'))
        self.args.category_vocab_src = pickle.load(open(os.path.join(user_proportions_data_path, 'category_vocab_src.pkl'), 'rb'))
        self.args.category_vocab_tgt = pickle.load(open(os.path.join(user_proportions_data_path, 'category_vocab_tgt.pkl'), 'rb'))
        self.args.brand_vocab_src = pickle.load(open(os.path.join(user_proportions_data_path, 'brand_vocab_src.pkl'), 'rb'))
        self.args.brand_vocab_tgt = pickle.load(open(os.path.join(user_proportions_data_path, 'brand_vocab_tgt.pkl'), 'rb'))
        self.args.processed_item2categories_tgt = pickle.load(open(os.path.join(user_proportions_data_path, 'processed_item2categories_tgt.pkl'), 'rb'))
        self.args.processed_item2brands_tgt = pickle.load(open(os.path.join(user_proportions_data_path, 'processed_item2brands_tgt.pkl'), 'rb'))

    def get_model_name(self):
        str_feature_type = '_'.join(self.args.feature_types)
        seed = str(self.args.seed)
        model_name = 'feature_type_{}_transfer_type_{}_seed_{}'.format(str_feature_type, self.args.transfer_types, seed)
        return os.path.join(self.args.user_proportions_data_model_path, model_name)

    def visualize(self):
        def show(embedding_matrix, title, color):
            # 使用 t-SNE 进行降维
            tsne = TSNE(n_components=2, verbose=1, init='pca', learning_rate='auto', perplexity=30)
            embedded_vectors = tsne.fit_transform(embedding_matrix)

            # 可视化
            plt.figure(figsize=(10, 8), dpi=300)
            plt.scatter(embedded_vectors[:, 0], embedded_vectors[:, 1], marker='.', c=color)
            # plt.title(title)
            # plt.xlabel('Dimension 1')
            # plt.ylabel('Dimension 2')
            plt.tight_layout()
            plt.savefig(os.path.join(self.args.user_proportions_data_model_path, title + '.png'))

        model = GRAPECDR(self.args).cuda()

        state_dict = torch.load(open(self.get_model_name(), 'rb'), map_location='cuda:{}'.format(self.args.local_rank))
        model.load_state_dict(state_dict)
        embedding_matrix = model.category_feature_generator.transfer_net.external_memory.values.data.detach().cpu().numpy()
        show(embedding_matrix, 'Category Memory Network'.format(self.args.src_category, self.args.tgt_category), color='#EA9823')
        embedding_matrix = model.brand_feature_generator.transfer_net.external_memory.values.data.detach().cpu().numpy()
        show(embedding_matrix, 'Brand Memory Network'.format(self.args.src_category, self.args.tgt_category), color='#03BE2B')
        embedding_matrix = model.aspect_feature_generator.transfer_net.external_memory.values.data.detach().cpu().numpy()
        show(embedding_matrix, 'Aspect Memory Network'.format(self.args.src_category, self.args.tgt_category), color='#02B3FF')

    def eval(self, model=None, test_loader=None):
        if model is None:
            model = GRAPECDR(self.args).cuda()
            state_dict = torch.load(open(self.get_model_name(), 'rb'), map_location='cuda:{}'.format(self.args.local_rank))
            model.load_state_dict(state_dict)
            test_set = TrainDataset(self.args, mode='test')
            test_loader = DataLoader(test_set, batch_size=self.args.batch_size * 5, collate_fn=test_set.collate_fn, shuffle=True)
        pred_rating_list, rating_list, loss_list = [], [], []
        model.eval()
        with torch.no_grad():
            for test_batch_data in test_loader:
                pred_rating, rating = model.compute_rating(test_batch_data)
                loss = model.compute_loss(pred_rating, rating)
                loss_list.append(loss.item())
                pred_rating_list.extend(pred_rating.squeeze(-1).tolist())
                rating_list.extend(rating.squeeze(-1).tolist())
        MAE = sklearn.metrics.mean_absolute_error(rating_list, pred_rating_list)
        RMSE = np.sqrt(sklearn.metrics.mean_squared_error(rating_list, pred_rating_list))
        valid_loss = np.mean(loss_list)
        model.train()
        # if is_test:
        #     self.train_logging.info('***************************************************************')
        #     self.train_logging.info(self.args_info)
        #     self.train_logging.info('MAE:{},RMSE:{}'.format(MAE, RMSE))
        return MAE, RMSE, valid_loss

    def train(self):
        self.train_logging.info('***************************************************************')
        self.train_logging.info(self.args_info)
        model = GRAPECDR(self.args).cuda()
        initialize_weights(model)
        train_set = TrainDataset(self.args, mode='train')
        optimizer = Adam(params=model.parameters(), lr=self.args.rate_learning)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(patience=1, mode='min', optimizer=optimizer)
        train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True, collate_fn=train_set.collate_fn, num_workers=self.args.num_workers, worker_init_fn=worker_init_fn)
        valid_set = TrainDataset(self.args, mode='validate')
        test_loader = DataLoader(valid_set, batch_size=1000, collate_fn=valid_set.collate_fn, shuffle=True)
        best_valid_loss = 1e5
        early_stop_tolerance = 0
        for epoch in range(self.args.num_epoch):
            epoch_iterator = tqdm(train_loader, ncols=100)
            loss_total = 0
            for batch_data in epoch_iterator:
                optimizer.zero_grad()
                loss = model(batch_data)
                loss_total += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                loss.backward()
                optimizer.step()
            model.eval()
            MAE, RMSE, valid_loss = self.eval(model, test_loader)
            model.train()
            scheduler.step(valid_loss)
            if best_valid_loss > valid_loss:
                torch.save(model.state_dict(), open(self.get_model_name(), 'wb'))
                early_stop_tolerance = 0
                best_valid_loss = valid_loss
            else:
                early_stop_tolerance += 1
            if early_stop_tolerance == 5:
                break
            self.train_logging.info('Epoch:{},valid_loss:{},best_valid_loss:{},lr:{}'.format(epoch, valid_loss, best_valid_loss, optimizer.param_groups[0]['lr']))

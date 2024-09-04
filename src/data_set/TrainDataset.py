import torch
from torch.nn.utils.rnn import pad_sequence


def padding_sequence(sequence):
    max_length = max(len(seq) for seq in sequence)

    # 对序列进行填充
    padded_sequences = []
    for seq in sequence:
        seq = list(seq)
        padded_seq = seq + [0] * (max_length - len(seq))
        padded_sequences.append(padded_seq)
    return padded_sequences


class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, args, mode='train'):
        super().__init__()
        self.args = args
        self.mode = mode
        self.data_preprocess()
        if mode == 'train':
            self.data = self.args.train_data
        elif mode == 'validate':
            self.data = self.args.valid_data
        else:
            self.data = self.args.test_data
        self.cache = {}

    def data_preprocess(self):
        self.args.udict_reverse = {v: k for k, v in self.args.user_dict.items()}
        self.args.idict_t_reverse = {v: k for k, v in self.args.idict_t.items()}
        self.args.idict_s_reverse = {v: k for k, v in self.args.idict_s.items()}

    def collate_fn(self, batches):
        (common_user, item, rating,
         user2categories_ids_src, user2categories_count_src, user2brands_ids_src, user2brands_count_src, user2aspects_ids_src, user2aspects_count_src,
         item2categories_ids_tgt, item2brands_ids_tgt, item2aspects_ids_tgt, item2aspects_polarity_tgt) = [i for i in zip(*batches)]

        user2categories_ids_src = padding_sequence(user2categories_ids_src)
        user2categories_count_src = padding_sequence(user2categories_count_src)
        user2brands_ids_src = padding_sequence(user2brands_ids_src)
        user2brands_count_src = padding_sequence(user2brands_count_src)
        user2aspects_ids_src = padding_sequence(user2aspects_ids_src)
        user2aspects_count_src = padding_sequence(user2aspects_count_src)

        item2categories_ids_tgt = padding_sequence(item2categories_ids_tgt)
        item2brands_ids_tgt = padding_sequence(item2brands_ids_tgt)
        item2aspects_ids_tgt = padding_sequence(item2aspects_ids_tgt)
        item2aspects_polarity_tgt = padding_sequence(item2aspects_polarity_tgt)

        # convert to tensor
        common_user = torch.tensor(common_user, dtype=torch.long)
        item = torch.tensor(item, dtype=torch.int)
        rating = torch.tensor(rating, dtype=torch.float)

        user2categories_ids_src = torch.tensor(user2categories_ids_src, dtype=torch.int)
        user2categories_count_src = torch.tensor(user2categories_count_src, dtype=torch.float)
        user2brands_ids_src = torch.tensor(user2brands_ids_src, dtype=torch.int)
        user2brands_count_src = torch.tensor(user2brands_count_src, dtype=torch.float)
        user2aspects_ids_src = torch.tensor(user2aspects_ids_src, dtype=torch.int)
        user2aspects_count_src = torch.tensor(user2aspects_count_src, dtype=torch.float)

        item2categories_ids_tgt = torch.tensor(item2categories_ids_tgt, dtype=torch.int)
        item2brands_ids_tgt = torch.tensor(item2brands_ids_tgt, dtype=torch.int)[:, 0]
        item2aspects_ids_tgt = torch.tensor(item2aspects_ids_tgt, dtype=torch.int)
        item2aspects_polarity_tgt = torch.tensor(item2aspects_polarity_tgt, dtype=torch.float)

        return (common_user, item, rating,
                user2categories_ids_src, user2categories_count_src, user2brands_ids_src, user2brands_count_src, user2aspects_ids_src, user2aspects_count_src,
                item2categories_ids_tgt, item2brands_ids_tgt, item2aspects_ids_tgt, item2aspects_polarity_tgt)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if index not in self.cache:
            user, item, rating = self.data[index]
            user_id = self.args.udict_reverse[user]
            item_id = self.args.idict_t_reverse[item]
            # 将用户编号转换为common_user2id,只有训练集才有。
            if self.mode == 'train':
                common_user = self.args.common_user2id[user_id]
            else:
                common_user = 0
            # user in source domain
            user2categories_src, user2categories_count_src = self.args.processed_user2categories_src[user_id]
            user2categories_ids_src = [self.args.category_vocab_src[i] for i in user2categories_src]
            user2brands_src, user2brands_count_src = self.args.processed_user2brands_src[user_id]
            user2brands_ids_src = [self.args.brand_vocab_src[i] for i in user2brands_src]
            user2aspects_src, user2aspects_count_src = self.args.processed_user2aspects_src[user_id]
            user2aspects_ids_src = [self.args.aspect_vocab_src[i] for i in user2aspects_src]

            # item in target domain
            item2categories_tgt = self.args.processed_item2categories_tgt.get(item_id, [])
            item2categories_ids_tgt = [self.args.category_vocab_tgt[i] for i in item2categories_tgt]
            item2brands_tgt = self.args.processed_item2brands_tgt.get(item_id, [])
            item2brands_ids_tgt = [self.args.brand_vocab_tgt[i] for i in item2brands_tgt]
            item_aspects = self.args.processed_item2aspects_tgt.get(item_id, [[], []])
            if len(item_aspects) == 0:
                item_aspects = [[], []]
            item2aspects_tgt, item2aspects_polarity_tgt = item_aspects
            item2aspects_ids_tgt = [self.args.aspect_vocab_tgt[i] for i in item2aspects_tgt]

            self.cache[index] = (common_user, item, rating,
                                 user2categories_ids_src, user2categories_count_src, user2brands_ids_src, user2brands_count_src, user2aspects_ids_src, user2aspects_count_src,
                                 item2categories_ids_tgt, item2brands_ids_tgt, item2aspects_ids_tgt, item2aspects_polarity_tgt)

        return self.cache[index]

from collections import defaultdict
import random
import tempfile
from recommenders.recommender import Recommender
from utils.item_id import ItemId
from transformers import BertConfig, BertForMaskedLM
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm
import mmh3


class MaskingCollator(object):
    def __init__(self, user_actions, sequence_length, val_users, masking_prob, pad_id, mask_id, ignore_val=-100,
                 mode='train', border_timestamp=None, val_ndcg_at=None, n_items=None) -> None:
        self.val_users = val_users
        self.sequence_length = sequence_length
        self.user_actions = user_actions
        self.masking_prob = masking_prob
        self.pad_id = pad_id
        self.ignore_val = ignore_val
        self.mask_id = mask_id
        self.mode = mode
        self.is_validation = mode == 'val'
        self.is_train = mode == 'train'
        self.is_test = mode == 'test'
        if border_timestamp is not None:
            self.border_timestamp = border_timestamp
        if val_ndcg_at is not None:
            self.val_ndcg_at = val_ndcg_at
        if n_items is not None:
            self.n_items = n_items

    def __call__(self, batch):
        seqs = []
        labels_all = []
        attns = []

        val_ratings_all = []
        v = 0
        val_not_null_seqs_after_ts_idxs = []

        for user_id in batch:
            if self.is_train:
                if user_id not in self.val_users:
                    seq = [x[1] for x in self.user_actions[user_id] if
                           x[2] > 0]  # sequence of the item id (x[1]) in the positive interactions of the user
                else:  # if we are using the global timestamp split, the items after the border timestamp are removed (they are used for the evaluation)
                    seq = [x[1] for x in self.user_actions[user_id] if (x[2] > 0 and x[0] < self.border_timestamp)][
                          :-1]  # the last one is one to evaluate the validation loss
                if len(seq) > self.sequence_length:
                    borderline = random.randint(self.sequence_length, len(seq))
                    seq = seq[borderline - self.sequence_length:borderline]
                seq = torch.tensor(seq, dtype=torch.long, requires_grad=False)

                num_masked_items = max(1, int(len(seq) * self.masking_prob)) - 1
                masked_positions = torch.randperm(len(seq))[:num_masked_items]
                last_position = torch.tensor([len(seq) - 1]).int()
                masked_positions = torch.cat((masked_positions, last_position))

                masked_mask = torch.zeros(len(seq), dtype=torch.long, requires_grad=False)
                masked_mask[
                    masked_positions] = 1  # the masked mask is set to 1 in the masked positions and to 0 in all the other ones
                labels = seq.clone() * masked_mask + self.ignore_val * (
                            1 - masked_mask)  # the label is set to the right one if the input token is a mask and to a ignore value otherwise
                seq[masked_positions] = self.mask_id  # the mask id is put in the masked positions of the sequence
                if (len(seq) < self.sequence_length):
                    ignore_pad = torch.tensor([self.ignore_val] * (self.sequence_length - len(seq)),
                                              requires_grad=False)
                    labels = torch.cat([ignore_pad, labels], dim=0)  # ignore values are added on the left of the labels

                labels_all.append(labels)

            elif self.is_validation:
                seq = [x[1] for x in self.user_actions[user_id] if (x[2] > 0 and x[0] < self.border_timestamp)]
                seq.append(
                    self.mask_id)  # if the collator is the validation one, the mask id is added to the sequence (is used to for the ncdg evaluation)
                if len(seq) > self.sequence_length:
                    seq = seq[
                          -self.sequence_length:]  # if the sequence is longer than the max sequence length, the firt exceeding items are removed from the sequence
                seq = torch.tensor(seq, dtype=torch.long, requires_grad=False)

                label = seq[-2].clone()
                masked_positions = torch.tensor([len(seq) - 2],
                                                requires_grad=False)  # it is masked the second to last item, that corresponds to the last one in the sequence before the border timestamp
                seq[masked_positions] = self.mask_id

                seq_after_ts = torch.tensor(
                    [(x[1], x[2]) for x in self.user_actions[user_id] if x[0] >= self.border_timestamp],
                    dtype=torch.long, requires_grad=False)
                if (seq_after_ts.shape[0]) > 0:
                    val_not_null_seqs_after_ts_idxs.append(v)
                v += 1

                ratings = torch.zeros(self.n_items, dtype=torch.long, requires_grad=False)
                if seq_after_ts.shape[0] != 0:
                    track_ids = seq_after_ts[:, 0]
                    track_ratings = seq_after_ts[:, 1]
                    likes_dislikes_mask = (track_ratings == -2) | (track_ratings == 2)
                    plays_skips_mask = ~likes_dislikes_mask
                    ratings.index_put_((track_ids[likes_dislikes_mask],), track_ratings[
                        likes_dislikes_mask])  # update ratings for likes and dislikes directly
                    current_ratings = ratings[
                        track_ids[plays_skips_mask]]  # get current ratings for tracks played or skipped
                    update_mask = (current_ratings != -2) & (
                                current_ratings != 2)  # mask for tracks that are not already liked or disliked
                    ratings.index_put_((track_ids[plays_skips_mask][update_mask],), track_ratings[plays_skips_mask][
                        update_mask])  # update ratings for plays and skips where applicable

                labels_all.append(label)
                val_ratings_all.append(ratings)
            elif self.is_test:
                seq = [x[1] for x in self.user_actions[user_id] if x[2] > 0]
                seq.append(self.mask_id)  # if the collator is the test one, the mask id is added to the sequence
                if len(seq) > self.sequence_length:
                    seq = seq[
                          -self.sequence_length:]  # if the sequence is longer than the max sequence length, the firt exceeding items are removed from the sequence
                seq = torch.tensor(seq, dtype=torch.long, requires_grad=False)

            attn = torch.ones_like(seq, requires_grad=False)  # initialization of the attention mask

            if (len(seq) < self.sequence_length):  # if the sequence is shorter than the max sequence length
                pad = torch.tensor([self.pad_id] * (self.sequence_length - len(seq)), requires_grad=False)
                zero_pad = torch.zeros_like(pad, requires_grad=False)
                seq = torch.cat([pad, seq], dim=0)  # pad ids are added on the left of the sequence
                attn = torch.cat([zero_pad, attn], dim=0)  # zero values are added on the left of the attention mask

            seqs.append(seq)
            attns.append(attn)

        if self.is_test:
            batch = {"seq": torch.stack(seqs), "attn": torch.stack(attns)}
        elif self.is_train:
            batch = {"seq": torch.stack(seqs), "attn": torch.stack(attns), "labels": torch.stack(labels_all)}
        elif self.is_validation:
            batch = {"seq": torch.stack(seqs), "attn": torch.stack(attns), "labels": torch.stack(labels_all),
                     "ratings": torch.stack(val_ratings_all),
                     "not_null_seqs_after_ts_idxs": torch.tensor(val_not_null_seqs_after_ts_idxs, requires_grad=False)}
        return batch


class BERT4RecPytorchRecommender(Recommender):
    def __init__(self,
                 embedding_size=64,
                 attention_probs_dropout_prob=0.2,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.2,
                 initializer_range=0.02,
                 num_attention_heads=2,
                 num_hidden_layers=3,
                 type_vocab_size=2,
                 masking_prob=0.2,
                 train_batch_size=128,
                 val_batch_size=128,
                 max_steps_per_epoch=128,
                 sequence_length=200,
                 max_epochs=10,
                 val_ndcg_at=10,
                 early_stop_epochs=200,
                 lr=0.001,
                 wd=0
                 ):
        super().__init__()
        self.users = ItemId()
        self.items = ItemId()
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_act = hidden_act
        self.embedding_size = embedding_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.type_vocab_size = type_vocab_size
        self.masking_prob = masking_prob
        self.sequence_length = sequence_length
        self.user_actions = defaultdict(list)
        self.train_batch_size = train_batch_size
        self.flags = {}
        self.max_steps_per_epoch = max_steps_per_epoch
        self.lr = lr
        self.wd = wd
        self.val_batch_size = val_batch_size
        self.val_ndcg_at = val_ndcg_at
        self.max_epochs = max_epochs
        self.early_stop_epochs = early_stop_epochs
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.positions = torch.arange(1, self.val_ndcg_at + 1, dtype=torch.float)
        self.ndcg_discounts = torch.unsqueeze(1 / torch.log2(self.positions + 1), 0)

    def get_tensorboard_dir(self):
        if self.tensorboard_dir is None:
            self.tensorboard_dir = tempfile.mkdtemp()
        return self.tensorboard_dir

    def add_action(self, action):
        user_id_internal = self.users.get_id(action.user_id)
        action_id_internal = self.items.get_id(action.item_id)
        self.user_actions[user_id_internal].append((action.timestamp, action_id_internal, action.rating))

    def set_val_users(self, val_users):
        return super().set_val_users(val_users)

    def sort_actions(self):
        for user_id in self.user_actions:
            self.user_actions[user_id].sort(key=lambda x: (x[0], mmh3.hash(f"{x[1]}_{user_id}")))

    def rebuild_model(self, val_border_timestamp=None):
        self.sort_actions()
        self.pad_item_id = self.items.size()  # the padding id is set to size of item list
        self.mask_item_id = self.items.size() + 1  # the mask id is set to size of item list+1
        self.bert_config = BertConfig(
            vocab_size=self.items.size() + 3,
            hidden_size=self.embedding_size,
            intermediate_size=self.embedding_size * 4,
            max_position_embeddings=self.sequence_length,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            initializer_range=self.initializer_range,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            type_vocab_size=self.type_vocab_size,
            pad_token_id=self.pad_item_id
        )
        self.val_border_timestamp = val_border_timestamp
        self.bert = BertForMaskedLM(self.bert_config).to(self.device)

        tensorboard_dir = self.get_tensorboard_dir()
        tb_writer = SummaryWriter(tensorboard_dir)

        all_users = list(self.user_actions.keys())  # list of all the user ids
        self.val_users_internal = set()
        for user in self.val_users:
            self.val_users_internal.add(self.users.get_id(user))

        train_collator = MaskingCollator(self.user_actions, self.sequence_length, self.val_users_internal,
                                         self.masking_prob, self.pad_item_id, self.mask_item_id, mode='train',
                                         border_timestamp=val_border_timestamp, val_ndcg_at=self.val_ndcg_at,
                                         n_items=self.items.size())
        train_loader = DataLoader(all_users, batch_size=self.train_batch_size, collate_fn=train_collator, shuffle=True)

        batches_per_epoch = min(self.max_steps_per_epoch, len(all_users) // self.train_batch_size)
        optimiser = torch.optim.Adam(self.bert.parameters(), lr=self.lr, weight_decay=self.wd)
        # optimiser = torch.optim.AdamW(self.bert.parameters(), lr=self.lr, weight_decay=self.wd)
        best_ndcg = float('-inf')
        best_epoch = -1
        best_model_weights = None
        epochs_since_best = 0

        for epoch in range(self.max_epochs):
            print(f"Epoch {epoch}")
            self.bert.train()
            pbar = tqdm(total=batches_per_epoch, ascii=True, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', position=0,
                        leave=True, ncols=70)
            epoch_loss_sum = 0
            for step, batch in enumerate(train_loader):
                optimiser.zero_grad()
                if step >= batches_per_epoch:
                    break

                seq = batch['seq'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attn'].to(self.device)

                loss = self.bert.forward(input_ids=seq, attention_mask=attention_mask, labels=labels).loss
                loss.backward()
                optimiser.step()
                epoch_loss_sum += loss.item()
                epoch_loss_mean = epoch_loss_sum / (step + 1)
                pbar.update()
                pbar.set_description(f"Loss: {epoch_loss_mean:.4f}")
            epoch_result = self.validate()
            if epoch_result['ndcg'] > best_ndcg:
                best_ndcg = epoch_result['ndcg']
                best_epoch = epoch
                best_model_weights = self.bert.state_dict().copy()
                epochs_since_best = 0
            else:
                epochs_since_best += 1  # the number of epochs since the best one is increased
            epochs_to_early_stop = self.early_stop_epochs - epochs_since_best
            print(f"Train loss: {epoch_loss_mean:.4f}")
            tb_writer.add_scalar("loss/train", epoch_loss_mean, epoch)
            print(f"Val loss: {epoch_result['loss']:.4f}")
            tb_writer.add_scalar("loss/val", epoch_result['loss'], epoch)
            print(f"Val NDCG@{self.val_ndcg_at}: {epoch_result['ndcg']:.4f}")
            tb_writer.add_scalar(f"ndcg@10/val", epoch_result['ndcg'], epoch)
            print(f"Best NDCG@{self.val_ndcg_at}: {best_ndcg:.4f} at epoch {best_epoch}")
            print(f"Epochs since best: {epochs_since_best}")
            steps_to_early_stop = self.max_steps_per_epoch * epochs_to_early_stop
            tb_writer.flush()
            if (steps_to_early_stop <= 0):
                print(f"Early stopping at epoch {epoch}")
                pbar.close()
                break
            pbar.close()
        print("Restoring best model from epoch", best_epoch)
        self.bert.load_state_dict(best_model_weights)
        self.test_collator = MaskingCollator(self.user_actions, self.sequence_length, self.val_users_internal,
                                             self.masking_prob, self.pad_item_id, self.mask_item_id, mode='test',
                                             n_items=self.items.size())

    def validate(self):
        self.bert.eval()
        ndcgs = []
        losses = []
        val_collator = MaskingCollator(self.user_actions, self.sequence_length, self.val_users_internal,
                                       self.masking_prob, self.pad_item_id, self.mask_item_id, mode='val',
                                       border_timestamp=self.val_border_timestamp, val_ndcg_at=self.val_ndcg_at,
                                       n_items=self.items.size())
        val_users_internal = list(self.val_users_internal)
        val_loader = DataLoader(val_users_internal, batch_size=self.val_batch_size, collate_fn=val_collator,
                                shuffle=False)
        for batch in val_loader:  # are passed only the validation users to the DataLoader
            recommendations = self.recommend_impl(batch, self.val_ndcg_at, 'val')
            items_for_ndcg = recommendations['items_for_ndcg']
            ratings = recommendations['ratings']

            true_scores = torch.gather(ratings, 1, items_for_ndcg)
            true_scores[
                true_scores < 0] = 0  # assign as rating 0 if the action is a skip (-1) or a dislike (-2) to work only with positive values in ndcg evaluation
            dcg = torch.sum(true_scores * self.ndcg_discounts, 1)

            best_ratings = torch.topk(ratings, self.val_ndcg_at, dim=1).values
            best_ratings[best_ratings < 0] = 0
            idcg = torch.sum(best_ratings * self.ndcg_discounts, 1)

            rec_ndcg = torch.nan_to_num(torch.div(dcg,
                                                  idcg))  # the nan values are the ones obtained when idcg is 0 (there are only dislikes and/or skips in the ratings)
            ndcgs.append(rec_ndcg)

            losses.append(-recommendations[
                'gt_logprobs'])  # the loss for each user recommendations is: - log probability(recommended_item = ground_truth_item)
        val_result = {
            "ndcg": torch.cat(ndcgs).mean().item(),
            "loss": torch.cat(losses).mean().item()
        }
        return val_result

    def recommend_impl(self, batch, limit, mode):
        with torch.no_grad():
            if mode == 'val':
                seq = batch['seq'].to(self.device)
                attn = batch['attn'].to(self.device)
                labels = batch['labels'].to(self.device)
                ratings = batch['ratings']
                retain_idxs = batch['not_null_seqs_after_ts_idxs'].tolist()

                ratings = ratings[retain_idxs]  # select only the ones having at least one explicit rating (!=0)

                logits = self.bert.forward(input_ids=seq, attention_mask=attn).logits
                logits_for_loss = logits[:, -2,
                                  :]  # to take the second to last one in the sequence (it is the one before border timestamp that we need for loss evaluation)
                logits_for_ndcg = logits[retain_idxs, -1,
                                  :]  # to take the last one in the sequence (the one we need for ndcg evaluation) for the sequences having at least one rating after the ts
                del logits

                logits_for_loss[:, self.items.size():] = float(
                    '-inf')  # the last three logits (self.items.size():) corresponding to the special tokens (pad, mask, ignore) are set to -inf
                log_probs = torch.nn.functional.log_softmax(logits_for_loss, dim=1)
                gt_logprobs = log_probs[range(len(seq)), labels]

                logits_for_ndcg[:, self.items.size():] = float('-inf')
                log_probs_for_ndcg = torch.nn.functional.log_softmax(logits_for_ndcg, dim=1)
                top_k = torch.topk(log_probs_for_ndcg, limit, dim=1)

                result = {
                    'gt_logprobs': gt_logprobs,
                    'items_for_ndcg': top_k.indices.cpu(),
                    'ratings': ratings
                }
            elif mode == 'test':
                seq = batch['seq'].to(self.device)
                attn = batch['attn'].to(self.device)

                logits = self.bert.forward(input_ids=seq, attention_mask=attn).logits[:, -1,
                         :]  # last one in the sequence
                logits[:, self.items.size():] = float(
                    '-inf')  # the last three logits (self.items.size():) corresponding to the special tokens (pad, mask, ignore) are set to -inf
                log_probs = torch.nn.functional.log_softmax(logits, dim=1)
                top_k = torch.topk(log_probs, limit,
                                   dim=1)  # the top k items are selected based on the log softmax of the logits
                result = {
                    'items': top_k.indices,
                    'scores': top_k.values
                }
            else:
                raise ValueError(f"Unknown mode {mode}")
            return result

    def recommend(self, user_id, limit, features=None):
        requests = [(user_id, features)]
        result = self.recommend_batch(requests, limit)
        return result[0]

    def recommend_batch(self, recommendation_requests, limit):
        self.bert.eval()
        user_ids = [x[0] for x in recommendation_requests]  # select the user ids in the recommendation requests
        internal_user_ids = [self.users.get_id(user_id) for user_id in user_ids]
        result = []
        test_loader = DataLoader(internal_user_ids, batch_size=self.val_batch_size, collate_fn=self.test_collator,
                                 shuffle=False)
        for users_batch in test_loader:
            recommendations = self.recommend_impl(users_batch, limit, 'test')
            rec_items = recommendations['items'].cpu().numpy()
            rec_scores = recommendations['scores'].cpu().numpy()
            for idx in range(len(users_batch['seq'])):
                items = [self.items.reverse_id(x.item()) for x in rec_items[idx]]
                scores = rec_scores[idx]
                user_result = list(zip(items, scores))
                result.append(user_result)
        return result

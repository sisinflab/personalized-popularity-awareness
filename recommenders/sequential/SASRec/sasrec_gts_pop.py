from collections import defaultdict
import tempfile
from recommenders.recommender import Recommender
from utils.item_id import ItemId
from recommenders.sequential.SASRec.sasrec import SASRec, SASRecConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm
import mmh3
import random


class MaskingCollator(object):
    def __init__(self, user_actions, sequence_length, val_users, pad_id, mode='train', border_timestamp=None,
                 val_ndcg_at=None, n_items=None) -> None:
        self.val_users = val_users
        self.sequence_length = sequence_length
        self.user_actions = user_actions
        self.pad_id = pad_id
        self.mode = mode
        self.ignore_val = -100
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

        val_ratings_all = []
        v = 0
        val_not_null_seqs_after_ts_idxs = []

        for user_id in batch:
            if self.is_train:
                if user_id not in self.val_users:
                    seq = [x[1] for x in self.user_actions[user_id] if x[
                        2] > 0]  # sequence of the item id (x[1]) in the actions of the user corresponding to play and like events
                else:  # if we are using the global timestamp split, the items after the border timestamp are removed (they are used for the evaluation)
                    seq = [x[1] for x in self.user_actions[user_id] if (x[2] > 0 and x[0] < self.border_timestamp)][
                          :-1]  # the last one is one to evaluate the validation loss
                if len(seq) > self.sequence_length:
                    borderline = random.randint(self.sequence_length, len(seq))
                    seq = seq[borderline - self.sequence_length:borderline]
                seq = torch.tensor(seq, dtype=torch.long, requires_grad=False)
                labels = seq.clone()

                if (len(seq) < self.sequence_length):
                    pad = torch.tensor([self.ignore_val] * (self.sequence_length - len(seq)), requires_grad=False)
                    labels = torch.cat([pad, labels], dim=0)  # pad values are added on the left of the labels
                    # the loss is only computed for labels in `[0, ..., config.vocab_size]`
                labels_all.append(labels[1:])  # the labels **are shifted**

            elif self.is_validation:
                seq = [x[1] for x in self.user_actions[user_id] if (x[2] > 0 and x[0] < self.border_timestamp)]
                if len(seq) > self.sequence_length:
                    seq = seq[
                          -self.sequence_length:]  # if the sequence is longer than the max sequence length, the firt exceeding items are removed from the sequence
                seq = torch.tensor(seq, dtype=torch.long, requires_grad=False)
                label = seq[-1].clone()

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
                if len(seq) > self.sequence_length:
                    seq = seq[
                          -self.sequence_length:]  # if the sequence is longer than the max sequence length, the firt exceeding items are removed from the sequence
                seq = torch.tensor(seq, dtype=torch.long, requires_grad=False)

            if (len(seq) < self.sequence_length):  # if the sequence is shorter than the max sequence length
                pad = torch.tensor([self.pad_id] * (self.sequence_length - len(seq)), requires_grad=False)
                seq = torch.cat([pad, seq], dim=0)  # pad ids are added on the left of the sequence

            seqs.append(seq)

        if self.is_test:
            batch = {"seq": torch.stack(seqs)}
        elif self.is_train:
            negatives = torch.randint(low=0, high=self.pad_id, size=(len(batch), self.sequence_length - 1),
                                      requires_grad=False)
            batch = {"seq": torch.stack(seqs), "labels": torch.stack(labels_all), "negatives": negatives}
        elif self.is_validation:
            negatives = torch.randint(low=0, high=self.pad_id, size=(len(batch),), requires_grad=False)
            batch = {"seq": torch.stack(seqs), "labels": torch.stack(labels_all), "negatives": negatives,
                     "ratings": torch.stack(val_ratings_all),
                     "not_null_seqs_after_ts_idxs": torch.tensor(val_not_null_seqs_after_ts_idxs, requires_grad=False)}
        return batch


class SASRecRecommender(Recommender):
    def __init__(self,
                 sequence_length=200,
                 embedding_size=256,
                 train_batch_size=128,
                 val_batch_size=128,
                 num_heads=4,
                 num_blocks=3,
                 dropout_rate=0.2,
                 max_epochs=10000,
                 max_steps_per_epoch=128,
                 early_stop_epochs=200,
                 val_ndcg_at=10,
                 lr=0.001,
                 wd=0,
                 eps=0.01
                 ):
        super().__init__()
        self.users = ItemId()
        self.items = ItemId()
        self.sequence_length = sequence_length
        self.user_actions = defaultdict(list)
        self.train_batch_size = train_batch_size
        self.lr = lr
        self.wd = wd
        self.max_epochs = max_epochs
        self.embedding_size = embedding_size
        self.val_batch_size = val_batch_size
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.max_steps_per_epoch = max_steps_per_epoch
        self.early_stop_epochs = early_stop_epochs
        self.val_ndcg_at = val_ndcg_at
        self.eps=eps

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

    def get_popularity_logits(self, seq):
        with torch.no_grad():
            item_count_matrix = torch.cumsum(torch.nn.functional.one_hot(seq, num_classes=self.items.size() + 2),
                                             dim=1) + self.eps
            item_count_matrix[:, :, self.items.size():] = self.eps

            probs_matrix = torch.div(item_count_matrix, torch.max(item_count_matrix, dim=-1).values.unsqueeze(-1))
            del item_count_matrix

            norm_probs_matrix = torch.div(probs_matrix,
                                          torch.sum(probs_matrix[:, :, :self.items.size()], dim=-1).unsqueeze(-1))
            del probs_matrix

            return -(torch.log(1 - norm_probs_matrix) - torch.log(norm_probs_matrix))

    def rebuild_model(self, val_border_timestamp=None):
        self.sort_actions()
        self.pad_item_id = self.items.size()  # the padding id is set to size of item list
        self.sasrec_config = SASRecConfig(
            num_items=self.items.size(),
            sequence_length=self.sequence_length,
            embedding_dim=self.embedding_size,
            num_heads=self.num_heads,
            num_blocks=self.num_blocks,
            dropout_rate=self.dropout_rate
        )
        self.val_border_timestamp = val_border_timestamp
        self.sasrec = SASRec(self.sasrec_config).to(self.device)

        tensorboard_dir = self.get_tensorboard_dir()
        tb_writer = SummaryWriter(tensorboard_dir)

        all_users = list(self.user_actions.keys())  # list of all the user ids
        self.val_users_internal = set()
        for user in self.val_users:
            self.val_users_internal.add(self.users.get_id(user))
        train_collator = MaskingCollator(self.user_actions, self.sequence_length, self.val_users_internal,
                                         self.pad_item_id, mode='train', border_timestamp=val_border_timestamp,
                                         val_ndcg_at=self.val_ndcg_at, n_items=self.items.size())
        train_loader = DataLoader(all_users, batch_size=self.train_batch_size, collate_fn=train_collator, shuffle=True)

        batches_per_epoch = min(self.max_steps_per_epoch, len(all_users) // self.train_batch_size)
        optimiser = torch.optim.Adam(self.sasrec.parameters(), lr=self.lr, weight_decay=self.wd)
        # optimiser = torch.optim.AdamW(self.sasrec.parameters(), lr=self.lr, weight_decay=self.wd)
        best_ndcg = float('-inf')
        best_epoch = -1
        best_model_weights = None
        epochs_since_best = 0

        for epoch in range(self.max_epochs):
            print(f"Epoch {epoch}")
            self.sasrec.train()
            pbar = tqdm(total=batches_per_epoch, ascii=True, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', position=0,
                        leave=True, ncols=70)
            epoch_loss_sum = 0
            for step, batch in enumerate(train_loader):
                optimiser.zero_grad()
                if step >= batches_per_epoch:
                    break
                seq = batch['seq'].to(self.device)
                labels = batch['labels'].to(self.device)
                negatives = batch['negatives'].to(self.device)  # are randomly selected items
                last_hidden_state, _ = self.sasrec.forward(seq)

                pos_neg_concat = torch.cat([labels.unsqueeze(-1), negatives.unsqueeze(-1)], dim=-1)
                pos_neg_concat[pos_neg_concat < 0] = self.pad_item_id + 1
                output_embeddings = self.sasrec.get_output_embeddings()
                # the embedding associated to positive and negative labels are retrieved using output_embeddings(pos_neg_concat)
                logits = torch.einsum('bse, bsne -> bsn', last_hidden_state[:, :-1, :], output_embeddings(
                    pos_neg_concat))  # logits computations (product of output embedding with label embedding)
                del output_embeddings

                all_popularity_logits = self.get_popularity_logits(seq[:, :-1])
                popularity_logits = torch.gather(all_popularity_logits, 2, pos_neg_concat)
                del pos_neg_concat

                if step == batches_per_epoch - 1:
                    all_logits = self.sasrec.get_logits(last_hidden_state[:, :-1, :]).detach()
                    gt_logits = all_logits.gather(-1, torch.relu(labels.unsqueeze(-1))).squeeze(-1).reshape(
                        seq.shape[0] * (seq.shape[1] - 1))
                    mean_gt_logits = ((gt_logits * (labels.reshape(labels.shape[0] * (labels.shape[1])) >= 0)).sum() / (
                                labels.reshape(labels.shape[0] * (labels.shape[1])) >= 0).sum())
                    del gt_logits

                    gt_pop_logits = all_popularity_logits.gather(-1, torch.relu(labels.unsqueeze(-1))).squeeze(
                        -1).reshape(seq.shape[0] * (seq.shape[1] - 1))
                    mean_gt_pop_logits = (
                                (gt_pop_logits * (labels.reshape(labels.shape[0] * (labels.shape[1])) >= 0)).sum() / (
                                    labels.reshape(labels.shape[0] * (labels.shape[1])) >= 0).sum())
                    del gt_pop_logits

                    not_gt_mask = torch.ones_like(all_logits, dtype=bool)
                    not_gt_mask.scatter_(-1, torch.relu(labels.unsqueeze(-1)), False)
                    not_gt_mask[:, :, self.items.size():] = False
                    not_gt_logits = all_logits[not_gt_mask]
                    mean_not_gt_logits = ((not_gt_logits * (
                                labels.reshape(labels.shape[0] * (labels.shape[1])).repeat_interleave(
                                    self.items.size() - 1) >= 0)).sum() / (labels.reshape(
                        labels.shape[0] * (labels.shape[1])).repeat_interleave(self.items.size() - 1) >= 0).sum())
                    del not_gt_logits
                    del all_logits

                    not_gt_pop_logits = all_popularity_logits[not_gt_mask]
                    mean_not_gt_pop_logits = ((not_gt_pop_logits * (
                                labels.reshape(labels.shape[0] * (labels.shape[1])).repeat_interleave(
                                    self.items.size() - 1) >= 0)).sum() / (labels.reshape(
                        labels.shape[0] * (labels.shape[1])).repeat_interleave(self.items.size() - 1) >= 0).sum())
                    del not_gt_pop_logits
                    del not_gt_mask

                    tb_writer.add_scalars("score/sasrec_contribution",
                                          {'gt': mean_gt_logits.cpu(), 'not_gt': mean_not_gt_logits.cpu()}, epoch)
                    tb_writer.add_scalars("score/popularity_contribution",
                                          {'gt': mean_gt_pop_logits.cpu(), 'not_gt': mean_not_gt_pop_logits.cpu()},
                                          epoch)

                del all_popularity_logits
                new_logits = logits + popularity_logits
                gt = torch.zeros_like(new_logits)
                gt[:, :, 0] = 1

                mask = (seq[:, :-1] != self.pad_item_id).float()  # used to not compute loss for pad id item
                loss_per_element = torch.nn.functional.binary_cross_entropy_with_logits(new_logits.to(torch.float64),
                                                                                        gt, reduction='none').mean(
                    -1) * mask
                loss = loss_per_element.sum() / mask.sum()

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
                best_model_weights = self.sasrec.state_dict().copy()
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
        self.sasrec.load_state_dict(best_model_weights)
        self.test_collator = MaskingCollator(self.user_actions, self.sequence_length, self.val_users_internal,
                                             self.pad_item_id, mode='test', n_items=self.items.size())

    def validate(self):
        self.sasrec.eval()
        ndcgs = []
        losses = []
        val_collator = MaskingCollator(self.user_actions, self.sequence_length, self.val_users_internal,
                                       self.pad_item_id, mode='val', border_timestamp=self.val_border_timestamp,
                                       val_ndcg_at=self.val_ndcg_at, n_items=self.items.size())
        val_users_internal = list(self.val_users_internal)
        val_loader = DataLoader(val_users_internal, batch_size=self.val_batch_size, collate_fn=val_collator,
                                shuffle=False)
        for batch in val_loader:  # are passed only the validation users to the DataLoader
            recommendations = self.recommend_impl(batch, self.val_ndcg_at, 'val')
            items_for_ndcg = recommendations['items_for_ndcg']
            ratings = recommendations['ratings']

            true_scores = torch.gather(ratings, 1, items_for_ndcg)
            true_scores[
                true_scores < 0] = 0  # assign as rating 0 if the action is a skip (-1) or a dislike (1) to work only with positive values in ndcg evaluation
            dcg = torch.sum(true_scores * self.ndcg_discounts, 1)

            best_ratings = torch.topk(ratings, self.val_ndcg_at, dim=1).values
            best_ratings[best_ratings < 0] = 0
            idcg = torch.sum(best_ratings * self.ndcg_discounts, 1)

            rec_ndcg = torch.nan_to_num(torch.div(dcg,
                                                  idcg))  # the nan values are the ones obtained when idcg is 0 (there are only dislikes and/or skips in the ratings)
            ndcgs.append(rec_ndcg)

            losses.append(recommendations['losses'])
        val_result = {
            "ndcg": torch.cat(ndcgs).mean().item(),
            "loss": torch.cat(losses).mean().item()
        }
        return val_result

    def recommend_impl(self, batch, limit, mode):
        with torch.no_grad():
            if mode == 'val':
                seq = batch['seq'].to(self.device)
                labels = batch['labels'].to(self.device)
                negatives = batch['negatives'].to(self.device)
                ratings = batch['ratings']
                retain_idxs = batch['not_null_seqs_after_ts_idxs'].tolist()

                ratings = ratings[retain_idxs]  # select only the ones having at least one explicit rating (!=0)

                last_hidden_state, _ = self.sasrec.forward(input=seq)
                logits = self.sasrec.get_logits(last_hidden_state) + self.get_popularity_logits(seq)
                del last_hidden_state

                all_logits_for_loss = logits[:, -2,
                                      :]  # to take the second to last one in the sequence (it is the second to last one before border timestamp that we need for loss evaluation)
                pos_neg_concat = torch.cat([labels.unsqueeze(-1), negatives.unsqueeze(-1)], dim=-1)
                logits_for_loss = all_logits_for_loss.gather(1, pos_neg_concat)
                del all_logits_for_loss
                gt = torch.zeros_like(logits_for_loss)
                gt[:, 0] = 1
                loss_per_element = torch.nn.functional.binary_cross_entropy_with_logits(logits_for_loss, gt,
                                                                                        reduction='none').mean(-1)

                logits_for_ndcg = logits[retain_idxs, -1,
                                  :]  # to take the last one in the sequence (the one we need for ndcg evaluation) for the sequences having at least one rating after the ts
                top_k = torch.topk(logits_for_ndcg, limit, dim=1)
                del logits_for_ndcg

                result = {
                    'losses': loss_per_element,
                    'items_for_ndcg': top_k.indices.cpu(),
                    'ratings': ratings
                }
            elif mode == 'test':
                seq = batch['seq'].to(self.device)

                last_hidden_state, _ = self.sasrec.forward(input=seq)
                logits = self.sasrec.get_logits(last_hidden_state)[:, -1, :] + self.get_popularity_logits(seq)[:, -1, :]
                del last_hidden_state

                top_k = torch.topk(logits, limit, dim=1)  # the top k items are selected based on the the logits

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
        self.sasrec.eval()
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

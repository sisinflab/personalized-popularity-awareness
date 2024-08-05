from recommenders.recommender import Recommender
from collections import Counter

from collections import defaultdict
from utils.item_id import ItemId
import mmh3


class MostPopular(Recommender):
    def __init__(self):
        super().__init__()
        self.items_counter = Counter()
        self.item_scores = {}
        self.user_actions = []
        self.most_common = {}
        self.users = ItemId()
        self.items = ItemId()

    def add_action(self, action):
        action_id_internal = self.items.get_id(action.item_id)
        self.user_actions.append((action.timestamp, action_id_internal, action.rating))

    def sort_actions(self):
        self.user_actions.sort(key=lambda x: x[0])

    def rebuild_model(self, val_border_timestamp = None):
        self.sort_actions()
        
        for _, item_id, rating in self.user_actions:
            if rating > 0:
                # self.items_counter[item_id] += int(rating)
                self.items_counter[item_id] += 1
        self.most_common = self.items_counter.most_common()
        for item, score in self.most_common:
            self.item_scores[item] = score     
    
    def recommend_impl(self, limit):
        items = []
        scores = []
        for item, score in self.item_scores.items():
            items.append(item)
            scores.append(score)
        items = items[:limit]
        scores = scores[:limit]
        result = {
            'items': items,
            'scores': scores,
            }
        return result

    def recommend(self, user_id, limit, features=None):
        requests = [(user_id, features)]
        result = self.recommend_batch(requests, limit)
        return result[0]

    def recommend_batch(self, recommendation_requests, limit):
        result = []
        recommendations = self.recommend_impl(limit)
        rec_items = recommendations['items']
        rec_scores = recommendations['scores']
        items = [self.items.reverse_id(x) for x in rec_items]
        scores = rec_scores
        for _ in range(len(recommendation_requests)):
            user_result = list(zip(items, scores))
            result.append(user_result)
        return result
    

class PersonalizedMostPopular(Recommender):
    def __init__(self, sequence_length):
        super().__init__()
        self.items_counter = defaultdict(Counter)
        self.item_scores = defaultdict(dict)
        self.user_actions = defaultdict(list)
        self.most_common = {}
        self.users = ItemId()
        self.items = ItemId()
        self.sequence_length = sequence_length

    def add_action(self, action):
        user_id_internal = self.users.get_id(action.user_id)
        action_id_internal = self.items.get_id(action.item_id)
        self.user_actions[user_id_internal].append((action.timestamp, action_id_internal, action.rating))

    def sort_actions(self):
        for user_id in self.user_actions:
            self.user_actions[user_id].sort(key=lambda x: (x[0], mmh3.hash(f"{x[1]}_{user_id}")) )

    def rebuild_model(self, val_border_timestamp = None):
        self.sort_actions()
        
        all_users = list(self.user_actions.keys())
        for user in all_users:
            last_positive_user_actions = [(item_id, rating) for _, item_id, rating in self.user_actions[user] if rating > 0][-self.sequence_length:]
            for item_id, rating in last_positive_user_actions:
            # for _, item_id, rating in self.user_actions[user]:
            #    if rating > 0:
                # self.items_counter[user][item_id] += int(rating)
                self.items_counter[user][item_id] += 1
            self.most_common[user] = self.items_counter[user].most_common()
            for item, score in self.most_common[user]:
                self.item_scores[user][item] = score
            
    
    def recommend_impl(self, user_ids, limit):
        internal_user_ids = [self.users.get_id(user_id) for user_id in user_ids]
        items = []
        scores = []
        for user in internal_user_ids:
            user_items = []
            user_scores = []
            for item, score in self.item_scores[user].items():
                user_items.append(item)
                user_scores.append(score)
            items.append(user_items[:limit])
            scores.append(user_scores[:limit])
        result = {
            'items': items,
            'scores': scores,
            }
        return result

    def recommend(self, user_id, limit, features=None):
        requests = [(user_id, features)]
        result = self.recommend_batch(requests, limit)
        return result[0]

    def recommend_batch(self, recommendation_requests, limit):
        user_ids = [x[0] for x in recommendation_requests] # select the user ids in the recommendation requests
        result = []
        recommendations = self.recommend_impl(user_ids, limit)
        rec_items = recommendations['items']
        rec_scores = recommendations['scores']
        for idx in range(len(user_ids)):
            items = [self.items.reverse_id(x) for x in rec_items[idx]]
            scores = rec_scores[idx] 
            user_result = list(zip(items, scores))
            result.append(user_result)
        return result

import math
from .metric import Metric
import numpy as np


class NDCG(Metric):
    def __init__(self, k):
        self.name = "ndcg@{}".format(k)
        self.k = k

    def __call__(self, recommendations, actual_actions):
        if (len(recommendations) == 0):
            return 0
        actual_set = set([action.item_id for action in actual_actions])
        recommended = [recommendation[0] for recommendation in recommendations[:self.k]]
        cool = set(recommended).intersection(actual_set)
        if len(cool) == 0:
            return 0
        ideal_rec = sorted(recommended, key=lambda x: not (x in actual_set))
        return NDCG.dcg(recommended, actual_set) / NDCG.dcg(ideal_rec, actual_set)

    @staticmethod
    def dcg(id_list, relevant_id_set):
        result = 0.0
        for idx in range(len(id_list)):
            i = idx + 1
            if (id_list[idx]) in relevant_id_set:
                result += 1 / math.log2(i + 1)
        return result


class NDCG_GTS(Metric):
    def __init__(self, k):
        self.name = "ndcg@{}".format(k)
        self.k = k
        positions = list(np.arange(1, k + 1))
        self.ndcg_discounts = [1 / math.log2(p + 1) for p in positions]

    def __call__(self, recommendations, actual_actions):
        if (len(recommendations) == 0):
            return 0

        actual_set = set([action.item_id for action in actual_actions])
        recommended = [recommendation[0] for recommendation in recommendations[:self.k]]
        cool = set(recommended).intersection(actual_set)
        if len(cool) == 0:
            return 0

        seq_after_ts = []
        for action in actual_actions:
            seq_after_ts.append((action.item_id, action.rating))

        ratings = {}
        for track_id, rating in seq_after_ts:
            # 1st approach: we don't need after to put negative elements to 0, 
            # but if a track is disliked by a user and then played the score 
            # becomes 1 even if it should remain -2
            #ratings[track_id] = max(ratings.get(track_id, 0), rating) 

            # 2nd approach: we do not update the rating of a disliked track with
            # 1 if the track is then played by the user, but we need to put 
            # negative elements to 0
            if track_id not in ratings or abs(rating) >= abs(ratings[track_id]):
                ratings[track_id] = rating

            # equivalent to 2nd approach
            # if rating == -2 or rating == 2:
            #     ratings[track_id] = rating
            # elif track_id in ratings.keys():
            #     if ratings[track_id] != -2 and ratings[track_id] != 2:
            #         ratings[track_id] = rating
            # else:
            #     ratings[track_id] = rating

        dcg = self.dcg(recommended, ratings)

        ordered_ratings = dict(sorted(ratings.items(), key=lambda item: item[1], reverse=True))
        ideal_rec_scores = list(ordered_ratings.values())[:self.k]

        # the following 4 lines are needed if using the 2nd approach
        for i, s in enumerate(ideal_rec_scores):
            if s < 0:
                # assign as rating 0 if the action is a skip (-1) or a dislike (1) to work only with positive values in ndcg evaluation
                ideal_rec_scores[i] = 0

        idcg = np.sum([a * b for a, b in zip(ideal_rec_scores, self.ndcg_discounts)])

        return np.nan_to_num(dcg / idcg)

    def dcg(self, id_list, ratings):
        true_scores = []
        actual_set = [r for r in ratings.keys()]
        for item in id_list:
            if item in actual_set:

                # the following 5 lines are needed if using the 2nd approach
                if ratings[item] > 0:
                    true_scores.append(ratings[item])
                else:
                    # assign as rating 0 if the action is a skip (-1) or a dislike (1) to work only with positive values in ndcg evaluation
                    true_scores.append(0)

                # single line to use with the 1st approach
                # true_scores.append(ratings[item])
            else:
                true_scores.append(0)

        result = np.sum([a * b for a, b in zip(true_scores, self.ndcg_discounts)])
        return result

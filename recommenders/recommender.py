import tempfile
from tqdm import tqdm
from api.item import Item

from api.items_ranking_request import ItemsRankingRequest
from api.user import User
from api.action import Action
import dill


class Recommender():
    def __init__(self):
        self.items_ranking_requests = []
        self.val_users = set() 
        self.tensorboard_dir = None
        self.flags = {}
        self.out_dir = None

    def name(self):
        raise NotImplementedError

    def add_action(self, action: Action):
        raise (NotImplementedError)

    def rebuild_model(self):
        raise (NotImplementedError)

    def recommend(self, user_id, limit: int, features=None):
        raise (NotImplementedError)

    # recommendation request = tuple(user_id, features)
    def recommend_batch(self, recommendation_requests, limit):
        results = []
        for user_id, features in tqdm(recommendation_requests, ascii=True, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', position=0, leave=True, ncols=70):
            results.append(self.recommend(user_id, limit, features))
        return results

    # many recommenders don't require users, so leave it doing nothing by default
    def add_user(self, user: User):
        pass

    # many recommenders don't require items, so leave it doing nothing by default
    def add_item(self, item: Item):
        pass

    def recommend_by_items(self, items_list, limit: int, filter_seen=True):
        raise (NotImplementedError)

    def recommend_by_items_multiple(self, user_actions, limit):
        result = []
        for actions in user_actions:
            result.append(self.recommend_by_items(actions, limit, False))
        return result

    def get_similar_items(self, item_id, limit: int):
        raise (NotImplementedError)

    def to_str(self):
        raise (NotImplementedError)

    def from_str(self):
        raise (NotImplementedError)

    def save(self, filename):
        with open(filename, 'wb') as output:
            dill.dump(self, output)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as input:
            recommender = dill.load(input) 
        return recommender

    #the directory where the recommender can save stuff, like logs
    def set_out_dir(self, out_dir):
        if self.out_dir is not None:
            raise Exception("out_dir already set")
        self.out_dir = out_dir

    def get_out_dir(self):
        if self.out_dir is None:
            self.out_dir = tempfile.mkdtemp()
        return self.out_dir
    
    #recommenders may save tensorboard logs there
    def set_tensorboard_dir(self, tensorboard_dir):
        self.tensorboard_dir = tensorboard_dir

    def get_metadata(self):
        return {}

    def set_val_users(self, val_users):
        self.val_users = val_users

    # class to run sample-based evaluation.
    # according to https://dl.acm.org/doi/abs/10.1145/3383313.3412259 it is not always correct strategy,
    # However we want to keep it in order to be able to produce results comparable with other
    # Papers in order to be able to understand that we implemented our methods correctly.
    # for example comparison table  in BERT4rec is based on sampled metrics (https://arxiv.org/pdf/1904.06690.pdf, page7)
    # we decompose requests and results because some of third-party libraries
    # it is hard to perform items ranking outside of their evaluation process
    def add_test_items_ranking_request(self, request: ItemsRankingRequest):
        self.items_ranking_requests.append(request)

    # should use item ranking requests produced added by add_test_itmes_ranking_requests
    def get_item_rankings(self):
        raise NotImplementedError
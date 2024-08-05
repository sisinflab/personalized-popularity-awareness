import os
import pickle
from datasets.dataset_utils import filter_popular_items, filter_random_items, filter_items_based_on_probs
from utils.os_utils import mkdir_p_local

from datasets.yandex import get_yandex_tracks_events
from datasets.lastfm1k import get_lastfm1k_events


class DatasetsRegister(object):
    DATA_DIR = "data/cache"

    _all_datasets = {
        "yandex": get_yandex_tracks_events,
        "yandex_mostpop30000": lambda: filter_popular_items(DatasetsRegister.get_from_cache("yandex")(), 15000),
        "yandex_random30000": lambda: filter_random_items(DatasetsRegister.get_from_cache("yandex")(), 30000),
        "yandex_30000": lambda: filter_items_based_on_probs(DatasetsRegister.get_from_cache("yandex")(), 30000),

        "lastfm-1k": get_lastfm1k_events,
        "lastfm-1k_mostpop30000": lambda: filter_popular_items(DatasetsRegister.get_from_cache("lastfm-1k")(), 30000),
        "lastfm-1k_random30000": lambda: filter_random_items(DatasetsRegister.get_from_cache("lastfm-1k")(), 30000),
        "lastfm-1k_30000": lambda: filter_items_based_on_probs(DatasetsRegister.get_from_cache("lastfm-1k")(), 30000),
    }

    @staticmethod
    def get_dataset_file(dataset_id):
        cache_dir = mkdir_p_local(DatasetsRegister.DATA_DIR)
        dataset_file = os.path.join(cache_dir, dataset_id + ".pickle")
        return dataset_file

    @staticmethod
    def get_from_cache(dataset_id):
        dataset_file = DatasetsRegister.get_dataset_file(dataset_id)
        if not os.path.isfile(dataset_file):
            DatasetsRegister.cache_dataset(dataset_file, dataset_id)
        return lambda: DatasetsRegister.read_from_cache(dataset_file)

    @staticmethod
    def cache_dataset(dataset_file, dataset_id):
        actions = [action for action in DatasetsRegister._all_datasets[dataset_id]()]
        with open(dataset_file, "wb") as output:
            pickle.dump(actions, output)

    @staticmethod
    def read_from_cache(dataset_file):
        with open(dataset_file, "rb") as input:
            actions = pickle.load(input)
        return actions

    def __getitem__(self, item):
        if item not in DatasetsRegister._all_datasets:
            raise KeyError(f"The dataset {item} is not registered")
        return DatasetsRegister.get_from_cache(item)

    def all_datasets(self):
        return list(DatasetsRegister._all_datasets.keys())

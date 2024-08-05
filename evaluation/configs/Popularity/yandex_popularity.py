from evaluation.metrics.ndcg import NDCG_GTS
from evaluation.split_actions import GlobalTemporalSplit
from recommenders.popularity_recommender import MostPopular, PersonalizedMostPopular

USERS_FRACTIONS = [1.0]

METRICS = [NDCG_GTS(5), NDCG_GTS(10), NDCG_GTS(40), NDCG_GTS(100)]

test_timestamps_fraction = 0.1
val_timestamps_fraction = 0.1


def MostPop():
    return MostPopular()


def PersonalizedMostPop():
    return PersonalizedMostPopular(sequence_length=50)


DATASET = "yandex_30000"

N_VAL_USERS = 2048
MAX_TEST_USERS = 20857  # with yandex dataset

SPLIT_STRATEGY = GlobalTemporalSplit(MAX_TEST_USERS, test_timestamps_fraction=test_timestamps_fraction,
                                     val_timestamp_fraction=val_timestamps_fraction)

RECOMMENDERS = {
    "MostPopular": MostPop,
    "PersonalizedMostPopular": PersonalizedMostPop,
}

if __name__ == "__main__":
    from evaluation.configs.test_configs import TestConfigs

    TestConfigs().validate_config(__file__)

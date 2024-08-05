from evaluation.metrics.ndcg import NDCG_GTS
from evaluation.split_actions import GlobalTemporalSplit

USERS_FRACTIONS = [1.0]

METRICS = [NDCG_GTS(5), NDCG_GTS(10), NDCG_GTS(40), NDCG_GTS(100)]

EMBEDDING_SIZE = 128
test_timestamps_fraction = 0.1
val_timestamps_fraction = 0.1


def BERT4Rec():
    from recommenders.sequential.BERT4Rec.bert4rec_gts import BERT4RecPytorchRecommender
    return BERT4RecPytorchRecommender(embedding_size=EMBEDDING_SIZE,
                                      sequence_length=150,
                                      train_batch_size=128, val_batch_size=128,
                                      max_epochs=10000, early_stop_epochs=200,
                                      num_hidden_layers=3,
                                      lr=0.0001,
                                      wd=0
                                      )


def BERT4Rec_ppl():
    from recommenders.sequential.BERT4Rec.bert4rec_gts_pop import BERT4RecPytorchRecommender
    return BERT4RecPytorchRecommender(embedding_size=EMBEDDING_SIZE,
                                      sequence_length=150,
                                      train_batch_size=128, val_batch_size=128,
                                      max_epochs=10000, early_stop_epochs=200,
                                      num_hidden_layers=3,
                                      lr=0.0001,
                                      wd=0
                                      )


DATASET = "yandex_30000"

N_VAL_USERS = 2048
MAX_TEST_USERS = 20857  # with yandex dataset

SPLIT_STRATEGY = GlobalTemporalSplit(MAX_TEST_USERS, test_timestamps_fraction=test_timestamps_fraction,
                                     val_timestamp_fraction=val_timestamps_fraction)

RECOMMENDERS = {
    "BERT4Rec": BERT4Rec,
    "BERT4Rec_ppl": BERT4Rec_ppl,
}

if __name__ == "__main__":
    from evaluation.configs.test_configs import TestConfigs

    TestConfigs().validate_config(__file__)

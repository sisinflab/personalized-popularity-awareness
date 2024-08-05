from evaluation.metrics.ndcg import NDCG_GTS
from evaluation.split_actions import GlobalTemporalSplit

USERS_FRACTIONS = [1.0]

METRICS = [NDCG_GTS(5), NDCG_GTS(10), NDCG_GTS(40), NDCG_GTS(100)]

EMBEDDING_SIZE=128
test_timestamps_fraction=0.1
val_timestamps_fraction=0.1


def SASRec():
    from recommenders.sequential.SASRec.sasrec_gts import SASRecRecommender
    return SASRecRecommender(embedding_size=EMBEDDING_SIZE,
                                sequence_length=150,
                                train_batch_size=128, val_batch_size=128,
                                max_epochs=10000, early_stop_epochs=500,
                                num_blocks=3,
                                lr=0.01,
                                wd=0
                                )

def SASRec_ppl():
    from recommenders.sequential.SASRec.sasrec_gts_pop import SASRecRecommender
    return SASRecRecommender(embedding_size=EMBEDDING_SIZE,
                                sequence_length=150,
                                train_batch_size=128, val_batch_size=128,
                                max_epochs=10000, early_stop_epochs=500,
                                num_blocks=3,
                                lr=0.0001,
                                wd=0
                                )

def gSASRec():
    from recommenders.sequential.SASRec.gsasrec_gts import gSASRecRecommender
    return gSASRecRecommender(embedding_size=EMBEDDING_SIZE,
                                sequence_length=150,
                                train_batch_size=128, val_batch_size=128,
                                max_epochs=10000, early_stop_epochs=500,
                                num_blocks=3,
                                lr=0.01,
                                wd=0,
                                negs_per_pos=256,
                                gbce_t=0.75
                                )

def gSASRec_ppl():
    from recommenders.sequential.SASRec.gsasrec_gts_pop import gSASRecRecommender
    return gSASRecRecommender(embedding_size=EMBEDDING_SIZE,
                                sequence_length=150,
                                train_batch_size=128, val_batch_size=128,
                                max_epochs=10000, early_stop_epochs=500,
                                num_blocks=3,
                                lr=0.0001,
                                wd=0,
                                negs_per_pos=256,
                                gbce_t=0.75
                                )

DATASET = "lastfm-1k_30000"

N_VAL_USERS=128
MAX_TEST_USERS=992 # with lastfm-1k dataset

SPLIT_STRATEGY = GlobalTemporalSplit(MAX_TEST_USERS, test_timestamps_fraction=test_timestamps_fraction, val_timestamp_fraction=val_timestamps_fraction)

RECOMMENDERS = {
    "SASRec": SASRec,
    "SASRec_ppl": SASRec_ppl,
    "gSASRec": gSASRec,
    "gSASRec_ppl": gSASRec_ppl,
     }

if __name__ == "__main__":
     from evaluation.configs.test_configs import TestConfigs
     TestConfigs().validate_config(__file__)
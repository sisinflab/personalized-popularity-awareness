import unittest


class TestLastfm1kDataset(unittest.TestCase):
    def test_lastfm1k_dataset(self):
        import json
        from datasets.dataset_stats import dataset_stats
        from datasets.datasets_register import DatasetsRegister

        lastfm1k = DatasetsRegister()["lastfm-1k"]()
        lastfm1k_30000 = DatasetsRegister()["lastfm-1k_30000"]()
        lastfm1k_random30000 = DatasetsRegister()["lastfm-1k_random30000"]()
        lastfm1k_mostpop30000 = DatasetsRegister()["lastfm-1k_mostpop30000"]()

        lastfm1k_stats = dataset_stats(lastfm1k,
                                       metrics=['num_users', 'num_items', 'num_interactions', 'average_session_len',
                                                'median_session_len', 'min_session_len', 'max_session_len', 'sparsity'])
        lastfm1k_30000_stats = dataset_stats(lastfm1k_30000, metrics=['num_users', 'num_items', 'num_interactions',
                                                                      'average_session_len', 'median_session_len',
                                                                      'min_session_len', 'max_session_len', 'sparsity'])
        lastfm1k_random30000_stats = dataset_stats(lastfm1k_random30000,
                                                   metrics=['num_users', 'num_items', 'num_interactions',
                                                            'average_session_len', 'median_session_len',
                                                            'min_session_len', 'max_session_len', 'sparsity'])
        lastfm1k_mostpop30000_stats = dataset_stats(lastfm1k_mostpop30000,
                                                    metrics=['num_users', 'num_items', 'num_interactions',
                                                             'average_session_len', 'median_session_len',
                                                             'min_session_len', 'max_session_len', 'sparsity'])

        print(json.dumps(lastfm1k_stats, indent=4))
        print(json.dumps(lastfm1k_30000_stats, indent=4))
        print(json.dumps(lastfm1k_random30000_stats, indent=4))
        print(json.dumps(lastfm1k_mostpop30000_stats, indent=4))


if __name__ == "__main__":
    unittest.main()

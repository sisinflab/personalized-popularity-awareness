import unittest


class TestYandexDataset(unittest.TestCase):
    def test_yandex_dataset(self):
        import json
        from datasets.dataset_stats import dataset_stats
        from datasets.datasets_register import DatasetsRegister

        yandex = DatasetsRegister()["yandex"]()
        yandex_30000 = DatasetsRegister()["yandex_30000"]()
        yandex_random30000 = DatasetsRegister()["yandex_random30000"]()
        yandex_mostpop30000 = DatasetsRegister()["yandex_mostpop30000"]()

        yandex_stats = dataset_stats(yandex,
                                     metrics=['num_users', 'num_items', 'num_interactions', 'average_session_len',
                                              'median_session_len', 'min_session_len', 'max_session_len', 'sparsity'])
        yandex_30000_stats = dataset_stats(yandex_30000,
                                           metrics=['num_users', 'num_items', 'num_interactions', 'average_session_len',
                                                    'median_session_len', 'min_session_len', 'max_session_len',
                                                    'sparsity'])
        yandex_random30000_stats = dataset_stats(yandex_random30000,
                                                 metrics=['num_users', 'num_items', 'num_interactions',
                                                          'average_session_len', 'median_session_len',
                                                          'min_session_len', 'max_session_len', 'sparsity'])
        yandex_mostpop30000_stats = dataset_stats(yandex_mostpop30000,
                                                  metrics=['num_users', 'num_items', 'num_interactions',
                                                           'average_session_len', 'median_session_len',
                                                           'min_session_len', 'max_session_len', 'sparsity'])

        print(json.dumps(yandex_stats, indent=4))
        print(json.dumps(yandex_30000_stats, indent=4))
        print(json.dumps(yandex_random30000_stats, indent=4))
        print(json.dumps(yandex_mostpop30000_stats, indent=4))


if __name__ == "__main__":
    unittest.main()

from collections import Counter
import gzip
import logging
import os
import random
import numpy as np

from utils.os_utils import get_dir, mkdir_p, shell


def filter_popular_items(actions_generator, max_items):
    actions = []
    items_counter = Counter()
    for action in actions_generator:
        actions.append(action)
        items_counter[action.item_id] += 1
    popular_items = set([item_id for (item_id, _) in items_counter.most_common(max_items)])
    return filter(lambda action: action.item_id in popular_items, actions)


def filter_items_based_on_probs(actions_generator, n_items):
    actions = []
    items_counter = Counter()
    for action in actions_generator:
        actions.append(action)
        items_counter[action.item_id] += 1
    items_counts = items_counter.most_common()
    items_ids = [item_id for item_id, _ in items_counts]
    item_counts = [item_count for _, item_count in items_counts]
    counts_sum = sum(item_counts)
    item_probs = [item_count / counts_sum for item_count in item_counts]
    np.random.seed(27)
    items = set(np.random.choice(a=items_ids, replace=False, p=item_probs, size=n_items))
    return filter(lambda action: action.item_id in items, actions)


def filter_random_items(actions_generator, n_items, min_count=1):
    actions = []
    items_counter = Counter()
    for action in actions_generator:
        actions.append(action)
        items_counter[action.item_id] += 1
    items_counts = items_counter.most_common()
    random.seed(27)
    random.shuffle(items_counts)
    items = set([item_id for (item_id, cnt) in items_counts if cnt >= min_count][:n_items])
    return filter(lambda action: action.item_id in items, actions)


def unzip(zipped_file, unzip_dir):
    full_dir_name = os.path.join(get_dir(), unzip_dir)
    if os.path.isdir(full_dir_name):
        logging.info(f"{unzip_dir} already exists, skipping")
    else:
        mkdir_p(full_dir_name)
        shell(f"unzip -o {zipped_file} -d {full_dir_name}")
    return full_dir_name


def gunzip(gzip_file):
    full_file_name = os.path.abspath(gzip_file)
    if not (gzip_file.endswith(".gz")):
        raise Exception(f"{gzip_file} is not a gzip file")
    unzipped_file_name = full_file_name[:-3]
    if os.path.isfile(unzipped_file_name):
        logging.info(f"{unzipped_file_name} already exists, skipping")
        return unzipped_file_name

    with gzip.open(full_file_name) as input:
        data = input.read()
        with open(unzipped_file_name, 'wb') as output:
            output.write(data)
    return unzipped_file_name

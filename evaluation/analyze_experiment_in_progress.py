import json
import pandas as pd
import re
import sys

pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', 256)


def is_experiment_start(line):
    return line.startswith('evaluating for')


def skip_n_experiments(input_file, experiment_num):
    current_experiment = 0
    while current_experiment < experiment_num:
        line = input_file.readline()
        if is_experiment_start(line):
            current_experiment += 1


def get_metrics(line):
    regexp = re.compile(r'[a-zA-Z0-9_]+\: [0-9\.\+\-eE]+')
    result = {}
    for metric_str in regexp.findall(line):
        metric, value = metric_str.split(': ')
        result[metric] = float(value)
    return result


def parse_experiment(experiment_log):
    current_recommender = None
    result = []
    cnt = 0
    metrics = []
    experiment_finished = True
    for line in experiment_log:
        if line.startswith('evaluating ') or line.startswith("!!!!!!!!!   evaluating"):
            current_recommender = line.split(' ')[-1]
            metrics = []
            experiment_finished = False
            epoch = 0
        if ('val_ndcg_at_' in line) or ('val_ndcg@' in line) or ('Val NDCG@' in line):
            epoch += 1
            epoch_metrics = get_metrics(line)
            epoch_metrics['epoch'] = epoch

        try:
            experiment_results = json.loads(line)
            experiment_results['model_name'] = current_recommender
            experiment_results['num_epochs'] = epoch
            experiment_results['metrics_history'] = metrics
            result.append(experiment_results)
            experiment_finished = True
        except:
            pass
    if not experiment_finished:
        experiment_results = {}
        experiment_results['model_name'] = current_recommender
        experiment_results['metrics_history'] = metrics
        experiment_results['num_epochs'] = epoch
        result.append(experiment_results)
    return result


def get_data_from_logs(logfile, experiment_num):
    with open(logfile) as input_file:
        skip_n_experiments(input_file, experiment_num)
        experiment_log = []
        for line in input_file:
            if is_experiment_start(line):
                break
            else:
                experiment_log.append(line.strip())
        return parse_experiment(experiment_log)


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        experiment_logs_file = "results/latest_experiment/stdout"
    else:
        experiment_logs_file = sys.argv[1]
    data = get_data_from_logs(experiment_logs_file, 0)
    df = pd.DataFrame(data).set_index('model_name')
    ranks = range(1, len(df) + 1)
    if len(sys.argv) > 2:
        main_metric = sys.argv[2]
    else:
        metric_names = df.columns
        main_metric = metric_names[0]

    df = df.sort_values(main_metric, ascending=False)

    df.insert(loc=0, column=f'rank by {main_metric}', value=ranks)
    try:
        del df['model_metadata']
    except:
        pass
    try:
        del df['metrics_history']
    except:
        pass

    if 'sampled_metrics' in df.columns:
        sampled_metrics_raw = list(df['sampled_metrics'])
        sampled_metrics = []
        for metrics in sampled_metrics_raw:
            if type(metrics) == dict:
                sampled_metrics.append(metrics)
            else:
                sampled_metrics.append(dict())

        del (df['sampled_metrics'])
        sampled_metrics_df = pd.DataFrame(sampled_metrics, index=df.index).sort_values(main_metric, ascending=False)
        sampled_metrics_df.insert(loc=0, column=f'rank by {main_metric}', value=ranks)

        print("sampled metrics: ")
        print(sampled_metrics_df)
        print("\n\n\n")

    print("unsampled metrics:")
    try:
        df['epoch_time'] = df['model_build_time'] / df['num_epochs']
        # if df['num_epochs'] != 0:
        #     df['epoch_time'] = df['model_build_time']/df['num_epochs']
        # else: 
        #     df['epoch_time'] = 0
    except:
        pass
    print(df.to_markdown())

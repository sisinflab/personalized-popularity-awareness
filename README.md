# Enhancing Sequential Music Recommendation with Personalised Popularity Awareness

This is the official repository for the paper "Enhancing Sequential Music Recommendation with Personalised Popularity Awareness", published at the 18th ACM Conference on Recommender Systems (RecSys '24).

In the realm of music recommendation, sequential recommender systems have shown promise in capturing the dynamic nature of music consumption. Nevertheless, traditional Transformer-based models, such as SASRec and BERT4Rec, while effective, encounter challenges due to the unique characteristics of music listening habits. In fact, existing models struggle to create a coherent listening experience due to rapidly evolving preferences. Moreover, music consumption is characterized by a prevalence of repeated listening, i.e. users frequently return to their favourite tracks, an important signal that could be framed as individual or personalized popularity. This paper addresses these challenges by introducing a novel approach that incorporates personalized popularity information into sequential recommendation. By combining user-item popularity scores with model-generated scores, our method effectively balances the exploration of new music with the satisfaction of user preferences.
Experimental results demonstrate that a Personalized Most Popular recommender, a method solely based on user-specific popularity, outperforms existing state-of-the-art models.
Furthermore, augmenting Transformer-based models with personalized popularity awareness yields superior performance, showing improvements ranging from 25.2% to 69.8%.

Find more details in the paper: https://dl.acm.org/doi/10.1145/3640457.3691719.

Our code is based on the `aprec` framework from this [reproducibility work](https://github.com/asash/bert4rec_repro), so you can use the original documentation to learn how to use the framework. 

If you use any part of this code, please cite the work:

```
@inproceedings{10.1145/3640457.3691719,
author = {Abbattista, Davide and Anelli, Vito Walter and Di Noia, Tommaso and Macdonald, Craig and Petrov, Aleksandr Vladimirovich},
title = {Enhancing Sequential Music Recommendation with Personalized Popularity Awareness},
year = {2024},
publisher = {Association for Computing Machinery},
url = {https://doi.org/10.1145/3640457.3691719},
doi = {10.1145/3640457.3691719},
booktitle = {Proceedings of the 18th ACM Conference on Recommender Systems},
pages = {1168–1173},
series = {RecSys '24}
}
```

# Environment setup

### Using venv, set up the environment as follows:

```
python3 -m venv <your working directory>/.venv
source .venv/bin/activate
pip install -r requirements.txt 
```

# Runnig experiments

### 1.  Go to the evaluation folder: 
```
cd <your working directory>
cd evaluation
```

### 2. Reproducing experiments from the paper
You need to run `run_n_experiments.sh` with the experiment configuration file.
The config files for experiments described in the paper are in the `configs/`. 
To run the experiments, please run.

**Yandex music event:**

```
sh run_n_experiments.sh configs/yandex_all.py
```

**Last.fm-1K:**

```
sh run_n_experiments.sh configs/lastfm1k_all.py
```

to analyse the results of the latest experiment run 

```
python3 analyze_experiment_in_progress.py
```

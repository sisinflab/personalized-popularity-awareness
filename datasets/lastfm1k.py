import os
import logging
import requests
from tqdm import tqdm
from api.track_event import TrackEvent
from utils.os_utils import get_dir, shell, mkdir_p_local

LASTFM1K_DIR = "data/lastfm-1k"
LASTFM1K_RAW_FILE = "lastfm-dataset-1K.tar.gz"
LASTFM1K_URL = "http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz"

events_file = "userid-timestamp-artid-artname-traid-traname.tsv"
users_file = "userid-profile.tsv"

LASTFM1K_DATA_DIR = os.path.join(get_dir(), LASTFM1K_DIR)
LASTFM1K_EVENTS_FILE = os.path.join(LASTFM1K_DATA_DIR, events_file)


def extract_lastfm1k_dataset():
    lastfm1k_tar_file = get_lastfm1k_tar_file()
    if os.path.isfile(LASTFM1K_EVENTS_FILE):
        return
    shell(f"tar xvf {lastfm1k_tar_file} -C {LASTFM1K_DATA_DIR}")
    shell(f"mv {LASTFM1K_DATA_DIR}/lastfm-dataset-1K/userid-profile.tsv {LASTFM1K_DATA_DIR}")
    shell(f"mv {LASTFM1K_DATA_DIR}/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv {LASTFM1K_DATA_DIR}")
    shell(f"mv {LASTFM1K_DATA_DIR}/lastfm-dataset-1K/README.txt {LASTFM1K_DATA_DIR}")
    shell(f"rm -r {LASTFM1K_DATA_DIR}/lastfm-dataset-1K")


def get_lastfm1k_tar_file():
    full_filename = os.path.join(LASTFM1K_DATA_DIR, LASTFM1K_RAW_FILE)
    if not (os.path.isfile(full_filename)):
        logging.info(f"downloading  {LASTFM1K_RAW_FILE} file")
        response = requests.get(LASTFM1K_URL, stream=True)
        mkdir_p_local(LASTFM1K_DATA_DIR)
        with open(full_filename, 'wb') as out_file:
            expected_length = int(response.headers.get('content-length'))
            with tqdm(total=expected_length, ascii=True) as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    out_file.write(chunk)
                    out_file.flush()
                    pbar.update(len(chunk))
        logging.info(f"{LASTFM1K_RAW_FILE} dataset downloaded")
    return full_filename


def get_lastfm1k_events():
    extract_lastfm1k_dataset()
    with open(LASTFM1K_EVENTS_FILE, 'r') as data_file:
        for line in data_file:
            user_id, timestamp, _, _, musicbrainz_track_id, _ = line.strip().split('\t')
            user_id = int(user_id.replace('user_', ''))
            if musicbrainz_track_id:
                yield TrackEvent(user_id, musicbrainz_track_id, timestamp, float(1))

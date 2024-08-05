import os
from api.track_event import TrackEvent
from utils.os_utils import get_dir, shell

YANDEX_DIR = "data/yandex"
YANDEX_RAW_FILE = "data.tar.gz"
YANDEX_URL = "https://www.kaggle.com/competitions/yandex-music-event-2019-02-16"

events_file = "user_events"

YANDEX_DATA_DIR = os.path.join(get_dir(), YANDEX_DIR)
YANDEX_EVENTS_FILE = os.path.join(YANDEX_DATA_DIR, events_file)


def extract_yandex_dataset():
    yandex_tar_file = get_yandex_tar_file()
    if os.path.isfile(YANDEX_EVENTS_FILE):
        return
    shell(f"tar xvf {yandex_tar_file} -C {YANDEX_DATA_DIR}")


def get_yandex_tar_file():
    full_filename = os.path.join(YANDEX_DATA_DIR, YANDEX_RAW_FILE)
    if not (os.path.isfile(full_filename)):
        raise Exception(f"We do not support automatic download for Yandex dataset.\n" +
                        f"Please download it manually from {YANDEX_URL} and put it into {YANDEX_DATA_DIR}")
    return full_filename


def get_yandex_tracks_events():
    extract_yandex_dataset()
    with open(YANDEX_EVENTS_FILE, 'r') as data_file:
        header = True
        for line in data_file:
            if header:
                header = False
            else:
                user_id, item_type, item_id, event, timestamp = line.strip().split(',')
                timestamp = int(timestamp)
                if item_type == "track":
                    rating = None
                    if event == "like":
                        rating = float(2)
                    elif event == "play":
                        rating = float(1)
                    elif event == "skip":
                        rating = float(-1)
                    elif event == "dislike":
                        rating = float(-2)
                    yield TrackEvent(user_id, item_id, timestamp, rating)
                else:
                    continue

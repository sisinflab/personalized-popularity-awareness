import json


class TrackEvent(object):
    def __init__(self, user_id, item_id, timestamp, rating=None):
        self.user_id = user_id
        self.item_id = item_id
        self.rating = rating
        self.timestamp = timestamp

    def to_str(self):
        result = "TrackEvent(uid={}, item={}, ts={}".format(
            self.user_id,
            self.item_id,
            self.timestamp)
        if self.rating is not None:
            result += ", rating={}".format(str(self.rating))
        result += ")"
        return result

    def to_json(self):
        return json.dumps({
            "user_id": self.user_id,
            "item_id": self.item_id,
            "rating": str(self.rating),
            "timestamp": self.timestamp
        })

    @staticmethod
    def from_json(action_str):
        doc = json.loads(action_str)
        return TrackEvent(doc["user_id"], doc["item_id"], doc["rating"], doc["timestamp"])

    def __str__(self):
        return self.to_str()

    def __repr__(self):
        return self.to_str()

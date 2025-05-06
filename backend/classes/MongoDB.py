import sys

from pymongo.mongo_client import MongoClient
import os

class MongoDB:

    def __init__(self):
        self._uri = None
        self._client = None

    def connect(self):
        try:
            self._uri = os.environ["MONGODB_URI"]
        except KeyError:
            print("Missing MONGODB_URI environment variable", file=sys.stderr)
            sys.exit(-1)
        self._client = MongoClient(self._uri)


    def insert_one(self, database_name, collection_name, data):
        if self._client is None:
            return None
        return self._client[database_name][collection_name].insert_one(data)

import sys

from pymongo.mongo_client import MongoClient, ConnectionFailure


class MongoDBLogger:

    def __init__(self, uri):
        self._uri = uri
        self._client = None

    def connect(self):
        self._client = MongoClient(self._uri)

        try:
            # The ping command is cheap and does not require auth.
            self._client.admin.command('ping')
        except ConnectionFailure:
            print("Connection Error", file=sys.stderr)

    def log(self, database_name, collection_name, data):
        if self._client is None:
            raise RuntimeError("No Connection Available")
        return self._client[database_name][collection_name].insert_one(data)

    def close(self):
        if self._client is not None:
            self._client.close()

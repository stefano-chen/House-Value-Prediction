import os
import sys

from pymongo.mongo_client import MongoClient, ConnectionFailure


class MongoDB:

    def __init__(self):
        self._uri = None
        self._client = None

    def connect(self, uri):
        try:
            self._uri = uri
        except KeyError:
            print("Missing MONGODB_URI environment variable", file=sys.stderr)
            sys.exit(-1)
        self._client = MongoClient(self._uri)

        try:
            # The ping command is cheap and does not require auth.
            self._client.admin.command('ping')
        except ConnectionFailure:
            print("Database not available", file=sys.stderr)
            sys.exit(-1)


    def insert_one(self, database_name, collection_name, data):
        if self._client is None:
            return None
        return self._client[database_name][collection_name].insert_one(data)

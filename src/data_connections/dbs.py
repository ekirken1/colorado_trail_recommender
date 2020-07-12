from pymongo import MongoClient


def get_mongo_data(creds):
    client = MongoClient(creds["url"], int(creds["port"]))
    db = client[creds["db"]]
    hikes = db[creds["table"]]

    return client, hikes

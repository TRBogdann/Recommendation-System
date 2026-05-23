from pymongo import MongoClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONNECTION = f"mongodb+srv://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@cluster0.qurflfl.mongodb.net/"
DB_SCHEMA = os.getenv('DB_SCHEMA')

mongo_client = MongoClient(
    DB_CONNECTION,
    server_api=ServerApi('1')
)
mongo_db = mongo_client[DB_SCHEMA]

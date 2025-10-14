import numpy  as np 
from config import settings
import chromadb
import redis 
import logging
import json 
import requests

logging.basicConfig(
    level=logging.INFO,              \
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

class recommender():
    def __init__(self, collection_name_http_address: str ="http://als_trainer:8000/collection_name", chroma_host: str = settings.CHROMA_HOST, chroma_port: int = settings.CHROMA_PORT, 
                 redis_host:str = settings.REDIS_HOST, redis_port:str =settings.REDIS_PORT):
       
        self.client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port
        )

        self.redis=redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        self.collection_name_http_address=collection_name_http_address

        self.collection_name = None 

    def _get_collection_name(self):
        """"
        Retrieving collection name from chromadb
        """

        try:
            resp = requests.get(self.collection_name_http_address)
            resp.raise_for_status()  
            data = resp.json()
            self.collection_name = data.get("collection_name")
        except Exception as e:
            log.exception("Error fetching collection name")
            self.collection_name = None 

    def _get_user_embedding(self, user_id:int):
        """"
        Finding user embedding in redis db
        """
        try:
            str_embedding = self.redis.get(f"user:{user_id}")
            if str_embedding:
                return json.loads(str_embedding)
            else:
                return json.loads(self.redis.get('user:mean'))
        except Exception:
            log.exception("Exception with redis")
            return []  
    def search_item(self, user_id: int, n_items: int):
        """
        Searching for good recommendation
        """

        self._get_collection_name()
        if not self.collection_name:
            raise RuntimeError("No active collection name available")

        if not isinstance(user_id, int) or user_id < 0:
            raise ValueError(f"user_id must be a non-negative integer, got {user_id}")
        if not isinstance(n_items, int) or n_items <= 0:
            raise ValueError(f"n_items must be a positive integer, got {n_items}")

        query_embedding = self._get_user_embedding(user_id)

        results = {}
        try:
            collection = self.client.get_collection(name=self.collection_name, embedding_function=None)
            results = collection.query(query_embeddings=[query_embedding], n_results=n_items)
        except Exception:
            log.exception("Chroma query failed for user %s", user_id)
            return [], []

        ids = results.get('ids', [[]])[0]
        distances = results.get('distances', [[]])[0]

        log.info("Search complete for user %s: %d items returned.", user_id, len(ids))
        return ids, distances

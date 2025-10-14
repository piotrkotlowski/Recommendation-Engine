import json
import logging
from typing import Tuple
import numpy as np
import pandas as pd
import torch
import chromadb
from chromadb.config import Settings as ChromaSettings
from psycopg.rows import dict_row
import psycopg
from .pyTorch_als import ALS
from config import settings
import redis
from typing import List 

logging.basicConfig(
    level=logging.INFO,              \
    format="%(asctime)s [%(levelname)s] %(message)s",
)

log = logging.getLogger(__name__)

class ALSRecommender:
    
    def __init__(self, collection_name: str, pg_dsn: str = settings.PG_DSN, chroma_host: str = settings.CHROMA_HOST, chroma_port: int = settings.CHROMA_PORT, 
                 als_param_path:str = settings.ALS_PARAM_PATH ,
                 redis_host:str = settings.REDIS_HOST, redis_port:str =settings.REDIS_PORT):
        """Initialize ALS recommender with database connection, ChromaDB client, and parameter paths."""

        
        self.pg_dsn = pg_dsn
        
        self.client = chromadb.HttpClient(
            host=chroma_host,
            port=chroma_port
        )

        self.redis=redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        self.collection_name = collection_name
        self.params_path=als_param_path
        self.X = None
        self.Y = None
        self.params = self._load_params()


  
    def _load_interactions(self) -> pd.DataFrame:
        """Load user-item interactions from PostgreSQL and return as a DataFrame."""
        try:
            with psycopg.connect(self.pg_dsn) as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute("SELECT * FROM interactions")
                    rows = cur.fetchall()
        except Exception as e:
            raise ALSRecommender.DataLoadException(f"Failed to load interactions: {e}")

        if not rows: 
            raise ALSRecommender.DataLoadException("No interaction data found in the database.")
        df = pd.DataFrame(rows)
        df = df[['user_id', 'item_id', 'rating']]
        df['user_id'] = df['user_id'].astype(int)
        df['item_id'] = df['item_id'].astype(int)
        df['rating'] = df['rating'].astype(int)
        return df

    def _load_params(self) -> Tuple[int, int]:
        """Load ALS parameters (rank, lambda) from JSON configuration file."""
        try:
            with open(self.params_path, 'r') as f:
                param_dict = json.load(f)
            rank = int(param_dict.get('rank', 10)) 
            lamb = int(param_dict.get('lambda', 1))
            if rank <= 0 or lamb < 0:
                raise ValueError("Invalid ALS params: rank must be >0, lambda >=0")
            return rank,lamb
        except FileNotFoundError:
            raise ALSRecommender.DataLoadException(f"Params file not found at {self.params_path}")
        except json.JSONDecodeError:
            raise ALSRecommender.DataLoadException("Params file is not valid JSON")
        except Exception as e:
            raise ALSRecommender.DataLoadException(f"Unexpected error loading ALS params: {e}")

    @staticmethod
    def factors_to_df(X: torch.Tensor, Y: torch.Tensor) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert ALS user/item factor matrices into DataFrames with embeddings."""
        df_users = pd.DataFrame({
            'id': range(X.shape[1]),
            'embedding': [X[:, i].cpu().numpy().tolist() for i in range(X.shape[1])]
        })

        mean_user_embedding=torch.mean(X,axis=1).cpu().numpy().tolist()


        df_items = pd.DataFrame({
            'id': range(Y.shape[1]),
            'embedding': [Y[:, i].cpu().numpy().tolist() for i in range(Y.shape[1])]
        })
        return df_users, df_items, mean_user_embedding

    def train_als(self, max_iter: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train the ALS model on user-item interactions and return factor matrices X, Y."""

        log.info('Loading interactions!')
        df = self._load_interactions()

        n_users_total = int(df['user_id'].max()) + 1
        m_items_total = int(df['item_id'].max()) + 1

        log.info('Loading params!')
        rank, lamb = self.params

        log.info("Training als!")

        als_trainer = ALS(
            n_users=n_users_total,
            n_items=m_items_total,
            rank=rank,
            lamb=lamb,
            max_iter=max_iter
        )

        self.X, self.Y = als_trainer.fit(
            df_users=df['user_id'], 
            df_items=df['item_id'], 
            df_ratings=df['rating'])
        
        log.info("Finished training als!")
        
        return self.X, self.Y

    def _ingest_items_to_chroma(self, df_items: pd.DataFrame, batch_size: int = 1000):
        """Ingest trained item embeddings into ChromaDB collection in batches."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None,
            configuration={"hnsw": {"space": "ip"}})
        
        log.info('Ingesting chroma!')
        
        ids_batch, embeddings_batch = [], []
        for _, row in df_items.iterrows():
            ids_batch.append(str(row['id']))
            embeddings_batch.append(row['embedding'])
            if len(ids_batch) >= batch_size:
                collection.add(ids=ids_batch, embeddings=embeddings_batch)
                ids_batch, embeddings_batch = [], []

        if ids_batch:
            collection.add(ids=ids_batch, embeddings=embeddings_batch)

        log.info('Finished ingesting chroma!')

    def _save_embeddings_to_pg(self,df_users: pd.DataFrame,df_items: pd.DataFrame):
        """"
        Ingesting embeddings into postgres database
        """
        try:
            with psycopg.connect(self.pg_dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS user_embeddings (
                            user_id bigint PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
                            embedding JSONB
                        );
                    """)
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS item_embeddings (
                            item_id bigint  PRIMARY KEY REFERENCES items(item_id) ON DELETE CASCADE,
                            embedding JSONB
                        );
                    """)
                    for _,row in df_items.iterrows():
                        cur.execute("""
                        INSERT INTO item_embeddings (item_id,embedding)
                        VALUES (%s,%s)
                        ON CONFLICT (item_id) DO UPDATE 
                            SET embedding = EXCLUDED.embedding;                    

                        """, (int(row['id']),json.dumps(row['embedding'])))
                    for _,row in df_users.iterrows():
                        cur.execute("""
                            INSERT INTO user_embeddings (user_id,embedding)
                            VALUES (%s,%s)
                            ON Conflict (user_id) DO UPDATE 
                                    SET embedding = EXCLUDED.embedding;
                            """,(int(row['id']),json.dumps(row['embedding'])))
                conn.commit()
            log.info('Saved embeddings into Postgres sucessfully')
        except Exception as e:
            log.exception('Failed saving embeddings to Postgres')

    

    def _save_embeddings_to_redis(self,df_users:pd.DataFrame,df_items: pd.DataFrame,mean_user_embedding:List[float]):
        """"
        Ingesting embeddings into redis
        """
        
        try:
            pipe = self.redis.pipeline(transaction=False)

            for _,row in df_users.iterrows():
                pipe.set(f"user:{row['id']}", json.dumps(row["embedding"]))
            
            pipe.set("user:mean", json.dumps(mean_user_embedding))

            for _,row in df_items.iterrows():
                pipe.set(f"item:{row['id']}", json.dumps(row["embedding"]))


            
            pipe.execute()
            log.info('Saved embedding into Redis sucessfully')
        except Exception as e :
            log.exception('Failed saving embeddings to Redis')

    def run_pipeline(self):
        """Full pipeline: train ALS, convert factors to DataFrames, ingest items to ChromaDB."""
        try:
            log.info("Starting ALS training pipeline...")
            X, Y = self.train_als()

            log.info("Converting factors ")
            df_users, df_items , mean_user_embedding= self.factors_to_df(X, Y)


            log.info("Ingesting item embeddings into ChromaDB...")
            self._ingest_items_to_chroma(df_items)

            log.info("Ingesting item embeddings into PG...")
            self._save_embeddings_to_pg(df_users,df_items)
            
            log.info("Ingesting item embeddings into Redis...")
            self._save_embeddings_to_redis(df_users,df_items,mean_user_embedding)

            log.info("Pipeline finished successfully.")
            return {"users": df_users, "items": df_items}

        except ALSRecommender.DataLoadException as e:
            log.error("Data loading failed: %s", e)
            return None
        except Exception as e:
            log.exception("Unexpected error during pipeline execution")
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    class DataLoadException(Exception):
        """Custom exception for handling data loading errors in ALS recommender."""
        pass

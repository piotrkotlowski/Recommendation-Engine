import argparse, os, time, subprocess, sys
import csv
import psycopg
from config import settings

PG_DSN = settings.PG_DSN

DDL_SQL = """
CREATE TABLE IF NOT EXISTS users (
    user_id  bigint PRIMARY KEY,
    name     varchar(200)
);
CREATE TABLE IF NOT EXISTS items (
    item_id     bigint PRIMARY KEY,
    item_name   varchar(400) NOT NULL,
    category    varchar(200) NOT NULL,
    brand       varchar(200) NOT NULL,
    description text NOT NULL,
    price       numeric CHECK (price>=0) NOT NULL
);
CREATE TABLE IF NOT EXISTS interactions (
    purchase_id bigint PRIMARY KEY,
    user_id     bigint REFERENCES users(user_id) ON DELETE CASCADE,
    item_id     bigint REFERENCES items(item_id) ON DELETE CASCADE,
    rating      int CHECK (rating BETWEEN 1 AND 5) NOT NULL,
    review_text text,
    verified    bool NOT NULL
   
);
Create INDEX IF NOT EXISTS  idx_interactions_user_id ON interactions(user_id);

CREATE TABLE IF NOT EXISTS chat_history (
    chat_id       BIGSERIAL PRIMARY KEY,
    user_id       bigint REFERENCES users(user_id) ON DELETE CASCADE,
    message_text  text,
    response_text text,
    recommended_products text[],
    recommendation_category text,
    ts            timestamptz DEFAULT now()
);
"""

def setup_schema():
    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(DDL_SQL)
    print("Database schema created/verified.")

def _to_bool(s: str) -> bool:
    s = (s or "").strip().lower()
    return s in {"true", "t", "1", "yes", "y"}

def ingest_data():
    conn = None
    try:
        conn = psycopg.connect(PG_DSN, autocommit=False) 
        with conn.cursor() as cur:
            #  Users 
            with open(settings.PG_Ingestion_Csv_Users, newline="") as f:
                reader = csv.reader(f)
                next(reader, None)  
                for row in reader:
                    user_id_str, name = row
                    user_id = int(user_id_str)
                    name = (name or "")[:200]
                    cur.execute(
                        "INSERT INTO users (user_id, name) VALUES (%s, %s)",
                        (user_id, name),
                    )
            print("Users staged.")

            #  Items 
            with open(settings.PG_Ingestion_Csv_Items, newline="") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    item_id_str, item_name, category, brand, description, price_str = row
                    item_id = int(item_id_str)
                    item_name = (item_name or "")[:400]    
                    category  = (category or "")[:200]
                    brand     = (brand or "")[:200]
                    description = (description or "")[:10000]
                    price = float(price_str)
                    cur.execute(
                        """
                        INSERT INTO items (
                            item_id, item_name, category, brand, description, price
                        )
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (item_id, item_name, category, brand, description, price),
                    )
            print("Items staged.")

            #  Interactions 
            with open(settings.PG_Ingestion_Csv_Interactions, newline="") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    purchase_id_str, user_id_str, item_id_str, rating_str, review_text, verified_str = row
                    purchase_id = int(purchase_id_str)
                    user_id     = int(user_id_str)
                    item_id     = int(item_id_str)
                    rating      = int(float(rating_str))
                    review_text = (review_text or "")[:10000]
                    verified    = _to_bool(verified_str)
                    cur.execute(
                        """
                        INSERT INTO interactions (
                            purchase_id, user_id, item_id, rating, review_text, verified
                        )
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (purchase_id, user_id, item_id, rating, review_text, verified),
                    )
            print("Interactions staged.")

        conn.commit()
        print("All data ingested successfully in one transaction.")

    except Exception as e:
        if conn is not None:
            conn.rollback()
        print(f"Ingestion failed, rolled back. Error: {e}")
        raise
    finally:
        if conn is not None:
            conn.close()


if __name__=='__main__':
    setup_schema()
    ingest_data()
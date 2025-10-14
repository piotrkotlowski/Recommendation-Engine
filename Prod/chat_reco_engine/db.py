import psycopg
from psycopg.rows import dict_row
from config import settings


class DatabaseError(Exception):
    """Custom exception for database errors."""
    pass


class Database:
    def __init__(self, dsn: str = settings.PG_DSN):
        self.dsn = dsn

    def _connect(self):
        """Create a new database connection."""
        return psycopg.connect(self.dsn)
    
    def get_purchased_items(self, user_id: int) -> list[int]:
        """ Fetch item Ids purchased by a given user."""
        try: 
            with self._connect() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        """
                        SELECT item_id from interactions 
                        where user_id = %s
                        """, (user_id,)
                    )
                    rows = cur.fetchall()
                    reviews = [str(row['item_id']) for row in rows]
            return reviews

        except psycopg.Error as e: 
            raise DatabaseError(f'Database operation failed: {e}')

    def get_users_history(self, user_id: int, limit: int = 5):
        """Fetch chat history for a given user."""
        try:
            with self._connect() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        """
                        SELECT message_text, response_text
                        FROM chat_history
                        WHERE user_id = %s
                        ORDER BY ts DESC
                        LIMIT %s
                        """,
                        (user_id, limit),
                    )
                    return cur.fetchall()
        except psycopg.Error as e:
            raise DatabaseError(f"Database operation failed: {e}")

    def add_chat_history(self,*, user_id: int, message_text: str, response_text: str,recommended_products: list[str],recommendation_category) -> None:
        """Insert a new chat history record."""
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO chat_history (user_id, message_text, response_text,recommended_products,recommendation_category)
                        VALUES (%s, %s, %s,%s,%s)
                        """,
                        (user_id, message_text, response_text,recommended_products,recommendation_category),
                    )
                    conn.commit()
        except psycopg.Error as e:
            raise DatabaseError(f"Database operation failed: {e}")

    def get_product_reviews(self, product_ids: list[int], limit: int = 5) -> dict[int, list[str]]:
       
        """Fetch random reviews for a list of product IDs."""
        reviews = {pid: None for pid in product_ids}
        try:
            with self._connect() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    for item_id in product_ids:
                        cur.execute(
                            """
                            SELECT review_text
                            FROM interactions 
                            WHERE item_id = %s
                            ORDER BY RANDOM()
                            LIMIT %s
                            """,
                            (int(item_id), limit),
                        )
                        rows = cur.fetchall()
                        reviews[item_id] = [row['review_text'] for row in rows]
            return reviews
        except psycopg.Error as e:
            raise DatabaseError(f"Database error: {e}")
        
    def get_last_category(self,user_id: int):
        """"
        Returning category of last searched products
        """
        try: 
            with self._connect() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        """
                        SELECT recommendation_category
                            FROM chat_history 
                            WHERE user_id = %s
                            ORDER BY ts DESC
                            LIMIT 1

                        """,(user_id,),
                    )
                    row = cur.fetchone()
                    print(row)
                return row["recommendation_category"] if row is not None else None 
        except psycopg.Error as e: 
             raise DatabaseError(f"Database error: {e}")



    def get_last_shown_products(self,user_id : int):
        """"
        Returning  last searched products
        """
        try: 
            with self._connect() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                        cur.execute(
                            """
                            SELECT recommended_products
                            FROM chat_history 
                            WHERE user_id = %s
                            ORDER BY ts DESC
                            LIMIT 1
                            """,
                            (user_id,),
                        )
                        row = cur.fetchone()
                        print(row)
            return  row["recommended_products"] if row is not None else None 
        except psycopg.Error as e:
            raise DatabaseError(f"Database error: {e}")
    

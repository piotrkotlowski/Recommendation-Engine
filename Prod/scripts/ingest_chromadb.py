from config import settings
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import json 
from typing import List, Dict, Any
import gc
import torch 


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for obj in data:
        yield obj

def create_rag_collection(client: chromadb.Client ,device: torch.device ):
  
    try:
        client.delete_collection(settings.RAG_CHROMA_COLLECTION)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=settings.RAG_CHROMA_COLLECTION,
        metadata={
            "hnsw:space": str(settings.RAG_CHROMA_METRIC),
            "hnsw:M": int(settings.RAG_HNSW_M),
            "hnsw:construction_ef": int(settings.RAG_HNSW_EF_CONSTRUCTION),
            "hnsw:search_ef": int(settings.RAG_HNSW_EF_SEARCH),
        },
        embedding_function =  SentenceTransformerEmbeddingFunction(
        model_name=settings.RAG_EMB_MODEL,
        device=str(device)
            )
    )

    print("Created collection")
   
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    print('Starting ingestion...')

    for rec in read_json(settings.RAG_PRODUCT_DATA_PATH):
        item_id=str(rec.get("item_id"))
        item_name = str(rec.get("item_name") or "").strip()
        brand     = str(rec.get("brand") or "").strip()
        category= str(rec.get("category") or "").strip()
        description= str(rec.get("description") or "").strip()
        price= float(rec.get("price"))
        avg_rating=float(rec.get("avg_rating"))
        num_ratings=int(rec.get("number_of_ratings"))

        prod_text = item_name + " " + brand + " " + description
        if prod_text:
            ids.append(item_id)
            metas.append({
                'item_name': item_name,
                'description': description,
                "category": category,
                "brand": brand,
                "price": price,
                "avg_rating": avg_rating,
                "num_ratings": num_ratings
            })
            docs.append(prod_text)
        

        if len(ids) >= settings.RAG_CHROMA_BATCH_SIZE:
            print(f'Ingesting batch... Collection count: {collection.count()}')
            collection.upsert(ids=ids, documents=docs, metadatas=metas)
            ids.clear(); docs.clear(); metas.clear()
            gc.collect()

    if ids:
        collection.upsert(ids=ids, documents=docs, metadatas=metas)

    print(f"Done.")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    client = chromadb.HttpClient(
        host="chroma_db",
        port=8000,
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    create_rag_collection(client,device)
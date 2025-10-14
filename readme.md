# Hybrid Recommendation System

This repository implements a **hybrid recommendation platform** with two components, both served via **FastAPI**.

---

## 1. ALS-based Recommendation Engine

* Uses the **Alternating Least Squares (ALS)** algorithm for collaborative filtering implemented in PyTorch.
* **Redis** is leveraged for fast retrieval of user embeddings.
* **ChromaDB** is used as a vector store for retrieving items based on embedding similarity (dot product).
* The ALS model is retrained every 30 minutes to adapt to new interactions and improve recommendation quality.
* The recommendation API is served via FastAPI through the `als_reco_engine` container.

---

## 2. Conversational Recommendation Engine (LLM + RAG)

* Integrates **Large Language Models (Mistral 7b)** with **Retrieval-Augmented Generation (RAG)** for context-aware, conversational recommendations.
* `all-MiniLM-L6-v2` is used for sentence embeddings, and `cross-encoder/ms-marco-MiniLM-L-6-v2` is used as a cross-encoder for reranking.
* **Ollama** is used to run and manage the LLM model for generating responses.
* Chat history is stored in **Postgres**.
* RAG queries support “WHERE” clauses to filter by brand, price, and minimum rating.
* The conversational API is served via FastAPI through the `chat_reco_engine` container.

---

## Containers in the Platform

| Container            | Role                                              |
| -------------------- | ------------------------------------------------- |
| `postgresdb`         | Stores user and chat data                         |
| `redis`              | Stores user embeddings for fast retrieval         |
| `chroma_db`          | Vector store for item embeddings                  |
| `ollama`             | Hosts and manages the LLM model                   |
| `als_trainer`        | Trains the ALS model periodically                 |
| `als_reco_engine`    | Serves ALS-based recommendations via FastAPI      |
| `chat_reco_engine`   | Serves conversational recommendations via FastAPI |
| `ingest_chroma_db`   | Ingests product embeddings into ChromaDB          |
| `ingest_postgres_db` | Ingests user and interaction data into Postgres   |
| `pull_ollama_model`  | Downloads and sets up the Ollama model            |

---

## How to Run the Application

To run the entire hybrid recommendation platform, navigate to the `Prod/` directory and run the `start_services.sh` script. This will build and launch the ALS engine, conversational chat engine, and all necessary data ingestion services automatically.

```bash
cd Prod
bash start_services.sh
```

Once the containers are up, the system will automatically initialize the ALS model training, set up Redis and Postgres for data storage, and launch the conversational recommendation API powered by the LLM + RAG pipeline.

---

## Project Structure

```
.
├── Modeling/                # Development/Research Artifacts
│   ├── Data/                # Data 
│   │   ├── Cleaned/         # Cleaned/Preprocessed data
│   │   │   ├── interactions.csv
│   │   │   ├── items.csv
│   │   │   ├── ProductsRAG.json
│   │   │   └── users.csv
│   │   └── Raw/             # Original, untouched source data
│   │       └── amazon_reviews.csv
│   ├── EDA/                 # Exploratory Data Analysis
│   │   └── 00_EDA.ipynb
│   └── RecoSystem/          # Core Recommendation System Development
│       ├── params/          # Model parameters/configs
│       │   └── als_params.json
│       ├── split/           # Train/Test data split
│       │   ├── test.csv
│       │   └── train.csv
│       ├── als_modeling.ipynb # ALS Model development and experimentation notebook
│       └── als_pytorch.py     # Prototype PyTorch ALS implementation
│
├── Prod/                    # Production/Deployment Code
│   ├── als_trainer/          # ALS training pipeline
│   │   ├── params/           # Hyperparameters/config files
│   │   ├── als_engine.py     # Core ALS training logic
│   │   ├── main.py           # Training entry point
│   │   └── pyTorch_als.py    # PyTorch ALS implementation
│   │
│   ├── als_reco_engine/      # ALS-based recommendation serving
│   │   ├── main.py           # Serving entry point
│   │   └── recommender.py    # Serving helper functions
│   │
│   ├── chat_reco_engine/     # Conversational recommendation engine
│   │   ├── __init__.py
│   │   ├── chat_engine.py    # LangChain helpers
│   │   ├── main.py           # Chat engine entry point
│   │   ├── db.py             # Postgres interaction functions
│   │   └── pydantic_helper.py # Validation schemas (Pydantic)
│   │
│   ├── data_ingest/          # Source datasets 
│   │   ├── interactions.csv  # User-item interactions
│   │   ├── items.csv         # Product metadata
│   │   ├── ProductsRAG.json  # Product catalog for RAG
│   │   └── users.csv         # User metadata
│   │
│   ├── docker_als_engine/    # Docker setup for ALS engine
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── docker_als_trainer/   # Docker setup for ALS trainer
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── docker_chat_engine/   # Docker setup for chat engine
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── docker_ingest_chroma/ # Docker setup for ingesting chroma_db
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── docker_ingest_pg/     # Docker setup for ingesting postgres_db
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── examples/             # usage_examples
│   │   ├── main_example_als.py
│   │   └── main_example_chat.py
│   │
│   └── scripts/              # ingestion scripts
│       ├── ingest_chromadb.py
│       └── ingest_postgres.py
│
├── .env                     # Environment variables
├── config.py                 # Global configuration file
├── docker-compose.yaml       # Orchestration for multi-container services
└── start_services.sh         # Script to start services
```

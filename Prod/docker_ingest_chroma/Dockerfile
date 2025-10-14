ARG PYTHON_IMAGE=python:3.11-slim
FROM ${PYTHON_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/ingest     

WORKDIR /ingest 
COPY docker_ingest_chroma/requirements.txt .

RUN pip install --timeout 600  --no-cache-dir --upgrade pip && pip install -r requirements.txt
RUN pip install torch==2.8.0  --extra-index-url https://download.pytorch.org/whl/cu116

COPY scripts/ingest_chromadb.py /ingest/
COPY config.py /ingest/config.py
CMD ["python", "ingest_chromadb.py"]
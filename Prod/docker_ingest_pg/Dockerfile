ARG PYTHON_IMAGE=python:3.11-slim
FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/ingest     

WORKDIR /ingest 
COPY docker_ingest_pg/requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY scripts/ingest_postgres.py /ingest/
COPY config.py /ingest/config.py
CMD ["python", "ingest_postgres.py"]
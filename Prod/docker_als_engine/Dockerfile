ARG PYTHON_IMAGE=python:3.11-slim
FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app        

WORKDIR /app

COPY docker_als_engine/requirements.txt .

RUN pip install --upgrade pip --timeout 600


RUN pip install --timeout 600 -r requirements.txt
RUN pip install torch==1.13.1+cu116  --extra-index-url https://download.pytorch.org/whl/cu116
COPY als_reco_engine/ /app/als_reco_engine/
COPY config.py /app/config.py
EXPOSE 8000

CMD ["uvicorn", "als_reco_engine.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers","4"]


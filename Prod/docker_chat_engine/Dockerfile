FROM  python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app        

WORKDIR /app

COPY docker_chat_engine/requirements.txt .

RUN pip install --upgrade pip --timeout 600

RUN pip install --timeout 1200 -r requirements.txt

RUN pip install torch==2.8.0+cpu  --index-url https://download.pytorch.org/whl/cpu

COPY chat_reco_engine/ /app/chat_reco_engine/
COPY config.py /app/config.py
EXPOSE 8000

CMD ["uvicorn", "chat_reco_engine.main:app", "--host", "0.0.0.0", "--port", "8000"]


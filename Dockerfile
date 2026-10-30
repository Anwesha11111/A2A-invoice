FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
ENV DB_PATH=/data/agent.db
RUN mkdir -p /data
EXPOSE 8000
# Single worker on purpose: idempotency locks are in-process.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]

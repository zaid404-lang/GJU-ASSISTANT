FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY index.py api.py eval.py store.py check.py ./
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8787 \
    HOST=0.0.0.0

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pass --build-arg INSTALL_NEURAL=1 to bake bge-small-en-v1.5 into the image (0.714 SciFact nDCG@10).
ARG INSTALL_NEURAL=0
RUN if [ "$INSTALL_NEURAL" = "1" ]; then \
      pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch sentence-transformers && \
      python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')"; \
    fi

COPY . .

EXPOSE 8787
CMD ["sh", "-c", "python3 backend/cli.py --host ${HOST} --port ${PORT}"]

# One image for the whole product: the API, the local embedding model, and the
# built React app, served from a single origin on $PORT.
#
#   docker build -t gapfolio .                      (from this directory)
#   docker run -p 8000:8000 -e GAPFOLIO_SECRET_KEY=$(openssl rand -hex 32) gapfolio
#
# Memory: ~170 MB for the model (ONNX Runtime, no PyTorch), so the whole service
# fits a 512 MB host such as Render's free tier. Not yet built in this repo's CI:
# the development machine has no Docker.

# ---- 1. build the React app -------------------------------------------------
FROM node:22-slim AS ui
WORKDIR /ui
COPY gapfolio-ui/package.json gapfolio-ui/package-lock.json ./
RUN npm ci
COPY gapfolio-ui/ ./
RUN npm run build

# ---- 2. the API + model -----------------------------------------------------
FROM python:3.12-slim
WORKDIR /srv/gapfolio
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/srv/hf

COPY gapfolio/requirements.txt .
RUN pip install -r requirements.txt

# Bake the ONNX model into the image so a cold start does not download 90 MB.
RUN python -c "from huggingface_hub import hf_hub_download as d; r='sentence-transformers/all-MiniLM-L6-v2'; d(r,'onnx/model.onnx'); d(r,'tokenizer.json')"

COPY gapfolio/app ./app
COPY gapfolio/scripts ./scripts
COPY gapfolio/db ./db
# app/main.py serves ../gapfolio-ui/dist relative to this directory.
COPY --from=ui /ui/dist /srv/gapfolio-ui/dist

# Set GAPFOLIO_SECRET_KEY at run time. Without GAPFOLIO_DATABASE_URL it uses
# SQLite in /tmp, which is lost on restart: point it at Postgres
# (postgresql+psycopg://...) for anything you want to keep.
ENV GAPFOLIO_ENV=production GAPFOLIO_EMBEDDING_PROVIDER=onnx GAPFOLIO_EMBEDDING_DIM=384
ENV GAPFOLIO_DATABASE_URL=sqlite:////tmp/gapfolio.db
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

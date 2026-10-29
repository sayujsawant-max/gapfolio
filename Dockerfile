# One image for the whole product: the API, the local SBERT model, and the built
# React app, served from a single origin on $PORT.
#
#   docker build -t gapfolio .                      (from this directory)
#   docker run -p 8000:8000 -e GAPFOLIO_SECRET_KEY=$(openssl rand -hex 32) gapfolio
#
# Hosts that read a Dockerfile (Render, Railway, Fly.io, Hugging Face Spaces in
# Docker mode) need nothing else. Not yet built or run in this repo's CI: the
# development machine has no Docker.

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
# CPU-only PyTorch: the default wheel pulls ~2 GB of CUDA libraries the model never uses.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# Bake the model into the image so a cold start does not download 90 MB.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY gapfolio/app ./app
COPY gapfolio/scripts ./scripts
COPY gapfolio/db ./db
# app/main.py serves ../gapfolio-ui/dist relative to this directory.
COPY --from=ui /ui/dist /srv/gapfolio-ui/dist

# Set GAPFOLIO_SECRET_KEY at run time. Without GAPFOLIO_DATABASE_URL it uses
# SQLite inside the container, which is lost on redeploy: attach a volume or point
# it at Postgres (postgresql+psycopg://...) for anything you want to keep.
ENV GAPFOLIO_ENV=production GAPFOLIO_EMBEDDING_PROVIDER=sbert GAPFOLIO_EMBEDDING_DIM=384
# Hosts like Hugging Face Spaces run the container as a non-root user and only
# /tmp is writable; override GAPFOLIO_DATABASE_URL for anything you want to keep.
ENV GAPFOLIO_DATABASE_URL=sqlite:////tmp/gapfolio.db
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

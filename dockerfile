# syntax=docker/dockerfile:1

########################
# Stage 1: Builder
########################
FROM python:3.11-slim AS builder

WORKDIR /build

# Build-time deps needed to compile some wheels (faiss/spacy deps etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install torch (CPU-only, via --index-url) and the rest of requirements.txt
# in a SINGLE resolver call. Installing them in separate pip invocations lets
# each call resolve numpy independently (torch's default vs. scipy/sklearn's
# default pulled in by sentence-transformers), which produced two
# incompatible numpy installs stomping on each other in the same --prefix and
# broke at runtime with "numpy._core.multiarray failed to import". One call
# with both indexes lets pip's resolver pick one consistent numpy for everyone.
RUN pip install --no-cache-dir --prefix=/install \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    torch==2.3.1 -r requirements.txt

# Download the small spaCy English model (used by StructureAwareIngestor).
# PYTHONPATH must point at the prefix populated above so pip can see the
# numpy/spacy versions already installed there - otherwise it can't tell
# they're satisfied and reinstalls a second, conflicting numpy on top,
# corrupting the site-packages (mixed numpy 1.x/2.x files, ABI errors at import).
RUN PYTHONPATH=/install/lib/python3.11/site-packages pip install --no-cache-dir --prefix=/install \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# Strip unnecessary bulk from installed packages to save space:
# - test directories, __pycache__, .dist-info RECORD caches, static libs
RUN find /install -type d -name "tests" -exec rm -rf {} + 2>/dev/null; \
    find /install -type d -name "test" -exec rm -rf {} + 2>/dev/null; \
    find /install -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
    find /install -name "*.pyc" -delete; \
    find /install -name "*.a" -delete; \
    find /install -type d -name "torch" -prune -false -o -name "*.so" -print | xargs -r strip --strip-unneeded 2>/dev/null || true; \
    rm -rf /install/lib/python3.11/site-packages/torch/test 2>/dev/null; \
    rm -rf /install/lib/python3.11/site-packages/torch/include 2>/dev/null; \
    rm -rf /install/lib/python3.11/site-packages/torch/utils/model_dump 2>/dev/null; \
    rm -rf /install/share 2>/dev/null


########################
# Stage 2: Runtime
########################
FROM python:3.11-slim AS runtime

# Minimal runtime system deps (pdfplumber/faiss need libgomp; ssl certs for requests)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pre-built, trimmed site-packages from builder
COPY --from=builder /install/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application source (includes ./local_cross_encoder — must be present
# in the build context; it's baked into the image per requirement)
COPY . .

# data/ is intentionally NOT copied — mount it as a volume at runtime:
#   docker run -v $(pwd)/data:/app/data ...
RUN mkdir -p /app/data

# Pre-download the sentence-transformers embedding model at BUILD time
# (before offline flags are set below) so the container can run fully
# offline at runtime without a first-run fetch. Requires network access
# during the build.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Cache dir the model above was just downloaded into (default ~/.cache/huggingface)
# is preserved as part of this layer/image.

# Offline / CPU-only env flags (mirrors what main.py sets, kept for early init too)
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    SENTENCE_TRANSFORMERS_BACKEND=torch \
    USE_TF=0 \
    TRANSFORMERS_NO_TF_IMPORT=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

VOLUME ["/app/data"]

ENTRYPOINT ["python", "main.py"]
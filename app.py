"""
Flask web frontend for the Unified Multi-RAG Pipeline.

Wraps the existing CLI pipeline (main.py) as a single in-process singleton
and exposes it over a small JSON API for the static frontend in `static/`.
Run with: python app.py
"""

# Import main first - it sets offline-mode env vars (HF_HUB_OFFLINE, blocks
# tensorflow, etc.) at module import time, before anything else touches
# sentence_transformers/transformers.
import main as cli

import os
import threading
import logging

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

import preprocessing

logger = logging.getLogger(__name__)

ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".eml"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25MB

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

if not os.path.exists("data"):
    os.makedirs("data")

cli.console.print("\n[bold blue]Initializing Unified RAG Pipeline (web)...[/]\n")
pipeline = cli.UnifiedRAGPipeline(data_folder="data")

_reindex_lock = threading.Lock()
_reindex_state = {"status": "idle", "detail": ""}


def _reindex_all():
    """Rebuilds KG/entities, FAISS, and BM25 from the current data/ folder.
    Runs on a background thread; callers should already hold _reindex_lock
    (acquired by the caller before starting the thread, released here when done)."""
    try:
        _reindex_state.update(status="running", detail="Re-running structure-aware ingestion...")
        pipeline.run_ingestion()

        _reindex_state.update(detail="Rebuilding vector index (FAISS)...")
        chunks = preprocessing.rebuild_faiss_index(pipeline.data_folder, embedder=cli.shared_encoder)

        if pipeline.vector_model:
            pipeline.vector_model.rebuild()

        if pipeline.bm25_model:
            _reindex_state.update(detail="Rebuilding BM25 index...")
            pipeline.bm25_model._load_documents_from_chunks(chunks["doc_chunks"], chunks["sources"])
            pipeline.bm25_model._build_index()

        _reindex_state.update(status="idle", detail=f"Indexed {chunks['num_chunks']} chunks.")
    except Exception as e:
        logger.exception("Reindex failed")
        _reindex_state.update(status="idle", detail=f"Reindex failed: {e}")
    finally:
        _reindex_lock.release()


def _start_reindex():
    """Acquires the reindex lock and starts _reindex_all in the background.
    Returns False without doing anything if a reindex is already running."""
    if not _reindex_lock.acquire(blocking=False):
        return False
    threading.Thread(target=_reindex_all, daemon=True).start()
    return True


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.before_request
def _reject_during_reindex():
    if request.path in ("/api/query", "/api/plan", "/api/compare") and _reindex_lock.locked():
        return jsonify({"error": "Reindex in progress, please retry shortly."}), 503


@app.route("/api/query", methods=["POST"])
def api_query():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400

    answer, confidence, elapsed, plan = pipeline.process_query(query)
    return jsonify({
        "answer": answer,
        "confidence": confidence,
        "elapsed": elapsed,
        "category": plan.get("category"),
    })


@app.route("/api/plan", methods=["POST"])
def api_plan():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400

    answer, confidence, elapsed, plan = pipeline.process_query(query, show_plan=True)
    return jsonify({
        "answer": answer,
        "confidence": confidence,
        "elapsed": elapsed,
        "plan": plan,
    })


@app.route("/api/compare", methods=["POST"])
def api_compare():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400

    results, winner = pipeline._compute_comparison(query)
    # Drop the raw retrieved chunk text - only needed for scoring, not the UI,
    # and can be large enough to bloat the response.
    slim_results = {
        name: {k: v for k, v in r.items() if k != "context"}
        for name, r in results.items()
    }
    return jsonify({"results": slim_results, "winner": winner})


@app.route("/api/ingest", methods=["POST"])
def api_ingest():
    if not _start_reindex():
        return jsonify({"status": "already_running"}), 409
    return jsonify({"status": "started"}), 202


@app.route("/api/scrape", methods=["POST"])
def api_scrape():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "url is required"}), 400
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        scraper = cli.PDFScraper(url, data_folder=pipeline.data_folder)
        saved = scraper.scrape_all_pdfs(delay=1)
    except Exception as e:
        return jsonify({"error": f"Scrape failed: {e}"}), 500

    reindexing = _start_reindex() if saved else False
    return jsonify({"saved_files": saved, "reindexing": reindexing})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files provided"}), 400

    saved = []
    rejected = []
    for f in files:
        if not f.filename:
            continue
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in ALLOWED_UPLOAD_EXTENSIONS:
            rejected.append(f.filename)
            continue
        safe_name = secure_filename(f.filename)
        f.save(os.path.join(pipeline.data_folder, safe_name))
        saved.append(safe_name)

    if rejected and not saved:
        return jsonify({
            "error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_UPLOAD_EXTENSIONS))}",
            "rejected": rejected,
        }), 400

    reindexing = _start_reindex() if saved else False
    return jsonify({"saved_files": saved, "rejected_files": rejected, "reindexing": reindexing})


@app.route("/api/reindex-status")
def api_reindex_status():
    return jsonify(_reindex_state)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)

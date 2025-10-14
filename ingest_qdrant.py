#!/usr/bin/env python3
import os
import json
import time
import re
import uuid
import glob
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import requests
import numpy as np

# Qdrant client
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# Sentence-transformers fallback (optional)
try:
    from sentence_transformers import SentenceTransformer
    EMB_FALLBACK_MODEL = SentenceTransformer(os.getenv("EMBED_FALLBACK_MODEL", "all-mpnet-base-v2"))
except Exception:
    EMB_FALLBACK_MODEL = None

# Optional tiktoken for token counting
try:
    import tiktoken
    TOK_ENCODER = tiktoken.get_encoding("cl100k_base")
    def token_len(text: str) -> int:
        return len(TOK_ENCODER.encode(text))
except Exception:
    TOK_ENCODER = None
    def token_len(text: str) -> int:
        return len(text.split())

# --- CONFIG ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL_OLLAMA = os.getenv("EMBED_MODEL_OLLAMA", "nomic-embed-text:latest")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "regulatory_summaries")
TARGET_DIM = int(os.getenv("EMBED_DIM", "1536"))
TOPIC_FOLDER = os.getenv("TOPIC_FOLDER", "topic-wise-extraction")
FALLBACK_METHOD = os.getenv("FALLBACK_METHOD", "proj_random")  # error | pad | proj_random

# --------------- Helpers ---------------

def extract_first_json_block(text: str) -> Dict[str, Any]:
    start = text.find('{')
    if start == -1:
        raise ValueError("No JSON object found in text.")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try:
                    return json.loads(candidate)
                except Exception:
                    m = re.search(r"\{[\s\S]*\}", text)
                    if m:
                        return json.loads(m.group(0))
                    raise
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return json.loads(m.group(0))
    raise ValueError("Could not extract JSON from text.")


def simple_fallback_summarizer(text: str, max_sentences: int = 4) -> Tuple[str, str]:
    if not text:
        return ("", "")
    sents = re.split(r'(?<=[\.?\!])\s+', text.strip())
    short = sents[0] if sents else text.strip()[:200]
    long = " ".join(sents[:max_sentences]) if sents else text.strip()[:800]
    return short, long

# --------------- Embeddings ---------------

_session = requests.Session()
_session.headers.update({"Content-Type": "application/json"})


def _ollama_embed_single(text: str, model: str, host: str) -> List[float]:
    """
    Robust call to Ollama embeddings.
    Prefers the official schema: {model, prompt}. Falls back to {model, input}.
    Returns a single embedding list of floats.
    """
    url = f"{host.rstrip('/')}/api/embeddings"

    # Try official format
    try:
        r = _session.post(url, json={"model": model, "prompt": text}, timeout=120)
        r.raise_for_status()
        resp = r.json()
        if isinstance(resp, dict) and "embedding" in resp:
            emb = resp["embedding"]
            return [float(x) for x in emb]
        # Some servers may return {data: [{embedding: ...}]}
        if isinstance(resp, dict) and "data" in resp and resp["data"]:
            first = resp["data"][0]
            if "embedding" in first:
                return [float(x) for x in first["embedding"]]
        # Some variants may return {embeddings: [[...]]}
        if isinstance(resp, dict) and "embeddings" in resp and resp["embeddings"]:
            return [float(x) for x in resp["embeddings"][0]]
    except Exception:
        pass

    # Fallback to {input: text}
    r = _session.post(url, json={"model": model, "input": text}, timeout=120)
    r.raise_for_status()
    resp = r.json()
    if isinstance(resp, dict) and "embedding" in resp:
        return [float(x) for x in resp["embedding"]]
    if isinstance(resp, dict) and "data" in resp and resp["data"]:
        first = resp["data"][0]
        if "embedding" in first:
            return [float(x) for x in first["embedding"]]
    if isinstance(resp, dict) and "embeddings" in resp and resp["embeddings"]:
        return [float(x) for x in resp["embeddings"][0]]

    raise RuntimeError("Ollama embeddings response missing embedding.")


def ollama_embeddings_or_raise(texts: List[str], model: str = EMBED_MODEL_OLLAMA, host: str = OLLAMA_HOST) -> List[List[float]]:
    embs: List[List[float]] = []
    for t in texts:
        emb = _ollama_embed_single(t, model, host)
        embs.append(emb)
    if len(embs) != len(texts):
        raise RuntimeError("Ollama returned unexpected number of embeddings.")
    return embs


def get_embeddings_or_fallback(texts: List[str]) -> List[List[float]]:
    try:
        return ollama_embeddings_or_raise(texts)
    except Exception as e:
        print(f"Ollama embed failed: {e}")
        if EMB_FALLBACK_MODEL is None:
            raise
        print("Falling back to sentence-transformers for embeddings.")
        embs_np = EMB_FALLBACK_MODEL.encode(texts, convert_to_numpy=True)
        return [list(map(float, v.tolist())) for v in np.atleast_2d(embs_np)]


_proj_matrix_cache: Dict[Tuple[int, int], np.ndarray] = {}

def ensure_embedding_dim(vec: List[float], target_dim: int = TARGET_DIM, method: str = "proj_random") -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    src = v.shape[0]
    if src == target_dim:
        return v
    if method == "error":
        raise ValueError(f"Embedding dim {src} != required {target_dim}.")
    if method == "pad":
        out = np.zeros((target_dim,), dtype=np.float32)
        out[: min(src, target_dim)] = v[: min(src, target_dim)]
        return out
    if method == "proj_random":
        key = (src, target_dim)
        if key not in _proj_matrix_cache:
            rng = np.random.RandomState(42)
            R = rng.normal(size=(target_dim, src)).astype(np.float32)
            try:
                q, _ = np.linalg.qr(R)
                R = q.astype(np.float32)
            except Exception:
                pass
            _proj_matrix_cache[key] = R
        R = _proj_matrix_cache[key]
        return R.dot(v.astype(np.float32))
    raise ValueError("Unknown method")

# --------------- IDs ---------------

def deterministic_summary_id(doc_id: str, topic_id: str, change_id: str = "", extra: str = "") -> str:
    ns = uuid.UUID("12345678-1234-5678-1234-567812345678")
    name = f"{doc_id}||{topic_id}||{change_id}||{extra}"
    return str(uuid.uuid5(ns, name))


def choose_point_id(summary_obj: Dict[str, Any], file_path: Optional[str] = None, index_in_file: Optional[int] = None) -> str:
    existing = summary_obj.get("summaryId") or summary_obj.get("summary_id")
    if existing:
        try:
            u = uuid.UUID(existing)
            return str(u)
        except Exception:
            pass
    doc_id = summary_obj.get("docId") or summary_obj.get("doc_id") or "unknown_doc"
    topic_id = summary_obj.get("topicId") or summary_obj.get("topic_id") or "unknown_topic"
    change_id = summary_obj.get("changeId") or summary_obj.get("change_id") or ""
    extra = ""
    if file_path:
        extra = os.path.basename(file_path)
    if index_in_file is not None:
        extra = f"{extra}::idx{index_in_file}"
    return deterministic_summary_id(doc_id, topic_id, change_id, extra)

# --------------- Normalization ---------------

def _compose_candidate_long(s: Dict[str, Any]) -> str:
    candidate_long = (
        s.get("summary_long")
        or s.get("topic_summary_long")
        or s.get("topic_summary")
        or s.get("chunk_summary_long")
        or s.get("text")
        or s.get("full_text")
        or s.get("extracted_text")
        or s.get("page_text")
        or s.get("content")
        or ""
    )

    # If candidate_long empty but there are chunk entries, join them
    if not candidate_long and isinstance(s.get("chunks"), list):
        candidate_long = " ".join([
            c.get("chunk_summary_long", "")
            for c in s.get("chunks", [])
            if isinstance(c, dict) and c.get("chunk_summary_long")
        ])

    # If still empty, build from metadata
    if not candidate_long:
        metadata_parts: List[str] = []
        preferred_keys = [
            "title",
            "name",
            "filename",
            "author",
            "page_title",
            "section",
            "s3_link",
            "url",
            "page_number",
            "page_count",
            "abstract",
        ]
        for k in preferred_keys:
            v = s.get(k)
            if v is None:
                continue
            if isinstance(v, (list, tuple)):
                v = ", ".join(map(str, v))
            if isinstance(v, (int, float)):
                v = str(v)
            if isinstance(v, str) and v.strip():
                metadata_parts.append(f"{k}: {v.strip()}")
        for k, v in s.items():
            if k in preferred_keys:
                continue
            if isinstance(v, str) and 30 <= len(v) <= 1000:
                metadata_parts.append(f"{k}: {v.strip()[:400]}")
        if metadata_parts:
            candidate_long = " | ".join(metadata_parts)

    if not candidate_long:
        minimal = {
            k: s.get(k)
            for k in (
                "docId",
                "topicId",
                "source",
                "source_file",
                "s3_link",
                "page_number",
                "page_count",
            )
            if s.get(k) is not None
        }
        candidate_long = (
            "metadata: " + json.dumps(minimal, ensure_ascii=False)
            if minimal
            else "metadata only - no readable text available."
        )
    return candidate_long


def normalize_summary_obj(raw: Dict[str, Any], file_path: str, index_in_file: int) -> Dict[str, Any]:
    s = dict(raw)

    # canonical doc/topic ids (fall back to filename if missing)
    s.setdefault("docId", s.get("doc_id") or s.get("docId") or os.path.splitext(os.path.basename(file_path))[0])
    s.setdefault("topicId", s.get("topic_id") or s.get("topicId") or s.get("topic") or s["docId"])

    candidate_long = _compose_candidate_long(s)

    # ensure normalized fields
    s["summary_long"] = candidate_long or s.get("summary_long", "")
    s["summary_short"] = (
        s.get("summary_short")
        or s.get("topic_summary_short")
        or s.get("chunk_summary_short")
        or (candidate_long[:250] if candidate_long else "")
        or s.get("summary", "")
    )

    # If still empty, synthesise minimal summaries
    if not s.get("summary_short") and not s.get("summary_long"):
        short, long = simple_fallback_summarizer(candidate_long)
        s["summary_short"] = short
        s["summary_long"] = long

    # Add a friendly alias expected by some consumers
    s["summary"] = s.get("summary") or s.get("summary_short") or s.get("summary_long") or ""

    # created_at
    s.setdefault("created_at", s.get("created_at") or datetime.utcnow().isoformat() + "Z")

    # provenance
    prov = s.get("provenance", {})
    if isinstance(prov, dict):
        prov.setdefault("source_file", os.path.basename(file_path))
        prov.setdefault("index_in_file", index_in_file)
        s["provenance"] = prov
    else:
        s["provenance"] = {"source_file": os.path.basename(file_path), "index_in_file": index_in_file}

    # external id
    s.setdefault("external_id", f"{s.get('docId')}::{s.get('topicId')}::{index_in_file}")

    return s

# --------------- Qdrant ---------------

def ensure_qdrant_collection(client: QdrantClient, collection_name: str, dim: int) -> None:
    coll_info = None
    try:
        coll_info = client.get_collection(collection_name)
    except Exception:
        coll_info = None

    if coll_info is None:
        print("Collection not present, creating:", collection_name)
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        return

    curr_dim = None
    try:
        curr_dim = coll_info.vectors_config.params.size
    except Exception:
        try:
            curr_dim = coll_info.vectors_config["params"]["size"]
        except Exception:
            curr_dim = None

    if curr_dim is None:
        print("Could not determine existing collection dim. Recreating to be safe.")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    elif curr_dim != dim:
        print(f"Collection '{collection_name}' dim {curr_dim} != desired {dim}. Recreating (deletes data).")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    else:
        print(f"Using existing collection '{collection_name}' with dim {dim}.")

# --------------- Ingest ---------------

def build_embed_text(summary_obj: Dict[str, Any]) -> str:
    long_text = (summary_obj.get("summary_long") or "").strip()

    actions_list = summary_obj.get("actionable_items") or summary_obj.get("action_items") or []
    acts = " ; ".join([
        a.get("description", a.get("action_text", ""))
        for a in actions_list
        if isinstance(a, dict)
    ]).strip()

    kw_list = summary_obj.get("keywords") or summary_obj.get("topics") or []
    if isinstance(kw_list, (list, tuple)):
        kw = ", ".join(map(str, kw_list))
    else:
        kw = str(kw_list or "")
    kw = kw.strip()

    prov = summary_obj.get("provenance", {})
    prov_text = ""
    if isinstance(prov, dict):
        prov_parts = []
        for pk in ("s3_link", "s3", "url", "page", "page_number", "page_range", "source_file"):
            if prov.get(pk):
                prov_parts.append(f"{pk}:{prov.get(pk)}")
        prov_text = " | ".join(prov_parts)

    pieces = []
    if long_text:
        pieces.append(long_text)
    if acts:
        pieces.append("Actions: " + acts)
    if kw:
        pieces.append("Keywords: " + kw)
    if prov_text:
        pieces.append("Provenance: " + prov_text)

    embed_text = "\n\n".join(pieces)

    # lightly truncate extremely long inputs to keep embedding cost bounded
    if TOK_ENCODER is not None:
        # rough cap ~2048 tokens
        if token_len(embed_text) > 2048:
            embed_text = embed_text[: 8000]  # crude char cap as last resort
    else:
        if len(embed_text) > 20000:
            embed_text = embed_text[:20000]

    return embed_text


def ingest():
    qdrant_client = QdrantClient(url=QDRANT_URL)
    ensure_qdrant_collection(qdrant_client, QDRANT_COLLECTION, TARGET_DIM)

    pattern = os.path.join(TOPIC_FOLDER, "*.json")
    files = glob.glob(pattern)
    print("Found topic JSON files:", len(files))
    if not files:
        raise SystemExit("No files found in TOPIC_FOLDER: " + TOPIC_FOLDER)

    seen_point_ids = set()
    upserted: List[str] = []
    errors: List[Tuple[str, str]] = []

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception as e:
            errors.append((path, f"JSON load error: {e}"))
            continue

        summaries = doc if isinstance(doc, list) else [doc]
        for idx, raw in enumerate(summaries):
            try:
                summary_obj = normalize_summary_obj(raw, path, idx)

                # Final guard: ensure at least one non-empty summary field
                if not (summary_obj.get("summary_long") or summary_obj.get("summary_short")):
                    short, long = simple_fallback_summarizer(summary_obj.get("summary", ""))
                    summary_obj["summary_short"] = summary_obj.get("summary_short") or short
                    summary_obj["summary_long"] = summary_obj.get("summary_long") or long

                # Ensure alias 'summary' exists (some downstream UIs expect this)
                summary_obj["summary"] = (
                    summary_obj.get("summary")
                    or summary_obj.get("summary_short")
                    or summary_obj.get("summary_long")
                    or ""
                )

                # Build embed text
                embed_text = build_embed_text(summary_obj)
                if not embed_text.strip():
                    errors.append((path, f"Empty embed text for index {idx}; skipping"))
                    continue

                # Get embedding (Ollama -> fallback)
                try:
                    emb = get_embeddings_or_fallback([embed_text])[0]
                except Exception as e:
                    errors.append((path, f"Embedding failed: {e}"))
                    continue

                # Ensure vector dim
                emb_fixed = ensure_embedding_dim(emb, target_dim=TARGET_DIM, method=FALLBACK_METHOD)

                # Choose valid UUID id (include index to avoid collisions)
                point_id = choose_point_id(summary_obj, file_path=path, index_in_file=idx)
                if point_id in seen_point_ids:
                    point_id = str(uuid.uuid4())
                seen_point_ids.add(point_id)

                # Keep a friendly external id in payload
                summary_obj.setdefault("external_id", f"{summary_obj.get('docId')}::{summary_obj.get('topicId')}::{idx}")

                payload = dict(summary_obj)

                # Upsert the point
                try:
                    qdrant_client.upsert(
                        collection_name=QDRANT_COLLECTION,
                        points=[
                            {
                                "id": point_id,
                                "vector": emb_fixed.tolist(),
                                "payload": payload,
                            }
                        ],
                    )
                    upserted.append(point_id)
                    print("Upserted:", point_id, "external:", payload.get("external_id"), "has summary:", bool(payload.get("summary")))
                except Exception as qerr:
                    errors.append((path, f"Qdrant upsert error for id {point_id}: {qerr}"))
                    print("Qdrant upsert error:", qerr)

                time.sleep(0.02)

            except Exception as e:
                errors.append((path, str(e)))
                print("Error processing", path, e)

    print("Done. Upserted:", len(upserted), "errors:", len(errors))
    if errors:
        print("Sample errors:", errors[:8])


if __name__ == "__main__":
    ingest()

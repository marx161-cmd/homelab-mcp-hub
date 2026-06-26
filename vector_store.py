from __future__ import annotations

import hashlib
import fnmatch
import glob
import gc
import json
import math
import os
import re
import sqlite3
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


TEXT_EXTENSIONS = {
    ".md",
    ".txt",
    ".conf",
    ".cfg",
    ".ini",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".py",
    ".sh",
    ".kt",
    ".kts",
    ".java",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".css",
    ".html",
    ".xml",
    ".service",
    ".timer",
}

IGNORE_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".gradle",
    "build",
    "dist",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".cache",
    ".opencode",
    ".continue",
    ".claude",
    ".gemini",
    ".vscode",
    ".rocprofv3",
    ".venvs",
    "tmp",
    "out",
    "logs",
    "models",
    "image_cache",
    "releases",
    "termux-build-logs",
    "binary-builds",
    "runtime-model-links",
    "googlcloud",
}


@dataclass(frozen=True)
class VectorConfig:
    chroma_dir: str
    collection_name: str
    embedding_backend: str
    embedding_model: str
    embedding_device: str
    embedding_batch_size: int
    ollama_url: str
    indexed_roots: tuple[str, ...]
    max_file_bytes: int
    max_chunk_chars: int
    chunk_overlap_lines: int
    incremental_enabled: bool
    btrfs_subvolume: str
    state_path: str
    embedding_unload_cooldown_seconds: int


def vector_config(config: Any) -> VectorConfig:
    return VectorConfig(
        chroma_dir=str(getattr(config, "chroma_dir", "/home/comrade/.local/share/mcp-hub/chroma")),
        collection_name=str(getattr(config, "vector_collection", "homelab_knowledge")),
        embedding_backend=str(getattr(config, "embedding_backend", "sentence-transformers")),
        embedding_model=str(getattr(config, "embedding_model", "Qwen/Qwen3-Embedding-0.6B")),
        embedding_device=str(getattr(config, "embedding_device", "cuda:0")),
        embedding_batch_size=int(getattr(config, "embedding_batch_size", 8)),
        ollama_url=str(getattr(config, "ollama_url", "http://127.0.0.1:11434")),
        indexed_roots=tuple(
            getattr(
                config,
                "indexed_roots",
                (
                    "/home/comrade/homelab",
                    "/home/comrade/bin",
                ),
            )
        ),
        max_file_bytes=int(getattr(config, "vector_max_file_bytes", 1_500_000)),
        max_chunk_chars=int(getattr(config, "vector_max_chunk_chars", 3_500)),
        chunk_overlap_lines=int(getattr(config, "vector_chunk_overlap_lines", 2)),
        incremental_enabled=bool(getattr(config, "vector_incremental_enabled", True)),
        btrfs_subvolume=str(getattr(config, "vector_btrfs_subvolume", "/home")),
        state_path=str(getattr(config, "vector_state_path", "/home/comrade/.local/share/mcp-hub/index-state.json")),
        embedding_unload_cooldown_seconds=int(getattr(config, "embedding_unload_cooldown_seconds", 60)),
    )


_EMBEDDER_LOCK = threading.RLock()
_EMBEDDER: Any | None = None
_EMBEDDER_KEY: tuple[str, str] | None = None
_EMBEDDER_UNLOAD_TIMER: threading.Timer | None = None


def load_embedder(model_name: str, device: str | None = None):
    from sentence_transformers import SentenceTransformer

    kwargs: dict[str, Any] = {"trust_remote_code": True}
    if device:
        kwargs["device"] = device
    return SentenceTransformer(model_name, **kwargs)


def load_embedder_for_config(cfg: VectorConfig):
    return load_embedder(cfg.embedding_model, cfg.embedding_device)


def _cleanup_torch_memory() -> None:
    gc.collect()
    try:
        import torch
    except Exception:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def get_cached_embedder(cfg: VectorConfig):
    global _EMBEDDER, _EMBEDDER_KEY, _EMBEDDER_UNLOAD_TIMER
    key = (cfg.embedding_model, cfg.embedding_device)
    with _EMBEDDER_LOCK:
        if _EMBEDDER_UNLOAD_TIMER is not None:
            _EMBEDDER_UNLOAD_TIMER.cancel()
            _EMBEDDER_UNLOAD_TIMER = None
        if _EMBEDDER is None or _EMBEDDER_KEY != key:
            _EMBEDDER = None
            _cleanup_torch_memory()
            _EMBEDDER = load_embedder_for_config(cfg)
            _EMBEDDER_KEY = key
        return _EMBEDDER


def unload_cached_embedder() -> None:
    global _EMBEDDER, _EMBEDDER_KEY, _EMBEDDER_UNLOAD_TIMER
    with _EMBEDDER_LOCK:
        if _EMBEDDER_UNLOAD_TIMER is not None:
            _EMBEDDER_UNLOAD_TIMER.cancel()
            _EMBEDDER_UNLOAD_TIMER = None
        _EMBEDDER = None
        _EMBEDDER_KEY = None
    _cleanup_torch_memory()


def schedule_embedder_unload(cfg: VectorConfig) -> None:
    global _EMBEDDER_UNLOAD_TIMER
    if cfg.embedding_backend in ("ollama", "llama-server"):
        return
    cooldown = max(0, int(cfg.embedding_unload_cooldown_seconds))
    if cooldown == 0:
        unload_cached_embedder()
        return
    with _EMBEDDER_LOCK:
        if _EMBEDDER is None:
            return
        if _EMBEDDER_UNLOAD_TIMER is not None:
            _EMBEDDER_UNLOAD_TIMER.cancel()
        _EMBEDDER_UNLOAD_TIMER = threading.Timer(cooldown, unload_cached_embedder)
        _EMBEDDER_UNLOAD_TIMER.daemon = True
        _EMBEDDER_UNLOAD_TIMER.start()


def embedder_status() -> dict[str, Any]:
    with _EMBEDDER_LOCK:
        return {
            "resident": _EMBEDDER is not None,
            "key": list(_EMBEDDER_KEY) if _EMBEDDER_KEY else None,
            "unload_timer_pending": _EMBEDDER_UNLOAD_TIMER is not None,
        }


def normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return vector
    return [value / norm for value in vector]


def ollama_embed(cfg: VectorConfig, texts: list[str]) -> list[list[float]]:
    payload = json.dumps({"model": cfg.embedding_model, "input": texts}).encode("utf-8")
    request = urllib.request.Request(
        f"{cfg.ollama_url.rstrip('/')}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.loads(response.read().decode("utf-8"))
        embeddings = data.get("embeddings")
        if embeddings:
            return [normalize_vector([float(value) for value in embedding]) for embedding in embeddings]
    except Exception:
        pass

    embeddings = []
    for text in texts:
        payload = json.dumps({"model": cfg.embedding_model, "prompt": text}).encode("utf-8")
        request = urllib.request.Request(
            f"{cfg.ollama_url.rstrip('/')}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.loads(response.read().decode("utf-8"))
        embeddings.append(normalize_vector([float(value) for value in data["embedding"]]))
    return embeddings


def llamaserver_embed(cfg: VectorConfig, texts: list[str]) -> list[list[float]]:
    # Batched call via OpenAI-compatible /v1/embeddings
    payload = json.dumps({"input": texts, "model": cfg.embedding_model}).encode("utf-8")
    request = urllib.request.Request(
        f"{cfg.ollama_url.rstrip('/')}/v1/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.loads(response.read().decode("utf-8"))
        items = sorted(data["data"], key=lambda x: x["index"])
        return [normalize_vector([float(v) for v in item["embedding"]]) for item in items]
    except Exception:
        # per-text fallback via legacy /embedding endpoint
        embeddings = []
        for text in texts:
            single_payload = json.dumps({"content": text}).encode("utf-8")
            single_req = urllib.request.Request(
                f"{cfg.ollama_url.rstrip('/')}/embedding",
                data=single_payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(single_req, timeout=600) as resp:
                    item = json.loads(resp.read().decode("utf-8"))
                result = item[0]["embedding"]
                if isinstance(result[0], list):
                    result = result[0]
                embeddings.append(normalize_vector([float(v) for v in result]))
            except Exception:
                embeddings.append([0.0])
        return embeddings


def embed_texts(cfg: VectorConfig, texts: list[str], embedder: Any | None = None) -> list[list[float]]:
    if cfg.embedding_backend == "ollama":
        return ollama_embed(cfg, texts)
    if cfg.embedding_backend == "llama-server":
        return llamaserver_embed(cfg, texts)
    if embedder is None:
        embedder = load_embedder_for_config(cfg)
    return embedder.encode(texts, normalize_embeddings=True, batch_size=cfg.embedding_batch_size).tolist()


def vector_config_for_collection(base: VectorConfig, name: str, roots: tuple[str, ...]) -> VectorConfig:
    """Clone a VectorConfig with a different collection name and indexed_roots."""
    state_base = Path(base.state_path)
    return VectorConfig(
        chroma_dir=base.chroma_dir,
        collection_name=name,
        embedding_backend=base.embedding_backend,
        embedding_model=base.embedding_model,
        embedding_device=base.embedding_device,
        embedding_batch_size=base.embedding_batch_size,
        ollama_url=base.ollama_url,
        indexed_roots=roots,
        max_file_bytes=base.max_file_bytes,
        max_chunk_chars=base.max_chunk_chars,
        chunk_overlap_lines=base.chunk_overlap_lines,
        incremental_enabled=base.incremental_enabled,
        btrfs_subvolume=base.btrfs_subvolume,
        state_path=str(state_base.parent / f"{name}-index-state.json"),
        embedding_unload_cooldown_seconds=base.embedding_unload_cooldown_seconds,
    )


def vector_search_multi(
    configs: list[VectorConfig],
    query: str,
    top_k: int = 8,
    paths: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Search multiple collections and merge results sorted by cosine distance."""
    all_results: list[dict[str, Any]] = []
    total_count = 0
    errors: list[str] = []
    for cfg in configs:
        try:
            result = vector_search(cfg, query=query, top_k=top_k, paths=paths)
            if result.get("ok"):
                total_count += result.get("count", 0)
                for r in result.get("results", []):
                    r.setdefault("collection", cfg.collection_name)
                    all_results.append(r)
        except Exception as exc:
            errors.append(f"{cfg.collection_name}: {exc}")
    all_results.sort(key=lambda r: r.get("distance", 1.0))
    return {
        "ok": True,
        "query": query,
        "total_indexed_chunks": total_count,
        "results": all_results[:top_k],
        "errors": errors,
    }


def chroma_client(chroma_dir: str):
    import chromadb

    Path(chroma_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=chroma_dir)


def chroma_collection(cfg: VectorConfig):
    client = chroma_client(cfg.chroma_dir)
    return client.get_or_create_collection(
        name=cfg.collection_name,
        metadata={"hnsw:space": "cosine", "stores_original_text": "false"},
    )


def should_index_file(path: Path, cfg: VectorConfig) -> bool:
    if any(part in IGNORE_DIRS for part in path.parts):
        return False
    if path.suffix.lower() not in TEXT_EXTENSIONS:
        return False
    try:
        st = path.stat()
    except OSError:
        return False
    if st.st_size <= 0 or st.st_size > cfg.max_file_bytes:
        return False
    return True


def path_matches_roots(path: Path, roots: Iterable[str]) -> bool:
    path_text = str(path)
    for root_text in roots:
        expanded = os.path.expanduser(root_text)
        if any(ch in expanded for ch in "*?["):
            if fnmatch.fnmatch(path_text, expanded):
                return True
            continue
        root = Path(expanded).resolve()
        if root.is_file() and path == root:
            return True
        if root.is_dir():
            try:
                path.relative_to(root)
                return True
            except ValueError:
                pass
    return False


def filter_index_files(cfg: VectorConfig, paths: Iterable[str]) -> list[Path]:
    filtered = []
    seen: set[Path] = set()
    for path_text in paths:
        path = Path(path_text).expanduser()
        if not path.is_absolute():
            path = Path(cfg.btrfs_subvolume).expanduser() / path
        resolved = path.resolve()
        if resolved in seen:
            continue
        if not path_matches_roots(resolved, cfg.indexed_roots):
            continue
        if not resolved.is_file() or not should_index_file(resolved, cfg):
            continue
        seen.add(resolved)
        filtered.append(resolved)
    return filtered


def iter_index_files(cfg: VectorConfig, paths: Iterable[str] | None = None) -> Iterable[Path]:
    roots = tuple(paths) if paths else cfg.indexed_roots
    yielded: set[Path] = set()
    for root_text in roots:
        if any(ch in root_text for ch in "*?["):
            for match in glob.glob(os.path.expanduser(root_text), recursive=True):
                path = Path(match)
                resolved = path.resolve()
                if resolved not in yielded and path.is_file() and should_index_file(path, cfg):
                    yielded.add(resolved)
                    yield resolved
            continue
        root = Path(root_text).expanduser()
        if root.is_file():
            resolved = root.resolve()
            if resolved not in yielded and should_index_file(root, cfg):
                yielded.add(resolved)
                yield resolved
            continue
        if not root.exists():
            continue
        for path in root.rglob("*"):
            resolved = path.resolve()
            if resolved not in yielded and path.is_file() and should_index_file(path, cfg):
                yielded.add(resolved)
                yield resolved


HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+|^\s*(class|def|fun|function)\s+\w+|^\s*\\[.+\\]\s*$")


def chunk_text(text: str, max_chars: int, overlap_lines: int) -> list[tuple[int, int, str]]:
    lines = text.splitlines()
    chunks: list[tuple[int, int, str]] = []
    start = 0
    current: list[str] = []
    current_start = 0

    def flush(end_line: int) -> None:
        nonlocal current, current_start
        body = "\n".join(current).strip()
        if body:
            chunks.append((current_start + 1, end_line, body))
        if overlap_lines > 0 and current:
            current = current[-overlap_lines:]
            current_start = max(0, end_line - len(current))
        else:
            current = []
            current_start = end_line

    for idx, line in enumerate(lines):
        boundary = bool(HEADER_RE.match(line)) and current and len("\n".join(current)) > 800
        too_large = len("\n".join(current + [line])) > max_chars
        if boundary or too_large:
            flush(idx)
        if not current:
            current_start = idx
        current.append(line)
        start = idx
    flush(start + 1 if lines else 0)
    return chunks


def stable_chunk_id(path: Path, start_line: int, end_line: int, digest: str) -> str:
    raw = f"{path}:{start_line}:{end_line}:{digest}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def summarize_chunk(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    heading = next((line for line in lines if line.startswith("#")), "")
    body = " ".join(lines[:8])
    summary = f"{heading} {body}".strip() if heading else body
    return summary[:900]


def read_lines(path: str, start_line: int, end_line: int, max_chars: int = 8_000) -> str:
    lines = []
    try:
        with Path(path).open("r", encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f, start=1):
                if idx < start_line:
                    continue
                if idx > end_line:
                    break
                lines.append(line.rstrip("\n"))
    except OSError as exc:
        return f"[unreadable: {exc}]"
    text = "\n".join(lines)
    return text[:max_chars] + ("\n[truncated]" if len(text) > max_chars else "")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def full_progress_path(cfg: VectorConfig) -> Path:
    state_path = Path(cfg.state_path).expanduser()
    return state_path.with_name("full-index-progress.jsonl")


def read_full_progress(cfg: VectorConfig) -> dict[str, dict[str, Any]]:
    path = full_progress_path(cfg)
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                path_text = str(record.get("path", ""))
                if path_text:
                    completed[path_text] = record
    except OSError:
        return completed
    return completed


def append_full_progress(cfg: VectorConfig, record: dict[str, Any]) -> None:
    path = full_progress_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def merge_collection_progress(
    cfg: VectorConfig,
    collection: Any,
    completed: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], int]:
    embedded, chunk_count = collection_metadata_fast(cfg, collection)
    merged = dict(completed)
    for path, meta in embedded.items():
        existing = merged.get(path)
        if existing and existing.get("sha256") == meta.get("sha256"):
            continue
        merged[path] = {
            "path": path,
            "sha256": meta.get("sha256"),
            "size": meta.get("size"),
            "mtime": meta.get("mtime"),
            "chunks": meta.get("chunks", 0),
            "source": "chroma",
        }
    return merged, chunk_count


def completed_matches(record: dict[str, Any] | None, digest: str, stat: os.stat_result) -> bool:
    if not record:
        return False
    try:
        size_matches = int(record.get("size", -1)) == int(stat.st_size)
    except (TypeError, ValueError):
        size_matches = False
    return bool(record.get("sha256") == digest and size_matches)


def index_paths(
    cfg: VectorConfig,
    paths: Iterable[str] | None = None,
    batch_size: int | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    cleanup: bool = True,
    resume: bool = False,
) -> dict[str, Any]:
    batch_size = batch_size or cfg.embedding_batch_size
    collection = chroma_collection(cfg)
    completed: dict[str, dict[str, Any]] = {}
    progress_log: str | None = None
    resume_chunk_count = 0
    if resume:
        completed = read_full_progress(cfg)
        completed, resume_chunk_count = merge_collection_progress(cfg, collection, completed)
        progress_log = str(full_progress_path(cfg))
        if progress:
            progress(
                {
                    "phase": "resume",
                    "files_seen": len(completed),
                    "chunks_indexed": 0,
                    "path": f"{len(completed)} completed files from progress/chroma",
                }
            )

    embedder = None
    files_seen = 0
    files_skipped = 0
    chunks_seen = 0
    chunks_indexed = 0
    errors: list[str] = []

    try:
        for path in iter_index_files(cfg, paths):
            files_seen += 1
            if progress and (files_seen == 1 or files_seen % 25 == 0):
                progress({"phase": "file", "files_seen": files_seen, "chunks_indexed": chunks_indexed, "path": str(path)})
            try:
                digest = file_hash(path)
                stat = path.stat()
                if resume and completed_matches(completed.get(str(path)), digest, stat):
                    files_skipped += 1
                    if progress and files_skipped % 100 == 0:
                        progress(
                            {
                                "phase": "skip",
                                "files_seen": files_seen,
                                "chunks_indexed": chunks_indexed,
                                "path": str(path),
                            }
                        )
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                chunks = chunk_text(text, cfg.max_chunk_chars, cfg.chunk_overlap_lines)
            except Exception as exc:
                errors.append(f"{path}: {exc}")
                continue

            old = collection.get(where={"path": str(path)}, include=[])
            if old.get("ids"):
                collection.delete(ids=old["ids"])

            ids: list[str] = []
            metas: list[dict[str, Any]] = []
            summaries: list[str] = []
            for start, end, chunk in chunks:
                summary = summarize_chunk(chunk)
                if not summary:
                    continue
                ids.append(stable_chunk_id(path, start, end, digest))
                metas.append(
                    {
                        "path": str(path),
                        "start_line": start,
                        "end_line": end,
                        "mtime": stat.st_mtime,
                        "size": stat.st_size,
                        "sha256": digest,
                        "summary": summary,
                    }
                )
                summaries.append(summary)
                chunks_seen += 1

            if summaries and cfg.embedding_backend not in ("ollama", "llama-server") and embedder is None:
                embedder = get_cached_embedder(cfg)

            for offset in range(0, len(ids), batch_size):
                batch_ids = ids[offset : offset + batch_size]
                batch_metas = metas[offset : offset + batch_size]
                batch_summaries = summaries[offset : offset + batch_size]
                embeddings = embed_texts(cfg, batch_summaries, embedder)
                collection.upsert(ids=batch_ids, embeddings=embeddings, metadatas=batch_metas)
                chunks_indexed += len(batch_ids)
                if progress and chunks_indexed % max(batch_size, 1) == 0:
                    progress({"phase": "embed", "files_seen": files_seen, "chunks_indexed": chunks_indexed, "path": str(path)})

            if resume:
                append_full_progress(
                    cfg,
                    {
                        "path": str(path),
                        "sha256": digest,
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                        "chunks": len(ids),
                        "completed_at": time.time(),
                        "model": cfg.embedding_model,
                        "collection": cfg.collection_name,
                    },
                )

        stale_deleted = cleanup_stale(collection) if cleanup else 0
        return {
            "ok": True,
            "files_seen": files_seen,
            "files_skipped": files_skipped,
            "chunks_seen": chunks_seen,
            "chunks_indexed": chunks_indexed,
            "collection_count": collection.count(),
            "stale_deleted": stale_deleted,
            "resume": resume,
            "resume_progress_path": progress_log,
            "resume_bootstrap_chunks": resume_chunk_count,
            "errors": errors[:50],
        }
    finally:
        schedule_embedder_unload(cfg)


def run_btrfs(cfg: VectorConfig, args: list[str], timeout: int = 60) -> dict[str, Any]:
    commands = [["btrfs", *args], ["sudo", "-n", "btrfs", *args]]
    last: dict[str, Any] | None = None
    for cmd in commands:
        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
        except Exception as exc:
            last = {"ok": False, "cmd": cmd, "stdout": "", "stderr": str(exc), "returncode": -1}
            continue
        result = {
            "ok": proc.returncode == 0,
            "cmd": cmd,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }
        if result["ok"]:
            return result
        last = result
    return last or {"ok": False, "cmd": commands[-1], "stdout": "", "stderr": "btrfs command failed", "returncode": -1}


def btrfs_current_generation(cfg: VectorConfig) -> dict[str, Any]:
    proc = run_btrfs(cfg, ["subvolume", "show", cfg.btrfs_subvolume])
    if not proc["ok"]:
        return {"ok": False, "error": proc.get("stderr", "").strip(), "cmd": proc.get("cmd")}
    for line in proc.get("stdout", "").splitlines():
        if "Generation:" not in line:
            continue
        _, value = line.split(":", 1)
        try:
            return {"ok": True, "generation": int(value.strip()), "subvolume": cfg.btrfs_subvolume}
        except ValueError:
            break
    return {"ok": False, "error": "could not parse btrfs generation", "stdout": proc.get("stdout", "")}


FIND_NEW_PATH_RE = re.compile(r"\spath\s+(.+)$")


def btrfs_find_new(cfg: VectorConfig, since_generation: int) -> dict[str, Any]:
    proc = run_btrfs(cfg, ["subvolume", "find-new", cfg.btrfs_subvolume, str(int(since_generation))], timeout=300)
    if not proc["ok"]:
        return {"ok": False, "error": proc.get("stderr", "").strip(), "cmd": proc.get("cmd"), "paths": []}
    paths = []
    for line in proc.get("stdout", "").splitlines():
        match = FIND_NEW_PATH_RE.search(line)
        if not match:
            continue
        path_text = match.group(1).strip()
        path = Path(path_text)
        if not path.is_absolute():
            path = Path(cfg.btrfs_subvolume) / path
        paths.append(str(path))
    return {"ok": True, "paths": paths, "raw_count": len(paths)}


def read_index_state(cfg: VectorConfig) -> dict[str, Any]:
    path = Path(cfg.state_path).expanduser()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_index_state(cfg: VectorConfig, state: dict[str, Any]) -> None:
    path = Path(cfg.state_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def update_index_generation(cfg: VectorConfig, mode: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    current = btrfs_current_generation(cfg)
    if not current.get("ok"):
        return current
    state = read_index_state(cfg)
    state.update(
        {
            "btrfs_subvolume": cfg.btrfs_subvolume,
            "last_transid": current["generation"],
            "last_mode": mode,
        }
    )
    if extra:
        state.update(extra)
    write_index_state(cfg, state)
    return {"ok": True, "last_transid": current["generation"], "state_path": cfg.state_path}


def index_incremental(
    cfg: VectorConfig,
    batch_size: int | None = None,
    fallback_full: bool = False,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if not cfg.incremental_enabled:
        return {"ok": False, "error": "incremental indexing is disabled"}
    state = read_index_state(cfg)
    last_transid = state.get("last_transid")
    if last_transid is None:
        if not fallback_full:
            return {"ok": False, "needs_full_index": True, "error": "no stored btrfs transid"}
        result = index_paths(cfg, batch_size=batch_size, progress=progress, cleanup=False, resume=True)
        if result.get("ok"):
            result["generation_update"] = update_index_generation(cfg, "full")
            result["mode"] = "full"
        return result

    changed = btrfs_find_new(cfg, int(last_transid))
    if not changed.get("ok"):
        return changed
    filtered = filter_index_files(cfg, changed.get("paths", []))
    if progress:
        progress(
            {
                "phase": "btrfs",
                "files_seen": len(filtered),
                "chunks_indexed": 0,
                "path": f"{changed.get('raw_count', 0)} raw changed paths",
            }
        )
    result = index_paths(cfg, paths=[str(path) for path in filtered], batch_size=batch_size, progress=progress, cleanup=False)
    if result.get("ok"):
        result["generation_update"] = update_index_generation(
            cfg,
            "incremental",
            {"last_raw_changed_count": changed.get("raw_count", 0), "last_filtered_changed_count": len(filtered)},
        )
    result.update(
        {
            "mode": "incremental",
            "previous_transid": int(last_transid),
            "raw_changed_count": changed.get("raw_count", 0),
            "filtered_changed_count": len(filtered),
        }
    )
    return result


def cleanup_stale(collection: Any) -> int:
    deleted = 0
    page = collection.get(include=["metadatas"], limit=10_000)
    ids = page.get("ids", [])
    metas = page.get("metadatas", [])
    stale = [id_ for id_, meta in zip(ids, metas) if meta and not Path(str(meta.get("path", ""))).exists()]
    if stale:
        collection.delete(ids=stale)
        deleted += len(stale)
    return deleted


def collection_metadata(collection: Any) -> tuple[dict[str, dict[str, Any]], int]:
    by_path: dict[str, dict[str, Any]] = {}
    chunk_count = 0
    offset = 0
    limit = 10_000
    while True:
        page = collection.get(include=["metadatas"], limit=limit, offset=offset)
        metas = page.get("metadatas", [])
        if not metas:
            break
        for meta in metas:
            if not meta:
                continue
            chunk_count += 1
            path = str(meta.get("path", ""))
            if not path:
                continue
            entry = by_path.setdefault(
                path,
                {
                    "path": path,
                    "chunks": 0,
                    "mtime": meta.get("mtime"),
                    "size": meta.get("size"),
                    "sha256": meta.get("sha256"),
                },
            )
            entry["chunks"] += 1
        if len(metas) < limit:
            break
        offset += limit
    return by_path, chunk_count


def collection_metadata_sqlite(cfg: VectorConfig) -> tuple[dict[str, dict[str, Any]], int]:
    db_path = Path(cfg.chroma_dir).expanduser() / "chroma.sqlite3"
    if not db_path.exists():
        return {}, 0
    by_path: dict[str, dict[str, Any]] = {}
    query = """
        WITH metadata_segment AS (
            SELECT s.id
            FROM segments s
            JOIN collections c ON c.id = s.collection
            WHERE c.name = ?
              AND s.type = 'urn:chroma:segment/metadata/sqlite'
            LIMIT 1
        )
        SELECT
            p.string_value AS path,
            max(sha.string_value) AS sha256,
            max(coalesce(sz.int_value, cast(sz.float_value AS INTEGER))) AS size,
            max(mt.float_value) AS mtime,
            count(distinct e.id) AS chunks
        FROM embeddings e
        JOIN metadata_segment ms ON ms.id = e.segment_id
        JOIN embedding_metadata p ON p.id = e.id AND p.key = 'path'
        LEFT JOIN embedding_metadata sha ON sha.id = e.id AND sha.key = 'sha256'
        LEFT JOIN embedding_metadata sz ON sz.id = e.id AND sz.key = 'size'
        LEFT JOIN embedding_metadata mt ON mt.id = e.id AND mt.key = 'mtime'
        GROUP BY p.string_value
    """
    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute(query, (cfg.collection_name,)).fetchall()
    chunk_count = 0
    for path, sha256, size, mtime, chunks in rows:
        if not path:
            continue
        chunk_total = int(chunks or 0)
        chunk_count += chunk_total
        by_path[str(path)] = {
            "path": str(path),
            "chunks": chunk_total,
            "mtime": mtime,
            "size": size,
            "sha256": sha256,
        }
    return by_path, chunk_count


def collection_metadata_fast(cfg: VectorConfig, collection: Any) -> tuple[dict[str, dict[str, Any]], int]:
    try:
        return collection_metadata_sqlite(cfg)
    except Exception:
        return collection_metadata(collection)


def vector_status(cfg: VectorConfig, paths: Iterable[str] | None = None, sample_limit: int = 50) -> dict[str, Any]:
    sample_limit = max(1, min(int(sample_limit), 500))
    collection = chroma_collection(cfg)
    embedded, chunk_count = collection_metadata_fast(cfg, collection)
    live_paths = [str(path) for path in iter_index_files(cfg, paths)]
    live_set = set(live_paths)
    embedded_set = set(embedded)

    missing_from_index = sorted(live_set - embedded_set)
    stale = sorted(embedded_set - live_set)
    changed = []
    errors = []
    for path_text in sorted(live_set & embedded_set):
        path = Path(path_text)
        meta = embedded[path_text]
        try:
            stat = path.stat()
        except OSError as exc:
            errors.append(f"{path_text}: {exc}")
            continue
        if int(meta.get("size") or -1) != stat.st_size:
            changed.append(path_text)
            continue
        stored_mtime = float(meta.get("mtime") or 0)
        if abs(stored_mtime - stat.st_mtime) > 0.001:
            try:
                digest = file_hash(path)
            except OSError as exc:
                errors.append(f"{path_text}: {exc}")
                continue
            if digest != meta.get("sha256"):
                changed.append(path_text)

    return {
        "ok": True,
        "collection": cfg.collection_name,
        "chroma_dir": cfg.chroma_dir,
        "chunk_count": chunk_count,
        "embedded_file_count": len(embedded_set),
        "live_candidate_count": len(live_set),
        "missing_from_index_count": len(missing_from_index),
        "changed_count": len(changed),
        "stale_count": len(stale),
        "missing_from_index": missing_from_index[:sample_limit],
        "changed": changed[:sample_limit],
        "stale": stale[:sample_limit],
        "errors": errors[:sample_limit],
    }


def vector_search(cfg: VectorConfig, query: str, top_k: int = 8, paths: Iterable[str] | None = None) -> dict[str, Any]:
    if not query.strip():
        return {"ok": False, "error": "query is required", "results": []}
    top_k = max(1, min(int(top_k), 30))
    try:
        embedder = None if cfg.embedding_backend in ("ollama", "llama-server") else get_cached_embedder(cfg)
        collection = chroma_collection(cfg)
        query_embedding = embed_texts(cfg, [query], embedder)[0]
        path_prefixes = [str(Path(p).expanduser().resolve()) for p in paths] if paths else []
        raw = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k * 3, 100) if path_prefixes else top_k,
            include=["metadatas", "distances"],
        )
        results = []
        for meta, distance in zip(raw.get("metadatas", [[]])[0], raw.get("distances", [[]])[0]):
            if not meta:
                continue
            path = str(meta.get("path", ""))
            if path_prefixes and not any(path.startswith(prefix) for prefix in path_prefixes):
                continue
            start = int(meta.get("start_line", 1))
            end = int(meta.get("end_line", start))
            results.append(
                {
                    "path": path,
                    "start_line": start,
                    "end_line": end,
                    "distance": distance,
                    "summary": meta.get("summary", ""),
                    "text": read_lines(path, start, end),
                }
            )
            if len(results) >= top_k:
                break
        return {"ok": True, "query": query, "count": collection.count(), "results": results}
    finally:
        schedule_embedder_unload(cfg)

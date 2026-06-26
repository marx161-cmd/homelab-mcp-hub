#!/usr/bin/env python3
from __future__ import annotations

import html
import base64
import datetime as dt
import json
import mimetypes
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from vector_store import index_paths as vector_index_paths
from vector_store import index_incremental as vector_index_incremental_impl
from vector_store import chroma_collection
from vector_store import cleanup_stale as vector_cleanup_stale_impl
from vector_store import embedder_status as vector_embedder_status_impl
from vector_store import update_index_generation as vector_update_generation
from vector_store import unload_cached_embedder as vector_unload_embedder_impl
from vector_store import vector_config
from vector_store import vector_config_for_collection
from vector_store import VectorConfig
from vector_store import vector_search as vector_search_impl
from vector_store import vector_search_multi as vector_search_multi_impl
from vector_store import vector_status as vector_status_impl

# ---------------------------------------------------------------------------
# Monkey-patch: disable the Accept header check in the upstream mcp library.
# Kai's MCP client sends no Accept header at all on JSON-RPC notifications
# (e.g. ``notifications/initialized``).  The stock check requires both
# ``application/json`` *and* ``text/event-stream`` on every POST and
# rejects missing/poor headers with 406, stalling the session handshake.
# We replace the validator with a no-op pass-through.
# ---------------------------------------------------------------------------
import mcp.server.streamable_http as _mcp_transport


async def _patched_validate_accept_header(self, request, scope, send):  # type: ignore[no-untyped-def]
    return True


_mcp_transport.StreamableHTTPServerTransport._validate_accept_header = _patched_validate_accept_header


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("HOMELAB_MCP_CONFIG", ROOT / "config.json"))


@dataclass(frozen=True)
class Config:
    bind_host: str = "127.0.0.1"
    port: int = 8765
    mcp_path: str = "/mcp"
    searxng_search_url: str = "https://comintern.taile6163a.ts.net:8444/search"
    searxng_engines: str = "duckduckgo"
    homelab_root: str = "/home/comrade/homelab"
    bin_dir: str = "/home/comrade/bin"
    call_log_path: str = "/home/comrade/logs/mcp-calls.jsonl"
    snapshot_root: str = "/root/Documents"
    snapshot_preview_chars: int = 12_000
    chroma_dir: str = "/home/comrade/.local/share/mcp-hub/chroma"
    vector_collection: str = "homelab_knowledge"
    embedding_backend: str = "sentence-transformers"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_device: str = "cuda:0"
    embedding_batch_size: int = 8
    embedding_unload_cooldown_seconds: int = 60
    ollama_url: str = "http://127.0.0.1:11434"
    indexed_roots: tuple[str, ...] = ("/home/comrade/homelab", "/home/comrade/bin")
    vector_max_file_bytes: int = 1_500_000
    vector_max_chunk_chars: int = 3_500
    vector_chunk_overlap_lines: int = 2
    vector_incremental_enabled: bool = True
    vector_btrfs_subvolume: str = "/home"
    vector_state_path: str = "/home/comrade/.local/share/mcp-hub/index-state.json"
    # Optional multi-collection override: tuple of (name, (root, ...)) pairs.
    # When set, each entry becomes a separate ChromaDB collection.
    # Falls back to (vector_collection, indexed_roots) when empty.
    vector_collections: tuple[tuple[str, tuple[str, ...]], ...] = ()


def load_config() -> Config:
    if not CONFIG_PATH.exists():
        return Config()
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return Config(
        bind_host=str(data.get("bind_host", Config.bind_host)),
        port=int(data.get("port", Config.port)),
        mcp_path=str(data.get("mcp_path", Config.mcp_path)),
        searxng_search_url=str(data.get("searxng_search_url", Config.searxng_search_url)),
        searxng_engines=str(data.get("searxng_engines", Config.searxng_engines)),
        homelab_root=str(data.get("homelab_root", Config.homelab_root)),
        bin_dir=str(data.get("bin_dir", Config.bin_dir)),
        call_log_path=str(data.get("call_log_path", Config.call_log_path)),
        snapshot_root=str(data.get("snapshot_root", Config.snapshot_root)),
        snapshot_preview_chars=int(data.get("snapshot_preview_chars", Config.snapshot_preview_chars)),
        chroma_dir=str(data.get("chroma_dir", Config.chroma_dir)),
        vector_collection=str(data.get("vector_collection", Config.vector_collection)),
        embedding_backend=str(data.get("embedding_backend", Config.embedding_backend)),
        embedding_model=str(data.get("embedding_model", Config.embedding_model)),
        embedding_device=str(data.get("embedding_device", Config.embedding_device)),
        embedding_batch_size=int(data.get("embedding_batch_size", Config.embedding_batch_size)),
        embedding_unload_cooldown_seconds=int(
            data.get("embedding_unload_cooldown_seconds", Config.embedding_unload_cooldown_seconds)
        ),
        ollama_url=str(data.get("ollama_url", Config.ollama_url)),
        indexed_roots=tuple(data.get("indexed_roots", Config.indexed_roots)),
        vector_max_file_bytes=int(data.get("vector_max_file_bytes", Config.vector_max_file_bytes)),
        vector_max_chunk_chars=int(data.get("vector_max_chunk_chars", Config.vector_max_chunk_chars)),
        vector_chunk_overlap_lines=int(data.get("vector_chunk_overlap_lines", Config.vector_chunk_overlap_lines)),
        vector_incremental_enabled=bool(data.get("vector_incremental_enabled", Config.vector_incremental_enabled)),
        vector_btrfs_subvolume=str(data.get("vector_btrfs_subvolume", Config.vector_btrfs_subvolume)),
        vector_state_path=str(data.get("vector_state_path", Config.vector_state_path)),
        vector_collections=tuple(
            (str(c["name"]), tuple(str(r) for r in c.get("roots", [])))
            for c in data.get("vector_collections", [])
            if c.get("name") and c.get("roots")
        ),
    )


CONFIG = load_config()


def all_vector_configs() -> list[VectorConfig]:
    """Return one VectorConfig per configured collection (falls back to single legacy config)."""
    base = vector_config(CONFIG)
    if CONFIG.vector_collections:
        return [vector_config_for_collection(base, name, roots) for name, roots in CONFIG.vector_collections]
    return [base]


def get_named_configs(collection: str | None) -> list[VectorConfig]:
    """Return configs for a specific collection name, or all configs if name is None."""
    configs = all_vector_configs()
    if collection is None:
        return configs
    matched = [c for c in configs if c.collection_name == collection]
    return matched  # empty list means unknown collection name


mcp = FastMCP("homelab-tools")

MAX_READ_BYTES = 2_000_000
DEFAULT_READ_CHARS = 12_000
MAX_READ_CHARS = 50_000
MAX_CMD_OUTPUT = 80_000


def log_tool_call(tool: str, args: dict[str, Any]) -> None:
    record = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "tool": tool,
        "args": args,
        "pid": os.getpid(),
    }
    try:
        path = Path(CONFIG.call_log_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    except Exception:
        pass


def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(int(value), high))


def normalize_string_list(value: list[str] | str | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item).strip()]
    if isinstance(parsed, str):
        return [parsed] if parsed.strip() else None
    return [part.strip() for part in text.split(",") if part.strip()]


def run_cmd(
    cmd: list[str],
    *,
    cwd: str | Path | None = None,
    timeout: int = 30,
    input_text: str | None = None,
) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            input=input_text,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc), "cmd": cmd}
    stdout = proc.stdout
    stderr = proc.stderr
    truncated = False
    if len(stdout) > MAX_CMD_OUTPUT:
        stdout = stdout[:MAX_CMD_OUTPUT] + "\n[truncated]"
        truncated = True
    if len(stderr) > MAX_CMD_OUTPUT:
        stderr = stderr[:MAX_CMD_OUTPUT] + "\n[truncated]"
        truncated = True
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "cmd": cmd,
        "stdout": stdout,
        "stderr": stderr,
        "truncated": truncated,
    }


def resolve_cwd(cwd: str | None = None) -> Path:
    if not cwd:
        return Path(CONFIG.homelab_root).expanduser().resolve()
    path = Path(cwd).expanduser()
    if not path.is_absolute():
        path = Path(CONFIG.homelab_root).expanduser() / path
    return path.resolve()


def repo_path(repo: str) -> Path:
    path = Path(repo).expanduser()
    if not path.is_absolute():
        path = Path(CONFIG.homelab_root).expanduser() / path
    return path.resolve()


def git_base_cmd(repo: str) -> tuple[Path, list[str]]:
    path = repo_path(repo)
    return path, ["git", "-C", str(path)]


def github_token() -> str | None:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token
    if shutil.which("gh"):
        proc = run_cmd(["gh", "auth", "token"], timeout=10)
        token = proc.get("stdout", "").strip()
        if proc["ok"] and token:
            return token
    return None


def gh_headers() -> dict[str, str]:
    token = github_token()
    if not token:
        raise RuntimeError("No GitHub token found via env or `gh auth token`")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "homelab-mcp-hub/0.2",
    }


def github_request(method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    url = f"https://api.github.com{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, headers=gh_headers(), method=method)
    if payload is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read().decode("utf-8")
        return json.loads(body) if body else {}


def github_owner() -> str:
    user = github_request("GET", "/user")
    return str(user["login"])


def hf_api() -> Any:
    try:
        from huggingface_hub import HfApi, get_token
    except Exception as exc:
        raise RuntimeError(f"huggingface_hub is not installed: {exc}") from exc
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN") or get_token()
    if not token:
        raise RuntimeError("No HuggingFace token found via env or huggingface_hub login cache")
    return HfApi(token=token)


def hf_token_configured() -> bool:
    try:
        from huggingface_hub import get_token
    except Exception:
        return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"))
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN") or get_token())


def normalize_url(url: str) -> str:
    url = url.strip()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("only http and https URLs are supported")
    if not parsed.netloc:
        raise ValueError("URL is missing a host")

    # GitHub's normal file view is HTML-heavy. For concrete files, raw content is
    # better evidence for the model and avoids page chrome.
    if parsed.netloc.lower() == "github.com":
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 5 and parts[2] == "blob":
            owner, repo, _, ref = parts[:4]
            file_path = "/".join(parts[4:])
            raw_path = f"/{owner}/{repo}/{ref}/{file_path}"
            return urllib.parse.urlunparse(("https", "raw.githubusercontent.com", raw_path, "", parsed.query, ""))

    return url


def fetch_bytes(url: str, *, timeout: int = 30) -> tuple[str, str, bytes]:
    normalized = normalize_url(url)
    request = urllib.request.Request(
        normalized,
        headers={
            "User-Agent": "homelab-mcp-hub/0.1 (+reader)",
            "Accept": "text/html,application/xhtml+xml,text/plain,application/json,application/xml;q=0.9,*/*;q=0.5",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        final_url = response.geturl()
        content_type = response.headers.get("Content-Type", "")
        body = response.read(MAX_READ_BYTES + 1)
    if len(body) > MAX_READ_BYTES:
        body = body[:MAX_READ_BYTES]
    return final_url, content_type, body


def decode_body(body: bytes, content_type: str) -> str:
    charset_match = re.search(r"charset=([^;\s]+)", content_type, re.I)
    charset = charset_match.group(1).strip('"') if charset_match else "utf-8"
    try:
        return body.decode(charset, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


class ReadableHtmlParser(HTMLParser):
    BLOCK_TAGS = {
        "article",
        "main",
        "section",
        "p",
        "div",
        "br",
        "li",
        "ul",
        "ol",
        "blockquote",
        "pre",
        "code",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "tr",
        "td",
        "th",
    }
    SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas", "nav", "footer", "header", "form"}

    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self._in_title = False
        self._skip_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = True
            return
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in self.BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        if self._skip_depth:
            return
        text = " ".join(data.split())
        if text:
            self._chunks.append(text)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
            return
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if not self._skip_depth and tag in self.BLOCK_TAGS:
            self._chunks.append("\n")

    def readable_text(self) -> str:
        raw = html.unescape(" ".join(self._chunks))
        raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
        raw = re.sub(r"\s*\n\s*", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        lines = [line.strip() for line in raw.splitlines()]
        return "\n".join(line for line in lines if line).strip()


def extract_readable_text(body: bytes, content_type: str, url: str) -> tuple[str, str, str]:
    text = decode_body(body, content_type)
    media_type = content_type.split(";", 1)[0].strip().lower()
    guessed_type = mimetypes.guess_type(urllib.parse.urlparse(url).path)[0] or ""
    if media_type in {"text/html", "application/xhtml+xml"} or guessed_type == "text/html" or "<html" in text[:1000].lower():
        parser = ReadableHtmlParser()
        parser.feed(text)
        return "html", html.unescape(" ".join(parser.title.split())), parser.readable_text()
    return media_type or guessed_type or "text", "", text.strip()


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars].rstrip() + "\n\n[truncated]", True


class SearxHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._capture: str | None = None
        self._chunks: list[str] = []
        self._in_result = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        classes = attrs_dict.get("class", "")
        if tag == "article" and "result" in classes:
            self._in_result = True
            self._current = {"title": "", "url": "", "snippet": ""}
            return
        if not self._in_result or self._current is None:
            return
        if tag == "a" and "url_header" in classes and not self._current["url"]:
            self._current["url"] = attrs_dict.get("href", "") or ""
        elif tag == "h3":
            self._capture = "title"
            self._chunks = []
        elif tag == "p" and "content" in classes:
            self._capture = "snippet"
            self._chunks = []

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._current is None:
            return
        if self._capture and ((tag == "h3" and self._capture == "title") or (tag == "p" and self._capture == "snippet")):
            text = " ".join("".join(self._chunks).split())
            self._current[self._capture] = html.unescape(text)
            self._capture = None
            self._chunks = []
        elif tag == "article" and self._in_result:
            if self._current.get("title") and self._current.get("url"):
                self.results.append(self._current)
            self._current = None
            self._in_result = False


def http_json(url: str, *, timeout: int = 20, data: dict[str, Any] | None = None) -> Any:
    body = None
    headers = {"User-Agent": "homelab-mcp-hub/0.1"}
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def web_search_impl(query: str, max_results: int = 5) -> dict[str, Any]:
    if not query.strip():
        return {"ok": False, "error": "query is required", "results": []}
    max_results = max(1, min(int(max_results), 10))
    params = {
        "q": query,
        "engines": CONFIG.searxng_engines,
    }
    url = f"{CONFIG.searxng_search_url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "homelab-mcp-hub/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return {"ok": False, "error": f"SearXNG HTTP {exc.code}", "results": []}
    except OSError as exc:
        return {"ok": False, "error": str(exc), "results": []}

    parser = SearxHtmlParser()
    parser.feed(body)
    return {
        "ok": True,
        "backend": "searxng",
        "engines": CONFIG.searxng_engines,
        "results": parser.results[:max_results],
    }


def read_url_impl(url: str, max_chars: int = DEFAULT_READ_CHARS) -> dict[str, Any]:
    max_chars = clamp_int(max_chars, 1_000, MAX_READ_CHARS)
    try:
        final_url, content_type, body = fetch_bytes(url)
        kind, title, text = extract_readable_text(body, content_type, final_url)
        text, truncated = truncate_text(text, max_chars)
    except urllib.error.HTTPError as exc:
        return {"ok": False, "url": url, "error": f"HTTP {exc.code}", "text": ""}
    except Exception as exc:
        return {"ok": False, "url": url, "error": str(exc), "text": ""}

    return {
        "ok": True,
        "url": url,
        "final_url": final_url,
        "content_type": content_type,
        "kind": kind,
        "title": title,
        "text": text,
        "chars": len(text),
        "truncated": truncated,
    }


@mcp.tool
def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search the web through the homelab SearXNG instance and return compact results."""
    log_tool_call("web_search", {"query": query, "max_results": max_results})
    return web_search_impl(query, max_results=max_results)


@mcp.tool
def read_url(url: str, max_chars: int = DEFAULT_READ_CHARS) -> dict[str, Any]:
    """Fetch a URL and return readable page, article, JSON, text, or source-code content."""
    log_tool_call("read_url", {"url": url, "max_chars": max_chars})
    return read_url_impl(url, max_chars=max_chars)


@mcp.tool
def web_search_and_read(query: str, max_results: int = 3, max_chars_per_page: int = 6_000) -> dict[str, Any]:
    """Search the web, fetch top result pages, and return readable text from each source."""
    log_tool_call(
        "web_search_and_read",
        {"query": query, "max_results": max_results, "max_chars_per_page": max_chars_per_page},
    )
    max_results = clamp_int(max_results, 1, 5)
    max_chars_per_page = clamp_int(max_chars_per_page, 1_000, 20_000)
    search = web_search_impl(query, max_results=max_results)
    if not search.get("ok"):
        return search

    pages = []
    for result in search.get("results", [])[:max_results]:
        url = result.get("url", "")
        if not url:
            continue
        page = read_url_impl(url, max_chars=max_chars_per_page)
        pages.append(
            {
                "search_title": result.get("title", ""),
                "search_snippet": result.get("snippet", ""),
                **page,
            }
        )

    return {
        "ok": True,
        "backend": search.get("backend"),
        "engines": search.get("engines"),
        "query": query,
        "pages": pages,
    }


@mcp.tool
def git_status(repo: str) -> dict[str, Any]:
    """Return local git working tree state for a repository."""
    log_tool_call("git_status", {"repo": repo})
    path, git = git_base_cmd(repo)
    status = run_cmd(git + ["status", "--short", "--branch"], timeout=15)
    branch = run_cmd(git + ["branch", "--show-current"], timeout=10)
    return {
        "ok": status["ok"],
        "repo": str(path),
        "branch": branch.get("stdout", "").strip(),
        "status": status.get("stdout", ""),
        "stderr": status.get("stderr", ""),
    }


def git_log_impl(repo: str, since: str | None = None, paths: list[str] | str | None = None, limit: int = 30) -> dict[str, Any]:
    path, git = git_base_cmd(repo)
    limit = clamp_int(limit, 1, 200)
    paths = normalize_string_list(paths)
    fmt = "%H%x1f%h%x1f%aI%x1f%an%x1f%s"
    cmd = git + ["log", f"--max-count={limit}", f"--pretty=format:{fmt}"]
    if since:
        cmd.append(f"--since={since}")
    if paths:
        cmd.extend(["--", *paths])
    proc = run_cmd(cmd, timeout=30)
    commits = []
    for line in proc.get("stdout", "").splitlines():
        parts = line.split("\x1f", 4)
        if len(parts) == 5:
            commits.append(
                {
                    "hash": parts[0],
                    "short_hash": parts[1],
                    "date": parts[2],
                    "author": parts[3],
                    "subject": parts[4],
                }
            )
    return {"ok": proc["ok"], "repo": str(path), "commits": commits, "stderr": proc.get("stderr", "")}


@mcp.tool
def git_log(repo: str, since: str | None = None, paths: list[str] | str | None = None, limit: int = 30) -> dict[str, Any]:
    """Return structured local git commit history."""
    log_tool_call("git_log", {"repo": repo, "since": since, "paths": paths, "limit": limit})
    return git_log_impl(repo, since=since, paths=paths, limit=limit)


@mcp.tool
def git_diff(repo: str, ref_a: str = "HEAD", ref_b: str | None = None, paths: list[str] | str | None = None) -> dict[str, Any]:
    """Return a local git diff between refs, or ref and working tree."""
    log_tool_call("git_diff", {"repo": repo, "ref_a": ref_a, "ref_b": ref_b, "paths": paths})
    path, git = git_base_cmd(repo)
    paths = normalize_string_list(paths)
    spec = f"{ref_a}..{ref_b}" if ref_b else ref_a
    cmd = git + ["diff", "--stat", "--patch", spec]
    if paths:
        cmd.extend(["--", *paths])
    proc = run_cmd(cmd, timeout=45)
    return {"ok": proc["ok"], "repo": str(path), "diff": proc.get("stdout", ""), "stderr": proc.get("stderr", "")}


@mcp.tool
def git_recent_changes(repo: str, days: int = 7) -> dict[str, Any]:
    """Return files and commits changed recently in a local git repository."""
    log_tool_call("git_recent_changes", {"repo": repo, "days": days})
    path, git = git_base_cmd(repo)
    days = clamp_int(days, 1, 365)
    since = f"{days} days ago"
    commits = git_log_impl(repo, since=since, limit=100)
    files_proc = run_cmd(git + ["log", f"--since={since}", "--name-only", "--pretty=format:"], timeout=30)
    files = sorted({line.strip() for line in files_proc.get("stdout", "").splitlines() if line.strip()})
    return {
        "ok": commits.get("ok") and files_proc.get("ok"),
        "repo": str(path),
        "days": days,
        "commits": commits.get("commits", []),
        "files": files,
        "stderr": files_proc.get("stderr", ""),
    }


@mcp.tool
def git_search_commits(repo: str, query: str, limit: int = 50) -> dict[str, Any]:
    """Search local git commit messages."""
    log_tool_call("git_search_commits", {"repo": repo, "query": query, "limit": limit})
    path, git = git_base_cmd(repo)
    limit = clamp_int(limit, 1, 200)
    proc = run_cmd(
        git + ["log", f"--max-count={limit}", "--regexp-ignore-case", "--grep", query, "--pretty=format:%H%x1f%h%x1f%aI%x1f%s"],
        timeout=30,
    )
    commits = []
    for line in proc.get("stdout", "").splitlines():
        parts = line.split("\x1f", 3)
        if len(parts) == 4:
            commits.append({"hash": parts[0], "short_hash": parts[1], "date": parts[2], "subject": parts[3]})
    return {"ok": proc["ok"], "repo": str(path), "commits": commits, "stderr": proc.get("stderr", "")}


@mcp.tool
def rg_search(
    query: str,
    paths: list[str] | str | None = None,
    file_glob: str | None = None,
    context_lines: int = 2,
    max_results: int = 80,
) -> dict[str, Any]:
    """Search local files with ripgrep and return structured matches."""
    log_tool_call(
        "rg_search",
        {"query": query, "paths": paths, "file_glob": file_glob, "context_lines": context_lines, "max_results": max_results},
    )
    if not query.strip():
        return {"ok": False, "error": "query is required", "matches": []}
    paths = normalize_string_list(paths)
    context_lines = clamp_int(context_lines, 0, 8)
    max_results = clamp_int(max_results, 1, 500)
    search_paths = [str(repo_path(p)) for p in paths] if paths else [CONFIG.homelab_root, CONFIG.bin_dir]
    cmd = ["rg", "--json", "-n", "-S", f"-C{context_lines}"]
    if file_glob:
        cmd.extend(["-g", file_glob])
    cmd.extend([query, *search_paths])
    proc = run_cmd(cmd, timeout=45)
    matches = []
    for line in proc.get("stdout", "").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "match":
            continue
        data = event.get("data", {})
        path_text = data.get("path", {}).get("text", "")
        lines_text = data.get("lines", {}).get("text", "").rstrip("\n")
        matches.append({"path": path_text, "line": data.get("line_number"), "text": lines_text})
        if len(matches) >= max_results:
            break
    ok = proc["ok"] or proc.get("returncode") == 1
    return {"ok": ok, "query": query, "matches": matches, "stderr": proc.get("stderr", "")}


SCRIPT_HEADER_RE = re.compile(r"^#\s*(DESC|TAGS|ARGS):\s*(.*)$")


@mcp.tool
def list_scripts(tag: str | None = None, query: str | None = None) -> dict[str, Any]:
    """List ~/bin scripts by DESC/TAGS/ARGS comment metadata."""
    log_tool_call("list_scripts", {"tag": tag, "query": query})
    scripts = []
    for path in sorted(Path(CONFIG.bin_dir).expanduser().glob("*")):
        if not path.is_file():
            continue
        meta = {"path": str(path), "name": path.name, "desc": "", "tags": [], "args": ""}
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[:30]
        except Exception:
            continue
        for line in lines:
            m = SCRIPT_HEADER_RE.match(line)
            if not m:
                continue
            key, value = m.group(1), m.group(2).strip()
            if key == "DESC":
                meta["desc"] = value
            elif key == "TAGS":
                meta["tags"] = [part.strip() for part in value.split(",") if part.strip()]
            elif key == "ARGS":
                meta["args"] = value
        haystack = " ".join([meta["name"], meta["desc"], meta["args"], " ".join(meta["tags"])]).lower()
        if tag and tag.lower() not in [str(t).lower() for t in meta["tags"]]:
            continue
        if query and query.lower() not in haystack:
            continue
        if meta["desc"] or meta["tags"] or meta["args"]:
            scripts.append(meta)
    return {"ok": True, "scripts": scripts}


@mcp.tool
def run_cli(command: str, cwd: str | None = None, timeout: int = 60) -> dict[str, Any]:
    """Run a shell command locally on comrade without nested SSH quoting."""
    log_tool_call("run_cli", {"command": command, "cwd": cwd, "timeout": timeout})
    if not command.strip():
        return {"ok": False, "error": "command is required", "stdout": "", "stderr": ""}
    workdir = resolve_cwd(cwd)
    if not workdir.exists():
        return {"ok": False, "error": f"cwd does not exist: {workdir}", "stdout": "", "stderr": ""}
    timeout = clamp_int(timeout, 1, 600)
    proc = run_cmd(["bash", "-lc", command], cwd=workdir, timeout=timeout)
    return {
        "ok": proc["ok"],
        "cwd": str(workdir),
        "returncode": proc.get("returncode"),
        "stdout": proc.get("stdout", ""),
        "stderr": proc.get("stderr", ""),
        "truncated": proc.get("truncated", False),
    }


@mcp.tool
def vector_index(
    paths: list[str] | str | None = None,
    resume: bool = True,
    collection: str | None = None,
) -> dict[str, Any]:
    """Index homelab knowledge into Chroma. Pass collection name (homelab/bin/notes/…) to target one DB; omit to index all."""
    log_tool_call("vector_index", {"paths": paths, "resume": resume, "collection": collection})
    paths = normalize_string_list(paths)
    configs = get_named_configs(collection)
    if not configs:
        return {"ok": False, "error": f"unknown collection: {collection}"}
    try:
        results = {}
        for cfg in configs:
            results[cfg.collection_name] = vector_index_paths(cfg, paths=paths, resume=bool(resume and paths is None))
        return {"ok": True, "collections": results} if len(results) > 1 else next(iter(results.values()))
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@mcp.tool
def vector_index_incremental(fallback_full: bool = False, collection: str | None = None) -> dict[str, Any]:
    """Index only files changed since the stored btrfs transid. Optionally target one collection."""
    log_tool_call("vector_index_incremental", {"fallback_full": fallback_full, "collection": collection})
    configs = get_named_configs(collection)
    if not configs:
        return {"ok": False, "error": f"unknown collection: {collection}"}
    try:
        results = {}
        for cfg in configs:
            results[cfg.collection_name] = vector_index_incremental_impl(cfg, fallback_full=fallback_full)
        return {"ok": True, "collections": results} if len(results) > 1 else next(iter(results.values()))
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@mcp.tool
def vector_mark_indexed_now(collection: str | None = None) -> dict[str, Any]:
    """Store the current btrfs generation as the vector index baseline. Optionally target one collection."""
    log_tool_call("vector_mark_indexed_now", {"collection": collection})
    configs = get_named_configs(collection)
    if not configs:
        return {"ok": False, "error": f"unknown collection: {collection}"}
    try:
        results = {}
        for cfg in configs:
            results[cfg.collection_name] = vector_update_generation(cfg, "manual-mark")
        return {"ok": True, "collections": results} if len(results) > 1 else next(iter(results.values()))
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@mcp.tool
def vector_search(
    query: str,
    top_k: int = 8,
    paths: list[str] | str | None = None,
    collection: str | None = None,
) -> dict[str, Any]:
    """Semantic search over local knowledge. Optionally filter by collection (homelab/bin/notes/…); searches all by default."""
    log_tool_call("vector_search", {"query": query, "top_k": top_k, "paths": paths, "collection": collection})
    paths = normalize_string_list(paths)
    configs = get_named_configs(collection)
    if not configs:
        return {"ok": False, "error": f"unknown collection: {collection}", "results": []}
    try:
        if len(configs) == 1:
            result = vector_search_impl(configs[0], query=query, top_k=top_k, paths=paths)
            for r in result.get("results", []):
                r.setdefault("collection", configs[0].collection_name)
            return result
        return vector_search_multi_impl(configs, query=query, top_k=top_k, paths=paths)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "results": []}


@mcp.tool
def vector_status(
    paths: list[str] | str | None = None,
    sample_limit: int = 50,
    collection: str | None = None,
) -> dict[str, Any]:
    """Report embedded files plus live files that are missing, changed, or stale. Optionally target one collection."""
    log_tool_call("vector_status", {"paths": paths, "sample_limit": sample_limit, "collection": collection})
    paths = normalize_string_list(paths)
    configs = get_named_configs(collection)
    if not configs:
        return {"ok": False, "error": f"unknown collection: {collection}"}
    try:
        results = {}
        for cfg in configs:
            results[cfg.collection_name] = vector_status_impl(cfg, paths=paths, sample_limit=sample_limit)
        return {"ok": True, "collections": results} if len(results) > 1 else next(iter(results.values()))
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@mcp.tool
def vector_cleanup_stale(collection: str | None = None) -> dict[str, Any]:
    """Delete Chroma vectors whose source file no longer exists. Optionally target one collection."""
    log_tool_call("vector_cleanup_stale", {"collection": collection})
    configs = get_named_configs(collection)
    if not configs:
        return {"ok": False, "error": f"unknown collection: {collection}"}
    try:
        results = {}
        for cfg in configs:
            deleted = vector_cleanup_stale_impl(chroma_collection(cfg))
            results[cfg.collection_name] = {"stale_deleted": deleted}
        return {"ok": True, "collections": results} if len(results) > 1 else {"ok": True, **next(iter(results.values()))}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@mcp.tool
def vector_embedder_status() -> dict[str, Any]:
    """Report whether the local sentence-transformers embedder is resident in this MCP process."""
    log_tool_call("vector_embedder_status", {})
    status = vector_embedder_status_impl()
    status.update(
        {
            "ok": True,
            "backend": CONFIG.embedding_backend,
            "model": CONFIG.embedding_model,
            "device": CONFIG.embedding_device,
            "unload_cooldown_seconds": CONFIG.embedding_unload_cooldown_seconds,
        }
    )
    return status


@mcp.tool
def vector_unload_embedder() -> dict[str, Any]:
    """Immediately drop the local sentence-transformers embedder and clear PyTorch GPU caches."""
    log_tool_call("vector_unload_embedder", {})
    try:
        before = vector_embedder_status_impl()
        vector_unload_embedder_impl()
        after = vector_embedder_status_impl()
        return {"ok": True, "before": before, "after": after}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def first_ok_output(commands: list[list[str]], timeout: int = 10) -> dict[str, Any]:
    for cmd in commands:
        if shutil.which(cmd[0]) is None:
            continue
        proc = run_cmd(cmd, timeout=timeout)
        if proc["ok"] and proc.get("stdout", "").strip():
            return proc
    return {"ok": False, "stdout": "", "stderr": "no command produced output"}


def newest_root_snapshot() -> dict[str, Any]:
    find_script = (
        "find \"$1\" -maxdepth 1 -type f "
        "\\( -name 'BIOS-ROCM-Fedora_SLIM_*.md' -o -name 'fedora-snapshot_system_*.md' \\) "
        "-printf '%T@\\t%s\\t%p\\n' 2>/dev/null | sort -nr | head -1"
    )
    proc = run_cmd(["sudo", "-n", "bash", "-lc", find_script, "bash", CONFIG.snapshot_root], timeout=15)
    if not proc["ok"] or not proc.get("stdout", "").strip():
        return {"ok": False, "error": proc.get("stderr", "no snapshot found").strip(), "root": CONFIG.snapshot_root}
    line = proc["stdout"].splitlines()[0]
    try:
        mtime_text, size_text, path = line.split("\t", 2)
    except ValueError:
        return {"ok": False, "error": f"unexpected snapshot listing: {line}", "root": CONFIG.snapshot_root}
    preview_chars = clamp_int(CONFIG.snapshot_preview_chars, 1_000, 80_000)
    text_proc = run_cmd(["sudo", "-n", "head", "-c", str(preview_chars), path], timeout=15)
    return {
        "ok": text_proc["ok"],
        "path": path,
        "size_bytes": int(float(size_text)),
        "mtime": dt.datetime.fromtimestamp(float(mtime_text), dt.timezone.utc).isoformat(),
        "preview_chars": preview_chars,
        "preview": text_proc.get("stdout", ""),
        "stderr": text_proc.get("stderr", ""),
    }


@mcp.tool
def homelab_state() -> dict[str, Any]:
    """Return the newest daily root snapshot plus a small live delta for comrade."""
    log_tool_call("homelab_state", {})
    snapshot = newest_root_snapshot()
    user_services = run_cmd(["systemctl", "--user", "--failed", "--no-pager", "--plain"], timeout=10)
    system_failed = run_cmd(["systemctl", "--failed", "--no-pager", "--plain"], timeout=10)
    journal = run_cmd(["journalctl", "-p", "err..alert", "--since", "2 hours ago", "--no-pager", "-n", "60"], timeout=15)
    disk = run_cmd(["df", "-hT", CONFIG.homelab_root, "/home/comrade", "/mnt/DDRXT4", "/mnt/pixel"], timeout=10)
    tailscale = run_cmd(["tailscale", "status", "--json"], timeout=15) if shutil.which("tailscale") else {"ok": False, "stdout": ""}
    sensors = first_ok_output([["sensors", "-j"], ["sensors"]], timeout=10)
    processes = run_cmd(
        ["pgrep", "-af", "vllm|ollama|iree|comfy|sunshine|ffmpeg|ffplay|litert|mcp|smbd|tailscaled"],
        timeout=10,
    )
    sunshine = run_cmd(["pgrep", "-af", "sunshine"], timeout=10)
    return {
        "ok": True,
        "host": os.uname().nodename,
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "daily_root_snapshot": snapshot,
        "systemd_user_failed": user_services.get("stdout", ""),
        "systemd_failed": system_failed.get("stdout", ""),
        "recent_journal_warnings": journal.get("stdout", ""),
        "disk": disk.get("stdout", ""),
        "tailscale": tailscale.get("stdout", ""),
        "sensors": sensors.get("stdout", ""),
        "processes_of_interest": processes.get("stdout", ""),
        "sunshine": sunshine.get("stdout", ""),
    }


@mcp.tool
def gh_list_repos() -> dict[str, Any]:
    """List GitHub repositories visible to the configured token."""
    log_tool_call("gh_list_repos", {})
    try:
        repos = github_request("GET", "/user/repos?per_page=100&sort=updated")
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "repos": [
            {
                "full_name": r.get("full_name"),
                "private": r.get("private"),
                "updated_at": r.get("updated_at"),
                "description": r.get("description"),
            }
            for r in repos
        ],
    }


@mcp.tool
def gh_create_repo(name: str, private: bool = True, description: str | None = None) -> dict[str, Any]:
    """Create a GitHub repository using the configured token."""
    log_tool_call("gh_create_repo", {"name": name, "private": private, "description": description})
    try:
        repo = github_request("POST", "/user/repos", {"name": name, "private": private, "description": description or ""})
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "repo": repo.get("full_name"), "html_url": repo.get("html_url")}


@mcp.tool
def gh_list_issues(repo: str, state: str = "open") -> dict[str, Any]:
    """List GitHub issues for owner/repo."""
    log_tool_call("gh_list_issues", {"repo": repo, "state": state})
    try:
        issues = github_request("GET", f"/repos/{repo}/issues?state={urllib.parse.quote(state)}&per_page=100")
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "issues": [{"number": i.get("number"), "title": i.get("title"), "state": i.get("state"), "url": i.get("html_url")} for i in issues]}


@mcp.tool
def gh_create_issue(repo: str, title: str, body: str = "") -> dict[str, Any]:
    """Create a GitHub issue for owner/repo."""
    log_tool_call("gh_create_issue", {"repo": repo, "title": title, "body": body})
    try:
        issue = github_request("POST", f"/repos/{repo}/issues", {"title": title, "body": body})
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "number": issue.get("number"), "url": issue.get("html_url")}


@mcp.tool
def gh_create_pr(repo: str, title: str, body: str, head: str, base: str = "main") -> dict[str, Any]:
    """Create a GitHub pull request for owner/repo."""
    log_tool_call("gh_create_pr", {"repo": repo, "title": title, "head": head, "base": base})
    try:
        pr = github_request("POST", f"/repos/{repo}/pulls", {"title": title, "body": body, "head": head, "base": base})
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "number": pr.get("number"), "url": pr.get("html_url")}


@mcp.tool
def gh_commit(repo: str, path: str, content: str, message: str, branch: str | None = None) -> dict[str, Any]:
    """Create or update one file in a GitHub repository."""
    log_tool_call("gh_commit", {"repo": repo, "path": path, "message": message, "branch": branch})
    try:
        query = f"?ref={urllib.parse.quote(branch)}" if branch else ""
        existing_sha = None
        try:
            existing = github_request("GET", f"/repos/{repo}/contents/{urllib.parse.quote(path)}{query}")
            existing_sha = existing.get("sha")
        except Exception:
            existing_sha = None
        payload: dict[str, Any] = {
            "message": message,
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        }
        if branch:
            payload["branch"] = branch
        if existing_sha:
            payload["sha"] = existing_sha
        result = github_request("PUT", f"/repos/{repo}/contents/{urllib.parse.quote(path)}", payload)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "commit": result.get("commit", {}).get("sha"), "url": result.get("content", {}).get("html_url")}


@mcp.tool
def gh_push(repo: str, branch: str | None = None) -> dict[str, Any]:
    """Push a local git repository with the system git CLI."""
    log_tool_call("gh_push", {"repo": repo, "branch": branch})
    path, git = git_base_cmd(repo)
    cmd = git + ["push"]
    if branch:
        cmd.extend(["origin", branch])
    proc = run_cmd(cmd, timeout=120)
    return {"ok": proc["ok"], "repo": str(path), "stdout": proc.get("stdout", ""), "stderr": proc.get("stderr", "")}


@mcp.tool
def hf_list_repos() -> dict[str, Any]:
    """List HuggingFace repos visible to the configured token."""
    log_tool_call("hf_list_repos", {})
    try:
        api = hf_api()
        user = api.whoami()
        username = user.get("name") or user.get("fullname")
        repos = []
        for repo_type, iterator in (
            ("model", api.list_models(author=username)),
            ("dataset", api.list_datasets(author=username)),
            ("space", api.list_spaces(author=username)),
        ):
            for repo in iterator:
                repos.append(
                    {
                        "id": getattr(repo, "id", None),
                        "type": repo_type,
                        "private": getattr(repo, "private", None),
                        "last_modified": str(getattr(repo, "last_modified", "")),
                    }
                )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "repos": repos}


@mcp.tool
def hf_create_repo(name: str, private: bool = True, repo_type: str = "model") -> dict[str, Any]:
    """Create a HuggingFace model, dataset, or space repo."""
    log_tool_call("hf_create_repo", {"name": name, "private": private, "repo_type": repo_type})
    try:
        url = hf_api().create_repo(repo_id=name, private=private, repo_type=repo_type, exist_ok=True)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "url": str(url)}


@mcp.tool
def hf_upload_file(repo: str, local_path: str, repo_path: str, repo_type: str = "model") -> dict[str, Any]:
    """Upload one local file to a HuggingFace repo."""
    log_tool_call("hf_upload_file", {"repo": repo, "local_path": local_path, "repo_path": repo_path, "repo_type": repo_type})
    try:
        info = hf_api().upload_file(path_or_fileobj=local_path, path_in_repo=repo_path, repo_id=repo, repo_type=repo_type)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "url": str(info)}


@mcp.tool
def hf_upload_folder(repo: str, local_dir: str, repo_type: str = "model") -> dict[str, Any]:
    """Upload a local folder to a HuggingFace repo."""
    log_tool_call("hf_upload_folder", {"repo": repo, "local_dir": local_dir, "repo_type": repo_type})
    try:
        info = hf_api().upload_folder(folder_path=local_dir, repo_id=repo, repo_type=repo_type)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "url": str(info)}


@mcp.tool
def hf_update_model_card(repo: str, content: str, repo_type: str = "model") -> dict[str, Any]:
    """Write README.md in a HuggingFace repo."""
    log_tool_call("hf_update_model_card", {"repo": repo, "repo_type": repo_type})
    try:
        info = hf_api().upload_file(
            path_or_fileobj=content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type=repo_type,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "url": str(info)}


@mcp.tool
def hf_delete_file(repo: str, path: str, repo_type: str = "model") -> dict[str, Any]:
    """Delete one file from a HuggingFace repo."""
    log_tool_call("hf_delete_file", {"repo": repo, "path": path, "repo_type": repo_type})
    try:
        info = hf_api().delete_file(path_in_repo=path, repo_id=repo, repo_type=repo_type)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "result": str(info)}


@mcp.tool
def hub_info() -> dict[str, Any]:
    """Return the homelab MCP hub configuration summary."""
    log_tool_call("hub_info", {})
    return {
        "ok": True,
        "mcp_path": CONFIG.mcp_path,
        "searxng_search_url": CONFIG.searxng_search_url,
        "searxng_engines": CONFIG.searxng_engines,
        "homelab_root": CONFIG.homelab_root,
        "bin_dir": CONFIG.bin_dir,
        "call_log_path": CONFIG.call_log_path,
        "snapshot_root": CONFIG.snapshot_root,
        "chroma_dir": CONFIG.chroma_dir,
        "vector_collection": CONFIG.vector_collection,
        "embedding_backend": CONFIG.embedding_backend,
        "embedding_model": CONFIG.embedding_model,
        "embedding_device": CONFIG.embedding_device,
        "embedding_batch_size": CONFIG.embedding_batch_size,
        "ollama_url": CONFIG.ollama_url,
        "indexed_roots": list(CONFIG.indexed_roots),
        "vector_collections": [
            {"name": name, "roots": list(roots)} for name, roots in CONFIG.vector_collections
        ] if CONFIG.vector_collections else None,
        "vector_incremental_enabled": CONFIG.vector_incremental_enabled,
        "vector_btrfs_subvolume": CONFIG.vector_btrfs_subvolume,
        "vector_state_path": CONFIG.vector_state_path,
        "vector_stores_original_text": False,
        "github_token_configured": bool(github_token()),
        "hf_token_configured": hf_token_configured(),
    }


def main() -> None:
    mcp.run(
        transport="streamable-http",
        host=CONFIG.bind_host,
        port=CONFIG.port,
        path=CONFIG.mcp_path,
    )


if __name__ == "__main__":
    main()

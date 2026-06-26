# MCP Hub Vector Store — Runbook

Quick reference for indexing, searching, and maintaining the three knowledge collections.

---

## What this is

Two retrieval consumers share this one indexed corpus:

| Consumer | Where | How it's used |
|---|---|---|
| **mcp-hub** (this) | port 8765 | Kai/Claude calls `vector_search` as an explicit MCP tool. The AI decides to search and sees results. |
| **linter-lm** | port 8099 | Invisible injection. Every prompt is silently prefixed with matching chunks before the LLM sees it. Model never knows. |

Both use the same iGPU Qwen3-Embedding-0.6B server (port 18084) and the same ChromaDB at `/home/comrade/.local/share/mcp-hub/chroma`. Index once, two consumers benefit automatically.

## What gets indexed

Files are split into **chunks** (up to 3 500 chars, 2-line overlap). Only chunk metadata (path, line range, sha256, summary) is stored in ChromaDB — the original text stays on disk. At search time, the matching line range is read live from the file.

So search results are **excerpt-sized** (a function, a section, a paragraph), not whole files.

---

## Collections

| Name | Root | State file |
|---|---|---|
| `homelab` | `/home/comrade/homelab` | `homelab-index-state.json` |
| `bin` | `/home/comrade/bin` | `bin-index-state.json` |
| `notes` | `/home/comrade/Obsidian/Vault` | `notes-index-state.json` |

State files live under `/home/comrade/.local/share/mcp-hub/`.  
ChromaDB: `/home/comrade/.local/share/mcp-hub/chroma/`.

Add a collection: append to `vector_collections` in `config.json` and run `--full` for that name.

---

## Indexer commands

All indexer commands use:

```bash
IDXR="HOMELAB_MCP_CONFIG=/home/comrade/homelab/mcp-hub/config.json \
  /home/comrade/comfyui-gpu/.venv/bin/python \
  /home/comrade/homelab/mcp-hub/indexer.py"
```

### List collections

```bash
$IDXR --list-collections
```

### Full index of one collection

```bash
$IDXR --collection homelab --full
$IDXR --collection bin    --full
$IDXR --collection notes  --full
```

Resumable by default — skips files already embedded at the same hash. Progress goes to stderr.

### Force full re-index from scratch (drop resume history)

```bash
$IDXR --collection homelab --full --no-resume
```

### Index a specific directory or file into a collection

```bash
$IDXR --collection homelab --path /home/comrade/homelab/Poll-E
$IDXR --collection homelab --path /home/comrade/homelab/linter-lm/server.py
```

Use this after working on a project to push only that project's changes without re-crawling everything.

### Incremental (btrfs find-new since last transid)

```bash
$IDXR --collection homelab --incremental
$IDXR --collection homelab --incremental --fallback-full   # full crawl if no baseline
```

The daily timer (`homelab-mcp-indexer.timer`) runs `--incremental` on all collections automatically at 00:45.

### Check coverage without embedding

```bash
$IDXR --collection homelab --status
$IDXR --collection notes   --status
```

Shows missing, changed, and stale file counts.

### Store current btrfs generation as incremental baseline

```bash
$IDXR --collection homelab --mark-indexed-now
```

Run this after a manual `--full` so the next incremental knows where to start.

### Delete stale vectors (source file deleted)

```bash
$IDXR --collection homelab --cleanup-stale
```

### Monitor background indexing

```bash
tail -f /tmp/mcp-idx-homelab.log
tail -f /tmp/mcp-idx-notes.log
```

---

## Background indexing (nohup)

```bash
IDXR_CMD="/home/comrade/comfyui-gpu/.venv/bin/python /home/comrade/homelab/mcp-hub/indexer.py"

nohup $IDXR_CMD --collection homelab --full --quiet > /tmp/mcp-idx-homelab.log 2>&1 &
nohup $IDXR_CMD --collection notes   --full --quiet > /tmp/mcp-idx-notes.log   2>&1 &
nohup $IDXR_CMD --collection bin     --full --quiet > /tmp/mcp-idx-bin.log     2>&1 &
```

---

## Service management

```bash
# Status
systemctl --user status homelab-mcp-hub.service
systemctl --user status homelab-mcp-indexer.timer

# Restart hub (after code changes)
systemctl --user restart homelab-mcp-hub.service

# Trigger indexer now (runs incremental for all collections)
systemctl --user start homelab-mcp-indexer.service

# Logs
journalctl --user -u homelab-mcp-hub.service -n 50
```

---

## MCP tool usage (from Kai/Claude)

```
# Search all collections, merged by relevance
vector_search(query="how does the embedding pipeline work", top_k=8)

# Search a specific collection only
vector_search(query="Obsidian daily notes format", collection="notes", top_k=5)
vector_search(query="lintr-toggle-context script", collection="bin", top_k=3)

# Index a specific path into a collection (triggered from Kai mid-session)
vector_index(paths=["/home/comrade/homelab/Poll-E"], collection="homelab")

# Incremental update
vector_index_incremental(collection="homelab")

# Coverage report
vector_status(collection="homelab")
```

---

## Add a new collection

1. Add an entry to `vector_collections` in `/home/comrade/homelab/mcp-hub/config.json`:

```json
{"name": "media", "roots": ["/home/comrade/path/to/lyrics"]}
```

2. Index it:

```bash
$IDXR --collection media --full
```

3. Restart the hub so the new collection is visible to MCP tools:

```bash
systemctl --user restart homelab-mcp-hub.service
```

---

## Embedding server

`llama-embed.service` — runs `llama-server` on the iGPU (`ROCR_VISIBLE_DEVICES=1`) at `127.0.0.1:18084`.  
Model: `Qwen/Qwen3-Embedding-0.6B` GGUF Q8_0, 1024-dim.

```bash
systemctl --user status llama-embed.service
curl -s http://127.0.0.1:18084/health
```

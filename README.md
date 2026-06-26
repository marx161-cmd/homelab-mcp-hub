# Homelab MCP Hub

Streamable HTTP MCP server for Kai 9000 and other MCP clients.

Endpoint:

```text
http://100.108.8.60:8765/mcp
```

Current tools:

- `web_search`: searches through the Dell SearXNG service.
- `read_url`: fetches a URL and extracts readable page, article, JSON, text, or source-code content.
- `web_search_and_read`: searches, opens the top result pages, and returns extracted source text.
- `git_status`, `git_log`, `git_diff`, `git_recent_changes`, `git_search_commits`: local git context.
- `rg_search`: structured ripgrep over `/home/comrade/homelab` and `/home/comrade/bin`.
- `list_scripts`: lists `~/bin` scripts by `DESC`/`TAGS`/`ARGS` headers.
- `vector_index`: indexes configured paths into Chroma. Pass `collection` to target one DB (homelab/bin/notes/…); omit to index all. `paths` overrides roots for targeted re-indexing of a single directory or file.
- `vector_index_incremental`: uses `btrfs subvolume find-new` from the stored transid and re-embeds only changed files. Also accepts `collection`.
- `vector_search`: semantic search over all collections (merged by cosine distance) or a single `collection`. Returns file path, line range, summary, and live resolved text.
- `vector_status`: reports Chroma coverage and live files that are missing, changed, or stale. Accepts `collection`.
- `vector_mark_indexed_now`: stores the current btrfs generation as the incremental baseline.
- `vector_cleanup_stale`: deletes vectors whose source file no longer exists.
- `run_cli`: runs a local shell command on comrade from JSON args, avoiding nested SSH quoting.
- `homelab_state`: newest daily root snapshot from `/root/Documents` plus a small live delta.
- `gh_*`: GitHub REST helpers using `GITHUB_TOKEN`/`GH_TOKEN`, falling back to `gh auth token`.
- `hf_*`: HuggingFace helpers using `HF_TOKEN`/`HUGGINGFACEHUB_API_TOKEN`, falling back to the normal `huggingface_hub` login cache.
- `hub_info`: shows current hub config.

Run manually:

```bash
/home/comrade/comfyui-gpu/.venv/bin/python /home/comrade/homelab/mcp-hub/server.py
```

Kai setup:

1. Open Kai settings.
2. Go to Tools.
3. Add MCP server.
4. Name: `Homelab`
5. URL: `http://100.108.8.60:8765/mcp`

Known issues / design notes:

- **Accept header workaround**: Kai 9000's MCP client omits the `Accept` header on
  JSON-RPC notifications (e.g. ``notifications/initialized``).  The upstream
  ``mcp`` library rejects those with ``406 Not Acceptable``, stalling the
  session handshake. `server.py` monkey-patches the streamable HTTP transport's
  Accept validator so Kai can finish the session handshake.

- SearXNG JSON is currently disabled on the Dell, so `web_search` parses the HTML result page.
- The hub pins SearXNG to `engines=duckduckgo` because the Dell SearXNG `wikidata` processor is currently broken and makes default searches return HTTP 500.
- Service-specific health tools were removed. Use `homelab_state` for coarse context, and use the normal CLI path for detailed service debugging.
- Chroma stores embeddings plus pointer metadata only: source path, line range, mtime, size, hash, and a short summary. Original source text stays on disk and is read live during `vector_search`.
- The vector backend is `Qwen/Qwen3-Embedding-0.6B` via the iGPU llama-server (`http://127.0.0.1:18084`, `ROCR_VISIBLE_DEVICES=1`), keeping the dGPU free for ROCm/ComfyUI. Batch size defaults to `8`; the batched `/v1/embeddings` endpoint is used so each batch is a single HTTP call.
- Every tool invocation is appended to `/home/comrade/logs/mcp-calls.jsonl`.

## Collections

Three separate Chroma collections, each independently indexable:

| Name | Roots |
|---|---|
| `homelab` | `/home/comrade/homelab` |
| `bin` | `/home/comrade/bin` |
| `notes` | `/home/comrade/Obsidian/Vault` |

Add more by appending to `vector_collections` in `config.json`.

Manual indexing (pick one collection at a time for fast targeted runs):

```bash
INDEXER=/home/comrade/comfyui-gpu/.venv/bin/python\ /home/comrade/homelab/mcp-hub/indexer.py

# List collections and their state files
$INDEXER --list-collections

# Full index of a single collection
$INDEXER --collection homelab --full
$INDEXER --collection bin    --full
$INDEXER --collection notes  --full

# Target a specific directory into a collection
$INDEXER --collection homelab --path /home/comrade/homelab/Poll-E

# Incremental (btrfs find-new) for a collection
$INDEXER --collection homelab --incremental

# Check coverage without embedding
$INDEXER --collection homelab --status
```

Full indexing is resumable by default — bootstraps from existing Chroma metadata.
Force a from-scratch pass:

```bash
$INDEXER --collection homelab --full --no-resume
```

Check what is embedded and what changed without loading Qwen:

```bash
/home/comrade/comfyui-gpu/.venv/bin/python /home/comrade/homelab/mcp-hub/indexer.py --status
```

Set the current `/home` btrfs generation as the baseline without embedding:

```bash
/home/comrade/comfyui-gpu/.venv/bin/python /home/comrade/homelab/mcp-hub/indexer.py --mark-indexed-now
```

Cleanup stale deleted-file pointers:

```bash
/home/comrade/comfyui-gpu/.venv/bin/python /home/comrade/homelab/mcp-hub/indexer.py --cleanup-stale
```

Incremental btrfs mode requires read-only access to:

```text
btrfs subvolume show /home
btrfs subvolume find-new /home <stored-transid>
```

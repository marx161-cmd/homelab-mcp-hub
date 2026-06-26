# MCP Comrade Upgrade Plan

## Context

Barebones MCP server already running on `comrade` (Ryzen 5 8600G, Fedora, Tailscale).
Goal: upgrade it into a genuinely useful context-awareness and retrieval system accessible from all clients (Claude Code, Kai APK, etc.) without needing one-shot SSH commands.

## Architecture Decisions

**Single monolithic server** — no trust boundary splitting. The sudoer rule is the one lever: passwordless sudo when active, normal user when disabled. No per-tool scoping needed.

**Tailscale-only binding** — server binds exclusively on the Tailscale interface, not `0.0.0.0`. Same trust zone as existing SSH.

**SSE transport** — so Kai and other HTTP clients can connect without SSH at all.

**Full call logging** — every tool invocation logged with full args to a persistent flat file / journald. Recovery aid, not security theatre.

---

## Tool Surface


### Git & Forge Tooling

**Local git — context/read-only**
- `git_log(repo, since?, paths?)` — structured commit list
- `git_diff(repo, ref_a, ref_b?)` — summarisable diff
- `git_recent_changes(repo, days)` — what changed where recently
- `git_search_commits(repo, query)` — grep commit messages
- `git_status(repo)` — working tree state, uncommitted changes

**GitHub — full CRUD via PAT**
- `gh_create_repo(name, private?, description?)` 
- `gh_commit(repo, path, content, message, branch?)` 
- `gh_push(repo, branch?)` 
- `gh_create_pr(repo, title, body, head, base)` 
- `gh_list_issues(repo)` / `gh_create_issue(repo, title, body)`
- `gh_list_repos()` — overview of your repos

Implementation: PyGithub or raw GitHub REST API with PAT in env var.

**HuggingFace — full CRUD via HF Hub library**

- `hf_create_repo(name, private?, repo_type?)` — model/dataset/space
- `hf_upload_file(repo, local_path, repo_path)` 
- `hf_upload_folder(repo, local_dir)` — bulk push
- `hf_update_model_card(repo, content)` 
- `hf_list_repos()` — your published models/datasets
- `hf_delete_file(repo, path)`

Implementation: `huggingface_hub` Python library, HF token in env var.

---

Both PAT and HF token just live in the server's environment, runit `env/` dir is the clean place to put them.

### 2. Ripgrep / Semantic Search

Fast indexed search across the homelab. Complements vector retrieval for exact-match cases.

Tools to implement:
- `rg_search(query, paths?, file_glob?, context_lines?)` — ripgrep with structured output (file, line, match, context)
- `list_scripts(tag?, query?)` — query `~/bin` DESC/TAGS/ARGS comment headers directly, returns matching scripts with metadata

### 3. Homelab State Snapshot

Single tool that returns a composite live picture of comrade. The thing Kai currently has zero visibility into without manual narration.

Tool:
- `homelab_state()` — returns JSON blob with:
  - Running/failed runit/systemd services
  - Recent journal errors (last N lines, filtered)
  - Disk usage (btrfs subvols or df)
  - Tailscale peer status
  - Active Sunshine sessions
  - GPU/CPU thermals (if sensors available)
  - Currently running processes of interest (vllm, ollama, iree, etc.)

### 4. Vector Retrieval (RAG)

Local semantic search over homelab knowledge. The context-awareness layer.

**Embedding model:** `Qwen3-Embedding-0.6B`
- Apache 2.0
- 32K context window
- 596M parameters, negligible on comrade
- Run via `sentence-transformers` or Ollama

**Vector store:** Chroma (local, on comrade)

**Storage pattern — pointer-based, no text duplication:**
- Vector + metadata stored in Chroma: `{"path": "~/homelab/poll-e/arch.md", "start_line": 3, "end_line": 42, "mtime": ...}`
- Original text lives on disk only — never copied into the store
- At retrieval time: MCP server resolves pointer → reads current file content → returns live text
- Dead pointer handling: existence check at retrieval, skip/flag gracefully

**Chunking strategy:**
- Split on structural boundaries (markdown headers, function definitions, comment blocks) not fixed token counts
- Each chunk passed through Gemma 3 1B (already on-device) for summarization
- Embed the **summary**, store pointer to **original**
- Gives semantic retrieval signal without noise

**What gets indexed:**
- `~/homelab/**` (markdown, txt, conf)
- `~/bin` DESC/TAGS/ARGS headers (already structured — nearly free)
- SKILL.md files
- Active project dirs (`/parrot`, etc.)
- Git commit messages + first paragraph of diffs (recent history only)
- runit service definitions

**What gets ignored (ignore list, define once):**
- `node_modules/`, `__pycache__/`, `.git/`
- Build artifacts, compiled outputs (`.vmfb`, `.so`, binaries)
- Large data files, model weights

**Re-indexing:**
- Nightly runit timer job
- Diff-based: check mtime/content hash, re-embed only changed chunks
- Stale pointer cleanup pass (delete vectors whose source files no longer exist)

Tool:
- `vector_search(query, top_k?, paths?)` — returns top-k chunks with file path, line range, summary, and live resolved text

---

## Infrastructure

### Runit Services

```
/etc/sv/mcp-comrade/
  run        # main server process
  log/run    # svlogd

/etc/sv/mcp-indexer/
  run        # nightly re-index script (oneshot triggered by timer)
```

### Call Logging

Every tool invocation appended to `~/logs/mcp-calls.jsonl`:
```json
{"ts": "...", "tool": "git_log", "args": {"repo": "/parrot", "days": 7}, "client": "kai"}
```

### Network

- Bind on Tailscale interface only (`--host <tailscale-ip>`)
- Port: pick one, hardcode in Tailscale DNS or `/etc/hosts` on clients
- No auth beyond Tailscale network membership

---

## Client Integration

**Claude Code** — connect via MCP config pointing at `http://comrade.tail…:PORT/sse`

**Kai APK** — replaces one-shot SSH commands with structured tool calls over the same SSE endpoint. Needs MCP client support wired into the existing tool-call infrastructure.

---

## Implementation Order

1. Add git tools to existing server
2. Add ripgrep + script registry tools
3. Add homelab_state composite tool
4. Set up Chroma + qwen3-embedding-0.6B
5. Write pointer-based indexer with Gemma summarization pass
6. Wire vector_search tool into MCP server
7. Runit units for server + nightly indexer
8. Test from Kai over Tailscale SSE

---

## Out of Scope

- Per-tool permission scoping (sudoer rule is the one lever)
- File read/write tools (shell already covers this)
- Any cloud dependencies — fully local

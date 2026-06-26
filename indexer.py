#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time

from server import CONFIG, all_vector_configs, get_named_configs
from vector_store import chroma_collection
from vector_store import cleanup_stale
from vector_store import index_incremental
from vector_store import index_paths
from vector_store import update_index_generation
from vector_store import vector_config
from vector_store import vector_status


def main() -> None:
    parser = argparse.ArgumentParser(description="Index homelab files into the MCP Chroma vector store.")
    parser.add_argument("--path", action="append", dest="paths", help="File or directory to index. May be repeated.")
    parser.add_argument("--batch-size", type=int, default=None, help="Embedding batch size. Defaults to config.")
    parser.add_argument("--quiet", action="store_true", help="Only print final JSON.")
    parser.add_argument("--status", action="store_true", help="Report index coverage and changed/stale files without embedding.")
    parser.add_argument("--sample-limit", type=int, default=50, help="Maximum changed/stale file samples to print.")
    parser.add_argument("--full", action="store_true", help="Index all configured roots. Required to avoid accidental full crawls.")
    parser.add_argument("--no-resume", action="store_true", help="Force a full reindex instead of skipping already embedded files.")
    parser.add_argument("--resume", action="store_true", help="Resume targeted --path indexing by skipping matching embedded files.")
    parser.add_argument("--incremental", action="store_true", help="Use btrfs find-new from the stored transid.")
    parser.add_argument("--fallback-full", action="store_true", help="Run a full index if no transid state exists.")
    parser.add_argument("--mark-indexed-now", action="store_true", help="Store the current btrfs generation as baseline.")
    parser.add_argument("--cleanup-stale", action="store_true", help="Delete vectors whose source path no longer exists.")
    parser.add_argument("--collection", metavar="NAME", help="Target a single named collection (e.g. homelab, bin, notes). Omit to target all.")
    parser.add_argument("--list-collections", action="store_true", help="Print configured collections and exit.")
    args = parser.parse_args()

    if args.list_collections:
        for cfg in all_vector_configs():
            print(json.dumps({"name": cfg.collection_name, "roots": list(cfg.indexed_roots), "state_path": cfg.state_path}))
        return

    configs = get_named_configs(args.collection)
    if not configs:
        print(json.dumps({"ok": False, "error": f"unknown collection: {args.collection}"}))
        return
    if len(configs) > 1 and not (args.incremental or args.full or args.status or args.mark_indexed_now or args.cleanup_stale):
        print(json.dumps({"ok": False, "error": "multiple collections require --collection NAME, --full, --incremental, --status, --mark-indexed-now, or --cleanup-stale"}))
        return

    cfg = configs[0] if len(configs) == 1 else vector_config(CONFIG)
    if args.mark_indexed_now:
        results = {c.collection_name: update_index_generation(c, "manual-mark") for c in configs}
        print(json.dumps(results if len(results) > 1 else next(iter(results.values())), indent=2, sort_keys=True))
        return
    if args.cleanup_stale:
        results = {c.collection_name: {"stale_deleted": cleanup_stale(chroma_collection(c))} for c in configs}
        print(json.dumps({"ok": True, **(results if len(results) > 1 else next(iter(results.values())))}, indent=2, sort_keys=True))
        return
    if args.status:
        results = {c.collection_name: vector_status(c, paths=args.paths, sample_limit=args.sample_limit) for c in configs}
        print(json.dumps(results if len(results) > 1 else next(iter(results.values())), indent=2, sort_keys=True))
        return

    if args.paths is None and not args.incremental and not args.full:
        result = {
            "ok": False,
            "error": "refusing accidental full index; use --full, --incremental, --status, --collection, or --path",
            "examples": [
                "indexer.py --collection homelab --full",
                "indexer.py --collection bin --full",
                "indexer.py --incremental",
                "indexer.py --path /home/comrade/homelab/Poll-E --collection homelab",
                "indexer.py --status",
                "indexer.py --list-collections",
            ],
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    started = time.monotonic()

    def progress(event: dict[str, object], label: str = "") -> None:
        elapsed = time.monotonic() - started
        prefix = f"[{label}] " if label else ""
        print(
            f"[{elapsed:7.1f}s] {prefix}{event.get('phase')} files={event.get('files_seen')} "
            f"chunks={event.get('chunks_indexed')} path={event.get('path')}",
            file=sys.stderr,
            flush=True,
        )

    if len(configs) > 1:
        # multi-collection run: iterate and print per-collection results
        all_results: dict[str, object] = {}
        for col_cfg in configs:
            col_progress = (lambda e, n=col_cfg.collection_name: progress(e, n)) if not args.quiet else None
            if args.incremental:
                col_result = index_incremental(col_cfg, batch_size=args.batch_size, fallback_full=args.fallback_full, progress=col_progress)
            else:
                col_result = index_paths(col_cfg, paths=args.paths, batch_size=args.batch_size, progress=col_progress,
                                         resume=(args.resume or (args.paths is None and not args.no_resume)))
                if col_result.get("ok") and args.paths is None:
                    col_result["generation_update"] = update_index_generation(col_cfg, "full")
            all_results[col_cfg.collection_name] = col_result
        print(json.dumps({"ok": True, "collections": all_results}, indent=2, sort_keys=True))
        return

    # single collection run
    col_progress = (lambda e: progress(e, cfg.collection_name)) if not args.quiet else None
    if args.incremental:
        result = index_incremental(cfg, batch_size=args.batch_size, fallback_full=args.fallback_full, progress=col_progress)
    else:
        result = index_paths(cfg, paths=args.paths, batch_size=args.batch_size, progress=col_progress,
                              resume=(args.resume or (args.paths is None and not args.no_resume)))
        if result.get("ok") and args.paths is None:
            result["generation_update"] = update_index_generation(cfg, "full")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate basic retrieval recall metrics for memory-mcp-server.

Metrics:
- Hit@1
- Hit@3
- MRR

This script creates a temporary document, inserts known memories, runs
labeled queries, computes metrics, and cleans up by default.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
import uuid
from dataclasses import dataclass
from random import Random
from typing import Any, Dict, Optional
from urllib import error, request


def parse_mcp_response(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    if not text:
        raise RuntimeError("Empty MCP response body")

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    chunk: list[str] = []
    for line in raw.splitlines():
        if line.startswith("data:"):
            chunk.append(line[5:].lstrip())
            continue
        if not line.strip() and chunk:
            candidate = "\n".join(chunk).strip()
            chunk = []
            if not candidate or candidate == "[DONE]":
                continue
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                continue

    if chunk:
        candidate = "\n".join(chunk).strip()
        if candidate and candidate != "[DONE]":
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

    preview = text[:220].replace("\n", "\\n")
    raise RuntimeError(f"Unable to parse MCP response as JSON or SSE JSON payload: {preview}")


@dataclass
class McpClient:
    server_url: str
    timeout_sec: float = 20.0
    snippet_len: int = 180
    verbose: bool = False
    _rpc_id: int = 1
    _session_id: Optional[str] = None

    def rpc(self, method: str, params: Optional[Dict[str, Any]] = None, notify: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        if not notify:
            payload["id"] = self._rpc_id
            self._rpc_id += 1

        body = json.dumps(payload).encode("utf-8")
        req = request.Request(self.server_url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json, text/event-stream")
        if self._session_id:
            req.add_header("Mcp-Session-Id", self._session_id)

        try:
            with request.urlopen(req, timeout=self.timeout_sec) as resp:
                session_id = resp.headers.get("Mcp-Session-Id") or resp.headers.get("mcp-session-id")
                if session_id:
                    self._session_id = session_id
                raw = resp.read().decode("utf-8")
        except error.HTTPError as e:
            msg = e.read().decode("utf-8", errors="replace") if e.fp else str(e)
            raise RuntimeError(f"HTTP {e.code} for {method}: {msg}") from e
        except error.URLError as e:
            raise RuntimeError(f"Failed to connect to MCP server: {e}") from e

        if notify:
            return {}

        if not raw.strip():
            raise RuntimeError(f"Empty response for method {method}")

        if self.verbose:
            snippet = raw.strip().replace("\n", "\\n")[: self.snippet_len]
            print(f"      [rpc:{method}] response snippet: {snippet}")

        data = parse_mcp_response(raw)
        if "error" in data:
            raise RuntimeError(f"MCP error on {method}: {json.dumps(data['error'], ensure_ascii=False)}")
        return data.get("result", {})

    def initialize(self) -> None:
        self.rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "mcp-recall-eval", "version": "1.0.0"},
            },
        )
        self.rpc("notifications/initialized", notify=True)

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return self.rpc("tools/call", {"name": name, "arguments": arguments})


def parse_tool_text(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return text


def extract_tool_result_payload(tool_result: Dict[str, Any]) -> Any:
    content = tool_result.get("content", [])
    for item in content:
        if item.get("type") == "text" and "text" in item:
            return parse_tool_text(item["text"])
    return content


def reciprocal_rank(rank: Optional[int]) -> float:
    if rank is None:
        return 0.0
    return 1.0 / float(rank)


def build_noise_dataset(count: int, seed: int = 42) -> list[Dict[str, str]]:
    rng = Random(seed)
    topics = [
        "gardening",
        "astronomy",
        "finance",
        "history",
        "cooking",
        "sports",
        "music",
        "travel",
        "education",
        "hardware",
    ]
    verbs = [
        "explains",
        "summarizes",
        "compares",
        "documents",
        "analyzes",
        "highlights",
        "describes",
        "reviews",
    ]
    objects = [
        "practical workflows",
        "common pitfalls",
        "maintenance checklists",
        "baseline metrics",
        "field observations",
        "cost tradeoffs",
        "daily routines",
        "operational notes",
    ]

    noise_rows: list[Dict[str, str]] = []
    for i in range(count):
        topic = rng.choice(topics)
        verb = rng.choice(verbs)
        obj = rng.choice(objects)
        token = rng.randint(100000, 999999)
        summary = f"Noise {i + 1:05d} {topic} {verb}"
        content = (
            f"Synthetic noise sample {i + 1}: this note {verb} {obj} in {topic}. "
            f"Marker token {token} to diversify lexical space and increase corpus size."
        )
        noise_rows.append({"summary": summary, "content": content})

    return noise_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Recall evaluation for memory-mcp-server")
    parser.add_argument("--url", default="http://127.0.0.1:9180", help="MCP server URL")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout in seconds")
    parser.add_argument("--topk", type=int, default=5, help="Top-K retrieved results per query")
    parser.add_argument("--noise-size", type=int, default=20, help="Number of synthetic noise memories to insert")
    parser.add_argument("--noise-seed", type=int, default=42, help="Random seed for deterministic noise generation")
    parser.add_argument("--keep-data", action="store_true", help="Keep temporary eval document")
    parser.add_argument("--verbose", action="store_true", help="Print RPC response snippets")
    args = parser.parse_args()

    topk = max(1, args.topk)
    noise_size = max(0, args.noise_size)
    client = McpClient(server_url=args.url, timeout_sec=args.timeout, verbose=args.verbose)

    dataset = [
        {
            "summary": "Tokio task spawning basics",
            "content": "Use tokio::spawn for concurrent async tasks and await JoinHandle.",
            "query": "tokio concurrent task spawn",
        },
        {
            "summary": "PostgreSQL btree index usage",
            "content": "B-tree index speeds up equality and range filters for scalar columns.",
            "query": "postgres equality range btree index",
        },
        {
            "summary": "Nginx websocket reverse proxy",
            "content": "Configure Upgrade and Connection headers with HTTP/1.1 for websocket proxy.",
            "query": "nginx websocket upgrade connection header",
        },
        {
            "summary": "pgvector HNSW cosine ops",
            "content": "For cosine operator <=> use vector_cosine_ops to accelerate ANN retrieval.",
            "query": "vector_cosine_ops cosine operator",
        },
        {
            "summary": "Hybrid BM25 and vector RRF",
            "content": "Hybrid retrieval combines keyword BM25 and vector ranking via reciprocal rank fusion.",
            "query": "bm25 vector reciprocal rank fusion",
        },
        {
            "summary": "Diesel migration rollback",
            "content": "Diesel migration redo and rollback can revert PostgreSQL schema changes.",
            "query": "diesel migration rollback postgres",
        },
    ]
    noise_dataset = build_noise_dataset(noise_size, seed=args.noise_seed)

    doc_id: Optional[int] = None

    try:
        print(f"Initialize MCP session: {args.url}")
        client.initialize()

        eval_doc_name = f"mcp_recall_eval_{uuid.uuid4().hex[:8]}"
        print(f"Create eval document: {eval_doc_name}")
        create_res = client.call_tool(
            "create_document",
            {"name": eval_doc_name, "description": "Temporary document for recall evaluation"},
        )
        create_data = extract_tool_result_payload(create_res)
        if not isinstance(create_data, dict) or "id" not in create_data:
            raise RuntimeError(f"Unexpected create_document result: {create_data}")
        doc_id = int(create_data["id"])

        print(f"Insert dataset: base={len(dataset)} noise={len(noise_dataset)} total={len(dataset) + len(noise_dataset)}")
        for row in dataset:
            client.call_tool(
                "insert_memory",
                {
                    "document_id": doc_id,
                    "summary": row["summary"],
                    "content": row["content"],
                },
            )
        for row in noise_dataset:
            client.call_tool(
                "insert_memory",
                {
                    "document_id": doc_id,
                    "summary": row["summary"],
                    "content": row["content"],
                },
            )

        hits_at_1 = 0
        hits_at_3 = 0
        mrr_sum = 0.0

        print(f"Run queries with topk={topk}")
        for i, row in enumerate(dataset, start=1):
            res = client.call_tool(
                "search_memory",
                {
                    "document_id": doc_id,
                    "query_summary": row["query"],
                    "query_content": row["query"],
                    "limit": topk,
                },
            )
            payload = extract_tool_result_payload(res)
            data = payload.get("results") if isinstance(payload, dict) else payload
            if not isinstance(data, list):
                raise RuntimeError(f"Unexpected search result for query {i}: {data}")

            rank: Optional[int] = None
            summaries = []
            for idx, item in enumerate(data, start=1):
                summary = str(item.get("summary", ""))
                summaries.append(summary)
                if summary == row["summary"] and rank is None:
                    rank = idx

            if rank == 1:
                hits_at_1 += 1
            if rank is not None and rank <= 3:
                hits_at_3 += 1
            mrr_sum += reciprocal_rank(rank)

            rank_show = rank if rank is not None else "MISS"
            top_show = " | ".join(summaries[:3])
            print(f"  Q{i}: rank={rank_show}; expected='{row['summary']}'; top3={top_show}")

        n = len(dataset)
        hit1 = hits_at_1 / n
        hit3 = hits_at_3 / n
        mrr = mrr_sum / n

        print("\nRecall Metrics")
        print(f"  Hit@1: {hits_at_1}/{n} = {hit1:.3f}")
        print(f"  Hit@3: {hits_at_3}/{n} = {hit3:.3f}")
        print(f"  MRR:   {mrr:.3f}")
        print("RECALL EVALUATION DONE")

        if args.keep_data:
            print(f"Keep data enabled, retained document id={doc_id}")
        else:
            client.call_tool("delete_document", {"document_id": doc_id})
            print(f"Cleanup complete, deleted document id={doc_id}")

        return 0

    except Exception as exc:
        print(f"RECALL EVALUATION FAILED: {exc}", file=sys.stderr)
        if doc_id is not None and not args.keep_data:
            try:
                client.call_tool("delete_document", {"document_id": doc_id})
                print(f"cleanup: deleted document id={doc_id}", file=sys.stderr)
            except Exception as cleanup_exc:
                print(f"cleanup warning: {cleanup_exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

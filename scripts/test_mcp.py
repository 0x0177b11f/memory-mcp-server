#!/usr/bin/env python3
"""Basic MCP smoke test for memory-mcp-server.

This script validates a minimal end-to-end flow:
- initialize MCP session
- list tools
- create document
- insert memory chunks
- search memory
- delete one memory chunk
- delete document

The script uses only Python standard library.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib import error, request


REQUIRED_TOOLS = {
    "create_document",
    "list_documents",
    "delete_document",
    "update_document",
    "insert_memory",
    "delete_memory",
    "search_memory_summary",
    "search_memory_content",
    "search_memory",
}


def parse_mcp_response(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    if not text:
        raise RuntimeError("Empty MCP response body")

    # First try plain JSON response.
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Fallback for Server-Sent Events payloads.
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

    preview = text[:200].replace("\n", "\\n")
    raise RuntimeError(f"Unable to parse MCP response as JSON or SSE JSON payload: {preview}")


@dataclass
class McpClient:
    server_url: str
    timeout_sec: float = 20.0
    snippet_len: int = 220
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
        # Streamable HTTP MCP servers may require both JSON and SSE in Accept.
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
                "clientInfo": {"name": "mcp-smoke-test", "version": "1.0.0"},
            },
        )
        self.rpc("notifications/initialized", notify=True)

    def tools_list(self) -> Dict[str, Any]:
        return self.rpc("tools/list")

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test memory-mcp-server over MCP streamable HTTP")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:9180",
        help="MCP server URL (example: http://127.0.0.1:9180 or http://127.0.0.1:9180/mcp)",
    )
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP request timeout in seconds")
    parser.add_argument(
        "--snippet-len",
        type=int,
        default=220,
        help="Max length of printed response snippet for each RPC",
    )
    parser.add_argument("--keep-data", action="store_true", help="Do not delete test document at the end")
    args = parser.parse_args()

    client = McpClient(server_url=args.url, timeout_sec=args.timeout, snippet_len=max(40, args.snippet_len))
    doc_id: Optional[int] = None

    try:
        print(f"[1/7] Initialize MCP session: {args.url}")
        client.initialize()

        print("[2/7] List tools")
        tools_res = client.tools_list()
        tools = tools_res.get("tools", [])
        tool_names = {t.get("name") for t in tools}
        missing = sorted(REQUIRED_TOOLS - tool_names)
        if missing:
            raise RuntimeError(f"Missing required tools: {missing}")
        print(f"      tools ok: {len(tool_names)} available")

        print("[3/7] Create document")
        name = f"mcp_smoke_{uuid.uuid4().hex[:8]}"
        create_res = client.call_tool("create_document", {"name": name, "description": "MCP smoke test"})
        create_data = extract_tool_result_payload(create_res)
        if not isinstance(create_data, dict) or "id" not in create_data:
            raise RuntimeError(f"Unexpected create_document result: {create_data}")
        doc_id = int(create_data["id"])
        print(f"      created document id={doc_id}")

        print("[4/7] Insert memory chunks")
        client.call_tool(
            "insert_memory",
            {
                "document_id": doc_id,
                "summary": "Tokio spawn basics",
                "content": "Use tokio::spawn to run async tasks concurrently and await JoinHandle.",
            },
        )
        client.call_tool(
            "insert_memory",
            {
                "document_id": doc_id,
                "summary": "pgvector cosine index",
                "content": "For cosine distance operator <=>, use vector_cosine_ops.",
            },
        )

        print("[5/7] Search memory content")
        search_res = client.call_tool(
            "search_memory_content",
            {"document_id": doc_id, "query_text": "tokio concurrent task spawn", "limit": 5},
        )
        search_payload = extract_tool_result_payload(search_res)
        search_data = search_payload.get("results") if isinstance(search_payload, dict) else search_payload
        if not isinstance(search_data, list) or not search_data:
            raise RuntimeError(f"Unexpected search result: {search_data}")
        top = search_data[0]
        print(f"      top hit id={top.get('id')} summary={top.get('summary')}")

        print("[6/7] Delete one memory chunk")
        memory_id = top.get("id")
        if memory_id is None:
            raise RuntimeError(f"Search result does not contain memory id: {top}")
        client.call_tool("delete_memory", {"memory_id": int(memory_id)})
        print(f"      deleted memory id={memory_id}")

        print("[7/7] Cleanup document")
        if args.keep_data:
            print(f"      keep-data enabled, retained document id={doc_id}")
        else:
            client.call_tool("delete_document", {"document_id": doc_id})
            print(f"      deleted document id={doc_id}")

        print("SMOKE TEST PASSED")
        return 0

    except Exception as exc:
        print(f"SMOKE TEST FAILED: {exc}", file=sys.stderr)
        if doc_id is not None and not args.keep_data:
            try:
                client.call_tool("delete_document", {"document_id": doc_id})
                print(f"cleanup: deleted document id={doc_id}", file=sys.stderr)
            except Exception as cleanup_exc:
                print(f"cleanup warning: {cleanup_exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

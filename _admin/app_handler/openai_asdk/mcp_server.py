"""
MCP server for BestBrain as OpenAI Apps SDK app.

Exposes /mcp endpoint per https://developers.openai.com/apps-sdk/quickstart.
Implements JSON-RPC 2.0: initialize, tools/list, tools/call so ChatGPT can
discover and invoke tools. Stateless mode (no session persistence).

Run: python -m _admin.app_handler.openai_asdk.mcp_server --port 8000
Test: npx @modelcontextprotocol/inspector@latest --server-url http://localhost:8000/mcp --transport http
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict, List

from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse, JSONResponse
from uvicorn import run as uvicorn_run

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MCP path required by ChatGPT connector (quickstart)
MCP_PATH = "/mcp"

# CSP for App Store submission
CSP_CONNECT_DOMAINS = "https://api.openai.com https://bestbrain.tech"

app = FastAPI(title="BestBrain MCP App", version="0.1.0")

# Load tool definitions from workflow (single source of truth)
def _get_tools() -> List[Dict[str, Any]]:
    try:
        from _admin.app_handler.openai_asdk.workflow import TOOLS
        return list(TOOLS)
    except Exception:
        return []


def _cors_headers() -> dict:
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS, DELETE",
        "Access-Control-Allow-Headers": "content-type, mcp-session-id",
        "Access-Control-Expose-Headers": "Mcp-Session-Id",
    }


# --- Health: GET / (quickstart) ---
@app.get("/")
async def root():
    return PlainTextResponse("BestBrain MCP server", headers=_cors_headers())


# --- CORS preflight: OPTIONS /mcp (quickstart) ---
@app.options(MCP_PATH)
async def mcp_options():
    return Response(status_code=204, headers=_cors_headers())


# --- Tool handlers (return content + structuredContent per Apps SDK) ---
def _handle_list_simulations(args: Dict[str, Any]) -> Dict[str, Any]:
    user_id = (args or {}).get("user_id") or "default"
    try:
        from qbrain.core.qbrain_manager import get_qbrain_table_manager
        qb = get_qbrain_table_manager()
        rows = qb.get_users_entries(user_id, "envs") or []
        sims = [{"env_id": r.get("id"), "status": r.get("status"), "user_id": r.get("user_id")} for r in rows[:50]]
    except Exception as e:
        sims = []
        logger.warning("list_simulations: %s", e)
    return {
        "content": [{"type": "text", "text": f"Found {len(sims)} simulation(s)."}],
        "structuredContent": {"simulations": sims, "count": len(sims)},
    }


def _handle_create_simulation(args: Dict[str, Any]) -> Dict[str, Any]:
    env_id = (args or {}).get("env_id") or ""
    modules = (args or {}).get("modules") or []
    if not env_id:
        return {
            "content": [{"type": "text", "text": "Missing env_id."}],
            "structuredContent": {"ok": False, "error": "env_id required"},
        }
    try:
        from qbrain.core.qbrain_manager import get_qbrain_table_manager
        qb = get_qbrain_table_manager()
        qb.set_item("envs", [{"id": env_id, "user_id": "default", "status": "active", "data": json.dumps({"modules": modules})}])
    except Exception as e:
        logger.warning("create_simulation: %s", e)
        return {
            "content": [{"type": "text", "text": str(e)}],
            "structuredContent": {"ok": False, "error": str(e)},
        }
    return {
        "content": [{"type": "text", "text": f"Created simulation env_id={env_id} with modules={modules}."}],
        "structuredContent": {"ok": True, "env_id": env_id, "modules": modules},
    }


TOOL_HANDLERS = {
    "list_simulations": _handle_list_simulations,
    "create_simulation": _handle_create_simulation,
}


def _mcp_initialize(params: Dict[str, Any]) -> Dict[str, Any]:
    """MCP initialize: return serverInfo and capabilities (tools)."""
    return {
        "protocolVersion": "2024-11-05",
        "serverInfo": {"name": "BestBrain", "version": "0.1.0"},
        "capabilities": {"tools": {}},
    }


def _mcp_tools_list() -> Dict[str, Any]:
    """MCP tools/list: return tools for ChatGPT."""
    tools = _get_tools()
    mcp_tools = []
    for t in tools:
        mcp_tools.append({
            "name": t.get("name"),
            "description": t.get("description") or t.get("title", ""),
            "inputSchema": t.get("inputSchema", {"type": "object", "properties": {}}),
        })
    return {"tools": mcp_tools}


def _mcp_tools_call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """MCP tools/call: run tool and return content + structuredContent."""
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return {
            "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
            "structuredContent": {"error": f"Unknown tool: {name}"},
            "isError": True,
        }
    try:
        return handler(arguments or {})
    except Exception as e:
        logger.exception("tools/call %s", name)
        return {
            "content": [{"type": "text", "text": str(e)}],
            "structuredContent": {"error": str(e)},
            "isError": True,
        }


# --- MCP endpoint: POST/GET/DELETE /mcp (quickstart) ---
@app.api_route(MCP_PATH, methods=["GET", "POST", "DELETE"])
async def mcp_handler(request: Request):
    """
    MCP endpoint. ChatGPT connects to https://<your-domain>/mcp.
    Implements JSON-RPC 2.0: initialize, tools/list, tools/call.
    """
    headers = _cors_headers()

    if request.method == "GET":
        return Response(content="OK", status_code=200, headers=headers)

    body = await request.body()
    try:
        data = json.loads(body) if body else {}
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content={"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}},
            headers=headers,
        )

    req_id = data.get("id")
    method = data.get("method", "")
    params = data.get("params") or {}

    try:
        if method == "initialize":
            result = _mcp_initialize(params)
        elif method == "tools/list":
            result = _mcp_tools_list()
        elif method == "tools/call":
            name = (params.get("name") or "").strip()
            arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
            result = _mcp_tools_call(name, arguments)
        else:
            result = {"message": f"Unsupported method: {method}"}
    except Exception as e:
        logger.exception("MCP %s", method)
        return JSONResponse(
            status_code=200,
            content={
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": str(e)},
            },
            headers=headers,
        )

    return JSONResponse(
        status_code=200,
        content={"jsonrpc": "2.0", "id": req_id, "result": result},
        headers=headers,
    )


# --- 404 for nested paths (quickstart: OAuth discovery etc. not used; avoid 502) ---
@app.api_route("/mcp/{rest:path}", methods=["GET", "POST", "OPTIONS", "DELETE"])
async def mcp_nested_404(rest: str):
    return Response(status_code=404, content="Not Found", headers=_cors_headers())


def main():
    parser = argparse.ArgumentParser(description="BestBrain MCP server for OpenAI Apps SDK")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    args = parser.parse_args()

    logger.info("BestBrain MCP server listening on http://%s:%s%s", args.host, args.port, MCP_PATH)
    uvicorn_run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

from __future__ import annotations

import asyncio
import argparse
import inspect
import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, Optional, get_type_hints

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

from mcp_server.service import MCPServerService


APP_NAME = "eq-storage"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_PATH = "/mcp"
_WINDOWS_BOOTSTRAP_DONE = False


def _patch_streamable_http_accept_wildcard() -> None:
    """
    Make StreamableHTTP Accept parsing treat '*/*' as accepting JSON and SSE.

    Some MCP clients send `Accept: */*`, which is valid HTTP semantics, but
    current MCP server parsing checks only explicit media types.
    """
    try:
        from mcp.server.streamable_http import StreamableHTTPServerTransport
    except Exception as exc:
        print(f"[DEBUG] Could not patch StreamableHTTP accept handling: {exc!r}")
        return

    if getattr(StreamableHTTPServerTransport, "_eq_accept_wildcard_patch_applied", False):
        return

    def _patched_check_accept_headers(self, request):  # type: ignore[no-untyped-def]
        accept_header = request.headers.get("accept", "")
        accept_types = [media_type.strip() for media_type in accept_header.split(",")]
        has_wildcard = any(media_type.startswith("*/*") for media_type in accept_types)
        has_json = any(media_type.startswith("application/json") for media_type in accept_types) or has_wildcard
        has_sse = any(media_type.startswith("text/event-stream") for media_type in accept_types) or has_wildcard
        return has_json, has_sse

    StreamableHTTPServerTransport._check_accept_headers = _patched_check_accept_headers  # type: ignore[method-assign]
    StreamableHTTPServerTransport._eq_accept_wildcard_patch_applied = True
    print("[DEBUG] Applied StreamableHTTP Accept wildcard compatibility patch")


def _to_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def create_mcp_server(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    path: str = DEFAULT_PATH,
    json_response: bool = True,
    stateless_http: bool = True,
) -> FastMCP:
    _patch_streamable_http_accept_wildcard()
    print(
        f"[DEBUG] create_mcp_server called host={host} port={port} "
        f"path={path} json_response={json_response} stateless_http={stateless_http}"
    )
    service = MCPServerService()
    server = FastMCP(
        name=APP_NAME,
        instructions="MCP tools for eq_storage graph, entry retrieval, upsert, and deletion.",
        host=host,
        port=port,
        streamable_http_path=path,
        json_response=json_response,
        stateless_http=stateless_http,
    )

    _register_public_service_methods(server=server, service=service)
    _register_debug_routes(server=server)
    _maybe_run_windows_bootstrap_checks(server=server, host=host, port=port, path=path)
    print("server creation... done")
    return server


def _register_public_service_methods(server: FastMCP, service: MCPServerService) -> None:
    for method_name in sorted(name for name in dir(service) if not name.startswith("_")):
        method = getattr(service, method_name, None)
        if not callable(method):
            continue

        signature = inspect.signature(method)
        annotations = _tool_annotations_for_method(method_name)
        description = (inspect.getdoc(method) or f"Auto-exposed method: MCPServerService.{method_name}").strip()
        tool_func = _build_tool_callable(service_method=method, signature=signature, method_name=method_name)

        server.tool(
            name=method_name,
            description=description,
            annotations=annotations,
        )(tool_func)
        print(f"[DEBUG] Registered MCP tool: {method_name}")


def _register_debug_routes(server: FastMCP) -> None:
    @server.custom_route("/health", methods=["GET"], include_in_schema=False)
    async def _health(_: Request) -> Response:
        return JSONResponse({"status": "ok", "app": APP_NAME, "mcp_path": DEFAULT_PATH})

    @server.custom_route("/dashboard", methods=["GET"], include_in_schema=False)
    async def _dashboard(request: Request) -> Response:
        base_url = f"{request.url.scheme}://{request.url.netloc}"
        mcp_url = f"{base_url}{DEFAULT_PATH}"
        health_url = f"{base_url}/health"
        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>{APP_NAME} dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #111; }}
    .ok {{ color: #0a7a27; font-weight: 700; }}
    code {{ background: #f2f2f2; padding: 0.1rem 0.3rem; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>EQ Storage MCP Dashboard</h1>
  <p class="ok">Server is running.</p>
  <ul>
    <li>Health endpoint: <a href="{health_url}">{health_url}</a></li>
    <li>MCP endpoint: <code>{mcp_url}</code></li>
  </ul>
  <p>Use an MCP client for <code>{DEFAULT_PATH}</code> (streamable HTTP / SSE semantics).</p>
</body>
</html>"""
        return HTMLResponse(content=html, status_code=200)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(by_alias=True)  # pydantic v2 models
        except Exception:
            return value.model_dump()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _maybe_run_windows_bootstrap_checks(*, server: FastMCP, host: str, port: int, path: str) -> None:
    global _WINDOWS_BOOTSTRAP_DONE
    if os.name != "nt" or _WINDOWS_BOOTSTRAP_DONE:
        return
    _WINDOWS_BOOTSTRAP_DONE = True

    print("[DEBUG][WIN] Running Windows MCP bootstrap diagnostics")
    try:
        tools = asyncio.run(server.list_tools())
        tool_payload = [_jsonable(tool) for tool in tools]
        hierarchy = {
            "server": {
                "name": APP_NAME,
                "host": host,
                "port": port,
                "path": path,
                "json_response": getattr(server.settings, "json_response", None),
                "stateless_http": getattr(server.settings, "stateless_http", None),
            },
            "tools": tool_payload,
        }
        print("[DEBUG][WIN] MCP structure/hierarchy payload:")
        print(json.dumps(hierarchy, indent=2, default=str))
    except Exception as exc:
        print(f"[DEBUG][WIN] Failed to print MCP hierarchy payload: {exc!r}")

    if _to_bool(os.getenv("MCP_RUN_BOOTSTRAP_TESTER"), False):
        try:
            from test import MCPWalkthroughTester

            print("[DEBUG][WIN] Running MCPWalkthroughTester from root test.py")
            asyncio.run(MCPWalkthroughTester(server=server).run())
            print("[DEBUG][WIN] MCPWalkthroughTester completed")
        except Exception as exc:
            print(f"[DEBUG][WIN] MCPWalkthroughTester failed: {exc!r}")
    else:
        print("[DEBUG][WIN] Skipping MCPWalkthroughTester (set MCP_RUN_BOOTSTRAP_TESTER=1 to enable)")


def _tool_annotations_for_method(method_name: str) -> ToolAnnotations:
    lowered = method_name.lower()
    return ToolAnnotations(
        readOnlyHint=lowered.startswith(("get", "list", "fetch", "read")),
        destructiveHint=("delete" in lowered) or ("remove" in lowered),
        idempotentHint=lowered.startswith(("get", "list", "fetch", "read")) or ("delete" in lowered),
        openWorldHint=False,
    )


def _build_tool_callable(
    *,
    service_method: Callable[..., Any],
    signature: inspect.Signature,
    method_name: str,
) -> Callable[..., Any]:
    resolved_hints = _resolve_type_hints(service_method=service_method, method_name=method_name)

    # Replace deferred/string annotations with concrete types so FastMCP/Pydantic
    # can build schemas without unresolved forward refs.
    resolved_params = [
        param.replace(annotation=resolved_hints.get(param_name, param.annotation))
        for param_name, param in signature.parameters.items()
    ]
    resolved_return = resolved_hints.get("return", signature.return_annotation)

    resolved_signature = signature.replace(
        parameters=resolved_params,
        return_annotation=resolved_return,
    )

    def tool_callable(**kwargs: Any) -> Any:
        print(f"[DEBUG] Tool '{method_name}' called")
        converted = {}
        for param_name, param in signature.parameters.items():
            if param_name not in kwargs:
                continue
            annotation = resolved_hints.get(param_name, param.annotation)
            converted[param_name] = _coerce_argument(kwargs[param_name], annotation)

        bound = signature.bind(**converted)
        bound.apply_defaults()
        result = service_method(*bound.args, **bound.kwargs)
        print(f"[DEBUG] Tool '{method_name}' raw result type: {type(result).__name__}")
        if is_dataclass(result):
            print(f"[DEBUG] Tool '{method_name}' returning dataclass as dict")
            return asdict(result)
        print(f"[DEBUG] Tool '{method_name}' returning result")
        return result

    tool_callable.__name__ = method_name
    tool_callable.__doc__ = inspect.getdoc(service_method)
    tool_callable.__signature__ = resolved_signature
    return tool_callable


def _resolve_type_hints(service_method: Callable[..., Any], method_name: str) -> Dict[str, Any]:
    # Bound methods can hide globals on __func__; resolve using the underlying function when present.
    target = getattr(service_method, "__func__", service_method)
    target_globals = getattr(target, "__globals__", {})
    localns = dict(target_globals)
    localns.update(globals())
    try:
        return get_type_hints(target, globalns=target_globals, localns=localns)
    except Exception as exc:
        print(f"[DEBUG] Failed to resolve type hints for tool '{method_name}': {exc!r}")
        return {}

def _coerce_argument(value: Any, annotation: Any) -> Any:
    if value is None or annotation is inspect.Signature.empty:
        return value
    if isinstance(annotation, type) and is_dataclass(annotation):
        if isinstance(value, annotation):
            return value
        if isinstance(value, dict):
            from_dict = getattr(annotation, "from_dict", None)
            if callable(from_dict):
                return from_dict(value)
            return annotation(**value)
    return value


def _read_runtime_config() -> Dict[str, Any]:
    host = os.getenv("MCP_HOST", DEFAULT_HOST)
    port = int(os.getenv("MCP_PORT", str(DEFAULT_PORT)))
    path = os.getenv("MCP_PATH", DEFAULT_PATH)
    json_response = _to_bool(os.getenv("MCP_JSON_RESPONSE"), True)
    stateless_http = _to_bool(os.getenv("MCP_STATELESS_HTTP"), True)
    return {
        "host": host,
        "port": port,
        "path": path,
        "json_response": json_response,
        "stateless_http": stateless_http,
    }


def main() -> None:
    env_cfg = _read_runtime_config()
    print(f"[DEBUG] Runtime config from env: {env_cfg}")
    parser = argparse.ArgumentParser(description="Hosted FastMCP server for eq_storage")
    parser.add_argument("--host", default=env_cfg["host"], help="Server host (env: MCP_HOST)")
    parser.add_argument("--port", type=int, default=env_cfg["port"], help="Server port (env: MCP_PORT)")
    parser.add_argument("--path", default=env_cfg["path"], help="MCP path (env: MCP_PATH)")
    parser.add_argument(
        "--json-response",
        action=argparse.BooleanOptionalAction,
        default=env_cfg["json_response"],
        help="Return JSON responses for streamable HTTP requests (env: MCP_JSON_RESPONSE)",
    )
    parser.add_argument(
        "--stateless-http",
        action=argparse.BooleanOptionalAction,
        default=env_cfg["stateless_http"],
        help="Enable stateless Streamable HTTP mode (env: MCP_STATELESS_HTTP)",
    )
    args = parser.parse_args()
    print(f"[DEBUG] Parsed CLI args: {args}")

    server = create_mcp_server(
        host=args.host,
        port=args.port,
        path=args.path,
        json_response=args.json_response,
        stateless_http=args.stateless_http,
    )
    print("[DEBUG] Starting MCP server with streamable-http transport")
    server.run(transport="streamable-http")


if __name__ == "__main__":
    main()


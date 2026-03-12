from __future__ import annotations

import os

from mcp_server.mcp_routes import create_mcp_server

def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

def main() -> None:
    host = os.getenv("MCP_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_PORT", "8000"))
    path = os.getenv("MCP_PATH", "/mcp")
    json_response = _to_bool(os.getenv("MCP_JSON_RESPONSE"), True)
    stateless_http = _to_bool(os.getenv("MCP_STATELESS_HTTP"), True)
    transport = (os.getenv("MCP_TRANSPORT", "stdio") or "stdio").strip().lower()
    if transport not in {"stdio", "streamable-http"}:
        print(f"[DEBUG] Invalid MCP_TRANSPORT='{transport}', falling back to 'stdio'")
        transport = "stdio"

    app = create_mcp_server(
        host=host,
        port=port,
        path=path,
        json_response=json_response,
        stateless_http=stateless_http,
    )

    print(f"[DEBUG] Starting MCP server with transport='{transport}'")
    if transport == "stdio":
        print("[DEBUG] stdio mode waits for an MCP client to connect via stdin/stdout.")
    else:
        print(f"[DEBUG] HTTP endpoint available at http://{host}:{port}{path}")
    print("finished init...")
    app.run(transport=transport)


if __name__ == "__main__":
    main()

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from mcp_server.mcp_routes import create_mcp_server


def create_app() -> FastMCP:
    return create_mcp_server()

app = create_app()


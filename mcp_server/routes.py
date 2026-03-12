from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp import ServerSession
from mcp.server import FastMCP

from _db.manager import DBManager
from mcp_server.service import MCPServerService
from mcp_server.types import DeleteRequest, UpsertRequest, GetGraphRequest, GetEntryRequest
from mcp.server.fastmcp import Context


@dataclass
class AppContext:
    """Application context with typed dependencies."""
    db: DBManager


@asynccontextmanager
async def lifespan(server: FastMCP):
    db = DBManager()
    yield AppContext(db=db)


app = FastMCP(
    "rewiredb",
    lifespan=lifespan
)

@app.tool()
async def upsert(
    request: UpsertRequest,
    ctx: Context[ServerSession, AppContext],

) -> dict:
    """
    "Execute database operations on the rewiredb equation storage system."
    """
    try:
        db = ctx.request_context.lifespan_context.db
        service = MCPServerService(db)
        result = service.upsert(request)
        print("upsert result:", result)
        result["session_id"] = ctx.session.session_id
        return result
    except Exception as exc:
        return {"status": "error"}


@app.tool()
async def get_entry(
        request: GetEntryRequest,
        ctx: Context[ServerSession, AppContext],
) -> dict:
    """
    "Execute database operations on the rewiredb equation storage system."
    """
    try:
        db = ctx.request_context.lifespan_context.db
        service = MCPServerService(db)
        result = service.get_entry(request)
        result["session_id"] = ctx.session.session_id
        return result
    except Exception as exc:
        print("ERR:", exc)
        return {"status": "error"}


@app.tool()
async def get_graph(
    request:GetGraphRequest,
    ctx: Context[ServerSession, AppContext],
) -> dict:
    """
    "Execute database operations on the rewiredb equation storage system."
    """
    try:
        db = ctx.request_context.lifespan_context.db
        service = MCPServerService(db)
        result = service.get_graph(request)
        result["session_id"] = ctx.session.session_id
        return result
    except Exception as exc:
        return {"status": "error"}

@app.tool()
async def delete_entries(
        request: DeleteRequest,
        ctx: Context[ServerSession, AppContext],
) -> dict:
    """
    "Execute database operations on the rewiredb equation storage system."
    """
    try:
        db = ctx.request_context.lifespan_context.db
        service = MCPServerService(db)
        result = service.delete_entries(request)
        result["session_id"] = ctx.session.session_id
        return result
    except Exception as exc:
        return {"status": str(exc)}


if __name__ == "__main__":
    app.run(transport="streamable-http")
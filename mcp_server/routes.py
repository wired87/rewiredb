from mcp.server import FastMCP

from _db.manager import DBManager
from mcp_server.service import MCPServerService
from mcp_server.types import DeleteRequest, UpsertRequest, GetGraphRequest, GetEntryRequest
from mcp.server.fastmcp import Context

app = FastMCP("rewiredb")
db = DBManager()

@app.tool()
async def upsert(request: UpsertRequest) -> dict:
    """
    "Execute database operations on the rewiredb equation storage system."
    """
    try:
        service = MCPServerService(db)
        result = service.upsert(request)
        print("upsert result:", result)
        return result
    except Exception as exc:
        return {"status": "error"}


@app.tool()
async def get_entry(request: GetEntryRequest) -> dict:
    """
    "Execute database operations on the rewiredb equation storage system."
    """
    try:
        service = MCPServerService(db)
        return service.get_entry(request)
    except Exception as exc:
        print("ERR:", exc)
        return {"status": "error"}


@app.tool()
async def get_graph(request:GetGraphRequest) -> dict:
    """
    "Execute database operations on the rewiredb equation storage system."
    """
    try:
        service = MCPServerService(db)
        return service.get_graph(request)

    except Exception as exc:
        return {"status": "error"}

@app.tool()
async def delete_entries(request: DeleteRequest) -> dict:
    """
    "Execute database operations on the rewiredb equation storage system."
    """
    try:
        service = MCPServerService(db)
        return service.delete_entries(request)
    except Exception as exc:
        return {"status": "error"}


if __name__ == "__main__":
    app.run(transport="streamable-http")









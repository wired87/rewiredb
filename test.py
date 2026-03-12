import base64

if __name__ == '__main__':
    import requests

    URL = "http://127.0.0.1:8000/mcp"

    HEADERS = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    def initialize_session():
        """
        Open a valid MCP session and return the server-issued session id.

        Why this exists:
        - `Mcp-Session-Id` must come from the server.
        - Sending a random UUID causes "Session not found".
        """
        body = {
            "jsonrpc": "2.0",
            "id": "init-1",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "eq-storage-test", "version": "0.1.0"},
            },
        }
        response = requests.post(URL, headers=HEADERS, json=body)
        response.raise_for_status()

        session_id = response.headers.get("Mcp-Session-Id")
        if not session_id:
            raise RuntimeError("Server did not return Mcp-Session-Id during initialize.")
        return session_id

    def call_tool(name, arguments, session_id):

        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        }

        headers = {**HEADERS, "Mcp-Session-Id": session_id}
        r = requests.post(URL, headers=headers, json=body, stream=True)

        for line in r.iter_lines():
            if line:
                print(line.decode())


    # -------------------------
    # UPSERT (UpsertRequest)
    # -------------------------

    session_id = initialize_session()

    call_tool("upsert", {
        "request": {
            "user_id": "user1",
            "module_id": "module1",
            "data": {
                "equation": "E = mc^2",
                "files": [
                    base64.b64encode(
                        open(
                            "data/ext.pdf",
                            "rb"
                        ).read()
                    ).decode()
                ]
            }
        }
    }, session_id=session_id)

    # -------------------------
    # GET ENTRY (GetEntryRequest)
    # -------------------------

    call_tool("get_entry", {
        "request": {
            "entry_id": "123",
            "table": "methods"
        }
    }, session_id=session_id)

    # -------------------------
    # GET GRAPH (GetGraphRequest)
    # -------------------------

    call_tool("get_graph", {
        "request": {
            "user_id": "user1"
        }
    }, session_id=session_id)

    # -------------------------
    # DELETE (DeleteRequest)
    # -------------------------

    call_tool("delete_entries", {
        "request": {
            "user_id": "user1",
            "table": "methods",
            "entry_id": "123"
        }
    }, session_id=session_id)
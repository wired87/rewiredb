import base64

if __name__ == '__main__':
    import requests
    import uuid



    URL = "http://127.0.0.1:8787/mcp"

    HEADERS = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "Mcp-Session-Id": str(uuid.uuid4())
    }


    def call_tool(name, arguments):

        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        }

        r = requests.post(URL, headers=HEADERS, json=body, stream=True)

        for line in r.iter_lines():
            if line:
                print(line.decode())


    # -------------------------
    # UPSERT (UpsertRequest)
    # -------------------------

    call_tool("upsert", {
        "user_id": "user1",
        "module_id": "module1",
        "data": {
            "equation": "E = mc^2",
            "files": [base64.b64encode(open("tmp_test_artifacts/exp.pdf", "rb").read()).decode()]
        }
    })

    # -------------------------
    # GET ENTRY (GetEntryRequest)
    # -------------------------

    call_tool("entry_get", {
        "entry_id": "123",
        "table": "methods"
    })

    # -------------------------
    # GET GRAPH (GetGraphRequest)
    # -------------------------

    call_tool("graph_get", {
        "user_id": "user1"
    })

    # -------------------------
    # DELETE (DeleteRequest)
    # -------------------------

    call_tool("entries_delete", {
        "user_id": "user1",
        "table": "methods",
        "entry_id": "123"
    })
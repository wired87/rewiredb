# eq_storage

Turn equations and text into a living knowledge graph you can query through MCP.

## What this project does

- Ingests equations and file content through MCP.
- Extracts methods, parameters, operators, and links between them.
- Stores results in local DuckDB for quick retrieval.
- Returns graph views to help you explore relationships, not just raw text.

## Why teams use it

- Faster technical onboarding: new people can "see the model" in graph form.
- Better reuse: avoid rewriting equations that already exist in prior work.
- Stronger consistency: compare formulas across files, modules, and owners.
- Traceable decisions: connect stored artifacts to extracted graph entities.

## Pathfinder possibilities (experimental)

The `pathfinder_manager` module extends the project from storage to discovery:

- Detect recurring patterns in time-series controller outputs.
- Persist reusable controller signatures for later comparison.
- Support equation-inference workflows from recurring event segments.

Use it as a "next layer" for insight generation after graph storage is in place.

## Product view at a glance

- **Capture**: your equations and files enter through MCP tools.
- **Structure**: entities and connections are normalized in the graph model.
- **Explore**: graph responses support analysis, QA, and downstream automation.
- **Evolve**: Pathfinder can add pattern intelligence on top of stored history.

## 2-minute run tutorial

### 1) Install dependencies

```bash
pip install -r r.txt
```

### 2) Start the MCP server

```bash
python -m mcp_server.mcp_routes --host 0.0.0.0 --port 8787 --path /mcp
```

### 3) Open an MCP client (optional check)

```bash
npx @modelcontextprotocol/inspector@latest --server-url http://localhost:8787/mcp --transport http
```

### 4) Use these core tools in order

1. `upsert` (store your equation/file content)
2. `get_graph` (view relationships)
3. `get_entry` (inspect one item)
4. `delete_entries` (cleanup when needed)

## Notes

- Built for your own data flow (no seeded data required).
- Default MCP endpoint is `/mcp` on port `8787`.
- Health page is available at `/health`.

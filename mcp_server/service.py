from __future__ import annotations

import importlib.util
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import networkx as nx

from graph.local_graph_utils import GUtils
from mcp_server.types import (
    DeleteRequest,
    DeleteResponse,
    EntryResponse,
    GraphEdgeOut,
    GraphNodeOut,
    GraphResponse,
    UpsertRequest,
    UpsertResponse,
)

from utils.id_gen import generate_id


class MCPServerService:
    def __init__(self, db) -> None:
        """Initialize DB access, extractor cache, and in-memory working graph."""
        self.db = db
        self._eq_extractor_cls = None
        self.g=GUtils(G=nx.MultiGraph())

    @staticmethod
    def _now() -> datetime:
        """Return UTC timestamp used for persisted records."""
        return datetime.utcnow()

    @staticmethod
    def _as_text(payload_bytes: bytes) -> str:
        """Decode bytes to text with UTF-8 fallback to latin-1."""
        try:
            return payload_bytes.decode("utf-8")
        except Exception:
            return payload_bytes.decode("latin-1", errors="ignore")

    @staticmethod
    def _extract_text_from_pdf_bytes(payload_bytes: bytes or str) -> str:
        """
        Lightweight PDF text extraction focused on text drawing operators.
        Falls back to decoded bytes if no text tokens are found.
        """
        raw = MCPServerService._as_text(payload_bytes)

        # Capture text blocks inside BT ... ET and extract (...) Tj / [...] TJ payloads.
        print("get chunks...", file=sys.stderr)
        chunks: List[str] = []
        for block in re.findall(r"BT(.*?)ET", raw, flags=re.DOTALL):
            chunks.extend(re.findall(r"\((.*?)\)\s*Tj", block, flags=re.DOTALL))
            for arr in re.findall(r"\[(.*?)\]\s*TJ", block, flags=re.DOTALL):
                chunks.extend(re.findall(r"\((.*?)\)", arr, flags=re.DOTALL))

        if not chunks:
            return raw

        cleaned = []
        for c in chunks:
            txt = c.replace(r"\(", "(").replace(r"\)", ")").replace(r"\\", "\\")
            txt = " ".join(txt.split())
            if txt:
                cleaned.append(txt)
        finalized = "\n".join(cleaned) if cleaned else raw
        print(finalized, file=sys.stderr)
        print("get chunks... done", file=sys.stderr)
        return finalized


    def _load_eq_extractor_class(self):
        """Lazily import and cache EqExtractor class from math/eq_extractor.py."""
        if self._eq_extractor_cls is not None:
            return self._eq_extractor_cls
        try:
            eq_path = Path(__file__).resolve().parents[1] / "math" / "eq_extractor.py"
            spec = importlib.util.spec_from_file_location("eq_storage_math_eq_extractor", str(eq_path))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                self._eq_extractor_cls = getattr(mod, "EqExtractor", None)
        except Exception:
            self._eq_extractor_cls = None
        return self._eq_extractor_cls

    def _extract_content_parts(self, text: str, file_name, user_id):
        """
        Run the equation extractor and append results into the internal graph.

        Args:
            text: Source text to analyze.
            file_name: Context/module identifier attached to extracted nodes.
            user_id: Owning user identifier used in graph metadata.
        """
        EqExtractorCls = self._load_eq_extractor_class()
        if EqExtractorCls is None:
            return []

        try:
            extractor = EqExtractorCls(debug=False)
            extractor.text_to_multigraph(
                text=text,
                context_id=file_name,
                module_id=file_name,
                user_id=user_id,
                g=self.g,
            )
            print("extracted", len(self.g.G.nodes), "nodes", file=sys.stderr)

        except Exception as e:
            print("Err", e, file=sys.stderr)


    def get_graph(self, user_id: str, test: bool = False) -> GraphResponse:
        """
        Build and return a graph view for a user from persisted data.

        Args:
            user_id: User identifier for graph filtering.
            test: If true, keep graph artifacts in project output directory.
        """
        print("[MCPServerService.get_graph] START", file=sys.stderr)
        print(f"[MCPServerService.get_graph] LOGIC_GATE user_id={user_id} test={test}", file=sys.stderr)
        if not user_id:
            return GraphResponse(status="error", user_id="", stats={"message": "user_id is required"})

        try:
            from graph.qa.visual_g import VisualizeGraph

            visualizer = VisualizeGraph(db=self.db)
            result = visualizer.run(user_id=user_id, test=test)

            nodes = [
                GraphNodeOut(id=str(n.get("id") or ""), attrs=dict(n.get("attrs") or {}))
                for n in (result.get("nodes") or [])
            ]
            edges = [
                GraphEdgeOut(
                    source=str(e.get("src") or ""),
                    target=str(e.get("trgt") or ""),
                    attrs=dict(e.get("attrs") or {}),
                )
                for e in (result.get("edges") or [])
            ]

            stats = dict(result.get("stats") or {})
            stats["artifacts"] = result.get("artifacts") or {}
            print("[MCPServerService.get_graph] END ok", file=sys.stderr)
            return GraphResponse(
                status="ok",
                user_id=user_id,
                nodes=nodes,
                edges=edges,
                stats=stats,
            )
        except Exception as exc:
            print(f"[MCPServerService.get_graph] END error={exc}", file=sys.stderr)
            return GraphResponse(
                status="error",
                user_id=user_id,
                stats={"message": str(exc)},
            )

    def upsert(self, request: UpsertRequest):
        """
        Ingest text/files, extract graph entities, and upsert rows into storage.
        Args:
            request: Upsert payload containing user_id, files, and optional equation.
        """
        print("upsert...", file=sys.stderr)
        user_id =request.user_id
        if not request.user_id:
            return UpsertResponse(status="error", message="user_id is required")

        normalized = [
            (f"file_{user_id}_{generate_id(20)}", file.encode("utf-8", errors="ignore"))
            for file in request.data.files
        ]
        print("files normalized", file=sys.stderr)

        file_ids: List[str] = []
        file_rows: List[Dict[str, Any]] = []
        method_rows: List[Dict[str, Any]] = []
        param_rows: List[Dict[str, Any]] = []
        operator_rows: List[Dict[str, Any]] = []

        for file_id, file_bytes in normalized:
            print("work file", file_id, file=sys.stderr)
            file_text = (
                self._extract_text_from_pdf_bytes(file_bytes)
                if file_bytes.startswith(b"%PDF")
                else self._as_text(file_bytes)
            )
            print("file_text...", file=sys.stderr)


            # EXTRACT EQUATIONS
            self._extract_content_parts(file_text, file_id, user_id)
            print("_extract_content_parts... done", file=sys.stderr)

            file_ids.append(file_id)
            file_rows.append(
                {
                    "id": file_id,
                    "user_id": request.user_id,
                    "content": file_bytes,
                    "created_at": self._now(),
                }
            )
            print("file id... done", file=sys.stderr)


        # EXTRACT EQUATIONS
        if request.data.equation:
            self._extract_content_parts(request.data.equation, f"{user_id}_{generate_id(20)}", user_id)
            print("request.data.equation... done", file=sys.stderr)

        check = {}
        for k, v in self.g.G.nodes(data=True):
            ntype = v["type"]
            if v["type"] == "METHOD":
                v["param_neighbors"] = self.g.get_neighbor_list(node=k, target_type="PARAM", just_ids=True)
                v["operator_neighbors"] = self.g.get_neighbor_list(node=k, target_type="OPERATOR", just_ids=True)
                if ntype not in check:
                    check[ntype] = []
                check[ntype].append(k)
                method_rows.append({"id":k,**v})

            elif v["type"] == "PARAM":
                v["method_neighbors"] = self.g.get_neighbor_list(node=k, target_type="METHOD", just_ids=True)
                v["operator_neighbors"] = self.g.get_neighbor_list(node=k, target_type="OPERATOR", just_ids=True)
                if ntype not in check:
                    check[ntype] = []
                check[ntype].append(k)
                param_rows.append({"id":k,**v})

            elif v["type"] == "OPERATOR":
                v["method_neighbors"] = self.g.get_neighbor_list(node=k, target_type="METHOD", just_ids=True)
                v["operator_neighbors"] = self.g.get_neighbor_list(node=k, target_type="OPERATOR", just_ids=True)
                if ntype not in check:
                    check[ntype] = []
                check[ntype].append(k)
                operator_rows.append({"id":k,**v})

        # EDGES
        edge_rows = []
        for src, trgt, attrs in self.g.G.edges(data=True):
            edge_rows.append({"src":src,"trgt":trgt,"attrs":attrs})

        self.db.insert("edges", edge_rows)
        self.db.insert("methods", method_rows)
        self.db.insert("params", param_rows)
        self.db.insert("operators", operator_rows)
        self.db.insert("files", file_rows)
        print(check, file=sys.stderr)
        print("upsert... done", file=sys.stderr)

    def get_entry(self, entry_id: str, table: str = "methods", user_id: Optional[str] = None) -> EntryResponse:
        """
        Fetch a single entry by id from a table, optionally scoped by user.

        Args:
            entry_id: Primary identifier of the requested row.
            table: Target table name.
            user_id: Optional owner filter.
        """
        try:
            row = self.db.row_from_id(entry_id, table=table, user_id=user_id)
        except ValueError as exc:
            return EntryResponse(status="error", table=table, message=str(exc))
        if not row:
            return EntryResponse(status="not_found1", table=table, message="Entry not found")
        return EntryResponse(status="ok", entry=row, table=table)


    def delete_entries(self, request: DeleteRequest) -> DeleteResponse:
        """
        Delete a single entry or all entries for a user in a table.
        Args:
            request: Delete payload with user_id, table, and optional entry_id.
        """
        if not request.user_id:
            return DeleteResponse(status="error", message="user_id is required")
        if request.entry_id:
            deleted:int = self.db.del_entry(nid=request.entry_id, table=request.table, user_id=request.user_id, )
            return DeleteResponse(status="ok", deleted_count=deleted, mode="single")
        self.db.delete(table=request.table, where_clause=f"WHERE user_id = ?", params={"user_id": request.user_id})
        return DeleteResponse(status="ok", deleted_count=-1, mode="all")

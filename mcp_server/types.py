from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


@dataclass
class UpsertDataIn:
    files: List[Any] = field(default_factory=list)
    equation: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "UpsertDataIn":
        return cls(
            files=list(payload.get("files") or []),
            equation=payload.get("equation"),
        )


@dataclass
class UpsertRequest:
    user_id: str
    data: UpsertDataIn
    module_id: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "UpsertRequest":
        return cls(
            user_id=str(payload.get("user_id") or "").strip(),
            data=UpsertDataIn.from_dict(payload.get("data") or {}),
            module_id=payload.get("module_id"),
        )


@dataclass
class UpsertResponse:
    status: str
    method_ids: List[str] = field(default_factory=list)
    param_ids: List[str] = field(default_factory=list)
    file_ids: List[str] = field(default_factory=list)
    extracted_equations: List[str] = field(default_factory=list)
    method_link_index: Dict[str, Any] = field(default_factory=dict)
    message: Optional[str] = None


@dataclass
class EntryResponse:
    status: str
    entry: Optional[Dict[str, Any]] = None
    table: str = "methods"
    message: Optional[str] = None


@dataclass
class GraphNodeOut:
    id: str
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdgeOut:
    source: str
    target: str
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphResponse:
    status: str
    user_id: str
    nodes: List[GraphNodeOut] = field(default_factory=list)
    edges: List[GraphEdgeOut] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

class GetEntryRequest(BaseModel):
    entry_id: str
    table: str

class GetGraphRequest(BaseModel):
    user_id: str

@dataclass
class DeleteRequest:
    user_id: str
    table: str
    entry_id: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DeleteRequest":
        return cls(
            user_id=str(payload.get("user_id") or "").strip(),
            table=str(payload.get("table") or "methods").strip() or "methods",
            entry_id=payload.get("entry_id"),
        )


@dataclass
class DeleteResponse:
    status: str
    deleted_count: int = 0
    mode: str = "single"
    message: Optional[str] = None


@dataclass
class MCPToolCallRequest:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MCPToolCallRequest":
        return cls(
            name=str(payload.get("name") or "").strip(),
            arguments=dict(payload.get("arguments") or {}),
        )


@dataclass
class MCPToolDescription:
    name: str
    description: str
    input_schema: Dict[str, Any]

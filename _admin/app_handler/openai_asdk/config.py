"""
Configuration schema for BestBrain OpenAI Apps SDK deployment.

Ref: https://developers.openai.com/apps-sdk/

Local demo paths (demo_video_path, demo_html_dir) are written by
_admin.record_qdash_demo when running: python -m _admin.main --record-qdash-demo.
The OpenAI app creator / submission workflow can use get_demo_paths() to obtain
the path to the recorded MP4 and HTML captures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AppConfig:
    """App-level configuration for ChatGPT Apps Directory."""

    name: str = "BestBrain"
    version: str = "0.1.0"
    description: str = "Physics simulation and scientific computing environment."
    # Required for submission
    privacy_policy_url: Optional[str] = None
    support_email: Optional[str] = None
    logo_url: Optional[str] = None


@dataclass
class MCPConfig:
    """MCP server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    path: str = "/mcp"
    # Stateless mode for simple deployments
    stateless: bool = True


@dataclass
class CSPConfig:
    """Content Security Policy for widget (required for app submission)."""

    connect_domains: List[str] = field(default_factory=lambda: ["https://api.openai.com"])
    resource_domains: List[str] = field(default_factory=list)
    frame_domains: List[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Local demo paths: canonical entry for the OpenAI app creator process.
# Written by _admin.record_qdash_demo (python -m _admin.main --record-qdash-demo).
# -----------------------------------------------------------------------------
DEMO_PATHS_FILE = Path(__file__).resolve().parent / "demo_paths.json"


def get_demo_paths() -> Dict[str, Any]:
    """
    Read local paths for demo video and HTML captures (from --record-qdash-demo).
    Used by the OpenAI app creator process when submitting or attaching demo assets.

    Returns:
        Dict with keys: demo_video_path (str), demo_html_dir (str), demo_video_webm (str | None).
        Paths are absolute. If demo_paths.json is missing, returns empty dict.
    """
    if not DEMO_PATHS_FILE.is_file():
        return {}
    try:
        data = json.loads(DEMO_PATHS_FILE.read_text(encoding="utf-8"))
        return {
            "demo_video_path": data.get("demo_video_path") or "",
            "demo_html_dir": data.get("demo_html_dir") or "",
            "demo_video_webm": data.get("demo_video_webm"),
        }
    except Exception:
        return {}


def get_demo_video_path() -> Optional[Path]:
    """Convenience: return Path to demo MP4 if it exists, else None."""
    paths = get_demo_paths()
    p = paths.get("demo_video_path")
    if not p:
        return None
    path = Path(p)
    return path if path.is_file() else None

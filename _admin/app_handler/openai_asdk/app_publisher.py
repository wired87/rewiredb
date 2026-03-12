"""
AppPublisher: Best practices class for publishing BestBrain as an OpenAI Apps SDK app.

Encapsulates requirements from:
- https://developers.openai.com/apps-sdk/
- https://developers.openai.com/apps-sdk/deploy
- https://developers.openai.com/apps-sdk/deploy/submission
- https://developers.openai.com/apps-sdk/app-submission-guidelines
- https://developers.openai.com/apps-sdk/guides/security-privacy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolAnnotation:
    """Tool hint annotations required for ChatGPT app submission.

    Ref: https://developers.openai.com/apps-sdk/reference#annotations
    """

    read_only_hint: bool = False  # True if tool only retrieves/lists, no side effects
    destructive_hint: bool = False  # True if irreversible (delete, overwrite, send)
    open_world_hint: bool = False  # True if writes to public internet (post, email, etc.)
    idempotent_hint: Optional[bool] = None  # True if same args = no extra effect


@dataclass
class CSPConfig:
    """Content Security Policy for widget iframe (required for app submission).

    Ref: https://developers.openai.com/apps-sdk/reference#component-resource-_meta-fields
    """

    connect_domains: List[str] = field(default_factory=list)  # fetch/XHR targets
    resource_domains: List[str] = field(default_factory=list)  # images, fonts, scripts
    frame_domains: List[str] = field(default_factory=list)  # iframe sources (opt-in, stricter review)


@dataclass
class AppMetadata:
    """App metadata for ChatGPT Apps Directory."""

    name: str
    version: str
    description: str
    logo_url: Optional[str] = None
    privacy_policy_url: Optional[str] = None
    support_contact: Optional[str] = None
    terms_of_service_url: Optional[str] = None
    company_url: Optional[str] = None
    subtitle: Optional[str] = None
    category: Optional[str] = None


class AppPublisher:
    """
    Best-practices class for publishing BestBrain as an OpenAI Apps SDK app.

    Workflow:
    1. Build MCP server with /mcp endpoint
    2. Wrap in Docker with stable HTTPS
    3. Test via MCP Inspector + ChatGPT developer mode
    4. Submit to OpenAI Platform Dashboard
    """

    # --- Deployment requirements (from deploy docs) ---
    MCP_PATH = "/mcp"
    REQUIRED_HTTP_METHODS = {"POST", "GET", "DELETE"}
    CORS_HEADERS = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
        "Access-Control-Allow-Headers": "content-type, mcp-session-id",
        "Access-Control-Expose-Headers": "Mcp-Session-Id",
    }

    def __init__(
        self,
        app_metadata: AppMetadata,
        mcp_server_factory: Optional[Callable[[], Any]] = None,
        docker_image_name: str = "bestbrain-mcp-app",
        port: int = 8000,
    ):
        """
        Args:
            app_metadata: App name, version, description for directory listing.
            mcp_server_factory: Callable that returns an MCP server instance.
            docker_image_name: Name for the Docker image.
            port: Port the MCP server listens on (default 8000 per quickstart).
        """
        self.app_metadata = app_metadata
        self.mcp_server_factory = mcp_server_factory
        self.docker_image_name = docker_image_name
        self.port = port

    # --- Pre-submission checklist (from app-submission-guidelines) ---

    def check_prerequisites(self) -> List[str]:
        """
        Returns list of missing prerequisites for app submission.
        Empty list = ready to submit.
        """
        issues: List[str] = []

        # Privacy (required)
        if not self.app_metadata.privacy_policy_url:
            issues.append("Provide a published privacy policy URL.")

        # Support (required)
        if not self.app_metadata.support_contact:
            issues.append("Provide customer support contact details.")

        return issues

    def get_prerequisite_checklist(self) -> List[str]:
        """
        Full checklist of items to complete before submission (informational).
        """
        return [
            "Complete identity verification in OpenAI Platform Dashboard (Settings → Organization → General).",
            "Define Content Security Policy (CSP) in resource _meta.ui.csp with connectDomains, resourceDomains.",
            "Host MCP server on publicly accessible HTTPS domain (no localhost).",
            "Provide a published privacy policy URL.",
            "Provide customer support contact details.",
        ]

    def validate_tool_annotations(self, tools: List[Dict[str, Any]]) -> List[str]:
        """
        Validates tool annotations per submission guidelines.
        Returns list of validation errors.
        """
        errors: List[str] = []

        for t in tools:
            name = t.get("name", "?")
            ann = t.get("annotations", {})
            if "readOnlyHint" not in ann:
                errors.append(f"Tool '{name}': missing readOnlyHint")
            if "destructiveHint" not in ann:
                errors.append(f"Tool '{name}': missing destructiveHint")
            if "openWorldHint" not in ann:
                errors.append(f"Tool '{name}': missing openWorldHint")

        return errors

    def get_deployment_options(self) -> Dict[str, str]:
        """
        Recommended deployment options from Apps SDK deploy docs.
        """
        return {
            "vercel": "Next.js starter, preview envs, automatic HTTPS",
            "cloud_run": "Google Cloud Run, scale-to-zero, watch cold starts",
            "fly_io": "Fly.io, managed containers, predictable streaming",
            "render": "Render, quick spin-up, automatic TLS",
            "railway": "Railway, managed containers",
            "kubernetes": "For teams with existing clusters; use ingress for SSE",
        }

    def get_docker_best_practices(self) -> List[str]:
        """
        Docker best practices for MCP app deployment.
        """
        return [
            "Expose single port for MCP (e.g. 8000); map to 80/443 via ingress.",
            "Use multi-stage build: builder for deps, slim runner for runtime.",
            "Set PYTHONUNBUFFERED=1 for real-time logs.",
            "Health check: GET / returns 200 (e.g. 'MCP server' plain text).",
            "OPTIONS /mcp must return 204 with CORS headers for preflight.",
            "Store secrets via env vars (not in image); use platform secret managers.",
        ]

    def get_security_checklist(self) -> List[str]:
        """
        Security & privacy checklist from guides/security-privacy.
        """
        return [
            "Defense in depth: validate all inputs; assume prompt injection.",
            "Redact PII before logging; use correlation IDs for debugging.",
            "Require human confirmation for irreversible operations.",
            "Verify and enforce scopes on every tool call.",
            "Keep dependencies patched; monitor anomalous traffic.",
        ]

    def get_testing_checklist(self) -> List[str]:
        """
        Regression checklist before launch (from deploy/testing).
        """
        return [
            "Unit test each tool handler with representative inputs.",
            "Use MCP Inspector: npx @modelcontextprotocol/inspector --server-url <url> --transport http",
            "Test in ChatGPT developer mode (Settings → Apps & Connectors).",
            "Run golden prompts (direct, indirect, negative); record precision/recall.",
            "Test mobile layouts (iOS/Android) if using custom UI.",
            "Verify structuredContent matches declared outputSchema.",
        ]

    def get_submission_steps(self) -> List[str]:
        """
        Steps to submit app from deploy/submission.
        """
        return [
            "1. Complete organization verification (Dashboard → Settings → General).",
            "2. Ensure Owner role in organization.",
            "3. Go to platform.openai.com/apps-manage → Submit for review.",
            "4. Fill: app name, logo, description, privacy policy, company URLs.",
            "5. Add MCP server URL (HTTPS + /mcp) and OAuth credentials if used.",
            "6. Provide demo account (no MFA) for authenticated apps.",
            "7. Add screenshots, test prompts, expected responses.",
            "8. Add localization info if supporting multiple locales.",
        ]

    def build_docker_cmd(self, project_root: Optional[Path] = None) -> str:
        """
        Returns docker build command for the MCP app image.
        """
        if project_root is None:
            # __file__ -> openai_asdk -> app_handler -> _admin -> project root
            project_root = Path(__file__).resolve().parent.parent.parent.parent
        dockerfile_rel = "_admin/app_handler/openai_asdk/Dockerfile.mcp"
        return f"docker build -t {self.docker_image_name} -f {dockerfile_rel} {project_root}"

    def run_local_cmd(self) -> str:
        """
        Returns command to run MCP server locally (for dev with ngrok).
        """
        return f"py -m app_handler.openai_asdk.mcp_server --port {self.port}"

    def ngrok_expose_cmd(self) -> str:
        """
        Returns ngrok command to expose local server for ChatGPT testing.
        """
        return f"ngrok http {self.port}"

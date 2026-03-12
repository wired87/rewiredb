"""
OpenAI Apps SDK Workflow: End-to-end production-ready workflow for publishing BestBrain.

This module orchestrates the full Apps SDK lifecycle with hardcoded test data
to stabilize understanding and enable reproducible runs.

Workflow steps (each documented inline):
  1. Load and validate hardcoded app metadata
  2. Run prerequisite checks for App Store submission
  3. Validate tool definitions and annotations
  4. Start MCP server and verify health
  5. Run golden prompt tests (direct, indirect, negative)
  6. Output submission checklist and next steps

Run: py -m app_handler.openai_asdk.workflow

Ref: https://developers.openai.com/apps-sdk/
     https://developers.openai.com/apps-sdk/deploy/testing
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# HARDCODED TEST DATA STRUCTURES
# =============================================================================
# These structures stabilize the workflow and match Apps SDK submission
# requirements. Do not change without updating submission form accordingly.
# Ref: https://developers.openai.com/apps-sdk/app-submission-guidelines

# --- App metadata: required for ChatGPT Apps Directory listing ---
# Must include: name, version, description, privacy_policy_url, support_contact
# Ref: https://developers.openai.com/apps-sdk/deploy/submission
APP_METADATA = {
    "name": "BestBrain",
    "version": "0.1.0",
    "description": "Physics simulation and scientific computing environment for lattice field theory, fermions, and gauge fields. Run and manage simulations, list saved environments, and create new physics simulations with configurable modules.",
    "privacy_policy_url": "https://bestbrain.tech/privacy",
    "terms_of_service_url": "https://bestbrain.tech/terms",
    "support_contact": "support@bestbrain.tech",
    "company_url": "https://bestbrain.tech",
    "logo_url": "https://bestbrain.tech/logo.png",
    "subtitle": "Physics simulation and scientific computing for lattice field theory",
    "category": "Developer tools",
}

# --- Tool definitions: MCP tools the app exposes to ChatGPT ---
# Each tool MUST have annotations: readOnlyHint, destructiveHint, openWorldHint.
# readOnlyHint=True: only retrieves data, no side effects.
# destructiveHint=True: irreversible (delete, overwrite, send).
# openWorldHint=True: writes to public internet (post, email, etc.).
TOOLS = [
    {
        "name": "list_simulations",
        "title": "List simulations",
        "description": "Returns the user's saved simulations without modifying them.",
        "inputSchema": {
            "type": "object",
            "properties": {"user_id": {"type": "string", "description": "User identifier"}},
            "required": [],
        },
        "annotations": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "openWorldHint": False,
        },
    },
    {
        "name": "create_simulation",
        "title": "Create simulation",
        "description": "Creates a new physics simulation with the given parameters.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "env_id": {"type": "string"},
                "modules": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["env_id", "modules"],
        },
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "openWorldHint": False,
        },
    },
]

# --- Golden prompts: test prompts for discovery validation ---
# Used when testing in ChatGPT developer mode. Record which tool the model
# selects and whether confirmation prompts appear. Ref: deploy/testing
# - direct: explicitly asks for app functionality
# - indirect: implies intent without naming the app
# - negative: should NOT trigger the app
GOLDEN_PROMPTS = {
    "direct": [
        "List my simulations",
        "Show me my saved simulations",
        "What simulations do I have?",
    ],
    "indirect": [
        "I want to see what I've run before",
        "Check my physics simulation history",
    ],
    "negative": [
        "What's the weather today?",
        "Send an email to john@example.com",
    ],
}

# --- Expected tool response shape: for structuredContent validation ---
# Tool results should include structuredContent (model-visible) and content.
# Ref: Apps SDK reference - Tool results
EXPECTED_TOOL_RESPONSE = {
    "structuredContent": {"simulations": [], "count": 0},
    "content": [{"type": "text", "text": "No simulations found."}],
}

# --- MCP server config: port and path for local testing ---
# ChatGPT connector URL format: https://<domain>/mcp
MCP_SERVER_PORT = 8000
MCP_SERVER_HOST = "127.0.0.1"
MCP_PATH = "/mcp"
HEALTH_PATH = "/"

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# STEP 1: Load and validate app metadata
# -----------------------------------------------------------------------------
def step_load_metadata() -> Tuple[Dict[str, Any], List[str]]:
    """
    STEP 1: Load hardcoded app metadata and validate required fields.

    Required for submission:
    - name, version, description
    - privacy_policy_url, support_contact (enforced in check_prerequisites)

    Returns:
        (metadata, errors) - errors empty if valid
    """
    errors: List[str] = []

    required = ["name", "version", "description"]
    for k in required:
        if not APP_METADATA.get(k):
            errors.append(f"APP_METADATA missing required field: {k}")

    if not APP_METADATA.get("privacy_policy_url"):
        errors.append("APP_METADATA.privacy_policy_url required for submission")

    if not APP_METADATA.get("support_contact"):
        errors.append("APP_METADATA.support_contact required for submission")

    return APP_METADATA, errors


# -----------------------------------------------------------------------------
# STEP 2: Run prerequisite checks
# -----------------------------------------------------------------------------
def step_check_prerequisites(metadata: Dict[str, Any]) -> List[str]:
    """
    STEP 2: Run prerequisite checklist for App Store submission.

    Checks:
    - Privacy policy URL present
    - Support contact present
    - (Informational) Org verification, CSP, HTTPS hosting

    Returns:
        List of missing items (empty = ready)
    """
    from _admin.app_handler.openai_asdk.app_publisher import AppMetadata, AppPublisher

    app_meta = AppMetadata(
        name=metadata["name"],
        version=metadata["version"],
        description=metadata["description"],
        privacy_policy_url=metadata.get("privacy_policy_url"),
        support_contact=metadata.get("support_contact"),
    )
    publisher = AppPublisher(app_metadata=app_meta)
    return publisher.check_prerequisites()


# -----------------------------------------------------------------------------
# STEP 3: Validate tool annotations
# -----------------------------------------------------------------------------
def step_validate_tools(tools: List[Dict[str, Any]]) -> List[str]:
    """
    STEP 3: Validate tool definitions have required annotations.

    Per submission guidelines, each tool must have:
    - readOnlyHint: True if read-only, False if write
    - destructiveHint: True if irreversible
    - openWorldHint: True if writes to public internet

    Returns:
        List of validation errors (empty = valid)
    """
    from _admin.app_handler.openai_asdk.app_publisher import AppMetadata, AppPublisher

    publisher = AppPublisher(app_metadata=AppMetadata(**APP_METADATA))
    return publisher.validate_tool_annotations(tools)


# -----------------------------------------------------------------------------
# STEP 4: Start MCP server and verify health
# -----------------------------------------------------------------------------
def step_start_server_and_health_check(
    port: int = MCP_SERVER_PORT,
    host: str = MCP_SERVER_HOST,
    timeout_start: float = 5.0,
) -> Tuple[Optional[subprocess.Popen], bool, str]:
    """
    STEP 4: Start MCP server and verify health endpoint.

    ChatGPT and MCP Inspector expect:
    - GET / returns 200 (health check)
    - OPTIONS /mcp returns 204 (CORS preflight)
    - POST /mcp accepts JSON-RPC 2.0

    This step spawns the server, polls GET / until 200, then returns.
    Caller must terminate the process when done.

    Returns:
        (process, ok, message) - process is None if not started
    """
    try:
        import urllib.request

        # Project root so _admin.app_handler.openai_asdk resolves
        admin_dir = Path(__file__).resolve().parent.parent.parent  # _admin
        project_root = admin_dir.parent
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

        # Spawn MCP server in subprocess; stdout/stderr piped to avoid cluttering workflow output
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "_admin.app_handler.openai_asdk.mcp_server",
                "--port",
                str(port),
                "--host",
                host,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(project_root),
            env=env,
        )

        # Poll until server responds to GET / (up to timeout_start seconds)
        for _ in range(int(timeout_start * 10)):
            time.sleep(0.1)
            try:
                with urllib.request.urlopen(f"http://{host}:{port}{HEALTH_PATH}", timeout=1) as r:
                    if r.status == 200:
                        return proc, True, "Health check OK"
            except OSError:
                pass

        proc.terminate()
        proc.wait(timeout=2)
        return None, False, "Health check timed out"
    except Exception as e:
        return None, False, str(e)


# -----------------------------------------------------------------------------
# STEP 5: Run MCP JSON-RPC test (tool call simulation)
# -----------------------------------------------------------------------------
def step_run_mcp_test(
    port: int = MCP_SERVER_PORT,
    host: str = MCP_SERVER_HOST,
) -> Tuple[bool, str]:
    """
    STEP 5: Send a minimal MCP JSON-RPC request to /mcp and verify response.

    The MCP protocol uses JSON-RPC 2.0 over HTTP. This step sends an
    "initialize" method (or any method) and checks that the response is
    valid JSON-RPC (has jsonrpc, id, result or error).

    Full MCP would require: initialize, tools/list, tools/call.
    Our stub responds to any method; this validates the endpoint is reachable.

    Returns:
        (success, message)
    """
    try:
        import urllib.request

        url = f"http://{host}:{port}{MCP_PATH}"
        req = urllib.request.Request(
            url,
            data=json.dumps(
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            resp = json.loads(r.read().decode())
            if resp.get("jsonrpc") == "2.0" and "id" in resp and ("result" in resp or "error" in resp):
                return True, "MCP JSON-RPC response valid"
            return False, f"Unexpected response: {resp}"
    except Exception as e:
        return False, str(e)


# -----------------------------------------------------------------------------
# STEP 6: Output submission checklist
# -----------------------------------------------------------------------------
def step_output_checklist() -> List[str]:
    """
    STEP 6: Output full submission checklist for next steps.

    Returns the checklist items for manual review.
    """
    from _admin.app_handler.openai_asdk.app_publisher import AppMetadata, AppPublisher

    app_meta = AppMetadata(**APP_METADATA)
    publisher = AppPublisher(app_metadata=app_meta)
    return publisher.get_submission_steps()


# -----------------------------------------------------------------------------
# MAIN WORKFLOW
# -----------------------------------------------------------------------------
def run_workflow(
    start_server: bool = True,
    port: int = MCP_SERVER_PORT,
) -> bool:
    """
    Run the complete OpenAI Apps SDK workflow.

    Args:
        start_server: If True, start MCP server and run HTTP tests.
        port: Port for MCP server.

    Returns:
        True if all steps pass, False otherwise.
    """
    all_ok = True
    server_proc: Optional[subprocess.Popen] = None

    logger.info("=" * 60)
    logger.info("OpenAI Apps SDK Workflow - BestBrain")
    logger.info("=" * 60)

    # --- STEP 1: Load metadata ---
    logger.info("")
    logger.info("[STEP 1] Load and validate app metadata")
    metadata, meta_errors = step_load_metadata()
    if meta_errors:
        for e in meta_errors:
            logger.error("  %s", e)
        all_ok = False
    else:
        logger.info("  OK: app=%s v=%s", metadata["name"], metadata["version"])

    # --- STEP 2: Prerequisites ---
    logger.info("")
    logger.info("[STEP 2] Prerequisite checks")
    prereqs = step_check_prerequisites(metadata)
    if prereqs:
        for p in prereqs:
            logger.warning("  Missing: %s", p)
        all_ok = False
    else:
        logger.info("  OK: prerequisites complete")

    # --- STEP 3: Tool validation ---
    logger.info("")
    logger.info("[STEP 3] Validate tool annotations")
    tool_errors = step_validate_tools(TOOLS)
    if tool_errors:
        for e in tool_errors:
            logger.error("  %s", e)
        all_ok = False
    else:
        logger.info("  OK: %d tools valid", len(TOOLS))

    # --- STEP 4 & 5: Server and MCP test ---
    if start_server:
        logger.info("")
        logger.info("[STEP 4] Start MCP server and health check")
        server_proc, health_ok, health_msg = step_start_server_and_health_check(port=port)
        if not health_ok:
            logger.error("  %s", health_msg)
            all_ok = False
        else:
            logger.info("  OK: %s", health_msg)

        logger.info("")
        logger.info("[STEP 5] MCP JSON-RPC test")
        mcp_ok, mcp_msg = step_run_mcp_test(port=port)
        if not mcp_ok:
            logger.error("  %s", mcp_msg)
            all_ok = False
        else:
            logger.info("  OK: %s", mcp_msg)

        if server_proc:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                server_proc.kill()
    else:
        logger.info("")
        logger.info("[STEP 4-5] Skipped (start_server=False)")

    # --- STEP 6: Golden prompts reference ---
    logger.info("")
    logger.info("[STEP 6a] Golden prompts (for manual ChatGPT testing)")
    logger.info("  Direct:   %s", GOLDEN_PROMPTS["direct"][0])
    logger.info("  Indirect: %s", GOLDEN_PROMPTS["indirect"][0])
    logger.info("  Negative: %s", GOLDEN_PROMPTS["negative"][0])

    # --- STEP 6b: Local demo path (for OpenAI app creator) ---
    try:
        from _admin.app_handler.openai_asdk.config import get_demo_paths
        demo = get_demo_paths()
        if demo.get("demo_video_path"):
            logger.info("")
            logger.info("[STEP 6b] Local demo assets (from --record-qdash-demo)")
            logger.info("  demo_video_path: %s", demo.get("demo_video_path"))
            logger.info("  demo_html_dir:   %s", demo.get("demo_html_dir"))
    except Exception:
        pass

    # --- STEP 6c: Submission checklist ---
    logger.info("")
    logger.info("[STEP 6c] Submission checklist")
    for step in step_output_checklist():
        logger.info("  %s", step)

    # --- STEP 7: Write submission manifest (all required data for App Store form) ---
    try:
        from _admin.app_handler.openai_asdk.submission_manifest import build_submission_manifest
        manifest_dir = Path(__file__).resolve().parent
        manifest_path = manifest_dir / "app_submission_manifest.json"
        build_submission_manifest(metadata=metadata, out_path=manifest_path)
        logger.info("")
        logger.info("[STEP 7] Submission manifest written: %s", manifest_path)
        logger.info("  Use this file when filling the form at %s", "https://platform.openai.com/apps-manage")
    except Exception as e:
        logger.warning("[STEP 7] Could not write submission manifest: %s", e)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Workflow complete: %s", "PASS" if all_ok else "FAIL")
    logger.info("=" * 60)

    return all_ok


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Parse args: --no-server skips steps 4-5 (useful when port in use)
    import argparse as _argparse

    _parser = _argparse.ArgumentParser(description="OpenAI Apps SDK workflow")
    _parser.add_argument(
        "--no-server",
        action="store_true",
        help="Skip MCP server start and HTTP tests (steps 4-5)",
    )
    _parser.add_argument("--port", type=int, default=MCP_SERVER_PORT, help="MCP server port")
    _args = _parser.parse_args()

    # Run full workflow with hardcoded test data
    success = run_workflow(
        start_server=not _args.no_server,
        port=_args.port,
    )
    sys.exit(0 if success else 1)

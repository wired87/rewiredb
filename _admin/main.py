#!/usr/bin/env python
"""
Admin workflow entry point: run with `python -m _admin.main`.

- Discovers all project dirs (Dockerfile + package.json / manage.py / requirements.txt).
- Classifies each as backend_drf | backend_fastapi | backend_py | frontend_react | mobile_react_native.
- With --local: start backend applications (DRF, FastAPI, pure Py) and React web frontends from inferred project dirs; React Native is skipped.
- Without --local, with --deploy: build each package (MiracleAI, qbrain, qdash) and deploy each Docker to Kubernetes.
  Deploy target: env LOCAL=true -> local cluster; else collect env vars and deploy to cloud (e.g. GKE).
- --run-local: same as --local (run apps natively). --scan-only: print discovered projects only.
"""
import argparse
import os
import sys
from pathlib import Path

# Package names to build and deploy when --deploy is used (no --local)
DEPLOY_PACKAGES = ("MiracleAI", "qbrain", "qdash")

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Load .env from project root
try:
    from dotenv import load_dotenv
    _env_path = _project_root / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Admin: discover projects; with --local: start backends and React web frontends from inferred dirs (React Native skipped)."
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Start backend apps (DRF, FastAPI, pure Py) and React web frontends from inferred project dirs. React Native is not started.",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Only print discovered and classified project dirs, then exit",
    )
    # Run locally (inferred start commands)
    parser.add_argument(
        "--run-local",
        action="store_true",
        help="Alias for --local: start backends and React web frontends from inferred dirs.",
    )
    parser.add_argument(
        "--run-local-project",
        type=str,
        default=None,
        metavar="PATH",
        help="Run only this project (path relative to project root or absolute). With --run-local only.",
    )
    parser.add_argument(
        "--run-local-port",
        type=str,
        default=None,
        metavar="PORT_OR_MAP",
        help="Port or mapping for run-local (e.g. 8000 or backend_drf:8000,frontend:3000). Defaults: backend 8000, frontend 3000.",
    )
    parser.add_argument(
        "--run-local-scan-only",
        action="store_true",
        help="Print discovered projects with inferred type, start command, and cwd; no execution.",
    )
    parser.add_argument(
        "--run-local-testing",
        action="store_true",
        help="Run with testing/debug mode (backend+frontend only, no pure Python apps); sets TESTING=1.",
    )
    parser.add_argument(
        "--separate-terminals",
        action="store_true",
        help="With --local: start each app (React, DRF) in its own terminal window; then exit (no Ctrl+C to stop all).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="With --local: print exact command, cwd, and env for each launched app (debug).",
    )
    # Deploy to Kubernetes (when --local is not set)
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Build packages (MiracleAI, qbrain, qdash) and deploy each Docker to K8s. Target: LOCAL=true -> local cluster; else cloud (GKE).",
    )
    parser.add_argument("--no-build", action="store_true", help="Skip build step (with --deploy).")
    parser.add_argument("--no-deploy", action="store_true", help="Skip deploy step (with --deploy).")
    parser.add_argument("--no-push", action="store_true", help="Do not push to Artifact Registry when deploying to cloud.")
    parser.add_argument("--no-npm-build", action="store_true", help="Do not run npm run build for frontend/mobile without Dockerfile.")
    parser.add_argument("--tag", default="latest", help="Docker tag (with --deploy).")
    parser.add_argument("--namespace", default="default", help="Kubernetes namespace (with --deploy).")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild Docker images even if they exist (with --deploy).")
    # Record qdash demo: start qdash, test (press buttons), capture HTML, save MP4 to project root; write local path for OpenAI app creator
    parser.add_argument(
        "--record-qdash-demo",
        action="store_true",
        help="Start qdash (cd qdash && npm start), test app (press buttons), capture HTML and record video; save MP4 in project root; write demo paths to OpenAI app creator config.",
    )
    parser.add_argument(
        "--qdash-demo-out",
        type=str,
        default=None,
        metavar="PATH",
        help="Output path for demo MP4 (with --record-qdash-demo). Default: project_root/qdash_demo.mp4",
    )
    # Gemini CLI settings.json (root + .gemini/) per https://geminicli.com/docs/reference/configuration
    parser.add_argument(
        "--write-gemini-settings",
        action="store_true",
        help="Write Gemini CLI settings.json to project root and .gemini/settings.json.",
    )
    # Build MCP Docker -> run MCP server + workflow -> app publish checklist (OpenAI Apps SDK)
    parser.add_argument(
        "--publish-app",
        action="store_true",
        help="Build MCP Docker image, start MCP server, run health + JSON-RPC tests, output App Store submission checklist.",
    )
    parser.add_argument("--publish-app-no-docker", action="store_true", help="With --publish-app: skip Docker build, only run workflow (MCP server + tests).")
    args = parser.parse_args()

    project_root = _project_root
    if args.run_local_scan_only:
        from _admin.run_local import run_local_scan_only
        run_local_scan_only(project_root)
        sys.exit(0)

    if args.write_gemini_settings:
        from _admin.gemini_settings import write_settings
        paths = write_settings(project_root=_project_root)
        print("[Admin] Wrote Gemini CLI settings:", paths)
        sys.exit(0)

    if args.scan_only:
        from _admin.admin import Admin
        admin = Admin(project_root=project_root)
        for path, image_name, project_type, has_docker in admin.scan_projects():
            try:
                rel = path.relative_to(admin.project_root)
            except ValueError:
                rel = path
            print(f"  {rel} -> image={image_name} type={project_type} dockerfile={has_docker}")
        sys.exit(0)

    if args.record_qdash_demo:
        from _admin.record_qdash_demo import run_record_qdash_demo
        out_path = Path(args.qdash_demo_out) if args.qdash_demo_out else None
        ok = run_record_qdash_demo(out_mp4=out_path)
        sys.exit(0 if ok else 1)

    if args.local or args.run_local:
        from _admin.run_local import run_local_execute
        run_local_execute(
            project_root,
            project_path=args.run_local_project,
            port_spec=args.run_local_port,
            testing=args.run_local_testing,
            skip_react_native=True,
            separate_terminals=args.separate_terminals,
            verbose=args.verbose,
        )
        sys.exit(0)

    if args.publish_app:
        # Build MCP Docker (optional) -> run OpenAI Apps SDK workflow (MCP server + health + test + checklist)
        project_root = _project_root
        if not args.publish_app_no_docker:
            import subprocess
            dockerfile = project_root / "_admin" / "app_handler" / "openai_asdk" / "Dockerfile.mcp"
            if not dockerfile.exists():
                print("[Admin] Missing Dockerfile:", dockerfile)
                sys.exit(1)
            cmd = [
                "docker", "build", "-t", "bestbrain-mcp-app",
                "-f", str(dockerfile),
                str(project_root),
            ]
            print("[Admin] Building MCP Docker image: bestbrain-mcp-app")
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print("[Admin] Docker build failed (is Docker Desktop running?). Continuing with workflow (MCP server + checklist).")
            else:
                print("[Admin] Docker build OK")
        from _admin.app_handler.openai_asdk.workflow import run_workflow
        success = run_workflow(start_server=True, port=8000)
        sys.exit(0 if success else 1)

    if args.deploy:
        from _admin.admin import Admin
        local_deploy = os.environ.get("LOCAL", "").strip().lower() in ("true", "1", "yes") or os.environ.get("DEPLOY_LOCAL", "").strip().lower() in ("true", "1", "yes")
        admin = Admin(project_root=project_root, tag=args.tag, local=local_deploy)
        out = admin.run(
            build=not args.no_build,
            push=False if args.no_push else None,
            deploy=not args.no_deploy,
            namespace=args.namespace,
            build_frontend_npm=not args.no_npm_build,
            force_rebuild=args.force_rebuild,
            packages_filter=list(DEPLOY_PACKAGES),
        )
        print("[Admin] Built (images):", len(out["built"]))
        print("[Admin] Pushed:", len(out["pushed_uris"]))
        print("[Admin] Deploy results:", out["deploy_results"])
        sys.exit(0)

    print("[Admin] No action (use --local to start backends and React web frontends from inferred project dirs, or --scan-only to list projects).")


if __name__ == "__main__":
    main()

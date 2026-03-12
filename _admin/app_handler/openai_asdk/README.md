# OpenAI Apps SDK – BestBrain Integration

Publish BestBrain as a ChatGPT app via the [OpenAI Apps SDK](https://developers.openai.com/apps-sdk/).

## Overview

- **MCP server**: Exposes `/mcp` endpoint for ChatGPT connector
- **Docker**: Wraps the app for deployment
- **App Store**: Submission checklist and best practices

## Quick Start

### 1. Run MCP server locally

```bash
py -m app_handler.openai_asdk.mcp_server --port 8000
```

### 2. Expose for ChatGPT (development)

```bash
ngrok http 8000
# Use https://<subdomain>.ngrok.app/mcp as Connector URL
```

### 3. Connect from ChatGPT

1. Settings → Apps & Connectors → Advanced → Enable developer mode
2. Create connector → paste `https://<your-ngrok>.ngrok.app/mcp`
3. Test in a new chat

## Docker

```bash
# Build
docker build -t bestbrain-mcp-app -f app_handler/openai_asdk/Dockerfile.mcp .

# Run
docker run -p 8000:8000 bestbrain-mcp-app
```

## AppPublisher Class

```python
from app_handler.openai_asdk import AppPublisher
from app_handler.openai_asdk.app_publisher import AppMetadata

metadata = AppMetadata(
    name="BestBrain",
    version="0.1.0",
    description="Physics simulation environment",
    privacy_policy_url="https://yoursite.com/privacy",
    support_contact="support@yoursite.com",
)

publisher = AppPublisher(app_metadata=metadata)

# Pre-submission checklist
for issue in publisher.check_prerequisites():
    print(issue)

# Deployment options
print(publisher.get_deployment_options())

# Docker build command
print(publisher.build_docker_cmd())
```

## Submission to OpenAI App Store

1. Complete [organization verification](https://platform.openai.com/settings/organization/general)
2. Implement full MCP protocol (add `mcp` package; see `mcp_server.py`)
3. Define CSP in resource `_meta.ui.csp`
4. Add privacy policy URL
5. Submit from [platform.openai.com/apps-manage](https://platform.openai.com/apps-manage)

## References

- [Apps SDK Docs](https://developers.openai.com/apps-sdk/)
- [Deploy your app](https://developers.openai.com/apps-sdk/deploy)
- [App submission](https://developers.openai.com/apps-sdk/deploy/submission)
- [App submission guidelines](https://developers.openai.com/apps-sdk/app-submission-guidelines)
- [Security & Privacy](https://developers.openai.com/apps-sdk/guides/security-privacy)

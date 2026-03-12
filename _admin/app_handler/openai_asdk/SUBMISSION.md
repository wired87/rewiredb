# Publish BestBrain to the OpenAI App Store

All requirements and data are prepared. Follow these steps to submit the app.

## 1. Run the workflow (already done)

```bash
py -m _admin.main --publish-app --publish-app-no-docker
```

This validates metadata, tools, MCP server, and writes **app_submission_manifest.json** with every field needed for the form.

## 2. Deploy the MCP server to a public HTTPS URL

- **Option A:** Deploy the Docker image to Cloud Run / Fly.io / Render and note the base URL.
- **Option B:** For testing, run `ngrok http 8000` and use `https://<your-subdomain>.ngrok.app`.

The Connector URL you will paste in the form must be: **https://&lt;your-domain&gt;/mcp**

## 3. Submit at the OpenAI Platform

1. Open **https://platform.openai.com/apps-manage**
2. Click **Submit for review**
3. Fill the form using the data in **app_submission_manifest.json**:
   - **App name:** BestBrain
   - **Subtitle, description, logo URL:** from manifest
   - **Privacy policy URL:** https://bestbrain.tech/privacy
   - **Terms of service:** https://bestbrain.tech/terms
   - **Support contact:** support@bestbrain.tech
   - **Company URL:** https://bestbrain.tech
   - **Category:** Developer tools
   - **MCP server URL:** https://&lt;your-deployed-domain&gt;/mcp
   - **Tools:** list_simulations, create_simulation (annotations in manifest)
   - **Test prompts and expected responses:** from manifest `test_prompts_and_responses`
   - **Screenshots:** upload app screenshots (see dashboard for dimensions)
   - **Demo video:** use `demo_video_path` from manifest (e.g. my_demo.webm) if the form accepts a file
   - **Localization:** en (default)

4. Complete organization verification and confirm all checkboxes.
5. Submit. You will receive an email with a Case ID. After review, click **Publish** to make the app live in the directory.

## Prerequisites (from manifest checklist)

- Organization verification completed (Dashboard → Settings → General)
- Owner role in organization
- MCP server on public HTTPS (no localhost)
- CSP defined (already set in the MCP server responses)
- Demo account without MFA if the app requires auth

## Files

| File | Purpose |
|------|---------|
| **app_submission_manifest.json** | All form data in one place; copy from here into the submission form |
| **demo_paths.json** | Paths to demo video (my_demo.webm) and HTML captures |
| **SUBMISSION.md** | This file |

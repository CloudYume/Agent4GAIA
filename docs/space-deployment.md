# Hugging Face Space Deployment

[Project overview](../README.md) · [Course submission](submission.md#course-submission)

The [`huggingface-space/`](../huggingface-space/) directory contains a Gradio app for manually entered questions and optional attachments. It does not load a GAIA task collection. Use a separate Space repository whose root contains the built bundle, rather than pointing a deployment at this repository's root.

## Build the bundle

From the checkout root:

```powershell
python scripts/build_space.py --output dist/space-bundle
```

The builder refuses an existing output directory. It copies only the Space README, app, requirements, package metadata, credential-free configuration template, and Python sources under `gaia_agent/`. Inspect that directory before uploading. Local `config.toml`, `.runs/`, downloaded tasks, answer files, tests, and caches are outside the allowlist.

## Configure and deploy

1. Create a Gradio Space in your Hugging Face account.
2. Upload the contents of `dist/space-bundle/` to the Space root; `README.md` must retain its YAML metadata and `app_file: app.py`.
3. Add a nonempty `SPACE_ACCESS_TOKEN` as a Space Secret. It is the shared access token entered in the app; it is separate from `HF_TOKEN` and your model-provider keys.
4. Configure the provider keys needed by your selected solver and reviewer using Space Secrets. The default adaptive review route needs both `DEEPSEEK_API_KEY` and `ANTHROPIC_API_KEY`. Add `OPENAI_API_KEY` only for a configured OpenAI route or transcription provider. `TAVILY_API_KEY` is optional.
5. Confirm the chosen model IDs and endpoints work with your provider. The bundled example is a template, not a service-availability guarantee. Nonsecret template changes can be made before deployment; keep credentials in Secrets.
6. Rebuild the Space, then try a manually entered question with your access token. Model calls can incur charges. Use the Space's public code URL in the course payload only after the code is actually available.

Space requirements install the package's audio/video extras and Gradio. They may download native dependencies or local transcription models. Fit the deployment to the Space's storage and memory limits. The app returns an answer, confidence, and evidence/warnings; it does not provide course submission, a leaderboard uploader, or cross-question learning.

The source app permits one submitted solve at a time through its Gradio event configuration. Review the deployment's visibility and access-token distribution before sharing a running instance. Hosting public source code does not require publishing benchmark inputs or credentials.

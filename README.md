# RunPod Serverless Sentiment Worker

A custom Serverless worker demonstrating RunPod's handler contract: module-level model loading, input validation, structured JSON output, and the full local → Docker → deploy lifecycle.

## What it does

Sentiment classification using DistilBERT (fine-tuned on SST-2). Small enough to run on CPU for local testing, fast on any GPU in production.

## Local development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# SDK auto-detects test_input.json and runs one local job
python handler.py

# Or pass input inline
python handler.py --test_input '{"input": {"text": "Great product!", "top_k": 2}}'
```

## Deploy to RunPod Serverless

```bash
# Build for x86_64 (required — RunPod hosts are x86_64, even from an ARM Mac)
docker build --platform=linux/amd64 -t youruser/sentiment-worker:v1.0.0 .
docker push youruser/sentiment-worker:v1.0.0

# Create template + endpoint via runpodctl
runpodctl template create --name sentiment-worker --serverless \
  --image youruser/sentiment-worker:v1.0.0 --container-disk-in-gb 10

runpodctl serverless create --template-id <TEMPLATE_ID> --name sentiment \
  --gpu-ids "AMPERE_24" --workers-min 0 --workers-max 1
```

## Request / response

```json
// Request
{"input": {"text": "RunPod is great!", "top_k": 2}}

// Response
{
  "labels": [
    {"label": "POSITIVE", "score": 0.9998},
    {"label": "NEGATIVE", "score": 0.0002}
  ],
  "input_length": 16
}
```

## Design notes

- Model loads at **module level** — once per worker cold start, reused across every job
- Input capped at 512 tokens (DistilBERT's max context)
- `top_k` controls how many labels are returned (default 2)
- Scale-to-zero means $0/hr idle, ~$0.69/hr on AMPERE_24 when serving
- `--platform=linux/amd64` in the Dockerfile build is the #1 gotcha for Mac developers

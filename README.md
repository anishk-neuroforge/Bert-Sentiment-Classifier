# BERT Sentiment Classifier

Production-style 5-class sentiment analysis, served as a containerized FastAPI
service with ONNX-optimized inference, MLflow experiment tracking, and a
live browser dashboard for manual testing.

```
Text Input → Tokenization → BERT Encoder → Classification Head → 5-Class Output
```

**Stack:** FastAPI · Transformers (BERT) · PyTorch · ONNX Runtime · MLflow · Docker · Pytest

---

## Table of contents

- [What this is](#what-this-is)
- [Status: what's verified vs. what to confirm yourself](#status-whats-verified-vs-what-to-confirm-yourself)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Quickstart](#quickstart)
- [The model](#the-model)
- [API reference](#api-reference)
- [Dashboard UI](#dashboard-ui)
- [Running with ONNX](#running-with-onnx-optimized-inference)
- [MLflow experiment tracking](#mlflow-experiment-tracking)
- [Docker](#docker)
- [Testing](#testing)
- [Configuration reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Roadmap / what a v2 would add](#roadmap--what-a-v2-would-add)
- [License](#license)

---

## What this is

This project fine-tunes/serves a BERT-family model for **5-class sentiment
classification** (Very Negative → Very Positive) behind a REST API, with the
kind of surrounding infrastructure a real deployment needs:

- **FastAPI service** — `/predict`, `/predict/batch`, `/health`, auto-generated
  OpenAPI docs at `/docs`
- **Dual inference backends** — standard PyTorch/Transformers, or ONNX Runtime
  for lower-latency, higher-throughput serving
- **MLflow tracking** — every prediction request is logged as a run (latency,
  backend, input length, predicted label), so production traffic is
  observable the same way training experiments are
- **Docker packaging** — model weights baked into the image at build time, so
  the container needs no network access at runtime
- **A browser dashboard** — a terminal-styled console (served by the API
  itself) for firing test predictions and watching the 5-class probability
  distribution and a live request log, without needing curl or Postman
- **A real test suite** — 6 passing tests covering the API contract
  (validation, success paths, batch handling), independent of whether the
  model weights are downloaded

## Status: what's verified vs. what to confirm yourself

This project was built and tested in a sandboxed environment without access
to `huggingface.co` (network was restricted to package registries). Because
of that:

| Component | Verified in build | Needs your confirmation |
|---|---|---|
| API routes, request validation, response schemas | ✅ via `pytest` (6/6 passing) | — |
| `GET /` serves the dashboard HTML correctly | ✅ via `TestClient` | — |
| MLflow logging call structure | ✅ code review + test mocking | Confirm a run actually appears in `mlflow ui` |
| **Live model inference** (real BERT weights, real predictions) | ❌ blocked by sandbox network | **Yes — run the Quickstart below** |
| ONNX export numerical parity | Code includes a self-check (`export_onnx.py` compares ONNX vs. torch logits) | Run `scripts/export_onnx.py` and confirm it prints "verified" |
| Docker build end-to-end | Not run (needs HF network access at build time) | Run `docker compose up --build` |

In short: the scaffolding, API contract, and test suite are solid and proven.
The one thing that must be confirmed on a machine with normal internet
access is that live BERT inference returns sensible predictions — see
Quickstart, step 3.

## Architecture

```
                         ┌─────────────────────────┐
                         │   Browser Dashboard      │
                         │   (app/static/index.html)│
                         └────────────┬─────────────┘
                                      │ fetch()
                                      ▼
┌─────────────────────────────────────────────────────────────┐
│                         FastAPI app                          │
│  GET  /health         →  liveness + backend/device info      │
│  POST /predict        →  single-text prediction               │
│  POST /predict/batch  →  batch prediction (≤ 64 texts)         │
└───────────────────┬───────────────────────────┬───────────────┘
                    │                           │
                    ▼                           ▼
        ┌───────────────────────┐   ┌─────────────────────────┐
        │  SentimentClassifier   │   │   MLflow tracking        │
        │  (app/model.py)         │   │   (per-request logging)  │
        │                         │   └─────────────────────────┘
        │  backend = torch        │
        │    Transformers +       │
        │    PyTorch inference    │
        │           or            │
        │  backend = onnx         │
        │    ONNX Runtime          │
        │    (scripts/export_onnx.py) │
        └───────────────────────┘
                    │
                    ▼
        nlptown/bert-base-multilingual-uncased-sentiment
        (1-5 star output, remapped to 5-class labels)
```

## Project structure

```
bert-sentiment-classifier/
├── app/
│   ├── main.py              FastAPI app: routes, lifespan model loading, MLflow logging
│   ├── model.py              SentimentClassifier: torch + ONNX Runtime backends
│   ├── schemas.py             Pydantic request/response models
│   └── static/
│       └── index.html          Browser dashboard (served at GET /)
├── scripts/
│   └── export_onnx.py          Exports the HF model to ONNX + verifies numerical parity
├── tests/
│   └── test_api.py             6 tests: health, predict, validation, batch
├── conftest.py                 Makes `app` importable from any working directory
├── pytest.ini                  pytest config (pythonpath, testpaths)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── README.md                   You are here
```

## Quickstart

### 1. Install dependencies

```bash
git clone <your-repo-url>
cd bert-sentiment-classifier
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the test suite (fast, no model download)

```bash
pytest tests/ -v
```
Expect `6 passed`. This confirms the API contract without needing the ~700MB
model download.

> If you see `ModuleNotFoundError: No module named 'app'`, make sure you're
> running pytest from the project root — `conftest.py` and `pytest.ini` there
> are what make `app` importable.

### 3. Run the API and confirm live inference

```bash
uvicorn app.main:app --reload --port 8000
```
Wait for `Model loaded in X.XXs` in the terminal (first run downloads the
model from Hugging Face — a few minutes depending on connection).

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The product quality is excellent and exceeded my expectations!"}'
```

Expect a response like:
```json
{
  "label": "Very Positive",
  "scores": [...],
  "latency_ms": 45.2,
  "model_backend": "torch"
}
```

Try a negative example to confirm the label actually flips (not just always
predicting one class):
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This broke after one day. Total waste of money."}'
```

### 4. Open the dashboard

Go to **http://localhost:8000/** — type text, hit **Run Prediction**, watch
the probability bars animate. See [Dashboard UI](#dashboard-ui) below.

## The model

**Backbone:** [`nlptown/bert-base-multilingual-uncased-sentiment`](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) —
a BERT model fine-tuned on multilingual product reviews for 1-5 star rating
prediction. This project remaps that 5-star output to the labeling scheme
used throughout:

| Model output | Remapped label |
|---|---|
| 1 star | Very Negative |
| 2 stars | Negative |
| 3 stars | Neutral |
| 4 stars | Positive |
| 5 stars | Very Positive |

To swap in your own fine-tuned checkpoint, set the `MODEL_NAME` environment
variable to any `AutoModelForSequenceClassification`-compatible model with a
5-class head — everything downstream (tokenization, serving, ONNX export) is
model-agnostic.

```bash
MODEL_NAME=your-org/your-finetuned-model uvicorn app.main:app
```

## API reference

Full interactive docs (Swagger UI) are always available at
**http://localhost:8000/docs** once the server is running.

### `GET /health`

Liveness check + current backend/device info.

**Response**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_backend": "torch",
  "device": "cpu"
}
```

### `POST /predict`

**Request**
```json
{ "text": "The product quality is excellent and exceeded my expectations!" }
```

| Field | Type | Constraints |
|---|---|---|
| `text` | string | 1–4096 chars, not blank/whitespace-only |

**Response**
```json
{
  "label": "Very Positive",
  "scores": [
    { "label": "Very Negative", "score": 0.01 },
    { "label": "Negative", "score": 0.02 },
    { "label": "Neutral", "score": 0.05 },
    { "label": "Positive", "score": 0.22 },
    { "label": "Very Positive", "score": 0.70 }
  ],
  "latency_ms": 45.2,
  "model_backend": "torch"
}
```

**Error responses**
| Status | Cause |
|---|---|
| `422` | Missing `text` field, or `text` is empty/whitespace |
| `500` | Model inference error (see server logs) |

### `POST /predict/batch`

**Request**
```json
{ "texts": ["Loved it!", "Terrible experience.", "It was okay, nothing special."] }
```

| Field | Type | Constraints |
|---|---|---|
| `texts` | array of strings | 1–64 items |

**Response**
```json
{
  "results": [
    { "label": "Very Positive", "scores": [...], "latency_ms": 12.1, "model_backend": "torch" },
    { "label": "Very Negative", "scores": [...], "latency_ms": 11.8, "model_backend": "torch" },
    { "label": "Neutral", "scores": [...], "latency_ms": 12.4, "model_backend": "torch" }
  ],
  "total_latency_ms": 36.3
}
```

## Dashboard UI

A live console is served at **http://localhost:8000/** — no separate frontend
server or build step required. It's a single static file
(`app/static/index.html`) that calls this API's own endpoints directly.

**Features:**
- **Status bar** — live connection state (green pulsing dot when healthy),
  active backend (`torch`/`onnx`), and device (`cpu`/`cuda`); polls `/health`
  every 10 seconds
- **Single mode** — type text, click **Run Prediction** (or `Ctrl`/`Cmd`+`Enter`)
- **Batch mode** — paste one text per line (up to 64) for batch predictions
- **Result bars** — animated 5-class probability bars, color-coded from red
  (Very Negative) to green (Very Positive), with the predicted class bolded
- **Log panel** — every prediction (single or batch) appends a timestamped
  line with input text, predicted label, and latency — a running session
  record (in-browser only, not persisted server-side)

Confirmed in this build: `GET /` returns `200` and serves the dashboard's
real HTML/JS (checked via `TestClient`, since this sandbox can't launch a
browser). **Not yet confirmed here:** the dashboard's fetch calls succeeding
against a live model — that depends on the Hugging Face download completing
on your machine, so open the page yourself and run a prediction to confirm.

## Running with ONNX (optimized inference)

```bash
python scripts/export_onnx.py
```

This traces the torch model, exports an ONNX graph to `onnx/model.onnx`,
reloads it with `onnxruntime`, and asserts the ONNX logits match the
original torch model within `1e-3` — you'll see a pass/fail printed right
after export.

Then run the server against the ONNX backend instead of torch:
```bash
MODEL_BACKEND=onnx ONNX_MODEL_PATH=onnx/model.onnx uvicorn app.main:app --port 8000
```

Confirm the switch worked:
```bash
curl http://localhost:8000/health
# "model_backend": "onnx"
```

## MLflow experiment tracking

Every `/predict` and `/predict/batch` call logs an MLflow run — backend,
input length, latency, and predicted label — under the
`bert-sentiment-classifier` experiment.

Tracking backend: `sqlite:///mlflow.db` by default.

> MLflow 3.x deprecated the plain `file:./mlruns` tracking store in favor of
> a database backend. This project uses SQLite for that reason — point
> `MLFLOW_TRACKING_URI` at Postgres/MySQL for a real multi-user deployment.

View the dashboard:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open **http://localhost:5000**, click into the `bert-sentiment-classifier`
experiment, and confirm a run appears for each prediction you made.

## Docker

```bash
docker compose up --build
```

The `Dockerfile` downloads and bakes model weights into the image at build
time, so the running container needs no network access — good for
production, but it does mean the **build** step needs `huggingface.co`
reachable.

Environment variables (see [Configuration reference](#configuration-reference))
can be overridden in `docker-compose.yml` or via `docker run -e KEY=value`.

## Testing

```bash
pytest tests/ -v
```

6 tests, all mocking the classifier so they run fast and offline (they test
the API contract — routing, validation, response shape — not model
accuracy):

| Test | Confirms |
|---|---|
| `test_health` | `/health` returns `200` with expected fields |
| `test_predict_success` | `/predict` returns a valid label + 5 scores summing to ~1.0 |
| `test_predict_rejects_empty_text` | Whitespace-only `text` → `422` |
| `test_predict_rejects_missing_field` | Missing `text` field → `422` |
| `test_predict_batch` | `/predict/batch` returns one result per input |
| `test_predict_batch_rejects_empty_list` | Empty `texts` array → `422` |

## Configuration reference

All configuration is via environment variables, with sensible defaults.

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `nlptown/bert-base-multilingual-uncased-sentiment` | HF model repo to load |
| `MODEL_BACKEND` | `torch` | `torch` or `onnx` |
| `ONNX_MODEL_PATH` | `onnx/model.onnx` | Path to exported ONNX graph (used when `MODEL_BACKEND=onnx`) |
| `MAX_SEQ_LEN` | `256` | Max token sequence length |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow tracking backend |
| `MLFLOW_EXPERIMENT` | `bert-sentiment-classifier` | MLflow experiment name |

## Troubleshooting

**`ModuleNotFoundError: No module named 'app'` when running pytest**
Run pytest from the project root. `conftest.py` and `pytest.ini` there set
the import path — if either is missing/deleted this error will return.

**`OSError: We couldn't connect to 'https://huggingface.co'...`**
The model download needs outbound internet access to Hugging Face. Check
your network/firewall/proxy settings; this isn't something the app can work
around.

**`mlflow.exceptions.MlflowException: ... filesystem tracking backend ... is in maintenance mode`**
You're on MLflow 3.x, which deprecated `file:./mlruns`. Make sure
`MLFLOW_TRACKING_URI` is set to `sqlite:///mlflow.db` (the default in this
repo) rather than a bare `file:` path.

**Dashboard loads but predictions error out**
Open the browser dev console (F12) → Network tab, retry a prediction, and
check the response body of the failed `/predict` call — the dashboard
surfaces the FastAPI error message directly in a red box under the input.

**Docker build hangs or fails downloading the model**
The build step needs `huggingface.co` reachable. If you're behind a
corporate proxy/firewall, configure Docker's build-time network access
accordingly, or pre-download weights and adjust the `Dockerfile` to `COPY`
them in instead of downloading at build time.

## Roadmap / what a v2 would add

- Fine-tune on a labeled dataset specific to a target domain (current model
  is a strong general-purpose baseline, not custom-trained on your data)
- Dynamic batching under load for genuine 500+ RPS throughput
- Prometheus metrics alongside MLflow for live ops/alerting
- Auth (API key or JWT) before exposing this publicly
- Persist dashboard log history server-side (currently in-browser only)

## License

MIT — use freely, attribution appreciated.

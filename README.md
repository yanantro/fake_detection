# Fake Reviews Detection

Web application for detecting potentially fake e-commerce reviews. The system is based on a FastAPI backend and a browser interface for checking one review, a list of reviews, or a CSV file.

## Features

- single review prediction;
- batch review ranking;
- CSV upload;
- fake probability and trust score calculation;
- three sensitivity modes;
- rule-based checks for obvious LLM-like patterns;
- browser interface served by FastAPI.

## Project Structure

```text
fake_review_detector_github/
├── main.py
├── model_service.py
├── requirements.txt
├── render.yaml
├── static/
│   └── index.html
└── models/
    ├── ensemble_params_ecommerce.json
    └── rubert_ecommerce/
```

The `models/rubert_ecommerce/` directory is required for the RuBERT backend. It contains the trained model files produced in the experimental notebook.

## Local Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

Then open:

```text
http://127.0.0.1:8000/
```

## Model Files

For local use, the application expects the model in:

```text
models/rubert_ecommerce/
```

and thresholds in:

```text
models/ensemble_params_ecommerce.json
```

If the model is stored outside the repository, set `PROJECT_DIR` to the directory that contains the `models` folder.

```bash
export PROJECT_DIR="/path/to/project"
export MODEL_BACKEND=rubert
python -m uvicorn main:app --reload --port 8000
```

## Deployment

The repository includes `render.yaml` for deployment on Render. The service command is:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

For public deployment, the trained model files must also be available in the deployed environment. Large model files should be uploaded using Git LFS or stored externally and copied into `models/rubert_ecommerce/` during deployment.

## API

Main endpoints:

| Endpoint | Method | Purpose |
|---|---:|---|
| `/` | GET | Web interface |
| `/health` | GET | Service health check |
| `/predict` | POST | Analyze one review |
| `/rank_reviews` | POST | Analyze and rank several reviews |
| `/rank_csv` | POST | Analyze reviews from a CSV file |

Example request:

```json
{
  "text": "Review text",
  "mode": "strict"
}
```

Example response:

```json
{
  "label": "fake",
  "prob_fake": 0.873,
  "prob_real": 0.127,
  "trust_score": 0.127,
  "status": "suspicious",
  "model": "rubert",
  "mode": "strict",
  "threshold": 0.505
}
```

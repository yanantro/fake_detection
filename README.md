---
title: Fake Reviews Detection
emoji: 🔎
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Fake Reviews Detection

Software project for detecting potentially fake e-commerce reviews using NLP methods.
It contains the experimental notebooks, datasets, and a FastAPI web application.

Demo: https://yanantro-fake-reviews-detection.hf.space

## Repository Layout

```text
notebooks/  training, dataset building, domain adaptation
data/       e-commerce_dataset.csv, ood_wb_50.csv
app/        FastAPI application and trained RuBERT backend
app/main.py
app/model_service.py
app/static/index.html
app/models/ensemble_params_ecommerce.json
app/models/rubert_ecommerce/
```

## Notebooks

- `notebooks/1_training_hotel_reviews.ipynb` trains models on MAiDE-up hotel reviews.
- `notebooks/2_dataset_builder.ipynb` builds the Wildberries e-commerce dataset.
- `notebooks/3_domain_adaptation.ipynb` evaluates domain shift, WB training, and OOD tests.

The notebooks use paths relative to the repository root.
The main dataset is `data/e-commerce_dataset.csv`.
The OOD set is `data/ood_wb_50.csv`.

Notebook 2 requires an Anthropic API key only if new synthetic reviews are generated:

```bash
export ANTHROPIC_API_KEY="your_key"
```

## Running the App Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r app/requirements.txt
cd app
python -m uvicorn main:app --reload --port 7860
```

Open `http://127.0.0.1:7860`.

The production backend is RuBERT after domain adaptation.
The model files are stored in `app/models/rubert_ecommerce/`.
Large model files are tracked with Git LFS.

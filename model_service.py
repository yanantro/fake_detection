from __future__ import annotations

import csv
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_PROJECT_DIR = Path(__file__).resolve().parent

THRESHOLD_MODES = {
    "strict": {},
    "moderation": {},
    "soft": {},
}

HARD_PATTERNS = [
    (
        "generation_intro",
        re.compile(
            r"\b(конечно|без проблем|держите|вот|ниже)\b.{0,80}"
            r"\b(\d+\s*)?(отзыв|отзыва|отзывов|вариант|варианта|вариантов)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "request_echo",
        re.compile(r"\b(по вашему запросу|как вы просили|для вашего товара|для карточки товара)\b", re.IGNORECASE),
    ),
    (
        "disclaimer",
        re.compile(r"\b(как ии|как языковая модель|я не могу|могу предложить|сгенерирую)\b", re.IGNORECASE),
    ),
]

SOFT_PATTERNS = [
    (
        "numbered_list",
        re.compile(r"(?m)^\s*\d+[).]\s+.{10,}"),
        0.25,
    ),
    (
        "long_dash",
        re.compile(r"—"),
        0.08,
    ),
    (
        "template_phrase",
        re.compile(
            r"\b(однозначно рекомендую|всем рекомендую|качество на высоте|не пожалеете|буду брать ещё|просто супер)\b",
            re.IGNORECASE,
        ),
        0.12,
    ),
]


@dataclass
class Prediction:
    text: str
    label: str
    prob_fake: float
    prob_real: float
    trust_score: float
    status: str
    model: str
    mode: str
    threshold: float
    model_prob_fake: float | None
    rule_flags: list[str]

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "label": self.label,
            "prob_fake": round(self.prob_fake, 4),
            "prob_real": round(self.prob_real, 4),
            "trust_score": round(self.trust_score, 4),
            "status": self.status,
            "model": self.model,
            "mode": self.mode,
            "threshold": round(self.threshold, 4),
            "model_prob_fake": None if self.model_prob_fake is None else round(self.model_prob_fake, 4),
            "rule_flags": self.rule_flags,
        }


class ReviewModelService:
    def __init__(self) -> None:
        self.project_dir = Path(os.getenv("PROJECT_DIR", str(DEFAULT_PROJECT_DIR)))
        self.model_backend = os.getenv("MODEL_BACKEND", "rubert").lower().strip()
        self.models_dir = self.project_dir / "models"
        self.params = self._load_params()
        self._model = None
        self._tokenizer = None

    def _load_params(self) -> dict:
        params_path = self.models_dir / "ensemble_params_ecommerce.json"
        if not params_path.exists():
            return {}
        with params_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def threshold_for_mode(self, mode: str = "strict") -> float:
        mode = self._normalize_mode(mode)
        base = self._base_threshold()
        if mode == "moderation":
            return min(base, 0.35)
        if mode == "soft":
            return min(base, 0.25)
        return base

    def _base_threshold(self) -> float:
        thresholds = self.params.get("thresholds", {})
        if self.model_backend == "tfidf":
            return float(thresholds.get("tfidf_logreg", 0.5))
        return float(
            thresholds.get(
                "rubert_domain_adapted_wb",
                thresholds.get("rubert_stage2", 0.5),
            )
        )

    def _normalize_mode(self, mode: str) -> str:
        mode = str(mode or "strict").lower().strip()
        if mode not in THRESHOLD_MODES:
            return "strict"
        return mode

    def _load(self) -> None:
        if self._model is not None:
            return

        if self.model_backend == "tfidf":
            self._load_tfidf()
            return

        self._load_rubert()

    def _load_tfidf(self) -> None:
        import joblib

        model_path = self.models_dir / "tfidf_logreg_wb.joblib"
        if not model_path.exists():
            raise FileNotFoundError("models/tfidf_logreg_wb.joblib not found")
        self._model = joblib.load(model_path)
        self._tokenizer = None

    def _load_rubert(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_path = self.models_dir / "rubert_ecommerce"
        if not model_path.exists():
            raise FileNotFoundError(f"RuBERT model not found: {model_path}")

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self._model.eval()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

    def predict_one(self, text: str, mode: str = "strict") -> Prediction:
        text = " ".join(str(text or "").split())
        if not text:
            raise ValueError("Text is empty")

        mode = self._normalize_mode(mode)
        rule_flags, hard_rule, rule_boost = self._detect_rule_flags(text)

        if hard_rule:
            model_prob_fake = None
            prob_fake = 0.98
        else:
            self._load()
            model_prob_fake = self._predict_fake_probability(text)
            prob_fake = min(0.99, model_prob_fake + rule_boost)

        threshold = self.threshold_for_mode(mode)
        label = "fake" if prob_fake >= threshold else "real"
        trust_score = 1.0 - prob_fake
        status = self._status(prob_fake, threshold)

        return Prediction(
            text=text,
            label=label,
            prob_fake=prob_fake,
            prob_real=1.0 - prob_fake,
            trust_score=trust_score,
            status=status,
            model=self.model_backend,
            mode=mode,
            threshold=threshold,
            model_prob_fake=model_prob_fake,
            rule_flags=rule_flags,
        )

    def _detect_rule_flags(self, text: str) -> tuple[list[str], bool, float]:
        flags: list[str] = []
        hard_rule = False
        boost = 0.0

        for name, pattern in HARD_PATTERNS:
            if pattern.search(text):
                flags.append(name)
                hard_rule = True

        for name, pattern, value in SOFT_PATTERNS:
            matches = pattern.findall(text)
            if not matches:
                continue
            if name == "long_dash" and len(matches) < 3:
                continue
            flags.append(name)
            boost += value

        return flags, hard_rule, min(boost, 0.35)

    def _status(self, prob_fake: float, threshold: float) -> str:
        if prob_fake >= threshold:
            return "suspicious"
        if prob_fake >= max(0.20, threshold - 0.15):
            return "uncertain"
        return "likely_real"

    def _predict_fake_probability(self, text: str) -> float:
        if self.model_backend == "tfidf":
            probs = self._model.predict_proba([text])[0]
            return float(probs[1])

        return self._predict_rubert_probability(text)

    def _predict_rubert_probability(self, text: str) -> float:
        import torch

        encoded = self._tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self._model(**encoded).logits
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        return float(probs[1])

    def rank_reviews(self, texts: Iterable[str], mode: str = "strict") -> list[dict]:
        predictions = [self.predict_one(text, mode=mode).to_dict() for text in texts if str(text).strip()]
        predictions.sort(key=lambda row: row["trust_score"], reverse=True)
        for rank, row in enumerate(predictions, start=1):
            row["rank"] = rank
        return predictions


def parse_csv_reviews(content: bytes, text_column: str | None = None) -> list[str]:
    decoded = content.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(decoded))
    if not reader.fieldnames:
        return []

    column = text_column if text_column in reader.fieldnames else None
    if column is None:
        column = "text" if "text" in reader.fieldnames else reader.fieldnames[0]

    return [row.get(column, "") for row in reader]

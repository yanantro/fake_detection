from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import BaseModel, Field

from model_service import ReviewModelService, THRESHOLD_MODES, parse_csv_reviews


BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="Fake Review Detector",
    version="0.2.0",
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

service = ReviewModelService()


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    mode: str = "strict"


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)
    mode: str = "strict"


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    with (BASE_DIR / "static" / "index.html").open("r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_backend": service.model_backend,
        "project_dir": str(service.project_dir),
        "threshold_modes": THRESHOLD_MODES,
    }


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    try:
        return service.predict_one(request.text, mode=request.mode).to_dict()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/rank_reviews")
def rank_reviews(request: BatchPredictRequest) -> dict:
    try:
        return {"items": service.rank_reviews(request.texts, mode=request.mode)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/rank_csv")
async def rank_csv(file: UploadFile = File(...), mode: str = Form("strict")) -> dict:
    try:
        content = await file.read()
        texts = parse_csv_reviews(content)
        return {"filename": file.filename, "items": service.rank_reviews(texts, mode=mode)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

FROM python:3.11-slim

WORKDIR /code

COPY app/requirements.txt /code/requirements.txt

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    grep -v '^torch$' /code/requirements.txt > /code/requirements-no-torch.txt && \
    python -m pip install --no-cache-dir -r /code/requirements-no-torch.txt

COPY app /code/app

WORKDIR /code/app

ENV MODEL_BACKEND=rubert
ENV PROJECT_DIR=/code/app

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]


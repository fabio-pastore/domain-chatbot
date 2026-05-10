FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8002

CMD ["uvicorn", "src.frontend:app", "--host", "0.0.0.0", "--port", "8002"]
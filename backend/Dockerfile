FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libmariadb-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-mark manual libmariadb3 2>/dev/null; \
    apt-get purge -y gcc && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8001

CMD ["uvicorn", "src.api_server.server:app", "--host", "0.0.0.0", "--port", "8001"]
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api_server.py config.py event_card_generator.py ./

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request, sys; \
    sys.exit(0 if urllib.request.urlopen('http://localhost:5000/health', timeout=3).status == 200 else 1)"

CMD ["gunicorn", "--bind", "0.0.0.0:5000", \
     "-k", "gthread", "-w", "1", "--threads", "8", \
     "--timeout", "120", "--graceful-timeout", "30", \
     "--access-logfile", "-", "--error-logfile", "-", \
     "api_server:app"]

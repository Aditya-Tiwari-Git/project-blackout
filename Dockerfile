# ---------------------------------------------------------------------------
# Project Blackout — Microgrid Power Dispatcher
# Dockerfile  |  Base: python:3.10-slim  |  Port: 7860
# ---------------------------------------------------------------------------

FROM python:3.10-slim

# ---- System deps ----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# ---- Working directory -----------------------------------------------------
WORKDIR /app

# ---- Python dependencies ---------------------------------------------------
# Copy requirements first to leverage layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- Application source ----------------------------------------------------
COPY openenv.yaml .
COPY models.py    .
COPY env.py       .
COPY grader.py    .
COPY inference.py .
COPY main.py      .
COPY tasks/       tasks/

# ---- Runtime environment ---------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ---- Expose FastAPI port (required by Hugging Face Spaces) -----------------
EXPOSE 7860

# ---- Default command -------------------------------------------------------
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
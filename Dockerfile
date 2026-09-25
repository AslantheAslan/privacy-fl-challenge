# Reproducible evaluator image: python run_submission.py --train ... --input ... --output ... --artifacts-dir ...
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN useradd --create-home --uid 1000 runner \
    && mkdir -p /app/outputs \
    && chown -R runner /app
USER runner

# Fail the build early if the image cannot run the tests offline.
RUN python -m pytest -q -p no:cacheprovider tests/test_extraction.py tests/test_secure_aggregation.py

ENTRYPOINT ["python", "run_submission.py"]
CMD ["--train", "data/train.jsonl", \
     "--input", "data/validation_inputs.jsonl", \
     "--output", "outputs/validation_predictions.jsonl", \
     "--artifacts-dir", "outputs/artifacts"]

# Reproducible environment for the ML pipeline and tests.
#
# NOT included here: scripts/render_pptx.py (PowerPoint COM automation --
# requires Windows + a real PowerPoint install, fundamentally incompatible
# with a Linux container) and the real X5 RetailHero dataset (data/raw/,
# excluded via .gitignore/.dockerignore -- see reports/data_download_report.md
# for how to obtain it and mount it into the container).
#
# Build:  docker build -t x5-uplift .
# Run tests:      docker run --rm x5-uplift
# Run the pipeline (mount your local data/ directory with the real dataset):
#   docker run --rm -v "$(pwd)/data:/app/data" x5-uplift python src/run_full_pipeline.py 15

FROM python:3.13-slim

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY src/ src/
COPY tests/ tests/
COPY docs/ docs/

# Default: run the unit test suite (fast, no dataset required) so `docker run`
# out of the box proves the environment actually works.
CMD ["python", "-m", "pytest", "tests/", "-v"]

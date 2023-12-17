FROM python:3.11-slim

RUN mkdir /setup
WORKDIR /setup
COPY pyproject.toml poetry.lock ./
RUN pip install --no-cache-dir poetry=='1.7.1' && poetry install --no-root --no-directory

COPY src/ ./

CMD ["poetry", "run", "python", "run.py"]
FROM python:3.11-slim

COPY pyproject.toml /root/src/pyproject.toml
COPY poetry.lock /root/src/poetry.lock

RUN chown -R root:root /root/

WORKDIR /root/src
RUN pip install --no-cache-dir poetry=='1.7.1' && poetry install --no-root --no-directory

COPY src/ ./
RUN chown -R root:root ./

ENV SECRET_KEY hello
ENV FLASK_APP run.py

RUN chmod +x run.py
CMD ["poetry", "run", "python", "run.py"]
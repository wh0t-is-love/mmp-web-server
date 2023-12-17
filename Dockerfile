FROM python:3.11-slim

RUN mkdir /setup
WORKDIR /setup
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry install

COPY src/ ./
RUN chown -R root:root ./

ENV SECRET_KEY hello
ENV FLASK_APP ml_server

RUN chmod +x run.py
CMD ["poetry", "run", "python", "run.py"]
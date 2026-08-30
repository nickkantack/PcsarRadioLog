FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libportaudio2 \
        portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the Python project and its dependencies.
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy the application source.
COPY *.py .
COPY *.html .
COPY *.js .
COPY *.css .

# The actual command is supplied by compose.yaml.

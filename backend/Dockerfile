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

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH}

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy the application source.
COPY *.py .
COPY *.html .
COPY *.js .
COPY *.css .

# The actual command is supplied by compose.yaml.

FROM python:3.11-slim AS build

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu/ \
    VENV_PATH=/opt/venv

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc protobuf-compiler curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY pyproject.toml uv.lock ./

COPY .env .env

RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install uv

# Install torch first (largest dep) to isolate failures
RUN uv pip install --system "torch @ https://download.pytorch.org/whl/cpu/torch-2.10.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl"

# Then sync the rest of the dependencies
RUN uv sync

RUN if [ -d ".venv" ]; then mkdir -p "$(dirname ${VENV_PATH})" && mv -T .venv "${VENV_PATH}"; fi


COPY service/ /src/service/
COPY proto/ /src/proto/

RUN if [ -d "./proto" ] && [ "$(ls -A proto 2>/dev/null)" != "" ]; then \
      "${VENV_PATH}/bin/python" -m grpc_tools.protoc -I ./proto --python_out=./service --grpc_python_out=./service proto/*.proto || true ; \
    fi

FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    WORKON_HOME=/opt/venv

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=build /opt/venv /opt/venv

COPY --from=build /src/service /app/service
COPY --from=build /src/.env /app/.env

EXPOSE 50051
CMD ["/opt/venv/bin/python", "-u", "service/inference_server.py"]
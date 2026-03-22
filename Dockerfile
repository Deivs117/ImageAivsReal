# Multi-stage build: instala una sola venv con todas las deps y construye ambos componentes
FROM python:3.11-slim AS build

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu/ \
    VENV_PATH=/opt/venv

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc protobuf-compiler curl git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Dependencias declaradas
COPY pyproject.toml uv.lock ./
# Opcional: si tu repo necesita .env para build-time, cópiala también (como en service Dockerfile)
COPY .env .env

RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install uv

# Instala torch primero (coincide con tu service Dockerfile)
RUN uv pip install --system "torch @ https://download.pytorch.org/whl/cpu/torch-2.10.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl"

# Sincroniza resto de dependencias
RUN uv sync

# Mueve .venv a /opt/venv (misma técnica que usas)
RUN if [ -d ".venv" ]; then mkdir -p "$(dirname ${VENV_PATH})" && mv -T .venv "${VENV_PATH}"; fi

# Copia código de ambos componentes y los protos
COPY service/ /src/service/
COPY app/ /src/app/
COPY proto/ /src/proto/

# Genera stubs para python (genera en proto/generated y también en service para compatibilidad)
RUN mkdir -p /src/proto/generated && \
    "${VENV_PATH}/bin/python" -m grpc_tools.protoc -I /src/proto \
      --python_out=/src/proto/generated --grpc_python_out=/src/proto/generated /src/proto/*.proto || true && \
    "${VENV_PATH}/bin/python" -m grpc_tools.protoc -I /src/proto \
      --python_out=/src/service --grpc_python_out=/src/service /src/proto/*.proto || true

# Runtime image
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Necesitamos bash (para el entrypoint), libsndfile1 (service), y ca-certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libsndfile1 bash \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia venv y artefactos creados en build
COPY --from=build /opt/venv /opt/venv
COPY --from=build /src/service /app/service
COPY --from=build /src/app /app/app
COPY --from=build /src/proto /app/proto
COPY --from=build /src/.env /app/.env

# Copia entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 50051 8501

ENTRYPOINT ["./entrypoint.sh"]
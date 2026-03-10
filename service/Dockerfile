FROM python:3.11-slim AS build

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Forzar índice CPU de PyTorch para evitar dependencias CUDA (opcional, uv.lock ya debe estar pinneado)
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu/ \
    VENV_PATH=/opt/venv

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc protobuf-compiler curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Copiar pyproject/lock (se resuelven las dependencias con uv)
COPY pyproject.toml uv.lock ./

# Si tienes un .env que quieras empaquetar en la imagen build-stage (opcional)
COPY .env .env

RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install uv \
    && uv sync

# Si uv crea .venv en la raíz del proyecto, lo movemos al path final (/opt/venv).
# Nos aseguramos de crear el directorio padre y usamos mv -T para evitar ambigüedades.
RUN if [ -d ".venv" ]; then mkdir -p "$(dirname ${VENV_PATH})" && mv -T .venv "${VENV_PATH}"; fi

# Copiar código fuente y .proto
COPY service/ /src/service/
COPY proto/ /src/proto/

# Si necesitas generar stubs con protoc:
RUN if [ -d "./proto" ] && [ "$(ls -A proto 2>/dev/null)" != "" ]; then \
      "${VENV_PATH}/bin/python" -m grpc_tools.protoc -I ./proto --python_out=./service --grpc_python_out=./service proto/*.proto || true ; \
    fi

# --- Stage runtime ---
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    WORKON_HOME=/opt/venv

# Instalar paquetes mínimos de sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar la virtualenv preparada (si existe)
COPY --from=build /opt/venv /opt/venv

# Copiar código de la app (incluye stubs generados)
COPY --from=build /src/service /app/service
# Copiamos .env si lo quieres empaquetar; si lo cargas por docker-compose env_file, puedes omitirlo
COPY --from=build /src/.env /app/.env

# Exponer puerto gRPC por defecto (ajusta si usas otro)
EXPOSE 50051

# Usar el Python del virtualenv por defecto; -u para salida sin buffering
CMD ["/opt/venv/bin/python", "-u", "service/inference_server.py"]
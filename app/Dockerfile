FROM python:3.11-slim AS build

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Refuerzo: índice CPU de PyTorch (si el lock ya está bien, igual no estorba)
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu/ \
    VENV_PATH=/opt/venv

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# 1) Copiar manifests (root)
COPY pyproject.toml uv.lock ./

# 2) Instalar uv y sincronizar deps (sin dev)
RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install uv && uv sync --no-dev

# 3) Mover .venv a /opt/venv si uv la creó localmente
RUN if [ -d ".venv" ]; then mkdir -p "$(dirname ${VENV_PATH})" && mv -T .venv "${VENV_PATH}"; fi

# 4) Copiar código necesario para el runtime
COPY app/ /src/app/
COPY proto/ /src/proto/

# 5) Generar stubs EN proto/generated (lo que tu clientGrpc.py espera)
RUN mkdir -p /src/proto/generated && \
    "${VENV_PATH}/bin/python" -m grpc_tools.protoc \
      -I /src/proto \
      --python_out=/src/proto/generated \
      --grpc_python_out=/src/proto/generated \
      /src/proto/inference.proto

# --- runtime ---
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app

# Dependencias runtime mínimas (Pillow/matplotlib a veces requieren libs extra;
# si te falla con errores de librerías, ajustamos aquí)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Copiar venv
COPY --from=build /opt/venv /opt/venv

# Copiar app + proto (incluye proto/generated)
COPY --from=build /src/app /app/app
COPY --from=build /src/proto /app/proto

EXPOSE 8501

# Usar módulo para no depender del script 'streamlit' en PATH
CMD ["/opt/venv/bin/python", "-u", "-m", "streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
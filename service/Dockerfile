# Dockerfile multi-stage para el servidor de inferencia (service/)
# - Build stage: instala dependencias, compila stubs desde proto/
# - Runtime stage: imagen más ligera que contiene solo el código y paquetes instalados

### STAGE: build
FROM python:3.12-slim AS build

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependencias del sistema necesarias para compilación y ejecución
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    protobuf-compiler \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Copiar pyproject y .env (opcionalmente)
COPY pyproject.toml ./
COPY .env ./

# Instalar pip y herramientas necesarias
RUN python -m pip install --upgrade pip setuptools

# Instalar dependencias principales en el build stage.
# NOTA: torch puede producir instalación pesada; si prefieres usar la rueda oficial
# de PyTorch o una imagen base distinta, cámbialo aquí.
RUN python -m pip install \
    "streamlit>=1.45.0" \
    "transformers>=4.52.0" \
    "torch>=2.7.0" \
    "accelerate>=1.7.0" \
    "Pillow>=11.2.1" \
    "pandas>=2.2.3" \
    "reportlab>=4.2.0" \
    "grpcio>=1.78.0" \
    "grpcio-tools>=1.78.0,<1.79" \
    "protobuf>=6.31.1,<7.0" \
    "mlflow>=3.10.0" \
    "python-dotenv>=1.0.0"

# Copiar el código de la aplicación y los proto files
COPY service/ /src/service/
COPY proto/ /src/proto/

# Generar stubs gRPC desde proto/ (si existen .proto)
RUN if [ -d "./proto" ] && [ "$(ls -A proto 2>/dev/null)" != "" ]; then \
      python -m grpc_tools.protoc -I ./proto --python_out=./service --grpc_python_out=./service proto/*.proto || true ; \
    fi

# Opcional: ejecutar tests o limpiezas aquí (omitido)

### STAGE: runtime
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1

# Dependencias de sistema necesarias en runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar paquetes instalados desde build (sitio de paquetes)
# Si la ruta cambia en futuras versiones de python, ajusta la ruta.
COPY --from=build /usr/local/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/

# Copiar código de aplicación (incluye stubs generados en build)
COPY --from=build /src/service /app/service
COPY --from=build /src/.env /app/.env 

# Exponer puerto gRPC (ajusta si usas otro)
EXPOSE 50051

# Comando por defecto: ejecutar el servidor de inferencia
# Ajusta esto si el entrypoint en tu repo es otro (ej.: python -m service.inference_server)
CMD ["python", "-u", "service/inference_server.py"]
FROM python:3.11-slim AS build

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu/ \
    VENV_PATH=/opt/venv

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY pyproject.toml uv.lock ./

RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install uv && uv sync --no-dev

RUN if [ -d ".venv" ]; then mkdir -p "$(dirname ${VENV_PATH})" && mv -T .venv "${VENV_PATH}"; fi

COPY app/ /src/app/
COPY proto/ /src/proto/

RUN mkdir -p /src/proto/generated && \
    "${VENV_PATH}/bin/python" -m grpc_tools.protoc \
      -I /src/proto \
      --python_out=/src/proto/generated \
      --grpc_python_out=/src/proto/generated \
      /src/proto/inference.proto

FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv

COPY --from=build /src/app /app/app
COPY --from=build /src/proto /app/proto

EXPOSE 8501

CMD ["/opt/venv/bin/python", "-u", "-m", "streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
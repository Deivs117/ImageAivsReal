# ImageAivsReal
Aplicación web en Streamlit que funciona como herramienta de apoyo a la verificación para clasificar imágenes como generadas por IA o reales.

Estructura Carpetas:
app/ (interfaz Streamlit)

service/ (servidor gRPC + inferencia)

proto/ (archivo .proto y stubs)

tests/ (pruebas funcionales)

docs/ (diagramas, decisiones, evidencias)

data/ (prueba locales)

para crear carpetas automáticamente usar comando

make create_dirs

## ⚙️ Configuración del entorno

### Opción recomendada: `uv` (gestor de dependencias)
```bash
# Instalar uv (si no lo tienes)
pip install uv

# Crear entorno e instalar dependencias base
uv sync
```

### Opción alternativa: `pip` + `requirements.txt`
```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Nota:** El `requirements.txt` contiene las **dependencias base** del proyecto.
> Podrán añadirse dependencias complementarias según las necesidades del desarrollo.
> La fuente de verdad es `pyproject.toml`.

## 🔌 gRPC Stubs

Los stubs gRPC se generan a partir de `proto/inference.proto` y se almacenan en `proto/generated/`.

### Generar los stubs
```bash
make grpc
```

Esto genera `proto/generated/inference_pb2.py` y `proto/generated/inference_pb2_grpc.py`.

### Regenerar los stubs
Si se modifica `proto/inference.proto`, regenerar con:
```bash
make grpc
```

Para eliminar los stubs generados:
```bash
make clean-grpc
```

### Tests de los stubs
Los tests en `tests/test_grpc_stubs.py` verifican que los stubs se importan correctamente y que los mensajes protobuf y el servicio gRPC funcionan como se espera.

```bash
uv run pytest tests/test_grpc_stubs.py
```
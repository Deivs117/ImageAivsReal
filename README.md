# ImageAivsReal

Aplicación web desarrollada con Streamlit que funciona como herramienta
de apoyo para clasificar imágenes como generadas por inteligencia
artificial o reales mediante un modelo de aprendizaje automático.

El sistema integra una arquitectura cliente-servidor basada en gRPC
para la comunicación entre la interfaz gráfica y el motor de inferencia.


# Estructura del proyecto

    app/        → Interfaz web en Streamlit  
    service/    → Servidor gRPC y motor de inferencia  
    proto/      → Definición del servicio y stubs gRPC  
    tests/      → Pruebas unitarias y funcionales  
    docs/       → Documentación, diagramas y evidencias  
    data/       → Imágenes de prueba locales  

Para crear las carpetas necesarias automáticamente:

``` bash
make create_dirs
```


# Configuración del entorno

El proyecto utiliza librerías de aprendizaje automático como PyTorch
y Hugging Face Transformers.

## Opción recomendada: uv

Se recomienda utilizar uv para la gestión de dependencias.

``` bash
pip install uv
uv sync
```

Esto creará el entorno e instalará automáticamente las dependencias
definidas en el proyecto.

También es posible ejecutar comandos mediante:

``` bash
uv run <comando>
```

Ejemplo:

``` bash
uv run python script.py
```


## Opción alternativa: pip

``` bash
python -m venv .venv
source .venv/bin/activate   # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Nota: requirements.txt contiene dependencias base.\
> La fuente principal de dependencias es pyproject.toml.


# Instalación, arranque y ejecución local

## 1. Clonar el repositorio

``` bash
git clone https://github.com/Deivs117/ImageAivsReal.git
cd ImageAivsReal
```


## 2. Instalar dependencias

Con uv (recomendado):

``` bash
make install
```

Esto ejecuta internamente:

``` bash
uv sync
```

Alternativamente con pip:

``` bash
pip install -r requirements.txt
```


## 3. Generar stubs gRPC

Los stubs gRPC se generan a partir de proto/inference.proto.

``` bash
make grpc
```

Archivos generados:

    proto/generated/inference_pb2.py
    proto/generated/inference_pb2_grpc.py

Si se modifica el .proto, regenerar con:

``` bash
make grpc
```

Para eliminar los stubs:

``` bash
make clean-grpc
```


## 4. Ejecutar servidor de inferencia

El servidor debe iniciarse antes de ejecutar la interfaz.

``` bash
make inference
```

También es posible ejecutar directamente el servidor gRPC:

``` bash
make grpc-server
```


## 5. Ejecutar la interfaz web

En otra terminal:

``` bash
make gui
```

o manualmente:

``` bash
streamlit run app/streamlit_app.py
```

La aplicación se abrirá en:

    http://localhost:8501

Desde la interfaz el usuario puede:

1.  cargar imágenes\
2.  ejecutar el análisis\
3.  visualizar los resultados\
4.  exportar resultados en CSV o PDF


# Primera descarga del modelo

Durante la primera ejecución, el modelo será descargado automáticamente
desde Hugging Face.

Este proceso puede tardar algunos minutos dependiendo de la conexión.\
Una vez descargado, el modelo quedará almacenado en caché local.


# Comandos útiles del proyecto

El proyecto incluye un Makefile con varios comandos para desarrollo:

``` bash
make help
```

Comandos principales:

    make install             → instalar dependencias con uv
    make gui                 → ejecutar interfaz Streamlit
    make grpc                → generar stubs gRPC
    make grpc-server         → iniciar servidor gRPC
    make inference           → ejecutar script de inferencia
    make test                → ejecutar todos los tests
    make test-inference      → tests del motor de inferencia
    make test-preprocessing  → tests de preprocesamiento
    make test-coverage       → reporte de cobertura
    make mlflow              → iniciar servidor MLflow
    make clean               → limpiar archivos temporales


# Limitaciones del sistema y uso responsable

## Disclaimer de uso responsable

El sistema desarrollado en este proyecto tiene como objetivo servir como
una herramienta de apoyo para la identificación preliminar de imágenes
potencialmente generadas por inteligencia artificial. Los resultados
proporcionados por el modelo corresponden a una clasificación
automática basada en aprendizaje automático, por lo que deben
interpretarse únicamente como una referencia orientativa.

Los resultados generados por la aplicación no constituyen una prueba
definitiva ni evidencia concluyente sobre el origen de una imagen. La
determinación final sobre la autenticidad de un contenido visual debe
apoyarse en análisis adicionales o herramientas especializadas de
análisis forense digital.

El uso de esta herramienta debe realizarse de forma responsable,
evitando utilizar sus resultados como única base para tomar decisiones
críticas o emitir afirmaciones categóricas sobre la autenticidad de una
imagen.


## Alcance del sistema

El sistema permite realizar la clasificación automática de imágenes
estáticas para estimar si presentan características asociadas a
imágenes reales o generadas por inteligencia artificial.

La aplicación permite además:

-   procesar múltiples imágenes en lote\
-   visualizar resultados en una interfaz gráfica\
-   exportar resultados en CSV y PDF


## Limitaciones del sistema

El sistema presenta algunas limitaciones inherentes al alcance del
proyecto.

En primer lugar, no garantiza una precisión del 100 %, ya que los
modelos de aprendizaje automático pueden cometer errores dependiendo de
la calidad de la imagen o del contenido analizado.

Asimismo, el sistema fue diseñado únicamente para analizar imágenes
estáticas, por lo que no incluye capacidades para analizar videos u
otros tipos de contenido multimedia.

Finalmente, el sistema no corresponde a una herramienta de análisis
forense digital, ya que no realiza análisis de metadatos, detección
avanzada de manipulación ni otros procedimientos especializados
utilizados en investigaciones profesionales.

Instalación, arranque y ejecución local de la aplicación
#Requisitos previos

Antes de ejecutar el sistema se requiere contar con:

1. Python 3.10 o superior

2. Conexión a internet (para la primera descarga del modelo)

3. Git (opcional, para clonar el repositorio)

El sistema utiliza librerías de aprendizaje automático y procesamiento de imágenes como PyTorch, Hugging Face Transformers y la interfaz web está desarrollada con Streamlit.

# 1. Clonar el repositorio
Primero se debe obtener el código del proyecto desde el repositorio.

git clone https://github.com/Deivs117/ImageAivsReal.git
cd ImageAivsReal

# 2. Crear y activar el entorno virtual

Se recomienda crear un entorno virtual para aislar las dependencias del proyecto.

python -m venv venv

Activar el entorno virtual:

Windows:

venv\Scripts\activate

Linux / Mac:

source venv/bin/activate

# 3. Instalar dependencias

Con el entorno virtual activado, instalar las dependencias:

pip install -r requirements.txt

# 4. Ejecutar el servidor de inferencia (gRPC)

El servidor de inferencia debe iniciarse antes de ejecutar la aplicación web. Este servidor es el encargado de cargar el modelo y procesar las solicitudes enviadas desde la interfaz.

Desde la raíz del proyecto ejecutar:

make inference

Este comando iniciará el servidor gRPC encargado de ejecutar el modelo de clasificación.

# 5. Ejecutar la aplicación Streamlit

En una segunda terminal, activar nuevamente el entorno virtual y ejecutar la interfaz gráfica.

venv\Scripts\activate
streamlit run app/streamlit_app.py

La aplicación se abrirá automáticamente en el navegador, normalmente en:

http://localhost:8501

Desde esta interfaz el usuario puede cargar imágenes, ejecutar el análisis y visualizar los resultados generados por el modelo.
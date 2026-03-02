import streamlit as st


def render_header():
    st.title("Prototipo: Clasificación de Imágenes (IA vs Real)")
    st.write(
        "Interfaz web (MVP) para apoyar la **clasificación probabilística** de imágenes como "
        "**Generadas por IA** o **Reales**, usando un modelo preentrenado en modo inferencia "
        "(sin fine-tuning)."
    )


def render_disclaimer():
    st.warning(
        "⚠️ **Disclaimer de uso responsable**\n\n"
        "- Esta herramienta es **de apoyo** para verificación preliminar.\n"
        "- **No** es certificación **forense/legal**.\n"
        "- Los resultados son **probabilísticos** y pueden fallar; no se garantiza exactitud del 100%.\n"
        "- No usar como única base para decisiones críticas."
    )


def render_placeholders():
    st.divider()

    st.header("1) Carga de imágenes (pendiente)")
    st.caption("Aquí irá la carga individual o por lote (JPG/JPEG/PNG).")
    st.info("📌 Placeholder: uploader + preview")

    st.header("2) Resultados (pendiente)")
    st.caption("Aquí se mostrará la predicción por imagen, confianza y tiempos.")
    st.info("📌 Placeholder: tabla + cards")

    st.header("3) Exportación (pendiente)")
    st.caption("Aquí se habilitará la descarga de **CSV y PDF** con resultados.")
    st.info("📌 Placeholder: download_button CSV")
    st.info("📌 Placeholder: download_button PDF (reporte)")


def main():
    st.set_page_config(page_title="AI vs Real Image Detector", layout="wide")

    render_header()
    render_disclaimer()
    render_placeholders()


if __name__ == "__main__":
    main()
"""Componentes visuales reutilizables para la app AI vs Real.

Responsabilidad: renderizar elementos de UI estaticos o de configuracion.
No contiene logica de negocio ni estado de sesion.
"""
import streamlit as st
import pandas as pd
from clientGrpc import GRPCClient, GRPCClientError
from report_pdf import build_pdf_bytes


def render_header() -> None:
    """Titulo y descripcion general de la app."""
    st.title("Prototipo: Clasificacion de Imagenes (IA vs Real)")
    st.write(
        "Interfaz web (MVP) para apoyar la **clasificacion probabilistica** de imagenes como "
        "**Generadas por IA** o **Reales**, usando un modelo preentrenado en modo inferencia "
        "(sin fine-tuning)."
    )


def render_disclaimer() -> None:
    """Aviso de uso responsable."""
    st.warning(
        "⚠️ **Disclaimer de uso responsable**\n\n"
        "- Esta herramienta es **de apoyo** para verificacion preliminar.\n"
        "- **No** es certificacion **forense/legal**.\n"
        "- Los resultados son **probabilisticos** y pueden fallar; no se garantiza exactitud del 100%.\n"
        "- No usar como unica base para decisiones criticas."
    )


def render_sidebar() -> GRPCClient | None:
    """Sidebar de configuracion gRPC. Retorna el cliente o None si falla."""
    with st.sidebar:
        st.header("Conexion gRPC")
        host = st.text_input("Host (vacio = localhost)", value="")
        port_str = st.text_input("Puerto (vacio = 50051)", value="")
        timeout_str = st.text_input("Timeout en segundos (vacio = 5)", value="")

    try:
        return GRPCClient(
            host=host or None,
            port=int(port_str) if port_str else None,
            timeout=int(timeout_str) if timeout_str else None,
        )
    except GRPCClientError as err:
        st.error(f"No se pudo conectar al servidor gRPC: {err}")
        return None


def render_summary(summary: dict) -> None:
    """Muestra resumen de resultados tras el analisis del lote."""
    exitosas = summary["exitosas"]
    fallidas = summary["fallidas"]
    total    = summary["total"]

    if fallidas == 0:
        st.success(f"Analisis completado: {exitosas} de {total} imagenes procesadas correctamente.")
    elif exitosas == 0:
        st.error(f"No se pudo procesar ninguna imagen. {fallidas} de {total} fallaron.")
    else:
        st.warning(f"{fallidas} de {total} imagenes no pudieron procesarse.")
        st.success(f"{exitosas} de {total} imagenes procesadas correctamente.")


def render_export_section(df: pd.DataFrame, builder) -> None:
    """Seccion 3: botones reales de exportacion CSV y PDF."""
    st.divider()
    st.header("3) Exportacion")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Descargar CSV",
            data=builder.to_csv_bytes(df),
            file_name="resultados_ai_vs_real.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        pdf_bytes = build_pdf_bytes(df)
        st.download_button(
            label="Descargar PDF",
            data=pdf_bytes,
            file_name="reporte_ai_vs_real.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
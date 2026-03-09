"""Streamlit app - AI vs Real image classifier.

Orquestador principal. Solo conecta modulos, sin logica propia.

Run with:
    streamlit run app/streamlit_app.py
"""
import streamlit as st

from ui_components import render_header, render_disclaimer, render_sidebar, render_summary, render_export_section
from batch_upload import BatchStore, BatchUploader
from batch_panel import inject_styles, render_batch_panel
from batch_runner import BatchRunner
from result_table import ResultsTableBuilder

st.set_page_config(page_title="AI vs Real Image Detector", layout="wide")
inject_styles()

render_header()
render_disclaimer()

client = render_sidebar()

# --- 1) Carga de imagenes ---
st.divider()
st.header("1) Carga de imagenes")

if "store" not in st.session_state:
    st.session_state.store = BatchStore()

store = st.session_state.store
uploader = BatchUploader(store)
uploader.render()

# --- 2) Analisis ---
st.divider()
st.header("2) Analisis")

items = store.items()

if items:
    if client is None:
        st.warning("No hay conexion al servidor gRPC. Verifica que este corriendo.")
        render_batch_panel(items)
    elif st.button("Analizar imagenes"):
        runner = BatchRunner(store=store, client=client)
        summary = runner.run()
        render_summary(summary)

        builder = ResultsTableBuilder()
        df = builder.from_batch_items(store.items())

        st.subheader("Resultados")
        st.dataframe(df, use_container_width=True)

        # --- 3) Exportacion ---
        render_export_section(df, builder)
    else:
        render_batch_panel(items)
else:
    st.info("Sube imagenes para comenzar.")
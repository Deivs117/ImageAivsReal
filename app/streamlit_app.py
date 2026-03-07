"""Streamlit integration example for the gRPC image classification client.

Demonstrates end-to-end usage of GRPCClient:
  - Configure host/port via sidebar (overrides .env values)
  - Upload one or multiple images
  - Send them to the inference server and display results as a DataFrame
  - Allow CSV export

Run with::

    streamlit run app/streamlit_app.py
"""
import pandas as pd
import streamlit as st

from clientGrpc import GRPCClient, GRPCClientError

st.set_page_config(page_title="AI vs Real – gRPC Example", layout="wide")
st.title("Image AI vs Real – Ejemplo de cliente gRPC")

# --- Connection settings (sidebar) ---
with st.sidebar:
    st.header("Conexión gRPC")
    host = st.text_input(
        "Host (vacío = GRPC_SERVER_HOST o localhost)", value=""
    )
    port_str = st.text_input(
        "Puerto (vacío = GRPC_SERVER_PORT o 50051)", value=""
    )
    timeout_str = st.text_input(
        "Timeout en segundos (vacío = GRPC_TIMEOUT o 5)", value=""
    )

# --- Connect to gRPC server ---
try:
    client = GRPCClient(
        host=host or None,
        port=int(port_str) if port_str else None,
        timeout=int(timeout_str) if timeout_str else None,
    )
except GRPCClientError as err:
    st.error(f"No se pudo conectar al servidor gRPC: {err}")
    st.stop()

# --- Image upload & inference ---
uploaded_files = st.file_uploader(
    "Sube una o varias imágenes (JPG/JPEG/PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("Analizar"):
    results = []
    with st.spinner("Clasificando imágenes…"):
        for f in uploaded_files:
            image_bytes = f.read()
            try:
                result = client.classify_image(image_bytes, filename=f.name)
                result["filename"] = f.name
                results.append(result)
            except GRPCClientError as err:
                results.append(
                    {
                        "filename": f.name,
                        "status": "error",
                        "error_message": str(err),
                    }
                )

    if results:
        df = pd.DataFrame(results)
        st.subheader("Resultados")
        st.dataframe(df, use_container_width=True)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar CSV",
            data=csv_bytes,
            file_name="resultados_ai_vs_real.csv",
            mime="text/csv",
        )

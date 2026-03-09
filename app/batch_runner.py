"""Orquesta la inferencia sobre el lote, actualizando el estado de cada imagen."""
import streamlit as st
from batch_upload import BatchStore
from batch_panel import render_batch_panel
from clientGrpc import GRPCClient
from result_table import utc_now_iso


class BatchRunner:
    """Recorre el lote, llama a inferencia y actualiza el estado de cada BatchImage."""

    def __init__(self, store: BatchStore, client: GRPCClient) -> None:
        self.store = store
        self.client = client

    def run(self) -> dict:
        """Ejecuta inferencia sobre el lote. Retorna resumen de resultados."""
        items = self.store.items()

        # Resetear campos para permitir re-analisis
        for item in items:
            if item.status != "error" or item.content:
                item.status = "pending"
                item.timestamp = None
                item.predicted_label = None
                item.prob_ai = None
                item.prob_real = None
                item.preprocess_time_ms = None
                item.inference_time_ms = None
                item.error_message = None

        panel_placeholder = st.empty()

        for item in items:
            # Saltar imagenes que fallaron en la carga (sin contenido)
            if item.status == "error" and not item.content:
                continue

            item.status = "processing"
            with panel_placeholder.container():
                render_batch_panel(items)

            result = self.client.classify_image_safe(
                item.content,
                filename=item.filename,
                image_id=item.id,
            )

            item.timestamp = utc_now_iso()
            item.predicted_label = result.get("predicted_label")
            item.prob_ai = result.get("prob_ai")
            item.prob_real = result.get("prob_real")
            item.preprocess_time_ms = result.get("preprocess_time_ms")
            item.inference_time_ms = result.get("inference_time_ms")

            if result.get("status") == "error":
                item.status = "error"
                item.error_message = result.get("error_message", "Error desconocido")
            else:
                item.status = "done"

            with panel_placeholder.container():
                render_batch_panel(items)

        # Resumen final
        exitosas = sum(1 for i in self.store.items() if i.status == "done")
        fallidas = sum(1 for i in self.store.items() if i.status == "error")

        return {"exitosas": exitosas, "fallidas": fallidas, "total": exitosas + fallidas}
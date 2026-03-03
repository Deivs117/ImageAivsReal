from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd


SCHEMA_COLUMNS = [
    "timestamp",
    "filename",
    "status",
    "predicted_label",
    "prob_ai",
    "prob_real",
    "preprocess_time_ms",
    "inference_time_ms",
    "error_message",
]


def utc_now_iso() -> str:
    # ISO 8601 en UTC sin microsegundos: 2026-03-02T05:12:10Z
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class ResultsTableBuilder:
    """
    Convierte una lista de items del lote (con estados y/o resultados) a DataFrame
    siguiendo el esquema E1 para visualización y export.
    """

    def __init__(self, columns: Optional[List[str]] = None) -> None:
        self.columns = columns or SCHEMA_COLUMNS

    def from_batch_items(self, items: List[Any]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        for it in items:
            # Soporta objetos dataclass (BatchImage) o dicts del session_state
            d = self._to_dict(it)

            filename = d.get("filename") or d.get("name") or "unknown"
            ui_status = d.get("status")  # pending/processing/done/error (GUI)
            timestamp = d.get("timestamp") or utc_now_iso()

            # Campos de predicción (pueden estar None si aún no hay inferencia)
            predicted_label = d.get("predicted_label")
            prob_ai = d.get("prob_ai")
            prob_real = d.get("prob_real")
            preprocess_time_ms = d.get("preprocess_time_ms")
            inference_time_ms = d.get("inference_time_ms")
            error_message = d.get("error_message")

            # ===== Normalización a esquema E1 =====
            # status E1: ok/error
            if ui_status == "error":
                status = "error"
                if not error_message:
                    error_message = "Unknown error"
            elif predicted_label is not None or prob_ai is not None or prob_real is not None:
                # Si hay predicción, consideramos ok
                status = "ok"
            else:
                # Aún no procesado: lo dejamos como error con mensaje explícito
                status = "error"
                error_message = error_message or "Not processed yet"

            rows.append(
                {
                    "timestamp": timestamp,
                    "filename": filename,
                    "status": status,
                    "predicted_label": predicted_label,
                    "prob_ai": prob_ai,
                    "prob_real": prob_real,
                    "preprocess_time_ms": preprocess_time_ms,
                    "inference_time_ms": inference_time_ms,
                    "error_message": error_message,
                }
            )

        df = pd.DataFrame(rows)

        # Asegurar columnas y orden (E1)
        for c in self.columns:
            if c not in df.columns:
                df[c] = None
        df = df[self.columns]

        # Tipos numéricos para tiempos y probabilidades (si vienen como string)
        for num_col in ["prob_ai", "prob_real", "preprocess_time_ms", "inference_time_ms"]:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

        return df

    def to_csv_bytes(self, df: pd.DataFrame) -> bytes:
        return df.to_csv(index=False).encode("utf-8")

    def _to_dict(self, it: Any) -> Dict[str, Any]:
        if is_dataclass(it):
            return asdict(it)
        if isinstance(it, dict):
            return dict(it)
        # fallback: atributos
        return {k: getattr(it, k) for k in dir(it) if not k.startswith("_")}
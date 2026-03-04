import os
import mlflow
from .model_loader import (
    init_inference_artifacts,
    report_loaded_to_mlflow,
)

hf_model_id = os.getenv("HF_MODEL_ID", "").strip()

mlflow.set_experiment("ImageAivsReal-Service-Health")

with mlflow.start_run(run_name="startup-model-load"):
    artifacts = init_inference_artifacts(
        hf_model_id=hf_model_id,
        device="cpu",
    )
    report_loaded_to_mlflow(artifacts=artifacts)
    print("✅ OK: reportado a MLflow")
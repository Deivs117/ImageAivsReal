import os
import sys
from pathlib import Path

# Allow running as a script directly (uv run service/inference/mlflow_health_check.py)
# as well as being imported as part of the package.
if __name__ == "__main__" and __package__ is None:
    # Add the project root (two levels up from this file) to sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from service.inference.model_loader import (
        init_inference_artifacts,
        report_loaded_to_mlflow,
    )
else:
    from .model_loader import (
        init_inference_artifacts,
        report_loaded_to_mlflow,
    )

import mlflow

hf_model_id = os.getenv("HF_MODEL_ID", "").strip()

mlflow.set_experiment("ImageAivsReal-Service-Health")

with mlflow.start_run(run_name="startup-model-load"):
    artifacts = init_inference_artifacts(
        hf_model_id=hf_model_id,
        device="cpu",
    )
    report_loaded_to_mlflow(artifacts=artifacts)
    print("✅ OK: reportado a MLflow")
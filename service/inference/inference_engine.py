import os

from dotenv import load_dotenv
from model_loader import init_inference_artifacts

load_dotenv()

hf_model_id = os.getenv("HF_MODEL_ID", "").strip()
print("HF_MODEL_ID:", hf_model_id)

artifacts = init_inference_artifacts(hf_model_id=hf_model_id, device="cpu")

print("device:", artifacts.device)
print("eval_mode:", not artifacts.model.training)
print("processor_type:", type(artifacts.processor))
print("model_type:", type(artifacts.model))
print("model_id_or_uri:", artifacts.model_id_or_uri)

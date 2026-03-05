"""
Modulo de inferencia central para el modelo dima806/ai_vs_real_image_detection.

Funcion reutilizable que ejecuta el modelo sobre una imagen y retorna
la prediccion (clase ganadora), las probabilidades por clase y los tiempos
de preprocesamiento e inferencia en milisegundos (ms), en formato
serializable (float), lista para ser usada desde gRPC o GUI.

Uso:
    from service.inference.inference_engine import run_inference
    result = run_inference(image, model, processor)

Comando Make asociado:
    make test-inference  ->  Ejecuta los tests unitarios del modulo

Unidades de tiempo:
    Todos los tiempos se expresan en milisegundos (ms) como float
    redondeado a 3 decimales. Se usa time.perf_counter() para maxima
    precision en medicion de intervalos cortos.
"""

import logging
import time
from typing import Union

import torch
from PIL import Image

from service.inference.preprocessing import preprocess_image

logger = logging.getLogger(__name__)

def run_inference(
    image: Union[Image.Image, bytes],
    model,
    processor,
) -> dict:
    """Ejecuta inferencia sobre una imagen usando el modelo ViT.

    Preprocesa la imagen, ejecuta el modelo en modo evaluacion, calcula
    softmax sobre los logits y retorna la prediccion con sus probabilidades
    por clase y tiempos de ejecucion como floats serializables.

    Args:
        image: Imagen fuente. Puede ser:
            - PIL.Image.Image: usada directamente.
            - bytes: decodificada internamente antes del preprocesamiento.
        model: Modelo de Hugging Face ya cargado (AutoModelForImageClassification
            o compatible). Debe tener model.config.id2label.
        processor: Instancia de AutoImageProcessor (o compatible) ya cargada.

    Returns:
        Dict con la siguiente estructura::

            {
                "label": "AI",          # str  - etiqueta de la clase predicha
                "label_id": 0,          # int  - indice de la clase predicha
                "scores": {             # dict - probabilidad por clase (float, suma ~1.0)
                    "AI": 0.9741,
                    "Real": 0.0259
                },
                "timing": {             # dict - tiempos de ejecucion en milisegundos (ms)
                    "preprocessing_ms": 12.345,   # float - tiempo de preprocesamiento
                    "inference_ms": 45.678,       # float - tiempo de inferencia del modelo
                    "total_ms": 58.023            # float - suma de preprocessing + inference
                }
            }

    Raises:
        TypeError: Si image no es PIL.Image.Image ni bytes (propagado desde
            preprocess_image).
        ValueError: Si los bytes no son una imagen valida, o si el modelo no
            retorna logits.
        RuntimeError: Si ocurre un error inesperado durante la inferencia.

    Example:
        >>> from PIL import Image
        >>> from transformers import AutoImageProcessor, AutoModelForImageClassification
        >>> processor = AutoImageProcessor.from_pretrained(
        ...     "dima806/ai_vs_real_image_detection"
        ... )
        >>> model = AutoModelForImageClassification.from_pretrained(
        ...     "dima806/ai_vs_real_image_detection"
        ... )
        >>> img = Image.open("photo.jpg")
        >>> result = run_inference(img, model, processor)
        >>> print(result["label"])                      # "AI" o "Real"
        >>> print(result["scores"])                     # {"AI": 0.97, "Real": 0.03}
        >>> print(result["timing"]["total_ms"])         # ej: 58.023
    """
    # 1. Preprocesar imagen -> inputs dict con pixel_values (con medicion de tiempo)
    t0_preprocess = time.perf_counter()
    inputs = preprocess_image(image, processor)
    t1_preprocess = time.perf_counter()
    preprocessing_ms = round((t1_preprocess - t0_preprocess) * 1000, 3)

    # 2. Inferencia en modo evaluacion sin gradientes (con medicion de tiempo)
    model.eval()
    t0_inference = time.perf_counter()
    try:
        with torch.no_grad():
            outputs = model(**inputs)
    except Exception as exc:
        raise RuntimeError(
            f"Error durante la inferencia del modelo: {exc}"
        ) from exc
    t1_inference = time.perf_counter()
    inference_ms = round((t1_inference - t0_inference) * 1000, 3)

    # 3. Validar que el modelo retorno logits
    if not hasattr(outputs, "logits"):
        raise ValueError(
            "El modelo no retorno 'logits'. "
            "Verifica que sea un modelo de clasificacion de imagenes."
        )

    logits = outputs.logits  # shape: (1, num_classes)

    # 4. Clase predicha (argmax)
    label_id = int(torch.argmax(logits, dim=-1).item())

    # 5. Probabilidades por clase (softmax -> float)
    probs = torch.softmax(logits, dim=-1).squeeze(0)  # shape: (num_classes,)

    # 6. Mapear id2label desde la configuracion del modelo
    id2label = model.config.id2label  # {0: "AI", 1: "Real"} o similar

    scores = {
        id2label[i]: round(float(probs[i]), 6)
        for i in range(len(probs))
    }

    label = id2label[label_id]

    # 7. Calcular total_ms como suma explicita de ambos tiempos
    total_ms = round(preprocessing_ms + inference_ms, 3)

    result = {
        "label": label,
        "label_id": label_id,
        "scores": scores,
        "timing": {
            "preprocessing_ms": preprocessing_ms,
            "inference_ms": inference_ms,
            "total_ms": total_ms,
        },
    }

    logger.debug(
        "Inferencia exitosa. label=%s, label_id=%d, scores=%s, "
        "preprocessing_ms=%.3f, inference_ms=%.3f, total_ms=%.3f",
        label,
        label_id,
        scores,
        preprocessing_ms,
        inference_ms,
        total_ms,
    )

    return result

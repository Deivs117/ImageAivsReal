"""
Modulo de preprocesamiento de imagen para el modelo dima806/ai_vs_real_image_detection.

Funcion reutilizable que transforma una imagen PIL (o bytes) en tensores
compatibles con el modelo ViT de Hugging Face, lista para inferencia.

Uso:
    from service.inference.preprocessing import preprocess_image
    inputs = preprocess_image(pil_image, processor)

Comando Make asociado:
    make test-preprocessing  ->  Ejecuta los tests unitarios del modulo
"""
"""gRPC inference server with real model inference.

Replaces the mock servicer with a real inference pipeline:
- Loads AutoModelForImageClassification + AutoImageProcessor at startup
- Delegates to run_inference() from service.inference.inference_engine
- Maps result scores to prob_ai / prob_human fields
- Returns ERROR status with error_message on inference failure (no crash)

Env vars:
    HF_MODEL_ID       - HuggingFace model ID (default: Ateeqq/ai-vs-human-image-detector)
    GRPC_LOG_LEVEL    - Logging level (default: INFO)
    GRPC_SERVER_HOST  - Server host (default: localhost)
    GRPC_SERVER_PORT  - Server port (default: 50051)

Usage:
    uv run python -m service.inference_server
"""
import logging
import os
import sys
from concurrent import futures

import grpc
from dotenv import load_dotenv
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Add proto/generated to path so gRPC stubs can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'proto', 'generated'))

import inference_pb2
import inference_pb2_grpc

from inference.inference_engine import run_inference

load_dotenv()

LOG_LEVEL = os.getenv('GRPC_LOG_LEVEL', 'INFO')
GRPC_SERVER_HOST = os.getenv('GRPC_SERVER_HOST', 'localhost')
GRPC_SERVER_PORT = int(os.getenv('GRPC_SERVER_PORT', '50051'))
HF_MODEL_ID = os.getenv('HF_MODEL_ID', 'Ateeqq/ai-vs-human-image-detector')

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


class AiVsRealClassifierServicer(inference_pb2_grpc.AiVsRealClassifierServicer):
    """gRPC servicer that performs real model inference using HuggingFace."""

    def __init__(self, model=None, processor=None):
        """Load model and processor at startup (injectable for tests).

        Args:
            model: Pre-loaded model. If None, loads from HF_MODEL_ID.
            processor: Pre-loaded processor. If None, loads from HF_MODEL_ID.
        """
        if model is not None and processor is not None:
            self.model = model
            self.processor = processor
            logger.info('Using injected model and processor.')
        else:
            logger.info('Loading model and processor from HuggingFace: %s', HF_MODEL_ID)
            self.processor = AutoImageProcessor.from_pretrained(HF_MODEL_ID)
            self.model = AutoModelForImageClassification.from_pretrained(HF_MODEL_ID)
            logger.info('Model and processor loaded successfully.')

    def ClassifyImage(self, request, context):
        logger.info(
            'Received ClassifyImage request: image_id=%s filename=%s',
            request.image_id,
            request.filename,
        )

        result = run_inference(request.image_data, self.model, self.processor)

        if result['status'] == 'error':
            logger.warning(
                'Inference error for image_id=%s: %s',
                request.image_id,
                result['error']['message'],
            )
            return inference_pb2.ClassificationResponse(
                image_id=request.image_id,
                status=inference_pb2.ERROR,
                predicted_label='',
                confidence=0.0,
                prob_ai=0.0,
                prob_human=0.0,
                metrics=inference_pb2.PerformanceMetrics(
                    preprocess_time_ms=0,
                    inference_time_ms=0,
                    total_time_ms=0,
                ),
                error_message=result['error']['message'],
            )

        # Map scores dict to proto fields - normalize label keys to lowercase
        scores = {k.lower(): v for k, v in result['scores'].items()}
        prob_ai = float(scores.get('ai', 0.0))
        prob_human = float(scores.get('human', 0.0))

        # Confidence = score of winning class
        predicted_label = result['label'].lower()
        confidence = float(scores.get(predicted_label, 0.0))

        timing = result['timing']
        metrics = inference_pb2.PerformanceMetrics(
            preprocess_time_ms=int(timing['preprocessing_ms']),
            inference_time_ms=int(timing['inference_ms']),
            total_time_ms=int(timing['total_ms']),
        )

        logger.info(
            'ClassifyImage OK: image_id=%s label=%s confidence=%.4f',
            request.image_id,
            predicted_label,
            confidence,
        )

        return inference_pb2.ClassificationResponse(
            image_id=request.image_id,
            status=inference_pb2.OK,
            predicted_label=predicted_label,
            confidence=confidence,
            prob_ai=prob_ai,
            prob_human=prob_human,
            metrics=metrics,
            error_message='',
        )


def serve(host=None, port=None, model=None, processor=None):
    """Start the gRPC server.

    Args:
        host: Server host (default from env).
        port: Server port (default from env).
        model: Optional pre-loaded model for testing (bypasses HF download).
        processor: Optional pre-loaded processor for testing.
    """
    host = host or GRPC_SERVER_HOST
    port = port or GRPC_SERVER_PORT

    servicer = AiVsRealClassifierServicer(model=model, processor=processor)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_AiVsRealClassifierServicer_to_server(servicer, server)
    address = f'{host}:{port}'
    server.add_insecure_port(address)
    server.start()
    logger.info('gRPC server started on port %d', port)
    return server


if __name__ == '__main__':
    server = serve()
    server.wait_for_termination()
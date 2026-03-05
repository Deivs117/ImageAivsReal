"""gRPC inference server using real HuggingFace model for image classification."""
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
from service.inference.inference_engine import run_inference

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
    """gRPC servicer that uses real HuggingFace model inference."""

    def __init__(self, model=None, processor=None):
        model_id = HF_MODEL_ID
        if model is not None and processor is not None:
            self.model = model
            self.processor = processor
        else:
            logger.info('Loading model and processor from HuggingFace: %s', model_id)
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModelForImageClassification.from_pretrained(model_id)
            self.model.eval()
        logger.info('Model loaded successfully.')

    def ClassifyImage(self, request, context):
        logger.info(
            'Received ClassifyImage request: image_id=%s filename=%s',
            request.image_id,
            request.filename,
        )

        result = run_inference(request.image_data, self.model, self.processor)

        if result['status'] == 'error':
            error_info = result.get('error') or {}
            error_message = error_info.get('message', 'Unknown error')
            logger.warning(
                'Inference error for image_id=%s: %s', request.image_id, error_message
            )
            timing = result.get('timing', {})
            metrics = inference_pb2.PerformanceMetrics(
                preprocess_time_ms=int(timing.get('preprocessing_ms', 0.0)),
                inference_time_ms=int(timing.get('inference_ms', 0.0)),
                total_time_ms=int(timing.get('total_ms', 0.0)),
            )
            return inference_pb2.ClassificationResponse(
                image_id=request.image_id,
                status=inference_pb2.ERROR,
                predicted_label='',
                confidence=0.0,
                prob_ai=0.0,
                prob_human=0.0,
                metrics=metrics,
                error_message=error_message,
            )

        label = result['label'].lower()
        scores = result.get('scores', {})
        prob_ai = float(scores.get('ai', 0.0))
        prob_human = float(scores.get('human', 0.0))
        confidence = max(prob_ai, prob_human)

        timing = result.get('timing', {})
        metrics = inference_pb2.PerformanceMetrics(
            preprocess_time_ms=int(timing.get('preprocessing_ms', 0.0)),
            inference_time_ms=int(timing.get('inference_ms', 0.0)),
            total_time_ms=int(timing.get('total_ms', 0.0)),
        )

        return inference_pb2.ClassificationResponse(
            image_id=request.image_id,
            status=inference_pb2.OK,
            predicted_label=label,
            confidence=confidence,
            prob_ai=prob_ai,
            prob_human=prob_human,
            metrics=metrics,
            error_message='',
        )


def serve(host=None, port=None, model=None, processor=None):
    """Start the gRPC server."""
    host = host or GRPC_SERVER_HOST
    port = port or GRPC_SERVER_PORT

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_AiVsRealClassifierServicer_to_server(
        AiVsRealClassifierServicer(model=model, processor=processor), server
    )
    address = f'{host}:{port}'
    server.add_insecure_port(address)
    server.start()
    logger.info('gRPC server started on port %d', port)
    return server


if __name__ == '__main__':
    server = serve()
    server.wait_for_termination()

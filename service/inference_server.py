"""gRPC inference server with mock responses for testing."""
import logging
import os
import random
import sys
from concurrent import futures

import grpc
from dotenv import load_dotenv

# Add proto/generated to path so gRPC stubs can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'proto', 'generated'))

import inference_pb2
import inference_pb2_grpc

load_dotenv()

LOG_LEVEL = os.getenv('GRPC_LOG_LEVEL', 'INFO')
GRPC_SERVER_HOST = os.getenv('GRPC_SERVER_HOST', 'localhost')
GRPC_SERVER_PORT = int(os.getenv('GRPC_SERVER_PORT', '50051'))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


class AiVsRealClassifierServicer(inference_pb2_grpc.AiVsRealClassifierServicer):
    """Mock gRPC servicer that returns random classification results."""

    def ClassifyImage(self, request, context):
        logger.info(
            'Received ClassifyImage request: image_id=%s filename=%s',
            request.image_id,
            request.filename,
        )

        predicted_label = random.choice(['AI', 'human'])
        confidence = round(random.uniform(0.70, 0.99), 4)
        prob_ai = round(random.uniform(0.0, 1.0), 4)
        prob_human = round(1.0 - prob_ai, 4)

        preprocess_time_ms = random.randint(10, 50)
        inference_time_ms = random.randint(100, 200)
        total_time_ms = preprocess_time_ms + inference_time_ms

        metrics = inference_pb2.PerformanceMetrics(
            preprocess_time_ms=preprocess_time_ms,
            inference_time_ms=inference_time_ms,
            total_time_ms=total_time_ms,
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


def serve(host=None, port=None):
    """Start the gRPC server."""
    host = host or GRPC_SERVER_HOST
    port = port or GRPC_SERVER_PORT

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_AiVsRealClassifierServicer_to_server(
        AiVsRealClassifierServicer(), server
    )
    address = f'{host}:{port}'
    server.add_insecure_port(address)
    server.start()
    logger.info('gRPC server started on port %d', port)
    return server


if __name__ == '__main__':
    server = serve()
    server.wait_for_termination()

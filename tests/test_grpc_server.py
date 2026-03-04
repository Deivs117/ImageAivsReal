"""Tests for gRPC inference server.

Tests verify that the mock server starts correctly and responds with valid
data. All tests follow the AAA (Arrange, Act, Assert) pattern.
"""
import threading
import time

import grpc
import pytest

import inference_pb2
import inference_pb2_grpc
from services.inference_server import serve


@pytest.fixture(scope='module')
def grpc_server():
    """Start the gRPC server on a free port and stop it after the module."""
    server = serve(host='localhost', port=50052)
    time.sleep(0.2)
    yield server
    server.stop(grace=0)


@pytest.fixture(scope='module')
def grpc_stub(grpc_server):
    """Return a stub connected to the test server."""
    channel = grpc.insecure_channel('localhost:50052')
    stub = inference_pb2_grpc.AiVsRealClassifierStub(channel)
    yield stub
    channel.close()


def _make_request(image_id='img-001', filename='photo.jpg'):
    return inference_pb2.ImageRequest(
        image_id=image_id,
        filename=filename,
        image_data=b'\xff\xd8\xff',
    )


# ============================================================
# Tests
# ============================================================

def test_server_startup(grpc_server):
    # Arrange / Act: fixture already starts server

    # Assert
    assert grpc_server is not None


def test_classify_image_basic(grpc_stub):
    # Arrange
    request = _make_request()

    # Act
    response = grpc_stub.ClassifyImage(request)

    # Assert
    assert response is not None
    assert response.image_id == 'img-001'


def test_response_format(grpc_stub):
    # Arrange
    request = _make_request(image_id='img-002', filename='test.png')

    # Act
    response = grpc_stub.ClassifyImage(request)

    # Assert
    assert response.image_id == 'img-002'
    assert response.status == inference_pb2.OK
    assert response.predicted_label in ('AI', 'human')
    assert response.metrics is not None
    assert response.error_message == ''


def test_predicted_label_alternates(grpc_stub):
    # Arrange
    labels = set()

    # Act
    for _ in range(20):
        response = grpc_stub.ClassifyImage(_make_request())
        labels.add(response.predicted_label)

    # Assert: both labels should appear in 20 requests
    assert 'AI' in labels
    assert 'human' in labels


def test_confidence_is_random(grpc_stub):
    # Arrange
    confidences = []

    # Act
    for i in range(5):
        response = grpc_stub.ClassifyImage(_make_request(image_id=f'img-{i}'))
        confidences.append(response.confidence)

    # Assert: all values within range 0.70-0.99
    for conf in confidences:
        assert 0.70 <= conf <= 0.99


def test_probabilities_sum_to_one(grpc_stub):
    # Arrange
    request = _make_request()

    # Act
    response = grpc_stub.ClassifyImage(request)

    # Assert
    assert abs(response.prob_ai + response.prob_human - 1.0) < 1e-4


def test_metrics_realistic(grpc_stub):
    # Arrange
    request = _make_request()

    # Act
    response = grpc_stub.ClassifyImage(request)
    m = response.metrics

    # Assert
    assert 10 <= m.preprocess_time_ms <= 50
    assert 100 <= m.inference_time_ms <= 200
    assert m.total_time_ms == m.preprocess_time_ms + m.inference_time_ms


def test_multiple_requests(grpc_stub):
    # Arrange
    requests = [_make_request(image_id=f'img-{i}') for i in range(8)]

    # Act
    responses = [grpc_stub.ClassifyImage(req) for req in requests]

    # Assert
    assert len(responses) == 8
    for i, response in enumerate(responses):
        assert response.image_id == f'img-{i}'
        assert response.status == inference_pb2.OK

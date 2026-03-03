"""Tests for gRPC stubs generated from proto/inference.proto.

These tests verify that the generated stubs (inference_pb2.py and
inference_pb2_grpc.py) can be imported and used correctly.
All tests follow the AAA (Arrange, Act, Assert) pattern.
"""
import importlib

import inference_pb2
import inference_pb2_grpc


def test_import_inference_pb2():
    # Arrange
    module_name = 'inference_pb2'

    # Act
    module = importlib.import_module(module_name)

    # Assert
    assert module is not None


def test_import_inference_pb2_grpc():
    # Arrange
    module_name = 'inference_pb2_grpc'

    # Act
    module = importlib.import_module(module_name)

    # Assert
    assert module is not None


def test_image_request_creation():
    # Arrange
    image_id = 'img-001'
    filename = 'photo.jpg'
    image_data = b'\xff\xd8\xff'

    # Act
    request = inference_pb2.ImageRequest(
        image_id=image_id,
        filename=filename,
        image_data=image_data,
    )

    # Assert
    assert request.image_id == image_id
    assert request.filename == filename
    assert request.image_data == image_data


def test_performance_metrics_creation():
    # Arrange
    preprocess_time_ms = 10
    inference_time_ms = 50
    total_time_ms = 60

    # Act
    metrics = inference_pb2.PerformanceMetrics(
        preprocess_time_ms=preprocess_time_ms,
        inference_time_ms=inference_time_ms,
        total_time_ms=total_time_ms,
    )

    # Assert
    assert metrics.preprocess_time_ms == preprocess_time_ms
    assert metrics.inference_time_ms == inference_time_ms
    assert metrics.total_time_ms == total_time_ms


def test_classification_response_creation():
    # Arrange
    metrics = inference_pb2.PerformanceMetrics(
        preprocess_time_ms=10,
        inference_time_ms=50,
        total_time_ms=60,
    )

    # Act
    response = inference_pb2.ClassificationResponse(
        image_id='img-001',
        status=inference_pb2.OK,
        predicted_label='ai',
        confidence=0.95,
        prob_ai=0.95,
        prob_human=0.05,
        metrics=metrics,
    )

    # Assert
    assert response.image_id == 'img-001'
    assert response.status == inference_pb2.OK
    assert response.predicted_label == 'ai'
    assert abs(response.confidence - 0.95) < 1e-5
    assert abs(response.prob_ai - 0.95) < 1e-5
    assert abs(response.prob_human - 0.05) < 1e-5
    assert response.metrics.total_time_ms == 60


def test_classification_response_with_error():
    # Arrange
    error_message = 'Image format not supported'

    # Act
    response = inference_pb2.ClassificationResponse(
        status=inference_pb2.ERROR,
        error_message=error_message,
    )

    # Assert
    assert response.status == inference_pb2.ERROR
    assert response.error_message == error_message


def test_result_status_enum():
    # Arrange
    result_status = inference_pb2.ResultStatus

    # Act
    unspecified = inference_pb2.RESULT_STATUS_UNSPECIFIED
    ok = inference_pb2.OK
    error = inference_pb2.ERROR

    # Assert
    assert unspecified == 0
    assert ok == 1
    assert error == 2


def test_aivs_real_classifier_service_exists():
    # Arrange
    servicer_class = inference_pb2_grpc.AiVsRealClassifierServicer

    # Act
    has_classify_image = hasattr(servicer_class, 'ClassifyImage')

    # Assert
    assert servicer_class is not None
    assert has_classify_image

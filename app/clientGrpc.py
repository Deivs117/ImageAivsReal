"""gRPC client for the AiVsReal image classification service.

Reads connection settings from .env (GRPC_SERVER_HOST, GRPC_SERVER_PORT,
GRPC_TIMEOUT) and allows overriding them via constructor parameters.

Usage::

    from app.clientGrpc import GRPCClient, GRPCClientError

    client = GRPCClient()
    result = client.classify_image(image_bytes, filename="photo.jpg")
    client.close()
"""
from __future__ import annotations

import logging
import os
import sys
import uuid
from typing import Any, Dict, Optional

import grpc
from dotenv import load_dotenv

# Add proto/generated to sys.path so the generated stubs can be imported
# regardless of the working directory.
_PROTO_GENERATED = os.path.join(os.path.dirname(__file__), "..", "proto", "generated")
if _PROTO_GENERATED not in sys.path:
    sys.path.insert(0, _PROTO_GENERATED)

try:
    import inference_pb2
    import inference_pb2_grpc
except ImportError as _exc:
    raise ImportError(
        "gRPC stubs not found. Run 'make proto-gen' to generate them from "
        "proto/inference.proto."
    ) from _exc

load_dotenv()

LOG = logging.getLogger(__name__)


class GRPCClientError(Exception):
    """Raised when the gRPC client cannot connect or an RPC call fails."""


class GRPCClient:
    """Reusable gRPC client for the AiVsRealClassifier inference service.

    Parameters
    ----------
    host:
        Server hostname. Falls back to ``GRPC_SERVER_HOST`` env var or
        ``"localhost"``.
    port:
        Server port. Falls back to ``GRPC_SERVER_PORT`` env var or ``50051``.
    timeout:
        Seconds to wait for channel readiness and per-RPC deadline. Falls back
        to ``GRPC_TIMEOUT`` env var or ``5``.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.host: str = host or os.getenv("GRPC_SERVER_HOST", "localhost")
        self.port: int = port or int(os.getenv("GRPC_SERVER_PORT", "50051"))
        self.timeout: int = (
            timeout if timeout is not None else int(os.getenv("GRPC_TIMEOUT", "5"))
        )
        self._channel: Optional[grpc.Channel] = None
        self._stub = None
        self._connect()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        target = f"{self.host}:{self.port}"
        try:
            self._channel = grpc.insecure_channel(target)
            grpc.channel_ready_future(self._channel).result(timeout=self.timeout)
            self._stub = inference_pb2_grpc.AiVsRealClassifierStub(self._channel)
            LOG.info("Connected to gRPC server at %s", target)
        except Exception as exc:
            LOG.exception("Could not connect to gRPC server at %s", target)
            raise GRPCClientError(
                f"Error connecting to gRPC server at {target}: {exc}"
            ) from exc

    def _parse_response(self, response: Any) -> Dict[str, Any]:
        """Convert a ``ClassificationResponse`` proto message to a plain dict.

        The returned dict uses the same field names expected by
        :class:`app.batch_upload.BatchImage` and
        :class:`app.result_table.ResultsTableBuilder` so it can be fed
        directly into the GUI layer or a CSV export.
        """
        metrics = response.metrics
        error_msg = response.error_message if response.error_message else None
        return {
            "image_id": response.image_id,
            "status": "ok" if response.status == inference_pb2.OK else "error",
            "predicted_label": response.predicted_label or None,
            "confidence": float(response.confidence),
            "prob_ai": float(response.prob_ai),
            # GUI / CSV schema uses "prob_real"; server returns "prob_human"
            "prob_real": float(response.prob_human),
            "preprocess_time_ms": metrics.preprocess_time_ms if metrics else None,
            "inference_time_ms": metrics.inference_time_ms if metrics else None,
            "error_message": error_msg,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify_image(
        self,
        image_bytes: bytes,
        filename: Optional[str] = None,
        image_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send image bytes to the server and return a parsed result dict.

        Parameters
        ----------
        image_bytes:
            Raw bytes of a JPG or PNG image.
        filename:
            Original filename (optional, informational).
        image_id:
            Unique identifier for this image. A UUID is generated if omitted.

        Returns
        -------
        dict with keys: ``image_id``, ``status``, ``predicted_label``,
        ``confidence``, ``prob_ai``, ``prob_real``, ``preprocess_time_ms``,
        ``inference_time_ms``, ``error_message``.

        Raises
        ------
        GRPCClientError
            If the client is not connected or if the RPC call fails.
        """
        if self._stub is None:
            raise GRPCClientError("Client is not connected")

        img_id = image_id or str(uuid.uuid4())
        try:
            request = inference_pb2.ImageRequest(
                image_id=img_id,
                filename=filename or "",
                image_data=image_bytes,
            )
            response = self._stub.ClassifyImage(request, timeout=self.timeout)
            return self._parse_response(response)
        except grpc.RpcError as rpc_err:
            LOG.exception("gRPC RpcError: %s", rpc_err)
            raise GRPCClientError(f"gRPC error: {rpc_err}") from rpc_err
        except GRPCClientError:
            raise
        except Exception as exc:
            LOG.exception("Error sending image: %s", exc)
            raise GRPCClientError(f"Error sending image: {exc}") from exc

    def close(self) -> None:
        """Close the underlying gRPC channel and release resources."""
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

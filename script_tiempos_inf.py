#!/usr/bin/env python3
"""
Medición básica de desempeño en CPU vía gRPC.

Este script:
- lee imágenes desde una carpeta local,
- envía cada imagen al servidor gRPC de inferencia,
- registra tamaño de archivo, dimensiones, etiqueta predicha y tiempos,
- guarda un CSV detallado por request,
- genera un resumen JSON y un resumen Markdown listo para informe.

No está pensado para pytest ni para `make test`.
Al dejarlo fuera de `tests/` y no nombrarlo `test_*.py`, no entra en la suite.

Ejemplo:
    uv run python script_tiempos_inf.py \
        --input-dir data/benchmark_cpu \
        --output-dir docs/evidencias/issue_29

Requisitos:
- servidor gRPC levantado en localhost:50051 (o el host/port que indiques)
- stubs generados en proto/generated
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import grpc
from PIL import Image

ROOT = Path(__file__).resolve().parent
PROTO_GEN = ROOT / "proto" / "generated"
if str(PROTO_GEN) not in sys.path:
    sys.path.insert(0, str(PROTO_GEN))

import inference_pb2  # type: ignore
import inference_pb2_grpc  # type: ignore

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class MeasurementRow:
    image_name: str
    repetition: int
    file_size_bytes: int
    width_px: int
    height_px: int
    grpc_status: str
    predicted_label: str
    confidence: float
    prob_ai: float
    prob_human: float
    preprocess_time_ms: int
    inference_time_ms: int
    total_time_ms: int
    round_trip_time_ms: float
    error_message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mide tiempos de inferencia CPU enviando imágenes al servidor gRPC."
    )
    parser.add_argument(
        "--input-dir",
        default="data/benchmark_cpu",
        help="Carpeta con imágenes a enviar. Default: data/benchmark_cpu",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/evidencias/issue_29",
        help="Carpeta donde se guardan CSV/JSON/MD. Default: docs/evidencias/issue_29",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host del servidor gRPC. Default: localhost",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="Puerto del servidor gRPC. Default: 50051",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Cuántas veces enviar cada imagen. Default: 1",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Cantidad de requests de calentamiento antes de medir. Default: 1",
    )
    return parser.parse_args()


def iter_images(input_dir: Path) -> Iterable[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de entrada: {input_dir}")

    files = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(
            f"No se encontraron imágenes válidas en {input_dir}. "
            f"Extensiones soportadas: {', '.join(sorted(VALID_EXTENSIONS))}"
        )
    return files


def image_dimensions(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def grpc_status_name(status_value: int) -> str:
    try:
        return inference_pb2.ResultStatus.Name(status_value)
    except Exception:
        return str(status_value)


def classify_once(
    stub: "inference_pb2_grpc.AiVsRealClassifierStub",
    image_path: Path,
    repetition: int,
) -> MeasurementRow:
    image_bytes = image_path.read_bytes()
    width, height = image_dimensions(image_path)

    request = inference_pb2.ImageRequest(
        image_id=f"{image_path.stem}-rep-{repetition}",
        filename=image_path.name,
        image_data=image_bytes,
    )

    t0 = time.perf_counter()
    response = stub.ClassifyImage(request)
    t1 = time.perf_counter()
    round_trip_ms = round((t1 - t0) * 1000, 3)

    return MeasurementRow(
        image_name=image_path.name,
        repetition=repetition,
        file_size_bytes=image_path.stat().st_size,
        width_px=width,
        height_px=height,
        grpc_status=grpc_status_name(response.status),
        predicted_label=response.predicted_label,
        confidence=round(float(response.confidence), 6),
        prob_ai=round(float(response.prob_ai), 6),
        prob_human=round(float(response.prob_human), 6),
        preprocess_time_ms=int(response.metrics.preprocess_time_ms),
        inference_time_ms=int(response.metrics.inference_time_ms),
        total_time_ms=int(response.metrics.total_time_ms),
        round_trip_time_ms=round_trip_ms,
        error_message=response.error_message,
    )


def compute_summary(rows: list[MeasurementRow]) -> dict:
    ok_rows = [r for r in rows if r.grpc_status == "OK"]
    error_rows = [r for r in rows if r.grpc_status != "OK"]

    def metric_stats(values: list[float]) -> dict:
        if not values:
            return {"avg": 0, "min": 0, "max": 0}
        return {
            "avg": round(statistics.mean(values), 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
        }

    summary = {
        "total_requests": len(rows),
        "ok_requests": len(ok_rows),
        "error_requests": len(error_rows),
        "preprocess_time_ms": metric_stats([r.preprocess_time_ms for r in ok_rows]),
        "inference_time_ms": metric_stats([r.inference_time_ms for r in ok_rows]),
        "server_total_time_ms": metric_stats([r.total_time_ms for r in ok_rows]),
        "client_round_trip_time_ms": metric_stats([r.round_trip_time_ms for r in ok_rows]),
        "images": [],
    }

    grouped: dict[str, list[MeasurementRow]] = {}
    for row in rows:
        grouped.setdefault(row.image_name, []).append(row)

    for image_name, image_rows in sorted(grouped.items()):
        ok_image_rows = [r for r in image_rows if r.grpc_status == "OK"]
        first = image_rows[0]
        image_summary = {
            "image_name": image_name,
            "file_size_bytes": first.file_size_bytes,
            "width_px": first.width_px,
            "height_px": first.height_px,
            "requests": len(image_rows),
            "ok_requests": len(ok_image_rows),
            "predicted_labels": sorted({r.predicted_label for r in ok_image_rows if r.predicted_label}),
            "preprocess_time_ms": metric_stats([r.preprocess_time_ms for r in ok_image_rows]),
            "inference_time_ms": metric_stats([r.inference_time_ms for r in ok_image_rows]),
            "server_total_time_ms": metric_stats([r.total_time_ms for r in ok_image_rows]),
            "client_round_trip_time_ms": metric_stats([r.round_trip_time_ms for r in ok_image_rows]),
        }
        summary["images"].append(image_summary)

    return summary


def write_csv(rows: list[MeasurementRow], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json(summary: dict, output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_markdown(summary: dict, output_md: Path, source_dir: Path) -> None:
    lines: list[str] = []
    lines.append("# Resumen de tiempos de inferencia CPU")
    lines.append("")
    lines.append(f"- Carpeta evaluada: `{source_dir}`")
    lines.append(f"- Requests totales: **{summary['total_requests']}**")
    lines.append(f"- Requests OK: **{summary['ok_requests']}**")
    lines.append(f"- Requests con error: **{summary['error_requests']}**")
    lines.append("")
    lines.append("## Promedios globales")
    lines.append("")
    lines.append(
        f"- Preprocesamiento: **{summary['preprocess_time_ms']['avg']} ms** "
        f"(min {summary['preprocess_time_ms']['min']} / max {summary['preprocess_time_ms']['max']})"
    )
    lines.append(
        f"- Inferencia: **{summary['inference_time_ms']['avg']} ms** "
        f"(min {summary['inference_time_ms']['min']} / max {summary['inference_time_ms']['max']})"
    )
    lines.append(
        f"- Total reportado por servidor: **{summary['server_total_time_ms']['avg']} ms** "
        f"(min {summary['server_total_time_ms']['min']} / max {summary['server_total_time_ms']['max']})"
    )
    lines.append(
        f"- Round trip cliente→servidor→cliente: **{summary['client_round_trip_time_ms']['avg']} ms** "
        f"(min {summary['client_round_trip_time_ms']['min']} / max {summary['client_round_trip_time_ms']['max']})"
    )
    lines.append("")
    lines.append("## Detalle por imagen")
    lines.append("")
    lines.append("| Imagen | Tamaño (bytes) | Resolución | Avg preprocess (ms) | Avg inference (ms) | Avg total servidor (ms) | Avg round trip (ms) |")
    lines.append("|---|---:|---|---:|---:|---:|---:|")
    for item in summary["images"]:
        resolution = f"{item['width_px']}x{item['height_px']}"
        lines.append(
            f"| {item['image_name']} | {item['file_size_bytes']} | {resolution} | "
            f"{item['preprocess_time_ms']['avg']} | {item['inference_time_ms']['avg']} | "
            f"{item['server_total_time_ms']['avg']} | {item['client_round_trip_time_ms']['avg']} |"
        )
    lines.append("")
    lines.append("## Texto breve para informe")
    lines.append("")
    lines.append(
        "Se ejecutó una medición básica de desempeño en CPU enviando un conjunto pequeño de imágenes "
        "al servidor gRPC del sistema. Para cada request se registraron el tamaño del archivo, la "
        "resolución de la imagen, el tiempo de preprocesamiento, el tiempo de inferencia y el tiempo "
        "total reportado por el servidor. Adicionalmente, se midió el tiempo end-to-end desde el cliente "
        "hasta recibir la respuesta. Con estos datos se calcularon promedios, mínimos y máximos como "
        "referencia de desempeño del prototipo."
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    image_paths = list(iter_images(input_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    address = f"{args.host}:{args.port}"
    rows: list[MeasurementRow] = []

    print(f"[INFO] Conectando a gRPC en {address}")
    print(f"[INFO] Leyendo imágenes desde {input_dir}")
    print(f"[INFO] Imágenes encontradas: {len(image_paths)}")
    print(f"[INFO] Repeticiones por imagen: {args.repetitions}")
    print(f"[INFO] Warmup requests: {args.warmup}")

    try:
        with grpc.insecure_channel(address) as channel:
            grpc.channel_ready_future(channel).result(timeout=10)
            stub = inference_pb2_grpc.AiVsRealClassifierStub(channel)

            warmup_image = image_paths[0]
            for i in range(args.warmup):
                _ = classify_once(stub, warmup_image, repetition=-(i + 1))
            if args.warmup:
                print(f"[INFO] Warmup completado con {warmup_image.name}")

            for image_path in image_paths:
                for rep in range(1, args.repetitions + 1):
                    row = classify_once(stub, image_path, repetition=rep)
                    rows.append(row)
                    print(
                        f"[OK] {row.image_name} rep={row.repetition} "
                        f"status={row.grpc_status} total={row.total_time_ms} ms "
                        f"rt={row.round_trip_time_ms} ms label={row.predicted_label or '-'}"
                    )

    except grpc.FutureTimeoutError:
        print(
            f"[ERROR] No se pudo conectar a {address}. "
            "Verifica que el servidor gRPC esté levantado.",
            file=sys.stderr,
        )
        return 1
    except grpc.RpcError as exc:
        print(
            f"[ERROR] gRPC RpcError: code={exc.code()} details={exc.details()}",
            file=sys.stderr,
        )
        return 1

    if not rows:
        print("[ERROR] No se registraron mediciones.", file=sys.stderr)
        return 1

    summary = compute_summary(rows)

    csv_path = output_dir / "mediciones_tiempos_cpu.csv"
    json_path = output_dir / "resumen_tiempos_cpu.json"
    md_path = output_dir / "resumen_tiempos_cpu.md"

    write_csv(rows, csv_path)
    write_json(summary, json_path)
    write_markdown(summary, md_path, input_dir)

    print("")
    print("[INFO] Archivos generados:")
    print(f"  - CSV:  {csv_path}")
    print(f"  - JSON: {json_path}")
    print(f"  - MD:   {md_path}")
    print("")
    print("[INFO] Promedios globales:")
    print(f"  - preprocess avg:   {summary['preprocess_time_ms']['avg']} ms")
    print(f"  - inference avg:    {summary['inference_time_ms']['avg']} ms")
    print(f"  - total server avg: {summary['server_total_time_ms']['avg']} ms")
    print(f"  - round trip avg:   {summary['client_round_trip_time_ms']['avg']} ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

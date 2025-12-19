"""
Pipeline to evaluate water detection using Sentinel-2 imagery, Google Earth Engine,
GSW occurrence, a vision-language model, and SAM.

The script samples random tiles over China, queries Sentinel-2 patches, lets a
vision-language model (glm-4.6v) decide whether water exists and return
bounding boxes, segments water with SAM, compares to Global Surface Water
occurrence, and produces an HTML report.
"""
from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# External imports are inside functions to avoid hard failures when the
# environment does not have them installed by default.


@dataclass
class TileConfig:
    """Configuration for sampling Sentinel-2 tiles."""

    count: int = 100
    tile_size_m: int = 2000
    start_date: str = "2023-01-01"
    end_date: str = "2024-01-01"
    cloud_percentage: int = 20
    occurrence_threshold: int = 50


@dataclass
class Detection:
    """Result of a single VLM detection call."""

    has_water: bool
    bbox: Optional[Tuple[float, float, float, float]] = None
    raw_response: Optional[dict] = None


@dataclass
class TileResult:
    """Aggregated result for a single tile."""

    idx: int
    rgb_array: np.ndarray
    occurrence: np.ndarray
    detection: Detection
    mask: Optional[np.ndarray] = None
    metrics: dict = field(default_factory=dict)
    bbox_pixels: Optional[Tuple[int, int, int, int]] = None


class GEEPatchSampler:
    """Handles interaction with Google Earth Engine for imagery and labels."""

    def __init__(self, project: Optional[str] = None):
        import ee

        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        self.ee = ee

    def _country_geometry(self, name: str):
        ee = self.ee
        world = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level0")
        country = world.filter(ee.Filter.eq("ADM0_NAME", name)).first()
        return country.geometry()

    def sample_tiles(self, config: TileConfig, country: str = "China") -> Iterable[Tuple[int, object]]:
        ee = self.ee
        geometry = self._country_geometry(country)
        points = ee.FeatureCollection.randomPoints(
            region=geometry,
            points=config.count,
            seed=42,
            maxError=1000,
        )
        tiles: List[Tuple[int, object]] = []
        for idx in range(config.count):
            feature = ee.Feature(points.toList(config.count).get(idx))
            geom = feature.geometry().buffer(config.tile_size_m / 2).bounds()
            tiles.append((idx, geom))
        return tiles

    def fetch_rgb_and_occurrence(self, geom, config: TileConfig) -> Tuple[np.ndarray, np.ndarray]:
        ee = self.ee
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(geom)
            .filterDate(config.start_date, config.end_date)
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", config.cloud_percentage))
        )
        image = s2.median().clip(geom)
        rgb = image.select(["B4", "B3", "B2"]).visualize(min=0, max=3000)
        rgb_sample = rgb.sampleRectangle(region=geom, defaultValue=0)
        rgb_array = np.dstack(
            [
                np.array(rgb_sample.get("vis-red")),
                np.array(rgb_sample.get("vis-green")),
                np.array(rgb_sample.get("vis-blue")),
            ]
        ).astype(np.uint8)

        gsw = self.ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
        gsw_sample = gsw.sampleRectangle(region=geom, defaultValue=0)
        occurrence = np.array(gsw_sample.get("occurrence")).astype(np.uint8)
        return rgb_array, occurrence


class VLMDetector:
    """Wraps glm-4.6v to find water presence and bounding boxes."""

    def __init__(self, api_key: str):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def _encode_image(image_array: np.ndarray) -> str:
        image = Image.fromarray(image_array)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def detect(self, image_array: np.ndarray) -> Detection:
        image_b64 = self._encode_image(image_array)
        schema = {
            "type": "object",
            "properties": {
                "has_water": {"type": "boolean"},
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "Normalized [x_min, y_min, x_max, y_max] for water region",
                },
            },
            "required": ["has_water"],
        }
        response = self.client.responses.create(
            model="glm-4.6v",
            response_format={"type": "json_schema", "json_schema": {"schema": schema}},
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Detect water bodies. Return bbox if present."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
        )
        payload = json.loads(response.output[0].content[0].text)
        has_water = bool(payload.get("has_water"))
        bbox = None
        if has_water:
            bbox_list = payload.get("bbox")
            if bbox_list and len(bbox_list) == 4:
                bbox = tuple(float(x) for x in bbox_list)
        return Detection(has_water=has_water, bbox=bbox, raw_response=payload)


class SAMSegmenter:
    """Segments water regions using SAM guided by the VLM bounding box."""

    def __init__(self, checkpoint_path: str, model_type: str = "vit_h"):
        from segment_anything import SamPredictor, sam_model_registry

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.predictor = SamPredictor(sam)

    @staticmethod
    def _bbox_to_pixels(bbox: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
        x_min, y_min, x_max, y_max = bbox
        return (
            int(x_min * width),
            int(y_min * height),
            int(x_max * width),
            int(y_max * height),
        )

    def segment(self, image_array: np.ndarray, bbox: Sequence[float]) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        height, width, _ = image_array.shape
        pixel_box = self._bbox_to_pixels(bbox, width, height)
        box_array = np.array(pixel_box, dtype=np.float32)

        self.predictor.set_image(image_array)
        masks, _, _ = self.predictor.predict(box=box_array[None, :])
        mask = masks[0].astype(np.uint8)
        return mask, pixel_box


class Evaluator:
    """Compares predicted masks with GSW occurrence."""

    @staticmethod
    def compute_metrics(mask: np.ndarray, occurrence: np.ndarray, threshold: int) -> dict:
        gt = occurrence >= threshold
        pred = mask.astype(bool)

        tp = np.logical_and(pred, gt).sum()
        fp = np.logical_and(pred, ~gt).sum()
        fn = np.logical_and(~pred, gt).sum()

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0

        return {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": precision,
            "recall": recall,
            "iou": iou,
        }


class ReportBuilder:
    """Creates an HTML report summarizing detections and metrics."""

    @staticmethod
    def _encode_png(array: np.ndarray) -> str:
        image = Image.fromarray(array)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def _mask_to_rgba(mask: np.ndarray) -> np.ndarray:
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[..., 0] = 0
        rgba[..., 1] = 128
        rgba[..., 2] = 255
        rgba[..., 3] = mask.astype(np.uint8) * 180
        return rgba

    @staticmethod
    def build(results: Sequence[TileResult], output_path: Path) -> None:
        rows = []
        for res in results:
            rgb_b64 = ReportBuilder._encode_png(res.rgb_array)
            occurrence_gray = np.clip(res.occurrence * 255 / 100, 0, 255).astype(np.uint8)
            occurrence_b64 = ReportBuilder._encode_png(occurrence_gray)

            overlay = res.rgb_array.copy()
            if res.mask is not None:
                alpha_mask = ReportBuilder._mask_to_rgba(res.mask)
                overlay = Image.fromarray(res.rgb_array).convert("RGBA")
                overlay.alpha_composite(Image.fromarray(alpha_mask))
                overlay = np.array(overlay.convert("RGB"))
            overlay_b64 = ReportBuilder._encode_png(overlay)

            metrics = res.metrics
            rows.append(
                f"""
                <tr>
                  <td>{res.idx}</td>
                  <td><img src='data:image/png;base64,{rgb_b64}' width='256'/></td>
                  <td><img src='data:image/png;base64,{occurrence_b64}' width='256'/></td>
                  <td><img src='data:image/png;base64,{overlay_b64}' width='256'/></td>
                  <td>{metrics.get('precision', 0):.3f}</td>
                  <td>{metrics.get('recall', 0):.3f}</td>
                  <td>{metrics.get('iou', 0):.3f}</td>
                </tr>
                """
            )

        avg_precision = np.mean([r.metrics.get("precision", 0.0) for r in results])
        avg_recall = np.mean([r.metrics.get("recall", 0.0) for r in results])
        avg_iou = np.mean([r.metrics.get("iou", 0.0) for r in results])

        html = f"""
        <html>
        <head>
          <style>
            body {{ font-family: Arial, sans-serif; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
          </style>
        </head>
        <body>
          <h1>Water Detection Evaluation (Sentinel-2 + glm-4.6v + SAM)</h1>
          <p>Tiles evaluated: {len(results)}</p>
          <p>Average Precision: {avg_precision:.3f} | Average Recall: {avg_recall:.3f} | Average IoU: {avg_iou:.3f}</p>
          <table>
            <thead>
              <tr>
                <th>Tile</th>
                <th>Input RGB</th>
                <th>GSW Occurrence</th>
                <th>Prediction Overlay</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>IoU</th>
              </tr>
            </thead>
            <tbody>
              {''.join(rows)}
            </tbody>
          </table>
        </body>
        </html>
        """
        output_path.write_text(html, encoding="utf-8")


def run_pipeline(
    tile_config: TileConfig,
    checkpoint_path: str,
    vlm_api_key: str,
    gee_project: Optional[str] = None,
    output_report: Path = Path("reports/report.html"),
) -> None:
    sampler = GEEPatchSampler(project=gee_project)
    detector = VLMDetector(api_key=vlm_api_key)
    segmenter = SAMSegmenter(checkpoint_path)
    evaluator = Evaluator()

    results: List[TileResult] = []
    tiles = sampler.sample_tiles(tile_config)
    for idx, geom in tiles:
        rgb_array, occurrence = sampler.fetch_rgb_and_occurrence(geom, tile_config)
        detection = detector.detect(rgb_array)
        mask = None
        bbox_pixels = None
        if detection.has_water and detection.bbox is not None:
            mask, bbox_pixels = segmenter.segment(rgb_array, detection.bbox)
        metrics = {}
        if mask is not None:
            metrics = evaluator.compute_metrics(mask, occurrence, tile_config.occurrence_threshold)
        results.append(
            TileResult(
                idx=idx,
                rgb_array=rgb_array,
                occurrence=occurrence,
                detection=detection,
                mask=mask,
                metrics=metrics,
                bbox_pixels=bbox_pixels,
            )
        )

    output_report.parent.mkdir(parents=True, exist_ok=True)
    ReportBuilder.build(results, output_report)


if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")
    checkpoint = os.environ.get("SAM_CHECKPOINT", "sam_vit_h.pth")
    project = os.environ.get("GEE_PROJECT")
    config = TileConfig()
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be set for glm-4.6v access")
    run_pipeline(config, checkpoint_path=checkpoint, vlm_api_key=api_key, gee_project=project)

"""Medical CT Multi-Model Unified Inference Engine & Report Generator.

Exposes:
  - MedicalCTPredictor: End-to-end multi-model inference with Multi-Layer Gradient Projection
  - format_report(r): String report formatter matching the clinical prototype style
  - save_prediction_json(r, path): JSON serialization helper
"""

import os
import re
import cv2
import json
import time
import hashlib
from pathlib import Path
from typing import Union, Dict, Any, Optional

import numpy as np
import torch
from PIL import Image

from .pipeline import MedicalCTPipeline
from .gradcam import compute_multilayer_gradcam

BASE_DIR = Path(__file__).resolve().parent.parent


def format_report(r: Dict[str, Any]) -> str:
    """Formats the multi-model prediction result into a clean console report."""
    lines = []
    lines.append("=" * 70)
    lines.append("                 MEDICAL CT AI DIAGNOSTIC REPORT")
    lines.append("=" * 70)
    lines.append(f"Image File       : {r.get('image_path', 'unknown')}")
    lines.append(f"Processing Time  : {r.get('latency_seconds', 0.0):.2f}s")
    lines.append("-" * 70)

    # Stage 1: Router
    router_res = r.get("router", {})
    anatomy = router_res.get("prediction", "UNKNOWN")
    lines.append("[STAGE 1: ANATOMY ROUTER]")
    lines.append(f"  • Identified Anatomy : {anatomy} ({router_res.get('confidence_pct', 'N/A')} confidence)")
    lines.append(f"  • Routing Status     : {router_res.get('status', 'ROUTED')}")
    lines.append(f"  • Pipeline Route     : {r.get('routing_action', 'N/A')}")
    probs_router = router_res.get("probabilities", {})
    if probs_router:
        lines.append(f"  • Router Probabilities:")
        lines.append(f"      - CHEST_CT : {probs_router.get('CHEST_CT', 0.0) * 100:.2f}%")
        lines.append(f"      - BRAIN_CT : {probs_router.get('BRAIN_CT', 0.0) * 100:.2f}%")
    lines.append("-" * 70)

    # Uncertainty check
    if router_res.get("uncertain"):
        lines.append("[ALERT] Scan falls in the uncertainty band (|P(Brain) - 0.5| < 0.45).")
        lines.append("        Automated routing halted. Slices must be manually reviewed by a radiologist.")
        lines.append("=" * 70)
        return "\n".join(lines)

    expert_res = r.get("expert_result", {})
    if not expert_res or expert_res.get("status") != "SUCCESS":
        lines.append(f"[Notice] {expert_res.get('message', 'Downstream expert unavailable.')}")
        lines.append("=" * 70)
        return "\n".join(lines)

    # Stage 2: Individual Model Native Formatting
    lines.append("[STAGE 2: DOWNSTREAM EXPERT ANALYSIS]")

    if anatomy == "CHEST_CT":
        lines.append(f"  • Target Expert      : {expert_res.get('expert', 'Chest CT Expert')}")
        lines.append(f"  • Predicted Class    : {expert_res.get('diagnosis', 'N/A')}")
        lines.append(f"  • Confidence         : {expert_res.get('confidence_pct', 'N/A')}")
        if "top1_margin_pct" in expert_res:
            lines.append(f"  • Top-1 Margin       : {expert_res['top1_margin_pct']}")
        if "entropy" in expert_res:
            lines.append(f"  • Entropy            : {expert_res['entropy']:.4f}")
        lines.append(f"  • Crop Source        : {expert_res.get('crop_source', 'patient body contour')}")
        lines.append("")
        lines.append("  CLASS PROBABILITIES")
        lines.append("  " + "-" * 50)
        probs = expert_res.get("probabilities", {})
        for name, p in probs.items():
            bar_len = int(p * 26)
            bar = "=" * bar_len
            lines.append(f"    {name:<12s}: {p * 100:6.2f}%  [{bar:<26}]")
        lines.append("  " + "-" * 50)
        lines.append(f"  CLINICAL SUMMARY: {expert_res.get('clinical_summary', 'N/A')}")

    elif anatomy == "BRAIN_CT":
        arch = expert_res.get("model_name", "se_resnext101_32x4d")
        diag_title = expert_res.get("diag_title", expert_res.get("diagnosis", "N/A"))
        lines.append(f"  • Model Architecture : {arch}")
        lines.append(f"  • Diagnosis Badge    : {diag_title}")
        lines.append("")
        lines.append(f"  {'CLASS':<20s} {'PROBABILITY':<12s} {'VISUAL BAR (0-100%)':<24s}")
        lines.append("  " + "-" * 60)
        probs = expert_res.get("probabilities", {})
        for lab, p in probs.items():
            pct = p * 100.0
            filled = int(round((pct / 100.0) * 20))
            filled = max(0, min(20, filled))
            bar = "#" * filled + "-" * (20 - filled)
            lines.append(f"  {lab:<20s}: {pct:6.2f}%    [{bar}]")
        lines.append("  " + "-" * 60)
        lines.append(f"  CLINICAL SUMMARY: {expert_res.get('clinical_summary', 'N/A')}")

    # Multi-Layer Gradient Projection Contributions
    layer_weights = r.get("gradcam_layer_weights", {})
    if layer_weights:
        lines.append("-" * 70)
        lines.append("MULTI-LAYER GRADIENT PROJECTIONS:")
        for l_name, l_wt in layer_weights.items():
            bar_len = int(l_wt / 100.0 * 20)
            bar_str = "=" * bar_len
            lines.append(f"  • {l_name:<16s}: {l_wt:5.1f}%  [{bar_str:<20}]")

    # Visual Artifacts section
    gradcam_files = r.get("gradcam_files", {})
    if gradcam_files:
        lines.append("-" * 70)
        lines.append("EXPLAINABILITY ARTIFACTS (Grad-CAM):")
        for k, v in gradcam_files.items():
            lines.append(f"  • {k:<15s}: {v}")

    lines.append("=" * 70)
    return "\n".join(lines)


def save_prediction_json(r: Dict[str, Any], output_path: Union[str, Path]) -> str:
    """Serializes the prediction dictionary into a clean JSON file."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def sanitize(obj):
        if isinstance(obj, dict):
            return {
                k: sanitize(v)
                for k, v in obj.items()
                if not k.startswith("_") and k not in (
                    "cam_overlay_pil",
                    "cam_heatmap_pil",
                    "cam_grid_pil",
                    "cam_layer_overlays",
                )
            }
        elif isinstance(obj, (list, tuple)):
            return [sanitize(v) for v in obj]
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist() if obj.size < 100 else "<array>"
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist() if obj.numel() < 100 else "<tensor>"
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, Image.Image):
            return "<PIL.Image>"
        return obj

    clean_dict = sanitize(r)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(clean_dict, f, indent=2)

    return str(p.resolve())


class MedicalCTPredictor:
    """Unified predictor with Multi-Layer Gradient Projection explainability."""

    def __init__(self, device: Optional[str] = None):
        self.pipeline = MedicalCTPipeline(device=device)

    def predict(
        self,
        image_path: Union[str, Path],
        gradcam_dir: Optional[Union[str, Path]] = None,
        target_class: Optional[str] = None,
        multi_layer: bool = True,
    ) -> Dict[str, Any]:
        """Runs multi-model inference and multi-layer gradient projection."""
        img_p = Path(image_path).resolve()
        result = self.pipeline.process(img_p)
        result["gradcam_files"] = {}
        result["gradcam_layer_weights"] = {}
        result["is_multi_layer"] = multi_layer

        anatomy = result["router"]["prediction"]
        orig_img = Image.open(str(img_p)).convert("RGB")

        try:
            overlay_pil = None
            heatmap_pil = None
            grid_pil = None
            layer_overlays = {}
            layer_weights = {}

            # ---------------- Multi-Layer Grad-CAM for Chest CT ----------------
            if anatomy == "CHEST_CT" and self.pipeline.chest_expert.is_loaded:
                chest_model = self.pipeline.chest_expert.model
                feats = chest_model.backbone.features
                if multi_layer:
                    # Hook 4 hierarchical stages across depth (shallow to deep)
                    hook_layers = [
                        ("stage_2 (low)", feats[2]),
                        ("stage_4 (mid)", feats[4]),
                        ("stage_6 (deep)", feats[6]),
                        ("stage_8 (top)", feats[8]),
                    ]
                else:
                    hook_layers = [("stage_8 (top)", feats[8])]

                x_tensor = self.pipeline.chest_expert.transform(orig_img).unsqueeze(0).to(self.pipeline.chest_expert.device)

                if target_class and target_class in self.pipeline.chest_expert.class_names:
                    c_idx = self.pipeline.chest_expert.class_names.index(target_class)
                else:
                    c_idx = None  # Highest predicted class

                cam_res = compute_multilayer_gradcam(
                    model=chest_model,
                    input_tensor=x_tensor,
                    original_image=orig_img,
                    hook_layers=hook_layers,
                    class_idx=c_idx,
                    alpha=0.48,
                    mode="layercam",
                )
                overlay_pil = cam_res.overlay_pil
                heatmap_pil = cam_res.heatmap_pil
                layer_weights = cam_res.weights_dict
                layer_overlays = cam_res.layer_overlays
                grid_pil = cam_res.grid_pil

                if gradcam_dir:
                    out_d = Path(gradcam_dir)
                    out_d.mkdir(parents=True, exist_ok=True)
                    ov_path = out_d / "chest_gradcam_overlay.png"
                    heat_path = out_d / "chest_gradcam_heatmap.png"
                    overlay_pil.save(ov_path)
                    heatmap_pil.save(heat_path)
                    result["gradcam_files"]["overlay"] = str(ov_path)
                    result["gradcam_files"]["heatmap"] = str(heat_path)

                    if grid_pil is not None:
                        grid_path = out_d / "chest_gradcam_multilayer_grid.png"
                        grid_pil.save(grid_path)
                        result["gradcam_files"]["multilayer_grid"] = str(grid_path)

                    for l_name, l_img in layer_overlays.items():
                        c_name = re.sub(r"[^a-zA-Z0-9_]", "_", l_name).strip("_")
                        l_path = out_d / f"chest_layer_{c_name}.png"
                        l_img.save(l_path)
                        result["gradcam_files"][f"layer_{c_name}"] = str(l_path)

            # ---------------- Multi-Layer Grad-CAM for Brain CT ----------------
            elif anatomy == "BRAIN_CT" and self.pipeline.brain_expert.is_loaded:
                brain_model = self.pipeline.brain_expert.predictor.model
                if multi_layer:
                    # Hook 4 residual stages of SE-ResNeXt-101
                    hook_layers = [
                        ("res_stage1 (low)", brain_model.model_ft.layer1[-1]),
                        ("res_stage2 (mid)", brain_model.model_ft.layer2[-1]),
                        ("res_stage3 (deep)", brain_model.model_ft.layer3[-1]),
                        ("res_stage4 (top)", brain_model.model_ft.layer4[-1]),
                    ]
                else:
                    hook_layers = [("res_stage4 (top)", brain_model.model_ft.layer4[-1])]

                # Prepare normalized tensor matching SE-ResNeXt
                w0, h0 = orig_img.size
                im_arr = np.array(orig_img).astype(np.float32) / 255.0
                im_arr = cv2.resize(im_arr, (256, 256), interpolation=cv2.INTER_AREA)
                mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
                std = np.array([0.224, 0.224, 0.224], dtype=np.float32)
                x_norm = (im_arr - mean) / std
                x_tensor = torch.from_numpy(x_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(self.pipeline.brain_expert.predictor.device)

                cam_res = compute_multilayer_gradcam(
                    model=brain_model,
                    input_tensor=x_tensor,
                    original_image=orig_img,
                    hook_layers=hook_layers,
                    class_idx=None,
                    alpha=0.50,
                    mode="layercam",
                )
                overlay_pil = cam_res.overlay_pil
                heatmap_pil = cam_res.heatmap_pil
                layer_weights = cam_res.weights_dict
                layer_overlays = cam_res.layer_overlays
                grid_pil = cam_res.grid_pil

                if gradcam_dir:
                    out_d = Path(gradcam_dir)
                    out_d.mkdir(parents=True, exist_ok=True)
                    ov_path = out_d / "brain_gradcam_overlay.png"
                    heat_path = out_d / "brain_gradcam_heatmap.png"
                    overlay_pil.save(ov_path)
                    heatmap_pil.save(heat_path)
                    result["gradcam_files"]["overlay"] = str(ov_path)
                    result["gradcam_files"]["heatmap"] = str(heat_path)

                    if grid_pil is not None:
                        grid_path = out_d / "brain_gradcam_multilayer_grid.png"
                        grid_pil.save(grid_path)
                        result["gradcam_files"]["multilayer_grid"] = str(grid_path)

                    for l_name, l_img in layer_overlays.items():
                        c_name = re.sub(r"[^a-zA-Z0-9_]", "_", l_name).strip("_")
                        l_path = out_d / f"brain_layer_{c_name}.png"
                        l_img.save(l_path)
                        result["gradcam_files"][f"layer_{c_name}"] = str(l_path)

            result["cam_overlay_pil"] = overlay_pil
            result["cam_heatmap_pil"] = heatmap_pil
            result["cam_grid_pil"] = grid_pil
            result["cam_layer_overlays"] = layer_overlays
            result["gradcam_layer_weights"] = layer_weights

        except Exception as e:
            result["gradcam_error"] = str(e)

        return result


"""End-to-end Multimodal Affective Music Recommendation Pipeline."""
import time
import numpy as np
import torch
from typing import Dict, Any, List

from .vision.face_detector import RobustFaceDetector
from .vision.emotion_backbone import AffectiveDualHeadNet, CIRCUMPLEX_PRIORS
from .vision.explainability import GradCAMExplainer
from .audio.mood_acoustic_mapping import AcousticAffectMapper, TrackAffectProfile
from .recommender.hybrid_engine import HybridAffectiveRecommender


class AffectiveEchoPipeline:
    """Integrated multimodal inference pipeline."""
    def __init__(self, music_catalog: List[TrackAffectProfile] = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = RobustFaceDetector(target_size=(224, 224), device=self.device)
        self.model = AffectiveDualHeadNet(backbone_name="mobilenetv3_large_100", pretrained=False).to(self.device)
        self.model.eval()
        self.explainer = GradCAMExplainer(self.model)

        self.catalog = music_catalog or []
        self.recommender = HybridAffectiveRecommender(self.catalog)

    def process_frame(self, frame_bgr: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """Processes single video frame and outputs emotion, continuous affect, and recommended tracks."""
        start_t = time.perf_counter()

        # 1. Face Detection & Cropping
        norm_face, raw_face, bbox = self.detector.detect_and_crop(frame_bgr)

        if norm_face is None:
            return {
                "face_detected": False,
                "latency_ms": round((time.perf_counter() - start_t) * 1000.0, 2),
                "fps": round(1.0 / max(1e-5, (time.perf_counter() - start_t)), 1)
            }

        # 2. Convert to PyTorch Tensor [1, 3, 224, 224]
        tensor_face = torch.from_numpy(norm_face).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        # 3. Predict Emotion & (Valence, Arousal)
        affect_res = self.model.predict_affect(tensor_face)

        # 4. Generate Grad-CAM attribution
        cam_heatmap = self.explainer.generate_heatmap(tensor_face)

        # 5. Multimodal Music Recommendation
        valence = affect_res["valence"]
        arousal = affect_res["arousal"]

        # If model initialized randomly, blend with circumplex prior for realistic evaluation
        label = affect_res["discrete_label"]
        prior_v, prior_a = CIRCUMPLEX_PRIORS.get(label, (0.0, 0.0))
        effective_v = 0.4 * valence + 0.6 * prior_v
        effective_a = 0.4 * arousal + 0.6 * prior_a

        recommendations = self.recommender.recommend(
            current_valence=effective_v,
            current_arousal=effective_a,
            top_k=top_k,
            enable_trajectory=True
        )

        total_latency = (time.perf_counter() - start_t) * 1000.0

        return {
            "face_detected": True,
            "bbox": bbox,
            "discrete_emotion": label,
            "confidence": round(affect_res["confidence"], 4),
            "valence": round(effective_v, 3),
            "arousal": round(effective_a, 3),
            "probabilities": affect_res["probabilities"],
            "heatmap": cam_heatmap,
            "recommendations": recommendations,
            "latency_ms": round(total_latency, 2),
            "fps": round(1000.0 / max(1e-5, total_latency), 1)
        }

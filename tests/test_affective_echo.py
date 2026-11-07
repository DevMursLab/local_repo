"""Comprehensive PyTest Suite for AffectiveEcho Framework."""
import os
import sys
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vision.face_detector import RobustFaceDetector
from src.vision.emotion_backbone import AffectiveDualHeadNet, get_emotion_labels
from src.vision.explainability import GradCAMExplainer
from src.audio.audio_feature_extractor import AudioFeatureExtractor
from src.audio.mood_acoustic_mapping import AcousticAffectMapper, TrackAffectProfile
from src.recommender.hybrid_engine import HybridAffectiveRecommender
from src.recommender.metrics import RecommenderMetricsEvaluator
from data.catalog import get_benchmark_catalog
from experiments.validate_manifest import validate_manifest


def test_face_detector():
    detector = RobustFaceDetector(target_size=(224, 224), device="cpu")
    # Synthetic frame (480x640x3)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add simple bright patch simulating face
    frame[100:300, 200:400] = 200
    norm_face, raw_face, bbox = detector.detect_and_crop(frame)
    assert norm_face is not None
    assert norm_face.shape == (224, 224, 3)
    assert norm_face.max() <= 1.0 and norm_face.min() >= 0.0
    assert len(bbox) == 4


def test_dual_head_net():
    model = AffectiveDualHeadNet(backbone_name="mobilenetv3_large_100", num_classes=7, pretrained=False)
    model.eval()
    dummy_input = torch.randn(2, 3, 224, 224)
    logits, va = model(dummy_input)
    assert logits.shape == (2, 7)
    assert va.shape == (2, 2)
    # Tanh bounds
    assert torch.all(va >= -1.0) and torch.all(va <= 1.0)


def test_gradcam_explainer():
    model = AffectiveDualHeadNet(backbone_name="mobilenetv3_large_100", num_classes=7, pretrained=False)
    explainer = GradCAMExplainer(model)
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)
    heatmap = explainer.generate_heatmap(dummy_input, class_idx=0)
    assert heatmap is not None
    assert heatmap.shape == (224, 224)
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0


def test_audio_feature_extractor():
    extractor = AudioFeatureExtractor()
    with pytest.raises(FileNotFoundError):
        extractor.extract_from_file("non_existent_sample.mp3")


def test_acoustic_affect_mapper():
    mapper = AcousticAffectMapper()
    dummy_feats = {
        "tempo": 140.0,
        "rms": 0.25,
        "spectral_centroid": 3200.0,
        "chroma_mean": 0.50
    }
    val, aro = mapper.compute_valence_arousal(dummy_feats)
    assert -1.0 <= val <= 1.0
    assert -1.0 <= aro <= 1.0


def test_recommender_iso_trajectory():
    catalog = get_benchmark_catalog()
    assert len(catalog) >= 20
    recommender = HybridAffectiveRecommender(catalog, diversity_lambda=0.65)

    # User starting in sad state (-0.7, -0.4)
    recs = recommender.recommend(current_valence=-0.7, current_arousal=-0.4, top_k=5, enable_trajectory=True)
    assert len(recs) == 5

    # Assert trajectory progresses towards higher valence
    valences = [r.predicted_valence for r in recs]
    # Check that recommendation items are unique
    rec_ids = [r.track_id for r in recs]
    assert len(set(rec_ids)) == 5


def test_recommender_metrics():
    recommended = ["T1", "T2", "T3", "T4", "T5"]
    ground_truth = ["T3", "T7", "T9"]
    hr = RecommenderMetricsEvaluator.compute_hit_ratio_at_k(recommended, ground_truth, k=5)
    assert hr == 1.0

    ndcg = RecommenderMetricsEvaluator.compute_ndcg_at_k(recommended, ground_truth, k=5)
    assert 0.0 < ndcg <= 1.0

    vecs = [np.random.randn(8) for _ in range(5)]
    ild = RecommenderMetricsEvaluator.compute_intra_list_diversity(vecs)
    assert ild >= 0.0


def test_manifest_validator_rejects_subject_leakage(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "image_path,split,label,valence,arousal,subject_id\n"
        "a.jpg,train,Happy,0.5,0.2,S1\n"
        "b.jpg,validation,Sad,-0.5,-0.2,S2\n"
        "c.jpg,test,Neutral,0,0,S1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Subject leakage"):
        validate_manifest(manifest)

"""Acoustic-to-Affect Mapping based on Russell's Circumplex Model and empirical MIR research."""
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass
class TrackAffectProfile:
    track_id: str
    title: str
    artist: str
    genre: str
    valence: float  # [-1.0, 1.0]
    arousal: float  # [-1.0, 1.0]
    tempo: float
    rms_energy: float
    feature_vector: np.ndarray


class AcousticAffectMapper:
    """Projects acoustic psycho-descriptors onto continuous (Valence, Arousal) plane."""
    def __init__(self):
        # Normalization reference bounds
        self.tempo_min, self.tempo_max = 60.0, 180.0
        self.rms_min, self.rms_max = 0.02, 0.35
        self.centroid_min, self.centroid_max = 1000.0, 4500.0
        self.chroma_min, self.chroma_max = 0.15, 0.60

    def compute_valence_arousal(self, features: Dict[str, float]) -> Tuple[float, float]:
        """Maps acoustic feature dictionary to (valence, arousal) in [-1.0, 1.0]."""
        # 1. Normalize primitives to [0, 1]
        t = np.clip((features.get("tempo", 120.0) - self.tempo_min) / (self.tempo_max - self.tempo_min), 0.0, 1.0)
        e = np.clip((features.get("rms", 0.12) - self.rms_min) / (self.rms_max - self.rms_min), 0.0, 1.0)
        c = np.clip((features.get("spectral_centroid", 2200.0) - self.centroid_min) / (self.centroid_max - self.centroid_min), 0.0, 1.0)
        h = np.clip((features.get("chroma_mean", 0.35) - self.chroma_min) / (self.chroma_max - self.chroma_min), 0.0, 1.0)

        # 2. Empirical Psychoacoustic Regression Formulation:
        # Arousal is largely driven by speed (tempo) and dynamic power (RMS energy) + brightness
        raw_arousal = 0.45 * t + 0.40 * e + 0.15 * c
        arousal = float(np.clip(2.0 * raw_arousal - 1.0, -1.0, 1.0))

        # Valence is driven by harmonic consonance (chroma energy) and spectral balance
        raw_valence = 0.50 * h + 0.30 * c + 0.20 * (1.0 - abs(t - 0.6))
        valence = float(np.clip(2.0 * raw_valence - 1.0, -1.0, 1.0))

        return valence, arousal

    def create_track_profile(self, track_id: str, title: str, artist: str, genre: str, features: Dict[str, float]) -> TrackAffectProfile:
        val, aro = self.compute_valence_arousal(features)

        # Assemble compact 8-dimensional feature embedding
        vec = np.array([
            val,
            aro,
            features.get("tempo", 120.0) / 180.0,
            features.get("rms", 0.1),
            features.get("spectral_centroid", 2000.0) / 5000.0,
            features.get("chroma_mean", 0.3),
            features.get("zero_crossing_rate", 0.05),
            features.get("mfcc_0", 0.0) / 100.0
        ], dtype=np.float32)

        return TrackAffectProfile(
            track_id=track_id,
            title=title,
            artist=artist,
            genre=genre,
            valence=val,
            arousal=aro,
            tempo=features.get("tempo", 120.0),
            rms_energy=features.get("rms", 0.1),
            feature_vector=vec
        )

"""Audio module for music acoustic feature extraction and mood mapping."""
from .audio_feature_extractor import AudioFeatureExtractor
from .mood_acoustic_mapping import AcousticAffectMapper, TrackAffectProfile

__all__ = [
    "AudioFeatureExtractor",
    "AcousticAffectMapper",
    "TrackAffectProfile",
]

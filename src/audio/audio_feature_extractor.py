"""Audio Feature Extractor using Librosa and acoustic psycho-acoustics."""
import os
import numpy as np
from typing import Dict, Optional

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


class AudioFeatureExtractor:
    """Extracts psycho-acoustic descriptors for emotion-aware music retrieval."""
    def __init__(self, sample_rate: int = 22050, n_mfcc: int = 13):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc

    def extract_from_file(self, audio_path: str) -> Dict[str, float]:
        """Extract multi-dimensional acoustic vector from an audio file.

        Args:
            audio_path: Path to mp3/wav/flac

        Returns:
            Dictionary of scalar summary features (mean MFCCs, tempo, rms, spectral centroid, etc.)
        """
        if not HAS_LIBROSA:
            raise RuntimeError("librosa is required for audio feature extraction; install it before running an experiment.")
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        try:
            # Load 30-second central snippet for efficiency
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=30.0, offset=10.0)
            if len(y) == 0:
                y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=30.0)

            # 1. Energy & Loudness
            rms = float(np.mean(librosa.feature.rms(y=y)))

            # 2. Spectral Brightness
            spec_cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            spec_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

            # 3. Timbral MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            mfcc_means = {f"mfcc_{i}": float(np.mean(mfccs[i])) for i in range(self.n_mfcc)}

            # 4. Tempo / Rhythm
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo_arr = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
            tempo = float(tempo_arr[0]) if len(tempo_arr) > 0 else 120.0

            # 5. Chroma / Harmonic Energy
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = float(np.mean(chroma))

            features = {
                "rms": rms,
                "tempo": tempo,
                "spectral_centroid": spec_cent,
                "spectral_rolloff": spec_rolloff,
                "zero_crossing_rate": zcr,
                "chroma_mean": chroma_mean,
                **mfcc_means
            }
            return features
        except Exception as exc:
            raise RuntimeError(f"Feature extraction failed for {audio_path}") from exc

"""Hybrid Affective Music Recommender Engine with Mood-Trajectory Optimization."""
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from ..audio.mood_acoustic_mapping import TrackAffectProfile


@dataclass
class RecommendationItem:
    rank: int
    track_id: str
    title: str
    artist: str
    genre: str
    predicted_valence: float
    predicted_arousal: float
    alignment_score: float
    target_step_valence: float


class HybridAffectiveRecommender:
    """Multimodal Recommender Engine implementing Iso-Principle Mood-Trajectory Regulation
    and Maximal Marginal Relevance (MMR) diversification.
    """
    def __init__(self, catalog: List[TrackAffectProfile], diversity_lambda: float = 0.65):
        self.catalog = catalog
        self.diversity_lambda = diversity_lambda

    def recommend(
        self,
        current_valence: float,
        current_arousal: float,
        target_emotion: str = "Positive_Calm",
        top_k: int = 5,
        enable_trajectory: bool = True
    ) -> List[RecommendationItem]:
        """Generate a personalized playlist aligned to user affective state.

        Args:
            current_valence: User detected face valence [-1.0, 1.0]
            current_arousal: User detected face arousal [-1.0, 1.0]
            target_emotion: Desired destination state (default: mildly positive & relaxed)
            top_k: Number of songs to recommend
            enable_trajectory: Apply Iso-principle gradual mood transition

        Returns:
            List of RecommendationItem
        """
        if not self.catalog:
            return []

        # Target attractor state in Valence-Arousal plane
        target_val = 0.60
        target_aro = 0.20

        # Construct mood pathway targets for each rank slot 0 to top_k-1
        u_start = np.array([current_valence, current_arousal], dtype=np.float32)
        u_target = np.array([target_val, target_aro], dtype=np.float32)

        candidates = list(self.catalog)
        selected: List[RecommendationItem] = []

        for step in range(top_k):
            # Compute interpolation parameter tau in [0, 1]
            if enable_trajectory and top_k > 1:
                tau = float(step) / float(top_k - 1)
            else:
                tau = 0.0  # purely matching current state

            # Current step desired ideal mood vector
            step_target = (1.0 - tau) * u_start + tau * u_target

            best_cand = None
            best_score = -float("inf")
            best_idx = -1

            for idx, cand in enumerate(candidates):
                cand_vec = np.array([cand.valence, cand.arousal], dtype=np.float32)

                # 1. Relevance: Euclidean / Cosine affinity to current trajectory step
                dist = np.linalg.norm(cand_vec - step_target)
                relevance = float(np.exp(- (dist ** 2) / (2 * 0.45 ** 2)))

                # 2. Diversity penalty against already selected tracks (MMR formulation)
                if selected:
                    redundancy = max(
                        float(np.dot(cand.feature_vector, s_prof.feature_vector) /
                              (np.linalg.norm(cand.feature_vector) * np.linalg.norm(s_prof.feature_vector) + 1e-8))
                        for s_prof in [self._find_cand(s.track_id) for s in selected]
                    )
                else:
                    redundancy = 0.0

                mmr_score = self.diversity_lambda * relevance - (1.0 - self.diversity_lambda) * redundancy

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_cand = cand
                    best_idx = idx

            if best_cand is not None:
                selected.append(RecommendationItem(
                    rank=step + 1,
                    track_id=best_cand.track_id,
                    title=best_cand.title,
                    artist=best_cand.artist,
                    genre=best_cand.genre,
                    predicted_valence=best_cand.valence,
                    predicted_arousal=best_cand.arousal,
                    alignment_score=round(float(best_score), 4),
                    target_step_valence=round(float(step_target[0]), 3)
                ))
                candidates.pop(best_idx)

        return selected

    def _find_cand(self, track_id: str) -> TrackAffectProfile:
        for c in self.catalog:
            if c.track_id == track_id:
                return c
        return self.catalog[0]

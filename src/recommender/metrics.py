"""Quantitative Metrics Evaluator for Affective Recommendation Systems."""
from typing import List, Dict
import numpy as np


class RecommenderMetricsEvaluator:
    """Evaluates top-K recommendations across standard Information Retrieval & RecSys metrics."""

    @staticmethod
    def compute_hit_ratio_at_k(recommended_ids: List[str], ground_truth_ids: List[str], k: int = 5) -> float:
        """Hit Ratio@K."""
        top_k = recommended_ids[:k]
        hits = any(item_id in ground_truth_ids for item_id in top_k)
        return 1.0 if hits else 0.0

    @staticmethod
    def compute_ndcg_at_k(recommended_ids: List[str], ground_truth_ids: List[str], k: int = 5) -> float:
        """Normalized Discounted Cumulative Gain@K."""
        top_k = recommended_ids[:k]
        dcg = 0.0
        for i, item_id in enumerate(top_k):
            if item_id in ground_truth_ids:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because i starts at 0

        # Ideal DCG
        ideal_hits = min(len(ground_truth_ids), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
        if idcg == 0.0:
            return 0.0
        return float(dcg / idcg)

    @staticmethod
    def compute_mrr(recommended_ids: List[str], ground_truth_ids: List[str]) -> float:
        """Mean Reciprocal Rank."""
        for i, item_id in enumerate(recommended_ids):
            if item_id in ground_truth_ids:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def compute_intra_list_diversity(recommended_vectors: List[np.ndarray]) -> float:
        """Intra-List Diversity (ILD) using pairwise cosine distance."""
        n = len(recommended_vectors)
        if n <= 1:
            return 0.0
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                v1, v2 = recommended_vectors[i], recommended_vectors[j]
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 > 0 and norm2 > 0:
                    cos_sim = np.dot(v1, v2) / (norm1 * norm2)
                    distances.append(1.0 - cos_sim)
                else:
                    distances.append(1.0)
        return float(np.mean(distances)) if distances else 0.0

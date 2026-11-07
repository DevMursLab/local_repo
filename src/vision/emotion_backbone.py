"""AffectiveDualHeadNet: Deep Multimodal Backbone for Facial Emotion & Continuous Valence-Arousal."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List
import timm

EMOTION_LABELS: List[str] = [
    "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"
]

CIRCUMPLEX_PRIORS: Dict[str, Tuple[float, float]] = {
    "Happy":    (0.81, 0.51),
    "Surprise": (0.40, 0.67),
    "Neutral":  (0.00, 0.00),
    "Sad":      (-0.63, -0.27),
    "Fear":     (-0.64, 0.60),
    "Angry":    (-0.51, 0.59),
    "Disgust":  (-0.60, 0.35)
}


def get_emotion_labels() -> List[str]:
    return list(EMOTION_LABELS)


class AffectiveDualHeadNet(nn.Module):
    """Deep Neural Backbone providing simultaneous categorical classification
    and continuous 2D Valence-Arousal projection.
    """
    def __init__(self, backbone_name: str = "mobilenetv3_large_100", num_classes: int = 7, pretrained: bool = False):
        super().__init__()
        self.num_classes = num_classes

        # Load vision backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=0.2
        )

        # Disable in-place activations to ensure autograd hooks compatibility
        for m in self.backbone.modules():
            if hasattr(m, 'inplace'):
                m.inplace = False

        # Determine output features dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            in_features = self.backbone(dummy).shape[1]

        # Shared bottleneck representation
        self.shared_proj = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3)
        )

        # Head 1: Categorical Emotion Classifier (Softmax logits)
        self.categorical_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, num_classes)
        )

        # Head 2: Continuous Dimensional Head (Valence, Arousal in [-1, 1])
        self.dimensional_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        embed = self.shared_proj(feats)
        logits = self.categorical_head(embed)
        va_coords = self.dimensional_head(embed)
        return logits, va_coords

    def predict_affect(self, x: torch.Tensor) -> Dict[str, any]:
        self.eval()
        with torch.no_grad():
            logits, va = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            pred_idx = torch.argmax(probs, dim=-1).item()
            pred_label = EMOTION_LABELS[pred_idx]
            confidence = probs[0, pred_idx].item()
            valence = va[0, 0].item()
            arousal = va[0, 1].item()

            return {
                "discrete_label": pred_label,
                "confidence": confidence,
                "probabilities": {label: probs[0, i].item() for i, label in enumerate(EMOTION_LABELS)},
                "valence": valence,
                "arousal": arousal
            }

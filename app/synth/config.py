"""Generator configurations.

Two generator *families* exist so accuracy can be reported honestly (see PLAN.md):

  TRAIN_CONFIG   the family the models train and early-stop on (train/val/test splits
                 differ only by seed namespace, i.e. in-distribution).
  HELDOUT_CONFIG a deliberately different generator (wider sizes, lighter/darker ink,
                 unseen paper textures, an extra elastic warp) that NO model ever trains
                 on. Metrics on this split measure cross-generator generalization and are
                 the headline number; in-distribution `test` is reported only for contrast.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GeneratorConfig:
    name: str
    band_h: tuple[int, int]  # band cell height range (px)
    pitch_jitter: float  # per-char horizontal jitter, fraction of pitch
    ink_value: tuple[int, int]  # ink darkness range (0=black)
    tint_p: float  # prob of a faint security tint on the paper
    elastic: bool  # apply an elastic warp the training family never sees
    tier_weights: dict[str, float] = field(
        default_factory=lambda: {"clean": 0.34, "medium": 0.33, "hard": 0.33}
    )


TRAIN_CONFIG = GeneratorConfig(
    name="train",
    band_h=(44, 56),
    pitch_jitter=0.03,
    ink_value=(10, 45),
    tint_p=0.25,
    elastic=False,
)

HELDOUT_CONFIG = GeneratorConfig(
    name="heldout",
    band_h=(38, 64),
    pitch_jitter=0.06,
    ink_value=(5, 70),
    tint_p=0.55,
    elastic=True,
)

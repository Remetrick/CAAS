from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RiskThresholds:
    low_max: int = 30
    medium_max: int = 65

    def get_risk_level(self, score_0_100: float) -> str:
        if score_0_100 <= self.low_max:
            return "Bajo"
        if score_0_100 <= self.medium_max:
            return "Medio"
        return "Alto"

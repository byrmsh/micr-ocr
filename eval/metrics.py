"""Recognition metrics: character error rate, full-line exact match, per-field accuracy.

Per-field accuracy parses both prediction and ground truth into MICR fields, so it credits
a read that gets the routing/account right even if a delimiter is off, the metric a check-
processing client actually cares about.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from app.synth.micr import parse_fields

FIELDS = ("routing", "account", "check_number", "amount")


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass
class Accumulator:
    n: int = 0
    exact: int = 0
    edit: int = 0
    chars: int = 0
    field_correct: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    field_total: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, pred: str, gt: str, gt_fields: dict[str, str | None]) -> None:
        self.n += 1
        self.exact += int(pred == gt)
        self.edit += levenshtein(pred, gt)
        self.chars += max(1, len(gt))
        pred_fields = parse_fields(pred)
        for f in FIELDS:
            if gt_fields.get(f) is not None:
                self.field_total[f] += 1
                self.field_correct[f] += int(pred_fields.get(f) == gt_fields.get(f))

    def summary(self) -> dict:
        out = {
            "n": self.n,
            "exact_match": self.exact / self.n if self.n else 0.0,
            "cer": self.edit / self.chars if self.chars else 0.0,
        }
        for f in FIELDS:
            tot = self.field_total[f]
            out[f"acc_{f}"] = self.field_correct[f] / tot if tot else None
        return out


def format_table(by_tier: dict[str, dict]) -> str:
    cols = ["n", "exact_match", "cer", "acc_routing", "acc_account", "acc_check_number"]
    head = f"{'split':10}" + "".join(f"{c:>16}" for c in cols)
    lines = [head, "-" * len(head)]
    for tier, s in by_tier.items():
        row = f"{tier:10}"
        for c in cols:
            v = s.get(c)
            row += f"{v:>16}" if isinstance(v, int) or v is None else f"{v:>16.4f}"
        lines.append(row)
    return "\n".join(lines)

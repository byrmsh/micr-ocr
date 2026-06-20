"""Compose plausible US-check MICR lines.

Produces a render string (with blank inter-field cells) and a label string (inked tokens
only, what the recognizer must output), plus the parsed fields for per-field evaluation.
Field grammar is randomized within real conventions: a Transit-bracketed 9-digit routing
number with a valid ABA check digit, an account number (sometimes dashed), a check number,
and optionally an auxiliary on-us check field at the front and a cleared-amount field at
the right. Symbols, not spaces, are the real delimiters, so the label carries no spaces.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .glyphs import AMOUNT, DASH, ONUS, TRANSIT

_ABA_WEIGHTS = (3, 7, 1, 3, 7, 1, 3, 7, 1)


@dataclass(frozen=True)
class MicrLine:
    render_text: str  # includes ' ' blank cells for inter-field gaps
    label: str  # inked tokens only (recognizer ground truth)
    routing: str
    account: str
    check_number: str
    amount: str | None


def aba_check_digit(first8: str) -> str:
    """Ninth routing digit so the ABA weighted checksum is 0 mod 10."""
    rest = sum(_ABA_WEIGHTS[i] * int(first8[i]) for i in range(8))
    return str((-rest) % 10)


def random_routing(rng: random.Random) -> str:
    # First two digits 00-12 / 21-32 / 61-72 etc. in reality; any 8 digits read fine
    # for a synthetic benchmark, and the check digit keeps it ABA-valid.
    first8 = "".join(str(rng.randint(0, 9)) for _ in range(8))
    return first8 + aba_check_digit(first8)


def random_account(rng: random.Random) -> str:
    n = rng.randint(6, 12)
    digits = "".join(str(rng.randint(0, 9)) for _ in range(n))
    if rng.random() < 0.15:  # occasional dashed account number
        cut = rng.randint(2, n - 2)
        digits = digits[:cut] + DASH + digits[cut:]
    return digits


def random_micr_line(rng: random.Random) -> MicrLine:
    routing = random_routing(rng)
    account = random_account(rng)
    check_number = "".join(str(rng.randint(0, 9)) for _ in range(rng.randint(3, 4)))
    aux = rng.random() < 0.35  # business-style auxiliary on-us at front
    amount_val = rng.random() < 0.40  # cleared checks carry an amount field at the right

    parts: list[str] = []
    if aux:
        parts.append(f"{ONUS}{check_number}{ONUS}")
    parts.append(f"{TRANSIT}{routing}{TRANSIT}")
    parts.append(f"{account}{ONUS}")
    if not aux:
        parts.append(check_number)

    amount: str | None = None
    if amount_val:
        amount = f"{rng.randint(0, 9_999_999):010d}"
        parts.append(f"{AMOUNT}{amount}{AMOUNT}")

    render_text = " ".join(parts)
    label = render_text.replace(" ", "")
    return MicrLine(
        render_text=render_text,
        label=label,
        routing=routing,
        account=account.replace(DASH, ""),
        check_number=check_number,
        amount=amount,
    )


def parse_fields(label: str) -> dict[str, str | None]:
    """Best-effort parse of a (possibly mis-recognized) token string into MICR fields.

    Tolerant by design: a recognizer error in one field should not blank the others.
    Mirrors the grammar in random_micr_line; returns None for a field it cannot locate.
    """
    routing = account = check_number = amount = None

    m = re.search(rf"{TRANSIT}(\d+){TRANSIT}", label)
    if m:
        routing = m.group(1)
    m = re.search(rf"{TRANSIT}\d+{TRANSIT}([0-9{DASH}]+){ONUS}", label)
    if m:
        account = m.group(1).replace(DASH, "")
    m = re.search(rf"{AMOUNT}(\d+){AMOUNT}", label)
    if m:
        amount = m.group(1)

    aux = re.match(rf"^{ONUS}(\d+){ONUS}", label)
    if aux:
        check_number = aux.group(1)
    else:  # trailing check number after the account's on-us, before any amount field
        tail = re.search(rf"{ONUS}(\d+)(?:{AMOUNT}|$)", label)
        if tail:
            check_number = tail.group(1)

    return {"routing": routing, "account": account, "check_number": check_number, "amount": amount}

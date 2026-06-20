import random

from app.synth.micr import aba_check_digit, parse_fields, random_micr_line


def test_aba_check_digit_known_routings():
    # Real routing numbers: 011000015 (FRB Boston), 021000021 (JPMorgan Chase).
    assert aba_check_digit("01100001") == "5"
    assert aba_check_digit("02100002") == "1"


def test_generated_routing_is_aba_valid():
    for s in range(100):
        line = random_micr_line(random.Random(s))
        assert len(line.routing) == 9
        assert aba_check_digit(line.routing[:8]) == line.routing[8]


def test_label_is_render_without_spaces():
    for s in range(100):
        line = random_micr_line(random.Random(s))
        assert line.label == line.render_text.replace(" ", "")


def test_parse_fields_roundtrip():
    for s in range(300):
        line = random_micr_line(random.Random(s))
        f = parse_fields(line.label)
        assert f["routing"] == line.routing
        assert f["account"] == line.account
        assert f["check_number"] == line.check_number
        assert f["amount"] == line.amount

"""Zufallsquelle: opr_rng.py ist die geprüfte Kopie der Vorlage (SplitMix64, plattformfeste Normalzahlen); bekannte Folge und Eigenschaften."""

from pathlib import Path

import pytest

import opr_rng as R

TEMPLATE = Path(__file__).resolve().parents[2] / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"


def test_kopie_gleicht_der_vorlage():
    if not TEMPLATE.exists():
        pytest.skip("Vorlage liegt nur im Projects-Ordner")
    assert Path(R.__file__).read_text(encoding="utf-8") == TEMPLATE.read_text(encoding="utf-8")


def test_bekannte_folge_von_splitmix64():
    """Referenzfolge des SplitMix64 (Vigna) für den Zufalls-Seed 1234567."""
    g = R.SplitMix64(1234567)
    assert [g.next_u64() for _ in range(3)] == [6457827717110365317, 3203168211198807973, 9817491932198370423]


def test_gleicher_seed_gleiche_folge_und_zufall_im_bereich():
    a, b = R.SplitMix64(5), R.SplitMix64(5)
    assert [a.random() for _ in range(5)] == [b.random() for _ in range(5)]
    g = R.SplitMix64(9)
    assert all(0.0 <= g.random() < 1.0 for _ in range(200)) and {g.below(3) for _ in range(200)} == {0, 1, 2}
    items = list(range(8))
    g.shuffle(items)
    assert sorted(items) == list(range(8))


def test_exakte_funktionen_gegen_mathe_bibliothek_im_rahmen_der_genauigkeit():
    import math
    for x in (0.001, 0.3, 1.0, 7.5):
        assert R.exact_log(x) == pytest.approx(math.log(x), rel=1e-15)
    for x in (0.0, 0.5, 3.0, 6.0):
        assert R.exact_cos(x) == pytest.approx(math.cos(x), abs=1e-14)

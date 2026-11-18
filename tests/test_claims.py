"""Claims-Tests: die Zahlen der Tabelle „Befunde“ der README, als Band (Test `test_claim_<Nr>`).

Zählungen als Band (± 3), Anteile und Prozentwerte als Band um den Wert des Baus, Ganzzahlen und Brüche exakt. Die Zahlen stammen aus den vorgerechneten Tabellen (data/*.json, Zufall
SplitMix64) und aus kleinen Neu-Rechnungen (exakte Zählungen, Ordnungsbedingung). Voraussetzungen wie in der README: Modell, Maß A, Termine nach den Mittelwerten, Mittel über 30 Instanzen je Zelle,
2 000 Szenarien, Zähleinheit Instanzen."""

from fractions import Fraction as F
from functools import lru_cache

import numpy as np
import pytest

import opr_analysis as AN
import opr_constants as C
import opr_evaluation as ev
import opr_exact as X
import opr_frozen as FR
import opr_kern as K


@lru_cache(maxsize=None)
def cell(n, level, r, ci=False):
    return ev.summarize(FR.o1_cell(n, level, r), ci=ci)


LMH = [(n, lv, r) for n in C.FROZEN_NS for lv in C.LEVELS for r in C.RS]


def test_claim_01_delta_zaehlung_und_beispiele():
    d = {k: cell(*k)["delta"] for k in LMH}
    assert len(d) == 60 and all(v < 0 for v in d.values())
    assert min(d.values()) * 100 == pytest.approx(-12.9, abs=0.5) and max(d.values()) * 100 == pytest.approx(-3.1, abs=0.5)
    assert min(d, key=d.get) == (8, "L", 0.5) and max(d, key=d.get) == (5, "L", 8.0)
    better = [cell(*k)["better"] for k in LMH]
    assert 17 <= min(better) <= 23 and max(better) == 30
    assert 2 <= max(cell(*k)["worse"] for k in LMH) <= 4
    assert 16 <= min(cell(*k)["better_B"] for k in LMH) <= 22 and max(cell(*k)["better_B"] for k in LMH) == 30
    for n, delta, lo, hi in ((5, -7.3, -9.7, -5.1), (7, -7.3, -8.9, -5.7), (8, -7.1, -8.8, -5.5), (12, -8.2, -9.4, -7.1)):
        s = cell(n, "M", 2.0, True)
        assert s["delta"] * 100 == pytest.approx(delta, abs=0.6)
        assert s["delta_lo"] * 100 == pytest.approx(lo, abs=0.9) and s["delta_hi"] * 100 == pytest.approx(hi, abs=0.9)
    for n in C.FROZEN_NS:                                           # Vorsprung (|Δ|) bei r = 8 in allen 12 Kombinationen aus n und Stufe am kleinsten
        for lv in C.LEVELS:
            by_r = {r: abs(cell(n, lv, r)["delta"]) for r in C.RS}
            assert min(by_r, key=by_r.get) == 8.0, (n, lv)


def test_claim_02_luecken_und_optimum_gleich_streuungsregel():
    small = [cell(n, lv, r) for n in (5, 7) for lv in C.LEVELS for r in C.RS]
    big = [cell(n, lv, r) for n in (8, 12) for lv in C.LEVELS for r in C.RS]
    pc = lambda cells, key, f: f(c[key] for c in cells) * 100
    assert pc(small, "gap_sd", min) == pytest.approx(0.05, abs=0.03) and pc(small, "gap_sd", max) == pytest.approx(0.27, abs=0.05)
    assert pc(small, "gap_spt", min) == pytest.approx(3.5, abs=0.5) and pc(small, "gap_spt", max) == pytest.approx(14.1, abs=0.8)
    assert pc(big, "gap_sd", min) == pytest.approx(0.12, abs=0.04) and pc(big, "gap_sd", max) == pytest.approx(0.38, abs=0.05)
    assert pc(big, "gap_spt", min) == pytest.approx(5.0, abs=0.5) and pc(big, "gap_spt", max) == pytest.approx(15.7, abs=0.8)
    o = [cell(n, lv, r)["opt_is_sd"] for n in (5, 7, 8) for lv in C.LEVELS for r in C.RS]
    assert 7 <= min(o) <= 13 and 22 <= max(o) <= 28
    o12 = [cell(12, lv, r)["opt_is_sd"] for lv in C.LEVELS for r in C.RS]
    assert min(o12) == 0 and 3 <= max(o12) <= 9


def test_claim_03_rangfolge_in_allen_60_zellen_und_spannen():
    for k in LMH:
        rel = cell(*k)["rel"]
        chain = [rel["OPT"], rel["SD_auf"], 1.0, rel["CV_auf"], rel["ZUFALL"], rel["LPT"], rel["SD_ab"]]
        assert all(a <= b + 1e-9 for a, b in zip(chain, chain[1:])), k
        assert chain[1] < 1.0 < chain[3]
    span = lambda key: (min(cell(*k)["rel"][key] for k in LMH), max(cell(*k)["rel"][key] for k in LMH))
    for key, lo, hi in (("ZUFALL", (1.09, 0.03), (1.45, 0.03)), ("LPT", (1.17, 0.03), (1.84, 0.04)), ("SD_ab", (1.20, 0.03), (1.89, 0.04))):
        got = span(key)
        assert got[0] == pytest.approx(lo[0], abs=lo[1]) and got[1] == pytest.approx(hi[0], abs=hi[1]), key


def test_claim_04_maass_b_spt_nie_schlechter_und_verhaeltnis():
    for n in C.FROZEN_NS:
        for lv in C.FROZEN_LEVELS:
            assert AN.maass_b_summary(FR.o1_maass_b(n, lv))["spt_le_sd"] == 30
    ratios = [AN.maass_b_summary(FR.o1_maass_b(n, lv))["ratio"] for n in C.FROZEN_NS for lv in C.LEVELS]
    assert len(ratios) == 12 and min(ratios) == pytest.approx(1.057, abs=0.01) and max(ratios) == pytest.approx(1.085, abs=0.01)
    assert max(AN.maass_b_summary(FR.o1_maass_b(n, lv))["ratio_max"] for n in C.FROZEN_NS for lv in C.LEVELS) == pytest.approx(1.41, abs=0.05)


def test_claim_05_kopplung():
    assert all(cell(n, "K", r)["equal"] == 30 and cell(n, "K", r)["delta"] == 0 for n in C.FROZEN_NS for r in C.RS)
    k, m, u = cell(8, "K", 2.0), cell(8, "M", 2.0), cell(8, "U", 2.0)
    assert k["rel"]["ZUFALL"] == pytest.approx(1.283, abs=0.03) and m["rel"]["ZUFALL"] == pytest.approx(1.305, abs=0.03) and u["rel"]["ZUFALL"] == pytest.approx(1.089, abs=0.03)
    assert m["delta"] * 100 == pytest.approx(-7.1, abs=1.0) and u["delta"] * 100 == pytest.approx(-24.1, abs=2.5)
    assert abs(u["better"] - 29) <= 3 and u["gap_sd"] * 100 == pytest.approx(1.3, abs=0.4) and m["gap_sd"] * 100 == pytest.approx(0.13, abs=0.1)
    corr = []
    for inst in range(30):
        mm, cv = K.gen_instance(8, "U", inst)
        corr.append(np.corrcoef(mm, mm * cv)[0, 1])
    assert np.mean(corr) == pytest.approx(0.15, abs=0.03) and min(corr) == pytest.approx(-0.39, abs=0.03) and max(corr) == pytest.approx(0.88, abs=0.03)
    for n in (5, 12):
        zs = [cell(n, "U", r)["rel"]["ZUFALL"] for r in C.RS]
        assert min(zs) > 0.99 and max(zs) < 1.01


def test_claim_06_vorlauf_zaehlung_und_kippstelle():
    row = {ek: AN.lead_summary(FR.lead_cell("M", 1.0, ek)) for ek in C.E_FROZEN}
    for ek, delta, better, tol in (("0", -8.1, 29, 1.0), ("80", -1.8, 25, 0.8), ("160", 0.3, 16, 0.8), ("∞", 4.3, 4, 1.0)):
        assert row[ek]["delta"] * 100 == pytest.approx(delta, abs=tol), ek
        assert abs(row[ek]["better"] - better) <= 3, ek
    for level in ("L", "M"):
        for r in (1.0, 4.0):
            assert ev.lead_tipping({ek: AN.lead_summary(FR.lead_cell(level, r, ek))["delta"] for ek in C.E_FROZEN}) == "160", (level, r)
    for r in (1.0, 4.0):
        d = {ek: AN.lead_summary(FR.lead_cell("H", r, ek))["delta"] for ek in C.E_FROZEN}
        assert d["160"] < 0 < d["∞"] and d["160"] * 100 == pytest.approx(-0.5, abs=0.5)
        assert abs(AN.lead_summary(FR.lead_cell("H", r, "160"))["better"] - 19) <= 3


EXACT_EXPECT = {  # n: (Zahl Instanzen, Streuungsregel exakt optimal bei r = 8, SPT optimal bei r = 0,5 / 1 / 2 / 4 / 8)
    3: (2000, 1004, (833, 833, 833, 926, 654)), 4: (2000, 1076, (415, 415, 415, 419, 344)),
    5: (1000, 405, (100, 100, 100, 100, 71)), 6: (300, 142, (6, 6, 6, 6, 2)),
}


def test_claim_07_exakte_zaehlung_ganzzahlig():
    for n, (count, sd8, spt) in EXACT_EXPECT.items():
        st = FR.exact_stats(n)
        assert all(st[r]["sd_opt"] == count for r in ("1/2", "1", "2", "4")), n
        assert st["8"]["sd_opt"] == sd8
        assert tuple(st[r]["spt_opt"] for r in ("1/2", "1", "2", "4", "8")) == spt
    for n in (3, 6):                                               # neu gerechnet aus dem Generator der Demo: exakt gleich (reine Ganzzahlarithmetik)
        new = X.exact_stats(n, C.EXACT_COUNTS[n])
        for r, st in FR.exact_stats(n).items():
            assert all(new[r][k] == st[k] for k in ("sd_opt", "spt_opt", "better", "equal", "worse")), (n, r)


def test_claim_08_handbeispiel_exakt():
    nm = lambda p: "".join(X.HAND_NAMES[i] for i in p)
    t2 = {nm(p): c for p, c in X.exact_table(X.HAND_LO, X.HAND_HI, X.HAND_M, F(2))}
    t8 = {nm(p): c for p, c in X.exact_table(X.HAND_LO, X.HAND_HI, X.HAND_M, F(8))}
    assert (t2["BAC"], t2["ABC"]) == (F(307, 4), F(337, 4)) and (t2["ABC"] - t2["BAC"]) / t2["ABC"] == F(30, 337)
    assert (t8["BCA"], t8["BAC"], t8["ABC"]) == (162, 172, F(359, 2)) and t8["BAC"] / t8["BCA"] - 1 == F(5, 81)
    for r, optima in ((F(1), ["BAC"]), (F(2), ["BAC"]), (F(4), ["BAC", "BCA"])):          # bei r = 4 liegt B-C-A genau gleichauf (beide 217/2)
        t = {nm(p): c for p, c in X.exact_table(X.HAND_LO, X.HAND_HI, X.HAND_M, r)}
        assert [k for k, c in t.items() if c == min(t.values())] == optima
    assert {nm(p): c for p, c in X.exact_table(X.HAND_LO, X.HAND_HI, X.HAND_M, F(4))}["BAC"] == F(217, 2)
    m = np.array(X.HAND_M, float)
    assert (K.wait_from_day_start(np.array([[0, 1, 2]]), m)[0], K.wait_from_day_start(np.array([[1, 0, 2]]), m)[0]) == (210.0, 240.0)
    pair = {r: X.exact_table(X.PAIR_LO, X.PAIR_HI, X.PAIR_M, r) for r in (F(1, 2), F(8))}
    assert (pair[F(1, 2)][0][1], pair[F(1, 2)][1][1]) == (F(189, 8), F(207, 4)) and pair[F(8)][0][1] == pair[F(8)][1][1] == 153


def test_claim_09_probe_auf_zweitem_szenariensatz():
    ratios = [cell(n, lv, r)["opt_B_over_A"] for n in C.FROZEN_NS for lv in C.FROZEN_LEVELS for r in C.RS]
    assert len(ratios) == 100 and min(ratios) == pytest.approx(0.997, abs=0.015) and max(ratios) == pytest.approx(1.017, abs=0.015)
    gaps = [cell(*k)["gap_sd"] for k in LMH]
    assert 0.0003 <= min(gaps) <= 0.0008 and 0.0033 <= max(gaps) <= 0.0043


TAU_EXPECT = {   # (r, cw): tau* je n = 2, 3, 4, 6, 8 (None = Mittelwert), Kosten bei tau* relativ zur Mittelwert-Planung auf Satz B
    (0.0, 1.0): ([0.5, None, 0.6, 0.65, 0.7], [0.986, 1.000, 0.980, 0.931, 0.883]),
    (0.0, 2.0): ([0.65, 0.7, 0.75, 0.75, 0.8], [0.963, 0.902, 0.829, 0.748, 0.686]),
}


def test_claim_10_quantil():
    for (r, cw), (stars, rels) in TAU_EXPECT.items():
        for n, star, rel in zip(C.TAU_NS, stars, rels):
            s = AN.tau_summary(n, r, cw)
            assert s["tau"] == star and s["rel_B"] == pytest.approx(rel, abs=0.012), (r, cw, n)
        assert AN.tau_summary(2, r, cw)["nv"] == pytest.approx(cw / (cw + 1))
    med = [AN.tau_summary(n, r, 2.0)["rel_B_median"] for n in (3, 4, 6, 8) for r in (0.0, 1.0)]
    assert min(med) == pytest.approx(1.06, abs=0.02) and max(med) == pytest.approx(1.20, abs=0.02)
    for r, lo, hi in ((0.0, 1.09, 1.20), (1.0, 1.06, 1.18)):    # die Spanne mischt beide Überziehpreise; einzeln gelesen
        one = [AN.tau_summary(n, r, 2.0)["rel_B_median"] for n in (3, 4, 6, 8)]
        assert min(one) == pytest.approx(lo, abs=0.02) and max(one) == pytest.approx(hi, abs=0.02), r
    knapp = AN.tau_summary(2, 0.0, 0.5)                         # 1 : 2: bei n = 2 lohnt eine knappere Planung als der Mittelwert
    assert knapp["tau"] == 0.35 and knapp["rel_B"] == pytest.approx(0.842, abs=0.012) and knapp["tau"] < FR.tau_phi(2)["mean"]
    assert AN.tau_summary(8, 1.0, 1.0)["rel_B"] == pytest.approx(0.937, abs=0.012)
    for r in (0.0, 1.0):
        assert all((AN.tau_summary(n, r, 0.5)["tau"] or 0) <= 0.60 for n in C.TAU_NS)
    assert AN.tau_summary(8, 1.0, 0.5)["tau"] is None
    assert FR.tau_phi(8)["mean"] == pytest.approx(0.564, abs=0.01)


def test_claim_11_ordnungsbedingung_von_de_kemp():
    for n, lv, expect in ((5, "M", 5.9), (7, "H", 0.55), (8, "L", 0.35), (12, "L", 0.0)):
        assert AN.dekemp_share(n, lv, 2000) * 100 == pytest.approx(expect, abs=0.4 if n < 12 else 0.15), (n, lv)
    assert AN.dekemp_share(8, "K", 200) == 1.0 and AN.dekemp_share(12, "K", 200) == 1.0


def test_claim_12_zeitungsverkaeufer_index_gegen_streuungsregel():
    diffs = [(cell(*k)["gap_nv"] - cell(*k)["gap_sd"]) * 100 for k in LMH]
    assert min(diffs) == pytest.approx(-0.08, abs=0.04) and max(diffs) == pytest.approx(0.07, abs=0.04)


# --- Aufwachbett (Befunde 13 bis 16): n = 6, r = 2, Stufen L / M / H ----------------------------------------------------------------------------

def _o3(level):
    return AN.o3_summary(level, 2.0)


def test_claim_13_aufwachbett_verhaeltnisse_und_blockadezeit():
    for level, b1 in zip(C.LEVELS, (1.88, 1.50, 1.28)):
        s = _o3(level)
        assert s["B1"]["ratio"] == pytest.approx(b1, abs=0.05)
        assert s["B2"]["ratio"] == pytest.approx({"L": 1.012, "M": 1.005, "H": 1.002}[level], abs=0.004) and s["B3"]["ratio"] == pytest.approx(1.0, abs=0.001)
    blk = {v: [_o3(lv)[v]["blk_spt"] * 100 for lv in C.LEVELS] for v in ("B1", "B2", "B3")}
    assert 5.5 <= min(blk["B1"]) <= 7.0 and 7.7 <= max(blk["B1"]) <= 9.1
    assert 0.08 <= min(blk["B2"]) <= 0.17 and 0.18 <= max(blk["B2"]) <= 0.32
    assert max(blk["B3"]) <= 0.01


def test_claim_14_wechselwirkung():
    dec = AN.o3_decomposition(2.0)
    assert dec["blockade_allein"] == pytest.approx(43.1, rel=0.08) and dec["fest_fest"] == pytest.approx(8.3, rel=0.12)
    for level, alone, both, plus, factor in (("L", 87.1, 150.4, 63.3, 1.47), ("M", 174.5, 243.5, 68.9, 1.60), ("H", 272.5, 339.6, 67.0, 1.56)):
        d = dec[level]
        assert d["streuung_allein"] == pytest.approx(alone, rel=0.05) and d["beides"] == pytest.approx(both, rel=0.05)
        assert d["zusatz"] == pytest.approx(plus, rel=0.1) and d["zusatz_vs_blockade_allein"] == pytest.approx(factor, abs=0.15)
    assert dec["blockade_allein"] < dec["M"]["streuung_allein"] and dec["blockade_allein"] / dec["M"]["streuung_allein"] == pytest.approx(0.25, abs=0.04)
    assert dec["blockade_allein"] / dec["fest_fest"] == pytest.approx(5.2, abs=0.8)


def test_claim_15_einfache_regel_verliert_bei_einem_bett():
    for level, one, none, two, spt, better, other in (("L", 17.0, 0.09, 0.34, 20.5, 19, 26), ("M", 5.6, 0.15, 0.21, 8.7, 22, 25), ("H", 1.9, 0.08, 0.09, 5.8, 23, 22)):
        s = _o3(level)
        assert s["B1"]["gap_SD_auf"] * 100 == pytest.approx(one, rel=0.25, abs=0.3) and s["ohne"]["gap_SD_auf"] * 100 == pytest.approx(none, abs=0.1)
        assert s["B2"]["gap_SD_auf"] * 100 == pytest.approx(two, abs=0.15) and s["B1"]["gap_SPT"] * 100 == pytest.approx(spt, rel=0.15)
        assert abs(s["B1"]["sd_better_spt"] - better) <= 3 and abs(s["ohne"]["sd_better_spt"] - {"L": 30, "M": 27, "H": 26}[level]) <= 3
        assert abs(s["B1"]["other_best"] - other) <= 3
        assert s["B1"]["gap_SD_auf"] > 10 * s["ohne"]["gap_SD_auf"]
    assert abs(sum(_o3(lv)["B1"]["other_best"] for lv in C.LEVELS) - 73) <= 6


def test_claim_16_weitere_regeln_im_aufwachbett():
    for level, one, two, none, alt in (("L", 58.8, 69.9, 69.6, 21.3), ("M", 50.8, 56.6, 55.6, 19.3), ("H", 57.9, 58.4, 57.1, 24.1)):
        s = _o3(level)
        assert s["B1"]["gap_LPT"] * 100 == pytest.approx(one, abs=4) and s["B2"]["gap_LPT"] * 100 == pytest.approx(two, abs=4) and s["ohne"]["gap_LPT"] * 100 == pytest.approx(none, abs=4)
        assert s["B1"]["gap_ALT"] * 100 == pytest.approx(alt, abs=3)
        assert s["ohne"]["alt_better_sd"] == 0 and s["B2"]["lpt_better_sd"] == 0 and s["ohne"]["lpt_better_sd"] == 0
    assert [_o3(lv)["B1"]["lpt_better_sd"] for lv in C.LEVELS] == [1, 0, 0]
    assert [_o3(lv)["B1"]["alt_better_sd"] for lv in C.LEVELS][0] in range(8, 15) and _o3("H")["B1"]["alt_better_sd"] == 0
    assert 2 <= _o3("M")["B1"]["alt_better_sd"] <= 7

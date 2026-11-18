"""Orakel-Tests für die exakte Rechnung (opr_exact): drei Rechenwege stimmen überein.

(1) Schleife mit Brüchen (`fractions`) über alle 2^n Ausgänge; (2) Fortschreiben der Verteilung der Endzeit (Wörterbuch Zeit -> Wahrscheinlichkeit); (3) die ganzzahlige Vektorrechnung der Demo;
dazu die Handwerte des Handbeispiels und die Gleitkomma-Rechnung des Kerns mit den Ausgängen als Szenarien. Das Orakel (1) und (2) ist unabhängig von numpy."""

import itertools
from fractions import Fraction as F

import numpy as np
import pytest

import opr_exact as X
import opr_kern as K
from opr_rng import SplitMix64


def loop_fractions(order, m, lo, hi, r):
    """Weg 1: alle Ausgänge einzeln, Brüche."""
    n = len(m)
    T = F(11, 10) * sum(m)
    tot = F(0)
    for bits in itertools.product((0, 1), repeat=n):
        f = s = F(0)
        w = F(0)
        for j in order:
            w += abs(f - s)
            f = max(f, s) + (hi[j] if bits[j] else lo[j])
            s += m[j]
        tot += w + r * max(F(0), f - T)
    return tot / 2 ** n


def distribution(order, m, lo, hi, r):
    """Weg 2: Verteilung der Endzeit fortschreiben."""
    dist = {F(0): F(1)}
    s = F(0)
    w = F(0)
    for j in order:
        new = {}
        for f, p in dist.items():
            w += p * abs(f - s)
            for d in (lo[j], hi[j]):
                e = max(f, s) + d
                new[e] = new.get(e, F(0)) + p / 2
        dist = new
        s += m[j]
    T = F(11, 10) * sum(m)
    return w + r * sum(p * max(F(0), f - T) for f, p in dist.items())


def test_drei_rechenwege_gleich_auf_zufallsinstanzen():
    g = SplitMix64(4711)
    for n in (3, 4):
        for _ in range(8):
            m, lo, hi = X.two_point_instance(g, n)
            for r in (F(1, 2), F(2), F(8)):
                table = X.exact_table(lo, hi, m, r)
                for order, cost in table[:: max(1, len(table) // 6)]:
                    assert cost == loop_fractions(order, m, lo, hi, r)
                    assert cost == distribution(order, m, lo, hi, r)


def test_handbeispiel_alle_sechs_reihenfolgen_exakt():
    nm = lambda p: "".join(X.HAND_NAMES[i] for i in p)
    t2 = {nm(p): c for p, c in X.exact_table(X.HAND_LO, X.HAND_HI, X.HAND_M, F(2))}
    assert t2 == {"ABC": F(337, 4), "ACB": F(197, 2), "BAC": F(307, 4), "BCA": F(327, 4), "CAB": F(101), "CBA": F(377, 4)}
    t8 = {nm(p): c for p, c in X.exact_table(X.HAND_LO, X.HAND_HI, X.HAND_M, F(8))}
    assert t8 == {"ABC": F(359, 2), "ACB": F(184), "BAC": F(172), "BCA": F(162), "CAB": F(179), "CBA": F(349, 2)}
    assert t8["BAC"] / t8["BCA"] - 1 == F(5, 81)                         # die Streuungsregel liegt bei r = 8 um 5/81 über dem Optimum
    assert (t2["ABC"] - t2["BAC"]) / t2["ABC"] == F(30, 337)             # und bei r = 2 um 30/337 unter SPT
    t1 = {nm(p): c for p, c in X.exact_table(X.HAND_LO, X.HAND_HI, X.HAND_M, F(1))}
    t4 = {nm(p): c for p, c in X.exact_table(X.HAND_LO, X.HAND_HI, X.HAND_M, F(4))}
    assert (t1["BAC"], t1["ABC"], t4["BAC"], t4["ABC"]) == (F(487, 8), F(547, 8), F(217, 2), F(116))
    for t in (t1, t2, t4):
        assert min(t, key=t.get) == "BAC"
    assert min(t8, key=t8.get) == "BCA"


def test_zwei_eingriffe_gleiches_mittel():
    for r, p_first, q_first in ((F(1, 2), F(189, 8), F(207, 4)), (F(1), F(129, 4), F(117, 2)), (F(2), F(99, 2), F(72)), (F(4), F(84), F(99)), (F(8), F(153), F(153))):
        t = {p: c for p, c in X.exact_table(X.PAIR_LO, X.PAIR_HI, X.PAIR_M, r)}
        assert t[(0, 1)] == p_first and t[(1, 0)] == q_first


def test_ganzzahlrechnung_gegen_gleitkomma_kern():
    g = SplitMix64(99)
    for n in (3, 4, 5):
        m, lo, hi = X.two_point_instance(g, n)
        D = X.outcomes(lo, hi).astype(float)
        P = X.all_perms_int(n)
        comp = K.comp_means(K.eval_perms(P, D, np.array(m, float), 1.1 * sum(m)))
        for r in (F(1, 2), F(4)):
            ref = K.cost_from(comp, float(r))
            got = [float(c) for _, c in X.exact_table(lo, hi, m, r)]
            assert got == pytest.approx(ref, rel=1e-12)


def test_regeln_exakt_und_zaehlungen_klein():
    m, lo, hi = list(X.HAND_M), list(X.HAND_LO), list(X.HAND_HI)
    assert X.rule_orders_exact(m, lo, hi) == ((0, 1, 2), (1, 0, 2))
    st = X.exact_stats(3, 300)
    for r, s in st.items():
        assert s["better"] + s["equal"] + s["worse"] == 300 and s["sd_opt"] <= 300 and s["spt_opt"] <= 300
    assert st["2"]["sd_opt"] == 300 and st["8"]["sd_opt"] < 300          # r <= 4: Streuungsregel immer exakt optimal (n = 3), bei r = 8 nicht
    assert st["8"]["gap_sd_max"] > 0 and st["8"]["worst_instance"] is not None
    assert X.exact_stats(3, 300) == st                                   # ganzzahlig und reproduzierbar

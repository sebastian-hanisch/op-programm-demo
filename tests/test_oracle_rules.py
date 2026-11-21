"""Orakel-Tests für Regeln, Aufzählung, lokale Verbesserung, Quantil und Newsvendor-Index.

Orakel: Schleife je Reihenfolge und Szenario (Aufzählung n = 5 mit 120 und n = 7 mit 5 040 Reihenfolgen, Stichprobe), `scipy.stats.lognorm` für das Quantil, `scipy.integrate.quad`
für die mittlere absolute Abweichung vom Median, Simulation (4 Standardfehler) für die geschlossene Form der erwarteten Wartezeit und des Leerlaufs bei n = 2."""

import math

import numpy as np
import pytest

import opr_kern as K
import opr_rules as R
from test_oracle_kern import py_day

scipy_stats = pytest.importorskip("scipy.stats")
integrate = pytest.importorskip("scipy.integrate")


def _inst(seed, n):
    rng = np.random.default_rng(seed)
    m = np.round(rng.uniform(30, 160, n))
    cv = rng.choice([0.2, 0.35, 0.5, 0.8], n)
    return m, cv, rng.standard_normal((120, n))


def test_aufzaehlung_n5_gegen_schleife_und_optimum_unter_jeder_regel():
    m, cv, Z = _inst(1, 5)
    D = K.durations(m, cv, Z)
    T = K.day_end(m)
    P = K.all_perms(5)
    assert len(P) == 120
    cm = K.enum_means(P, D, m, T)
    cost = K.cost_from(cm, 2.0)
    for p in range(0, 120, 7):
        ref = np.mean([py_day(P[p], D[s], m, T)[0] + py_day(P[p], D[s], m, T)[1] + 2.0 * py_day(P[p], D[s], m, T)[2] for s in range(len(D))])
        assert cost[p] == pytest.approx(ref, rel=1e-12)
    ro = R.rule_orders(m, cv, 5)
    index = {tuple(int(x) for x in q): i for i, q in enumerate(P)}
    for key in R.RULES:
        assert cost.min() <= cost[index[tuple(int(x) for x in ro[key])]] + 1e-12, key
    assert cost.min() < cost.max()


def test_aufzaehlung_n7_blockweise_gleich_am_stueck():
    m, cv, Z = _inst(2, 7)
    D = K.durations(m, cv, Z)
    P = K.all_perms(7)
    assert len(P) == 5040
    a = K.enum_means(P, D, m, K.day_end(m), chunk=500)
    b = K.enum_means(P[:600], D, m, K.day_end(m), chunk=100)
    for key in ("wait", "idle", "over"):
        assert a[key][:600] == pytest.approx(b[key], abs=1e-9)
    c = K.cost_from(a, 4.0)
    order = R.rule_orders(m, cv, 1)["SD_auf"]
    j = int(np.where((P == order).all(axis=1))[0][0])
    comp = [py_day(order, D[s], m, K.day_end(m)) for s in range(len(D))]
    assert c[j] == pytest.approx(np.mean([w + i + 4.0 * o for w, i, o, _, _ in comp]), rel=1e-12)


def test_lokale_verbesserung_ist_lokales_optimum_und_nie_besser_als_aufzaehlung():
    for seed in range(5):
        m, cv, Z = _inst(10 + seed, 5)
        D = K.durations(m, cv, Z)
        T = K.day_end(m)
        fn = lambda comp: K.cost_from(comp, 2.0)
        start = R.rule_orders(m, cv, 1)["LPT"]
        best, cost = R.local_improvement(start, D, m, T, fn)
        cm = K.enum_means(K.all_perms(5), D, m, T)
        enum_cost = K.cost_from(cm, 2.0)
        assert cost >= enum_cost.min() - 1e-12
        start_cost = float(K.cost_from(K.comp_means(K.eval_perms(np.array([start]), D, m, T)), 2.0)[0])
        assert cost <= start_cost + 1e-12
        nb = R.neighbors(best)
        nb_cost = K.cost_from(K.comp_means(K.eval_perms(nb, D, m, T)), 2.0)
        assert nb_cost.min() >= cost - 1e-9, "ein Nachbar ist besser: kein lokales Optimum"


def test_nachbarn_sind_vertauschen_und_verschieben():
    nb = {tuple(x) for x in R.neighbors([0, 1, 2])}
    assert nb == {(1, 0, 2), (2, 1, 0), (0, 2, 1), (1, 2, 0), (2, 0, 1)}      # alle Permutationen außer der Ausgangsreihenfolge
    # 15 Vertauschungen + 30 Verschiebungen, davon 10 Verschiebungen um eine Stelle = 5 benachbarte Vertauschungen (je zwei Verschiebungen)
    assert len(R.neighbors(list(range(6)))) == 15 + 30 - 10


def test_quantil_gegen_scipy():
    m, cv = np.array([90.0, 45.0, 150.0]), np.array([0.5, 0.2, 0.8])
    mu, sg = K.ln_params(m, cv)
    for tau in (0.3, 0.5, 0.67, 0.9):
        ref = np.array([scipy_stats.lognorm.ppf(tau, s=s, scale=math.exp(u)) for u, s in zip(mu, sg)])
        assert K.quantile_plan(m, cv, tau) == pytest.approx(ref, rel=1e-9)
    assert K.quantile_plan(m, cv, 0.5) == pytest.approx(m / np.sqrt(1 + cv ** 2), rel=1e-12)       # Median


def test_erwartete_wartezeit_und_leerlauf_bei_zwei_eingriffen_geschlossen_gegen_simulation():
    m, cv = np.array([90.0, 60.0]), np.array([0.5, 0.3])
    Z = np.random.default_rng(4).standard_normal((200000, 2))
    D = K.durations(m, cv, Z)
    for q in (60.0, 90.0, 110.0):
        comp = K.eval_perms(np.array([[0, 1]]), D, np.array([q, 60.0]), 1e9)
        for key, closed in (("wait", K.lognormal_over(m[0], cv[0], q)), ("idle", K.lognormal_under(m[0], cv[0], q))):
            x = comp[key][0]
            assert abs(x.mean() - float(closed)) < 4 * x.std() / math.sqrt(len(x)), (key, q)


def test_bestes_quantil_bei_zwei_eingriffen_ist_das_zeitungsverkaeufer_verhaeltnis():
    m, cv = 90.0, 0.5
    for cw in (1.0, 2.0, 4.0):
        grid = np.arange(20.0, 300.0, 0.25)
        cost = np.array([cw * float(K.lognormal_over(m, cv, q)) + float(K.lognormal_under(m, cv, q)) for q in grid])
        q_best = grid[cost.argmin()]
        mu, sg = K.ln_params(m, cv)
        q_theory = scipy_stats.lognorm.ppf(cw / (cw + 1.0), s=float(sg), scale=math.exp(float(mu)))
        assert abs(q_best - q_theory) <= 0.25 + 1e-9


def test_zeitungsverkaeufer_index_gegen_numerische_integration():
    for m, cv in ((90.0, 0.2), (45.0, 0.5), (150.0, 0.8)):
        mu, sg = K.ln_params(m, cv)
        mu, sg = float(mu), float(sg)
        med = math.exp(mu)
        pdf = lambda x: scipy_stats.lognorm.pdf(x, s=sg, scale=math.exp(mu))
        ref = integrate.quad(lambda x: abs(x - med) * pdf(x), 0, med, epsabs=1e-10)[0] + integrate.quad(lambda x: abs(x - med) * pdf(x), med, math.inf, epsabs=1e-10)[0]
        assert float(R.newsvendor_index(np.array([m]), np.array([cv]))[0]) == pytest.approx(ref, rel=1e-7)


def test_index_ist_proportional_zur_streuung_bei_symmetrischen_zweipunkt_dauern_und_ordnet_schiefe_anders_als_die_varianz():
    def index(vals, probs):
        return min(sum(p * max(0.0, d - q) + p * max(0.0, q - d) for d, p in zip(vals, probs)) for q in vals)

    def var(vals, probs):
        mean = sum(d * p for d, p in zip(vals, probs))
        return sum(p * (d - mean) ** 2 for d, p in zip(vals, probs))

    sym = [(30, 90), (75, 105), (80, 160)]
    idx = [index(v, (0.5, 0.5)) for v in sym]
    assert idx == pytest.approx([30.0, 15.0, 40.0])                      # = halbe Spannweite = Streuung
    x, y = ((30, 60, 90), (0.5, 0.25, 0.25)), ((30, 60, 120), (2 / 3, 1 / 6, 1 / 6))
    assert var(*x) < var(*y) and index(*x) > index(*y)                  # Varianz setzt X vor Y, der Index Y vor X
    assert var(*x) == pytest.approx(618.75) and var(*y) == pytest.approx(1100.0) and index(*x) == pytest.approx(22.5) and index(*y) == pytest.approx(20.0)


def test_regeln_ordnen_wie_definiert_und_brechen_gleichstaende_nach_index():
    m = np.array([90.0, 45.0, 90.0, 150.0])
    cv = np.array([0.2, 0.5, 0.2, 0.3])
    ro = R.rule_orders(m, cv, 1)
    assert list(ro["SPT"]) == [1, 0, 2, 3]              # Gleichstand der Mittel 90: Index 0 vor Index 2
    assert list(ro["LPT"]) == [3, 0, 2, 1]
    assert list(ro["CV_auf"]) == [0, 2, 3, 1]
    assert list(ro["SD_auf"]) == [0, 2, 1, 3]           # Streuungen 18, 22,5, 18, 45
    assert list(ro["SD_ab"]) == [3, 1, 0, 2]
    assert list(R.alternating(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))) == [0, 4, 1, 3, 2]
    z = ro["ZUFALL"]
    assert z.shape == (20, 4) and all(sorted(row) == [0, 1, 2, 3] for row in z)
    assert (R.random_orders(4, 1) == R.random_orders(4, 1)).all()

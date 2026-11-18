"""Orakel-Tests für den Kern (opr_kern): anderer Rechenweg als die Demo.

Orakel: (1) eine reine Python-Schleife je Szenario (ohne numpy-Vektorisierung) für Wartezeit, Leerlauf, Überziehen und Blockade, ohne und mit 1, 2, 3 Betten;
(2) die Bilanz Ende = Summe der Dauern + Leerlauf (+ Blockade); (3) die Tandem-Blockade gegen alle 64 Ausgänge (zwei Dauern und zwei Aufwachdauern je Eingriff);
(4) die Handwerte des Handbeispiels (Brüche) mit der Gleitkomma-Rechnung des Kerns; (5) die Vorlauf-Identität je Szenario; (6) Maß B gegen Simulation;
(7) die Lognormalverteilung gegen ihre Eingaben (Zufallszahlen des Orakels aus numpy); (8) die Normalzahlen des ganzzahligen Generators gegen Mittel und Streuung.
Das Orakel nutzt numpy-Zufall, der Kern nur den SplitMix64-Generator; verglichen werden Rechenwege, keine Zufallsfolgen."""

import itertools
import math

import numpy as np
import pytest

import opr_constants as C
import opr_kern as K


def py_day(order, D_row, planned, T, R_row=None, beds=None):
    """Ein Tag in reinem Python: (Wartezeit, Leerlauf, Überziehen, Blockade, Ende)."""
    f = s = 0.0
    wait = idle = blk = 0.0
    free = [0.0] * beds if beds else None
    for j in order:
        if f > s:
            wait += f - s
        else:
            idle += s - f
        end = max(f, s) + D_row[j]
        if beds:
            i = min(range(beds), key=lambda x: free[x])
            dep = max(end, free[i])
            blk += dep - end
            free[i] = dep + R_row[j]
            f = dep
        else:
            f = end
        s += planned[j]
    return wait, idle, max(0.0, f - T), blk, f


def _instance(seed, n=6, scen=40):
    rng = np.random.default_rng(seed)
    m = np.round(rng.uniform(30, 160, n))
    cv = rng.choice([0.1, 0.3, 0.5, 0.8], n)
    Z = rng.standard_normal((scen, 2 * n))
    return m, cv, Z


@pytest.mark.parametrize("beds", [None, 1, 2, 3])
def test_vektor_gleich_schleife_ohne_und_mit_betten(beds):
    for seed in range(4):
        m, cv, Z = _instance(seed)
        n = len(m)
        D, Rt = K.durations(m, cv, Z), K.recovery_durations(n, Z)
        T = K.day_end(m)
        rng = np.random.default_rng(100 + seed)
        perms = np.array([rng.permutation(n) for _ in range(30)])
        comp = K.eval_perms(perms, D, m, T, Rt if beds else None, beds)
        for p, order in enumerate(perms):
            for s in range(D.shape[0]):
                ref = py_day(order, D[s], m, T, Rt[s], beds)
                got = (comp["wait"][p, s], comp["idle"][p, s], comp["over"][p, s], comp["blk"][p, s], comp["end"][p, s])
                assert got == pytest.approx(ref, abs=1e-9), (beds, seed, p, s)


def test_bilanz_ende_ist_summe_dauern_plus_leerlauf_plus_blockade():
    for beds in (None, 1, 2):
        m, cv, Z = _instance(7)
        n = len(m)
        D, Rt = K.durations(m, cv, Z), K.recovery_durations(n, Z)
        perms = np.array(list(itertools.islice(itertools.permutations(range(n)), 0, 720, 37)))
        comp = K.eval_perms(perms, D, m, K.day_end(m), Rt if beds else None, beds)
        total_d = np.stack([D[:, p].sum(axis=1) for p in perms])
        assert comp["end"] == pytest.approx(total_d + comp["idle"] + comp["blk"], abs=1e-9)


def test_genug_betten_geben_dasselbe_wie_keine_blockade():
    m, cv, Z = _instance(3)
    n = len(m)
    D, Rt = K.durations(m, cv, Z), K.recovery_durations(n, Z)
    P = K.all_perms(n)[::23]
    a = K.eval_perms(P, D, m, K.day_end(m))
    b = K.eval_perms(P, D, m, K.day_end(m), Rt, n)
    for key in ("wait", "idle", "over", "end"):
        assert b[key] == pytest.approx(a[key], abs=1e-9)
    assert not b["blk"].any()


def test_handbeispiel_handwerte_aus_allen_acht_ausgaengen():
    """Drei Eingriffe, zwei Dauern je Eingriff (Handrechnung im Plan), Termine = Summe der Mittel, T = 297: B-A-C kostet 307/4, SPT A-B-C 337/4 bei r = 2."""
    lo, hi, m = (30, 75, 80), (90, 105, 160), (60.0, 90.0, 120.0)
    D = np.array([[hi[j] if b[j] else lo[j] for j in range(3)] for b in itertools.product((0, 1), repeat=3)], dtype=float)
    perms = np.array([[1, 0, 2], [0, 1, 2], [1, 2, 0]])
    comp = K.comp_means(K.eval_perms(perms, D, np.array(m), 297.0))
    cost2 = K.cost_from(comp, 2.0)
    assert cost2[0] == pytest.approx(307 / 4, rel=1e-12)
    assert cost2[1] == pytest.approx(337 / 4, rel=1e-12)
    cost8 = K.cost_from(comp, 8.0)
    assert cost8[2] == pytest.approx(162.0, rel=1e-12)       # B-C-A ist bei r = 8 das Optimum
    assert cost8[0] == pytest.approx(172.0, rel=1e-12)       # Streuungsregel B-A-C
    assert cost8[1] == pytest.approx(359 / 2, rel=1e-12)     # SPT


def test_tandem_blockade_gegen_alle_64_ausgaenge():
    """Drei Eingriffe, je zwei Dauern und zwei Aufwachdauern: Erwartung des Kerns = Mittel über alle 64 Ausgänge der Schleife."""
    lo, hi, rl, rh = (40, 60, 30), (70, 100, 50), 30, 90
    outs = list(itertools.product((0, 1), repeat=6))
    D = np.array([[hi[j] if o[j] else lo[j] for j in range(3)] for o in outs], dtype=float)
    Rt = np.array([[rh if o[3 + j] else rl for j in range(3)] for o in outs], dtype=float)
    m = np.array([55.0, 80.0, 40.0])
    perms = np.array(list(itertools.permutations(range(3))))
    for beds in (1, 2, 3):
        comp = K.comp_means(K.eval_perms(perms, D, m, 1e9, Rt, beds))
        for p, order in enumerate(perms):
            ref = np.mean([py_day(order, D[s], m, 1e9, Rt[s], beds)[3] for s in range(64)])
            assert comp["blk"][p] == pytest.approx(ref, abs=1e-9)
    b1 = K.comp_means(K.eval_perms(perms, D, m, 1e9, Rt, 1))["blk"]
    b3 = K.comp_means(K.eval_perms(perms, D, m, 1e9, Rt, 3))["blk"]
    assert (b1 > 0).all(), "bei einem Bett blockiert die Aufwachdauer den Saal in jeder Reihenfolge"
    assert not b3.any(), "bei drei Betten und drei Eingriffen blockiert nie ein Bett (belegte Nullen, kein Fehler)"


def test_blockade_handwerte_25_und_20_minuten():
    """Zwei Eingriffe à 40 min, Aufwachdauer 30 oder 90 (je ½): ein Bett 25 min mittlere Blockade, feste Aufwachdauer 60 gleiches Mittel 20 min, zwei Betten 0."""
    D = np.full((4, 2), 40.0)
    Rt = np.array([[a, b] for a in (30.0, 90.0) for b in (30.0, 90.0)])
    perm = np.array([[0, 1]])
    plan = np.array([40.0, 40.0])
    assert K.comp_means(K.eval_perms(perm, D, plan, 1e9, Rt, 1))["blk"][0] == pytest.approx(25.0)
    assert K.comp_means(K.eval_perms(perm, D, plan, 1e9, Rt, 2))["blk"][0] == 0.0
    fix = np.full((1, 2), 60.0)
    assert K.comp_means(K.eval_perms(perm, D[:1], plan, 1e9, fix, 1))["blk"][0] == pytest.approx(20.0)


def test_vorlauf_identitaet_je_szenario():
    """Wartezeit ab Ankunft (Ankunft = max(0, Termin - e)) = Wartezeit nach dem Termin + min(Termin, e), je Szenario exakt."""
    m, cv, Z = _instance(11, n=6, scen=50)
    D = K.durations(m, cv, Z)
    rng = np.random.default_rng(5)
    perms = np.array([rng.permutation(6) for _ in range(20)])
    for e in (0.0, 20.0, 80.0, 160.0, 1e9):
        comp = K.eval_perms(perms, D, m, K.day_end(m))
        lead = K.lead_term(perms, m, e)
        starts_arrival = np.zeros(comp["wait"].shape)
        for p, order in enumerate(perms):
            for s in range(D.shape[0]):
                f = t = 0.0
                tot = 0.0
                for j in order:
                    start = max(t, f)
                    tot += start - max(0.0, t - e)
                    f = start + D[s, j]
                    t += m[j]
                starts_arrival[p, s] = tot
        assert starts_arrival == pytest.approx(comp["wait"] + lead[:, None], abs=1e-9)


def test_maass_b_formel_gegen_simulation():
    """Maß B: alle sofort da, keine Termine: Summe der Wartezeiten = Summe_k Summe_{j<k} m im Mittel (Simulation, 4 Standardfehler)."""
    m = np.array([45.0, 90.0, 150.0, 60.0, 100.0])
    cv = np.array([0.5, 0.35, 0.3, 0.8, 0.2])
    rng = np.random.default_rng(1)
    D = K.durations(m, cv, rng.standard_normal((40000, 5)))
    for perm in ([0, 1, 2, 3, 4], [2, 4, 1, 3, 0]):
        p = np.array([perm])
        wait = K.eval_perms(p, D, np.zeros(5), 1e12)["wait"][0]      # alle Termine bei 0: jeder wartet bis zum Ende aller Vorgänger
        closed = K.wait_from_day_start(p, m)[0]
        assert abs(wait.mean() - closed) < 4 * wait.std() / math.sqrt(len(wait))
    assert K.wait_from_day_start(np.array([[0, 1, 2]]), np.array([60.0, 90.0, 120.0]))[0] == 210.0


def test_lognormal_gegen_eingaben():
    m = np.array([45.0, 90.0, 150.0, 90.0])
    cv = np.array([0.1, 0.35, 0.8, 0.5])
    Z = np.random.default_rng(2).standard_normal((400000, 4))
    D = K.durations(m, cv, Z)
    ln = np.log(D)
    mu, sg = K.ln_params(m, cv)
    assert ln.mean(axis=0) == pytest.approx(mu, abs=0.004)
    assert ln.std(axis=0) == pytest.approx(sg, abs=0.004)
    assert np.all(np.abs(D.mean(axis=0) - m) < 4 * cv * m / math.sqrt(len(D)))
    assert np.exp(mu) == pytest.approx(m / np.sqrt(1 + cv ** 2))                # Median = m / sqrt(1 + c^2)
    rec = K.recovery_durations(4, np.random.default_rng(3).standard_normal((400000, 8)))
    assert rec.mean() == pytest.approx(60.0, rel=0.003) and rec.std() / rec.mean() == pytest.approx(0.5, abs=0.01)
    assert (K.recovery_durations(4, Z[:5], r_cv=0) == 60.0).all()


def test_normalzahlen_des_ganzzahligen_generators():
    Z = K.normals(2, 5000, 12345)
    assert abs(Z.mean()) < 4 / math.sqrt(Z.size) and Z.std() == pytest.approx(1.0, abs=0.03)
    assert (K.normals(2, 5, 7) == K.normals(2, 5, 7)).all() and not (K.normals(2, 5, 7) == K.normals(2, 5, 8)).all()

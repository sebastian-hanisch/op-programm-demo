"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen: n = 1 und n = 2 (Zweig `n < 2`), Gleichstände, Variationskoeffizient 0, r = 0, ein Szenario, Tagesende knapp unter der Summe der Mittel."""

from fractions import Fraction

import numpy as np
import pytest

import opr_analysis as AN
import opr_constants as C
import opr_evaluation as ev
import opr_exact as X
import opr_kern as K
import opr_rules as R


def _D(m, cv, scen=30, seed=1):
    return K.durations(np.asarray(m, float), np.asarray(cv, float), K.normals(len(m), scen, seed))


def test_genau_ein_eingriff():
    m, cv = np.array([90.0]), np.array([0.3])
    D = _D(m, cv)
    comp = K.eval_perms(np.array([[0]]), D, m, K.day_end(m))
    assert not comp["wait"].any() and not comp["idle"].any()          # ein Eingriff wartet nie und lässt nichts leer
    assert list(R.rule_orders(m, cv)["SPT"]) == [0]
    perm, cost = R.local_improvement([0], D, m, K.day_end(m), lambda c: K.cost_from(c, 2.0))
    assert list(perm) == [0] and cost >= 0
    res = ev.evaluate_instance(m, cv, D, rs=(1.0,))
    assert res["r"][1.0]["SPT"] == res["r"][1.0]["SD_auf"] == res["r"][1.0]["OPT"]
    assert res["mB"]["SPT"] == 0.0


def test_genau_zwei_eingriffe_ein_nachbar_und_zwei_reihenfolgen():
    m, cv = np.array([90.0, 45.0]), np.array([0.5, 0.2])
    D = _D(m, cv)
    assert [tuple(x) for x in R.neighbors([0, 1])] == [(1, 0)]
    res = ev.evaluate_instance(m, cv, D, rs=(2.0,))
    rec = res["r"][2.0]
    assert rec["OPT"] <= min(rec["SPT"], rec["SD_auf"], rec["LPT"], rec["SD_ab"]) + 1e-12
    assert len(K.all_perms(2)) == 2
    perm, _ = R.local_improvement([0, 1], D, m, K.day_end(m), lambda c: K.cost_from(c, 2.0))
    assert sorted(perm) == [0, 1]
    table = X.exact_table(X.PAIR_LO, X.PAIR_HI, X.PAIR_M, Fraction(1))
    assert len(table) == 2


def test_alle_mittel_gleich_alle_streuungen_gleich_gleichstand_nach_index():
    m, cv = np.full(4, 90.0), np.full(4, 0.35)
    ro = R.rule_orders(m, cv, 1)
    for key in ("SPT", "SD_auf", "CV_auf", "NV"):
        assert list(ro[key]) == [0, 1, 2, 3], key
    assert list(ro["LPT"]) == [0, 1, 2, 3] and list(ro["SD_ab"]) == [0, 1, 2, 3]      # absteigend mit Gleichstand: auch nach Index
    res = ev.evaluate_instance(m, cv, _D(m, cv), rs=(1.0,))
    assert res["r"][1.0]["SPT"] == pytest.approx(res["r"][1.0]["SD_auf"], abs=1e-12)


def test_variationskoeffizient_null_und_r_null():
    m, cv = np.array([60.0, 90.0, 45.0]), np.zeros(3)
    res = ev.evaluate_instance(m, cv, _D(m, cv), rs=(0.0, 1.0))
    for r in (0.0, 1.0):
        assert all(res["r"][r][k] == pytest.approx(0, abs=1e-9) for k in R.RULES + ("OPT",))
    m2, cv2 = np.array([60.0, 90.0, 45.0]), np.array([0.5, 0.2, 0.8])
    D = _D(m2, cv2)
    a = ev.evaluate_instance(m2, cv2, D, rs=(0.0,))["r"][0.0]
    comp = K.comp_means(K.eval_perms(np.array([R.rule_orders(m2, cv2)["SPT"]]), D, m2, K.day_end(m2)))
    assert a["SPT"] == pytest.approx(comp["wait"][0] + comp["idle"][0], rel=1e-12)          # r = 0: Überziehen kostet nichts


def test_ein_szenario_und_tagesende_knapp_unter_der_summe_der_mittel():
    m, cv = np.array([60.0, 90.0, 45.0]), np.array([0.2, 0.2, 0.2])
    D = _D(m, cv, scen=1)
    res = ev.evaluate_instance(m, cv, D, rs=(2.0,))
    assert res["r"][2.0]["se_diff"] == 0.0                              # ein Szenario: kein Standardfehler, aber auch kein Fehler
    T = m.sum() - 1.0
    comp = K.eval_perms(np.array([[0, 1, 2]]), np.tile(m, (5, 1)), m, T)
    assert comp["over"][0] == pytest.approx(np.full(5, 1.0))            # Dauern = Mittel: Ende = Summe der Mittel, 1 min über T
    assert K.day_end(m) == pytest.approx(1.1 * m.sum())


def test_regler_grenzen_n_klein_und_gross_live():
    for n in (C.N_MIN, C.N_MAX):
        m, cv = K.gen_instance(n, "H", 3)
        D = K.durations(m, cv, K.normals(n, 40, 5))
        live = AN.live_order(m, cv, D, None, 8.0)
        assert set(live["comps"]) == set(R.RULES) | {"OPT"} and len(live["rec"]["opt_perm"]) == n
        assert live["rec"]["OPT"] <= min(live["rec"][k] for k in R.RULES) + 1e-9


def test_zusammenfassung_mit_nullkosten_teilt_nicht_durch_null():
    rec = {"SPT": 0.0, "SD_auf": 0.0, "OPT": 1.0, "se_diff": 0.0, "CV_auf": 1.0, "NV": 1.0, "LPT": 1.0, "SD_ab": 1.0, "ZUFALL": 1.0, "opt_is_sd": 1, "opt_is_spt": 1}
    s = ev.summarize([rec, rec], with_b=False, ci=False)
    assert s["delta"] == 0.0 and s["equal"] == 2


def test_verdict_text_bei_gleichen_kosten_meldet_keinen_effekt():
    assert ev._cmp(0.0) == "gleich teuer wie SPT" and ev._cmp(-0.089) == "8,9 % weniger als SPT" and ev._cmp(0.004) == "0,4 % mehr als SPT"


def test_nachbar_zelle_naechstliegendes_n():
    assert ev.nearest_frozen_n(4) == 5 and ev.nearest_frozen_n(6) == 5 and ev.nearest_frozen_n(9) == 8 and ev.nearest_frozen_n(10) == 8 and ev.nearest_frozen_n(11) == 12

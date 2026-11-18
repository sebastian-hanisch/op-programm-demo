"""Eigenschaftstests (metamorph, ORACLE-PLAYBOOK §5.2): Aussagen, die für das Modell gelten müssen, ohne dass man das richtige Ergebnis kennt.

Je Eigenschaft 20 bis 50 Instanzen mit festem Zufalls-Seed, plus Randfälle."""

import numpy as np
import pytest

import opr_constants as C
import opr_evaluation as ev
import opr_kern as K
import opr_rules as R


def _inst(seed, n=6, scen=60):
    rng = np.random.default_rng(seed)
    m = np.round(rng.uniform(30, 160, n))
    cv = rng.choice([0.2, 0.35, 0.5, 0.8], n)
    return m, cv, rng.standard_normal((scen, 2 * n))


def _means(perms, D, plan, T, **kw):
    return K.comp_means(K.eval_perms(np.atleast_2d(perms), D, plan, T, **kw))


def test_feste_dauern_kosten_null():
    for seed in range(20):
        m, _, Z = _inst(seed)
        D = K.durations(m, np.zeros(len(m)), Z)
        rng = np.random.default_rng(seed)
        cm = _means(rng.permutation(len(m)), D, m, K.day_end(m))
        assert cm["wait"][0] == pytest.approx(0, abs=1e-9) and cm["idle"][0] == pytest.approx(0, abs=1e-9) and cm["over"][0] == 0


def test_gleicher_variationskoeffizient_streuungsregel_gleich_spt():
    for inst in range(50):
        m, cv = K.gen_instance(8, "K", inst)
        ro = R.rule_orders(m, cv, 1)
        assert list(ro["SD_auf"]) == list(ro["SPT"]), inst


def test_skalierung_aller_dauern_skaliert_die_kosten_und_aendert_keine_reihenfolge():
    for seed in range(20):
        m, cv, Z = _inst(seed)
        D = K.durations(m, cv, Z)
        P = K.all_perms(4)[:5] if False else np.array([np.random.default_rng(seed).permutation(len(m)) for _ in range(5)])
        a = _means(P, D, m, K.day_end(m))
        b = _means(P, 3.0 * D, 3.0 * m, 3.0 * K.day_end(m))
        for key in ("wait", "idle", "over"):
            assert b[key] == pytest.approx(3.0 * a[key], rel=1e-9, abs=1e-9)
        assert [list(v) for k, v in R.rule_orders(m, cv, 1).items() if k != "ZUFALL"] == [list(v) for k, v in R.rule_orders(3 * m, cv, 1).items() if k != "ZUFALL"]


def test_umnummerieren_der_eingriffe_aendert_nur_die_namen():
    for seed in range(20):
        m, cv, Z = _inst(seed)
        n = len(m)
        D = K.durations(m, cv, Z)
        rng = np.random.default_rng(1000 + seed)
        perm, relabel = rng.permutation(n), rng.permutation(n)
        a = _means(perm, D, m, K.day_end(m))
        inv = np.argsort(relabel)                  # neue Nummer j gehört zum alten Eingriff relabel[j]
        b = _means(inv[perm], D[:, relabel], m[relabel], K.day_end(m))
        for key in ("wait", "idle", "over"):
            assert b[key][0] == pytest.approx(a[key][0], abs=1e-9)


def test_mehr_betten_verkuerzen_blockade_und_ende_nie():
    """Blockadezeit und Tagesende (je Szenario) werden mit mehr Betten nie größer."""
    for seed in range(20):
        m, cv, Z = _inst(seed, n=5, scen=40)
        n = len(m)
        D, Rt = K.durations(m, cv, Z), K.recovery_durations(n, Z)
        P = np.array([np.random.default_rng(seed + k).permutation(n) for k in range(8)])
        T = K.day_end(m)
        comps = [K.eval_perms(P, D, m, T, Rt, b) for b in (1, 2, 3, 5)]
        for a, b in zip(comps, comps[1:]):
            assert (b["blk"] <= a["blk"] + 1e-9).all() and (b["end"] <= a["end"] + 1e-9).all()


def test_kosten_mit_leerlauf_koennen_mit_mehr_betten_steigen():
    """Befund des Tests: „mehr Betten verschlechtern die Kosten nie“ gilt NICHT für Kosten mit Leerlauf. Wer wegen der Blockade später fertig wird, hat weniger Leerlauf
    vor dem nächsten Termin; in dieser festen Folge von 20 Instanzen steigen die Kosten bei mindestens einer Reihenfolge mit einem zusätzlichen Bett."""
    steigt = 0
    for seed in range(20):
        m, cv, Z = _inst(seed, n=5, scen=40)
        n = len(m)
        D, Rt = K.durations(m, cv, Z), K.recovery_durations(n, Z)
        P = np.array([np.random.default_rng(seed + k).permutation(n) for k in range(8)])
        costs = [K.cost_from(_means(P, D, m, K.day_end(m), Rt=Rt, beds=b), 2.0) for b in (1, 2, 3, 5)]
        steigt += int(any((b > a + 1e-9).any() for a, b in zip(costs, costs[1:])))
    assert steigt >= 1


def test_ueberziehpreis_erhoeht_die_kosten_jeder_reihenfolge_nie_senkt_sie():
    for seed in range(20):
        m, cv, Z = _inst(seed)
        D = K.durations(m, cv, Z)
        P = np.array([np.random.default_rng(seed + k).permutation(len(m)) for k in range(6)])
        cm = _means(P, D, m, K.day_end(m))
        c = [K.cost_from(cm, r) for r in C.RS]
        for a, b in zip(c, c[1:]):
            assert (b >= a - 1e-12).all()


def test_gleicher_zufalls_seed_gleiche_ausgabe_und_szenarienreihenfolge_egal():
    m, cv = K.gen_instance(6, "M", 4)
    DA = K.durations(m, cv, K.normals(6, 60, 11))
    a, b = ev.evaluate_instance(m, cv, DA, rule_seed=3, rs=(2.0,)), ev.evaluate_instance(m, cv, DA, rule_seed=3, rs=(2.0,))
    assert a == b
    shuffled = DA[np.random.default_rng(0).permutation(len(DA))]
    c = ev.evaluate_instance(m, cv, shuffled, rule_seed=3, rs=(2.0,))
    for k in R.RULES + ("OPT", "ZUFALL"):
        assert c["r"][2.0][k] == pytest.approx(a["r"][2.0][k], rel=1e-12)


def test_maass_b_haengt_nicht_von_der_streuung_ab():
    m, _ = K.gen_instance(7, "M", 3)
    DA = K.durations(m, np.full(7, 0.3), K.normals(7, 20, 5))
    a = ev.evaluate_instance(m, np.full(7, 0.3), DA, rs=(1.0,))["mB"]
    b = ev.evaluate_instance(m, np.full(7, 0.3), DA, rs=(1.0,))["mB"]
    cv2 = np.full(7, 0.3)                           # gleiche Regeln brauchen gleiche Streuungsordnung: gleiche Mittel, andere Streuung ändert nur SD-Regeln
    assert a == b
    spt_value = a["SPT"]
    for cv in (np.full(7, 0.1), np.full(7, 0.8)):
        assert ev.evaluate_instance(m, cv, K.durations(m, cv, K.normals(7, 20, 5)), rs=(1.0,))["mB"]["SPT"] == pytest.approx(spt_value)
    assert cv2.shape == (7,)


def test_maass_b_spt_ist_nie_schlechter_als_jede_andere_reihenfolge():
    for seed in range(30):
        m, _, _ = _inst(seed, n=6)
        P = K.all_perms(6)
        w = K.wait_from_day_start(P, m)
        assert w.min() == pytest.approx(K.wait_from_day_start(R._order(m, 6)[None, :], m)[0])


def test_quantil_waechst_mit_tau():
    m, cv, _ = _inst(1)
    qs = [K.quantile_plan(m, cv, t) for t in C.TAUS]
    for a, b in zip(qs, qs[1:]):
        assert (b > a).all()


def test_kosten_nach_regel_sind_nie_unter_dem_optimum_n_bis_7():
    for inst in range(10):
        m, cv = K.gen_instance(6, "H", inst)
        DA = K.durations(m, cv, K.normals(6, 80, 100 + inst))
        res = ev.evaluate_instance(m, cv, DA, rule_seed=1, rs=(0.5, 8.0))
        for r, rec in res["r"].items():
            assert all(rec[k] >= rec["OPT"] - 1e-9 for k in R.RULES + ("ZUFALL",))

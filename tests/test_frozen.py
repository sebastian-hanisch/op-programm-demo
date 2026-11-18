"""Die vorgerechneten Tabellen (data/*.json): Form, innere Widersprüche, Zweig-Tests (keine Spalte, die nie wirkt) und stichprobenartiges Neu-Rechnen als Band.

Die Tabellen entstehen mit tools/opr_make_*.py (je mit --check); hier steht, was jede Tabelle erfüllen muss, ohne die Zahlen der README zu wiederholen (das tut test_claims.py)."""

import json

import numpy as np
import pytest

import opr_analysis as AN
import opr_constants as C
import opr_frozen as F
import opr_kern as K
import opr_rules as R

RULES = ("SPT", "LPT", "SD_auf", "SD_ab", "CV_auf", "NV", "ZUFALL")


def test_dateien_vorhanden_und_zellen_vollstaendig():
    assert F.files_present()
    for n in C.FROZEN_NS:
        for level in C.FROZEN_LEVELS:
            assert len(F.o1_maass_b(n, level)) == C.N_INST
            for r in C.RS:
                cell = F.o1_cell(n, level, r)
                assert [x["inst"] for x in cell] == list(range(C.N_INST))


def test_optimum_ist_nie_teurer_als_eine_regel_und_gleich_auf_beiden_saetzen_geordnet():
    for n in C.FROZEN_NS:
        for level in C.FROZEN_LEVELS:
            for r in C.RS:
                for x in F.o1_cell(n, level, r):
                    best_rule = min(x[k] for k in RULES[:6])
                    assert x["OPT"] <= best_rule + 1e-3, (n, level, r, x["inst"])      # Zahlen sind auf 4 Stellen gerundet
                    if n <= C.N_ENUM_MAX:
                        assert x["OPT"] <= x["ZUFALL"] + 1e-3


def test_stufe_k_streuungsregel_ist_spt_in_jeder_instanz():
    for n in C.FROZEN_NS:
        for r in C.RS:
            assert all(x["SD_auf"] == x["SPT"] for x in F.o1_cell(n, "K", r))
            assert all(x["SD_auf_B"] == x["SPT_B"] for x in F.o1_cell(n, "K", r))
        assert all(a["SD_auf"] == a["SPT"] for a in F.o1_maass_b(n, "K"))          # Maß B: auch die Wartezeit ab Tagesbeginn ist dieselbe


def test_zweige_jede_variante_weicht_auf_mindestens_einer_instanz_ab():
    """Eine Spalte mit lauter gleichen Werten hieße: die Variante läuft nie (toter Zweig). Gezählt über die ganze Tabelle."""
    differs = lambda a, b, cells: sum(int(abs(x[a] - x[b]) > 1e-3) for cell in cells for x in cell)
    cells = [F.o1_cell(n, lv, r) for n in C.FROZEN_NS for lv in ("L", "M", "H", "U") for r in C.RS]
    for a, b in (("SD_auf", "SPT"), ("CV_auf", "SPT"), ("NV", "SPT"), ("LPT", "SPT"), ("SD_ab", "SD_auf"), ("ZUFALL", "SPT"), ("OPT", "SD_auf")):
        assert differs(a, b, cells) > 0, (a, b)
    assert differs("NV", "SD_auf", [F.o1_cell(7, "H", r) for r in C.RS]) > 0       # der Index ordnet bei schiefen Dauern anders als die Streuung
    for col in ("opt_is_sd", "opt_is_spt", "se_diff"):
        assert sum(x[col] for cell in cells for x in cell) > 0, col
    for level in ("L", "M", "H", "U"):
        assert sum(int(x["SD_auf"] != x["SPT"]) for r in C.RS for x in F.o1_cell(8, level, r)) > 0


def test_maass_b_spt_ist_in_jeder_instanz_hoechstens_so_teuer_wie_jede_regel():
    for n in C.FROZEN_NS:
        for level in C.FROZEN_LEVELS:
            for x in F.o1_maass_b(n, level):
                assert all(x["SPT"] <= x[k] + 1e-6 for k in ("LPT", "SD_auf", "SD_ab", "CV_auf", "NV"))


def test_maass_b_werte_stimmen_mit_dem_generator_ueberein():
    for n, level, inst in ((5, "M", 3), (8, "U", 11), (12, "H", 29), (7, "K", 0)):
        m, cv = K.gen_instance(n, level, inst)
        ro = R.rule_orders(m, cv, 1)
        x = F.o1_maass_b(n, level)[inst]
        for k in ("SPT", "SD_auf", "LPT"):
            assert x[k] == pytest.approx(K.wait_from_day_start(ro[k][None, :], m)[0], abs=1e-3)


def test_vorlauf_tabelle_stimmt_bei_e_null_mit_der_haupttabelle_ueberein_und_waechst_mit_e():
    for level in C.LEVELS:
        for r in C.RS:
            main = F.o1_cell(7, level, r)
            zero = F.lead_cell(level, r, "0")
            for a, b in zip(main, zero):
                assert (a["SPT"], a["SD_auf"], a["OPT"]) == pytest.approx((b["SPT"], b["SD_auf"], b["OPT"]), abs=2e-4)
            prev = np.array([x["SPT"] for x in zero])
            for ekey in C.E_FROZEN[1:]:
                cur = np.array([x["SPT"] for x in F.lead_cell(level, r, ekey)])
                assert (cur >= prev - 1e-6).all(), (level, r, ekey)          # der Vorlauf kann die Kosten nur erhöhen
                prev = cur


def test_exakt_datei_zaehlungen_sind_konsistent():
    for n in C.EXACT_NS:
        for r, st in F.exact_stats(n).items():
            assert st["n_inst"] == C.EXACT_COUNTS[n] and st["better"] + st["equal"] + st["worse"] == st["n_inst"]
            assert 0 <= st["sd_opt"] <= st["n_inst"] and 0 <= st["spt_opt"] <= st["n_inst"]
            assert st["gap_sd_max"] >= st["gap_sd_mean"] >= 0 and st["gap_spt_mean"] >= 0


def test_quantil_datei_form_und_plan_null_ist_der_mittelwert():
    for n in C.TAU_NS:
        d = F.tau_data(n)
        assert d["A"].shape == d["B"].shape == (C.N_INST, 1 + len(C.TAUS), 3)
        assert np.isfinite(d["A"]).all() and (d["A"] >= 0).all()
    d = F.tau_data(2)
    assert (d["A"][:, 0, 2] >= 0).all()
    s = AN.tau_summary(8, 0.0, 1.0)
    assert len(s["curve_A"]) == 1 + len(C.TAUS) and s["curve_A"][0] == pytest.approx(1.0) and s["curve_B"][0] == pytest.approx(1.0)


def test_aufwachbett_datei_zellen_und_belegte_nullen():
    for level in ("D", "L", "M", "H"):
        for v in ("ohne", "B3", "B2", "B1"):
            assert len(F.o3_cell(level, v, 2.0)) == C.N_INST
    assert len(F.o3_cell("D", "B1d", 2.0)) == len(F.o3_cell("D", "B2d", 2.0)) == C.N_INST
    assert all(x["blk_spt"] == 0 for x in F.o3_cell("M", "ohne", 2.0))                          # ohne Blockade keine Blockadezeit (belegte Null)
    assert max(x["blk_spt"] for x in F.o3_cell("M", "B3", 2.0)) < 0.002                          # drei Betten: höchstens 0,2 % der Eingriffsdauern
    assert min(x["blk_spt"] for x in F.o3_cell("M", "B1", 2.0)) > 0                              # ein Bett: in jeder Instanz positiv


def test_neu_rechnen_zweier_instanzen_im_band():
    """Wie `tools/opr_make_o1.py --check`, aber in der Suite: n = 5, Stufe M, Instanz 3 gegen die Datei (Regeln relativ 1e-6, Optimum relativ 2e-3)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import opr_make_o1 as M
    za, zb = M.blocks(5, 3)
    res = M.one_instance(5, "M", 3, za, zb, False)
    rows, _ = M.rows_of(5, "M", 3, res)
    stored = {tuple(r[:4]): r for r in json.loads((F.DATA / "opr_o1_tabelle.json").read_text(encoding="utf-8"))["rows"]}
    for row in rows:
        old = stored[tuple(row[:4])]
        for col, a, b in zip(M.COLS[4:], row[4:], old[4:]):
            if col in ("opt_is_sd", "opt_is_spt"):
                continue
            assert a == pytest.approx(b, rel=2e-3 if col.startswith("OPT") else 1e-6, abs=1e-4), (row[3], col)

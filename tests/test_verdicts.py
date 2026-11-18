"""Verdikte an von Hand gesetzten Werten: jede Schwelle und jede Zahl im Text steht hier (DEMO-PLAYBOOK §12)."""

from fractions import Fraction as F

import opr_evaluation as ev


def _cell(better=28, equal=2, worse=0, delta=-0.0944, gap_sd=0.0015, gap_spt=0.098, better_b=27):
    return {"better": better, "equal": equal, "worse": worse, "delta": delta, "gap_sd": gap_sd, "gap_spt": gap_spt, "better_B": better_b}


LIVE = {"seed": 7, "scen": 500, "delta": -0.0883, "se_rel": 0.0123, "gap_sd": 0.0021, "gap_spt": 0.0987, "enum": True, "n": 7}
INFO = {"n_cell": 7, "level": "M", "r": 2.0}


def test_verdict_order_nennt_live_zahlen_zaehlung_und_luecken():
    kind, text = ev.verdict_order(LIVE, _cell(), INFO)
    assert kind == "success"
    assert "Zufalls-Seed 7, 500 Szenarien" in text and "8,8 % weniger als SPT (± 1,2 %)" in text
    assert "5.040 Reihenfolgen" in text and "0,21 % über dem Stichproben-Optimum" in text and "SPT 9,87 %" in text
    assert "besser in 28, gleich in 2, schlechter in 0" in text and "mittlere Differenz -9,4 %" in text
    assert "besser in 27 von 30" in text and "n = 7, Stufe M, r = 2,0" in text and "in der Literatur bekannt" in text
    assert "Lücke zum Stichproben-Optimum: SPT 9,80 %, Streuungsregel 0,15 %" in text


def test_verdict_order_schwelle_20_von_30_ist_inklusive():
    assert ev.verdict_order(LIVE, _cell(better=20, equal=5, worse=5), INFO)[0] == "success"
    assert ev.verdict_order(LIVE, _cell(better=19, equal=6, worse=5), INFO)[0] == "info"


def test_verdict_order_gleich_teuer_und_lokale_verbesserung_ohne_optimum_wort():
    live = dict(LIVE, delta=0.0, enum=False, n=12)
    kind, text = ev.verdict_order(live, _cell(better=0, equal=30, delta=0.0), dict(INFO, n_cell=12, level="K"))
    assert kind == "info" and "gleich teuer wie SPT" in text
    assert "beste lokale Verbesserung (kein Optimum" in text and "Lücke zur besten lokalen Verbesserung (kein Optimum)" in text
    assert "Stichproben-Optimum (alle" not in text


def _lead(deltas):
    return {k: {"better": 28 - i, "equal": 0, "worse": 2 + i, "delta": d} for i, (k, d) in enumerate(zip(("0", "10", "20", "40", "80", "160", "∞"), deltas))}


MB = {"ratio": 1.065, "spt_le_sd": 30, "n_inst": 30, "ratio_min": 1.0, "ratio_max": 1.1}


def test_verdict_mass_kipppunkt_zwischen_zwei_rasterwerten():
    kind, text = ev.verdict_mass("M", 1.0, _lead([-0.094, -0.074, -0.061, -0.044, -0.020, 0.006, 0.045]), MB, "80")
    assert kind == "info"
    assert "Vorlauf 0 min: Streuungsregel besser als SPT in 28 von 30 (Δ -9,4 %)" in text
    assert "zwischen e = 80 und e = 160 min" in text and "Lage der Kippstelle" in text and "nicht belegt" in text
    assert "1,065fache Wartezeit" in text and "30 von 30 Instanzen höchstens so teuer" in text and "n = 7" in text


def test_verdict_mass_ohne_vorzeichenwechsel_und_wechsel_schon_bei_null():
    assert "wechselt das Vorzeichen von Δ nicht" in ev.verdict_mass("L", 1.0, _lead([-0.1] * 7), MB, "0")[1]
    assert "Schon bei e = 0" in ev.verdict_mass("L", 1.0, _lead([0.01] * 7), MB, "0")[1]
    assert ev.lead_tipping({"0": -1, "10": -1, "20": -1, "40": -1, "80": -1, "160": -1, "∞": 0.1}) == "∞"


def _per(zk=0.288, zm=0.295, zu=0.049):
    mk = lambda z, d, e, gap: {"equal": e, "better": 0 if e == 30 else 30, "delta": d, "gap_sd": gap, "rel": {"ZUFALL": 1 + z}}
    return {"K": mk(zk, 0.0, 30, 0.001), "M": mk(zm, -0.089, 0, 0.0013), "U": mk(zu, -0.259, 0, 0.019)}


def test_verdict_coupling_nennt_drei_kopplungen_und_kennzeichnet_die_deutung():
    kind, text = ev.verdict_coupling(8, 2.0, _per())
    assert kind == "info"
    assert "dieselbe Reihenfolge" in text and "in 30 von 30" in text
    assert "Zufall 28,8 % über SPT" in text and "Zufall 29,5 % über SPT" in text and "Zufall +4,9 % gegen SPT" in text
    assert "Deutung, nicht gemessen als Ursache" in text


def _lines(sd, spt, opt, same):
    return {"opt_name": "B–A–C" if same else "B–C–A", "opt_cost": F(opt), "sd_name": "B–A–C", "sd_cost": F(sd), "spt_name": "A–B–C", "spt_cost": F(spt)}


def test_verdict_exact_streuungsregel_optimal_und_nicht_optimal():
    kind, text = ev.verdict_exact("Handbeispiel", F(2), _lines(F(307, 4), F(337, 4), F(307, 4), True), None)
    assert kind == "success" and "76,75 = 307/4" in text and "84,25 = 337/4" in text and "exakt optimal" in text
    assert "SPT kostet 9,8 % mehr" in text and "8,9 % weniger als SPT" in text
    kind, text = ev.verdict_exact("Handbeispiel", F(8), _lines(172, F(359, 2), 162, False), None)
    assert kind == "warning" and "Optimum ist B–C–A mit 162" in text and "6,2 % darüber" in text


def test_verdict_exact_zaehlung_nennt_die_voraussetzung_symmetrische_zweipunkt_verteilung():
    stat = {"n_inst": 2000, "n": 3, "sd_opt": 1004, "spt_opt": 654}
    _, text = ev.verdict_exact("Zufallsinstanz", F(8), _lines(F(5), F(6), F(5), True), stat)
    assert "In 1.004 von 2.000 zufälligen Zweipunkt-Instanzen mit n = 3" in text and "SPT in 654" in text and "symmetrische Zweipunkt-Verteilung" in text


def test_verdict_tau_nennt_tau_stern_gewinn_und_median():
    summ = {"tau": 0.7, "rel_B": 0.883, "rel_B_median": 1.119, "nv": 0.5}
    kind, text = ev.verdict_tau(8, 0.0, "1 : 1", summ, {"mean": 0.568})
    assert kind == "info" and "bestes τ* = 0,70" in text and "0,883fache" in text and "1,119fache" in text and "Quantil 0,568" in text and "Zeitungsverkäufer-Verhältnis" in text
    assert "Der Gewinn ist hier klein" not in text
    assert "Der Gewinn ist hier klein" in ev.verdict_tau(3, 1.0, "1 : 1", dict(summ, rel_B=0.996), {"mean": 0.57})[1]
    assert "Hier bleibt der Mittelwert die beste geplante Dauer" in ev.verdict_tau(3, 1.0, "1 : 2", dict(summ, tau=None, rel_B=1.0), {"mean": 0.57})[1]


def test_verdict_beds_nennt_verhaeltnisse_zaehlungen_und_wechselwirkung():
    blk = lambda ratio, sd, lpt, other, lbs: {"ratio": ratio, "gap_SD_auf": sd, "gap_LPT": lpt, "other_best": other, "lpt_better_sd": lbs}
    s = {"B1": blk(1.53, 0.05, 0.564, 22, 0), "B2": blk(1.006, 0.0018, 0.634, 10, 0), "B3": blk(1.0, 0.0016, 0.62, 3, 0), "ohne": {"gap_SD_auf": 0.0016, "gap_LPT": 0.622}}
    dec = {"M": {"zusatz": 76.0, "zusatz_vs_blockade_allein": 1.6}, "blockade_allein": 46.5, "fest_fest": 11.4}
    kind, text = ev.verdict_beds("M", 2.0, s, dec)
    assert kind == "info" and "das 1,53fache" in text and "bei zwei Betten 1,006" in text and "bei drei 1,000" in text
    assert "5,0 % über dem Optimum" in text and "in 22 von 30 Instanzen eine andere" in text
    assert "76,0 Kosteneinheiten" in text and "das 1,6fache" in text and "(46,5)" in text and "56 % über dem Optimum" in text and "in 0 von 30 Instanzen billiger" in text
    assert "tritt in diesem Modell auch ohne Blockade auf" in text and "(Stichproben-Optimum)" in text and "Eigenschaft des Maßes" not in text

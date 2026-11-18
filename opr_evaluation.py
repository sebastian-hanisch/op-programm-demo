"""Auswertung: Regeln und Optimum je Instanz (Live und gefroren), Zusammenfassungen über 30 Instanzen, Reiter „Geplante Dauer“ und „Aufwachbett“.

Verdikte (reine Funktionen `verdict_*` -> (Art, Text)) stehen am Ende der Datei; die App ruft `getattr(st, Art)(Text)`."""

import math
from fractions import Fraction

import numpy as np

import opr_constants as C
import opr_kern as K
import opr_rules as R
from opr_rng import SplitMix64

RULE_KEYS = R.RULES                      # SPT, LPT, SD_auf, SD_ab, CV_auf, NV


def fmt(x: float, digits: int = 1) -> str:
    """Dezimalzahl mit Komma."""
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x: float, digits: int = 1, sign: bool = False) -> str:
    s = f"{x * 100:+.{digits}f}" if sign else f"{x * 100:.{digits}f}"
    return s.replace(".", ",") + " %"


def fmt_fraction(x: Fraction) -> str:
    """Bruch als Dezimalzahl und, falls nicht ganz, als Bruch: '76,75 = 307/4'."""
    dec = f"{float(x):.4f}".rstrip("0").rstrip(".").replace(".", ",")
    return dec if x.denominator == 1 else f"{dec} = {x.numerator}/{x.denominator}"


# --- Instanz: Regeln, Optimum, Probe auf zweitem Szenariensatz ------------------------------------------------------------------------------

def scenario_seed(which: str, n: int, tag: int) -> int:
    return K.seed_of(C.SCEN_SEED[which], n, tag)


def evaluate_instance(m, cv, DA, DB=None, rule_seed: int = 1, rs=C.RS, lead=False, enum_max: int = C.N_ENUM_MAX) -> dict:
    """Kosten der Regeln und des (Stichproben-)Optimums für eine Instanz, je Überziehpreis r (Maß A, Termine nach den Mittelwerten).

    DA: Szenarien, auf denen das Optimum gewählt und bewertet wird; DB (optional): unabhängiger zweiter Satz, der nur bewertet.
    Rückgabe: {"r": {r: record}, "mB": {Regel: Wartezeit-Summe Maß B (geschlossen)}, "lead": {...} (nur mit lead=True und n <= enum_max)}."""
    m = np.asarray(m, float)
    cv = np.asarray(cv, float)
    n = len(m)
    T = K.day_end(m)
    ro = R.rule_orders(m, cv, rule_seed)
    rule_perms = np.array([ro[k] for k in RULE_KEYS] + list(ro["ZUFALL"]))
    nr = len(RULE_KEYS)
    iSPT, iSD = RULE_KEYS.index("SPT"), RULE_KEYS.index("SD_auf")
    compA = K.eval_perms(rule_perms, DA, m, T)
    cmA = K.comp_means(compA)
    cmB = K.comp_means(K.eval_perms(rule_perms, DB, m, T)) if DB is not None else None
    enum = n <= enum_max
    if enum:
        P = K.all_perms(n)
        cmAll = K.enum_means(P, DA, m, T)
    cache: dict = {}
    out = {"r": {}, "mB": {k: float(K.wait_from_day_start(ro[k][None, :], m)[0]) for k in RULE_KEYS}}
    for r in rs:
        cS = K.cost_from(compA, r)
        cM = K.cost_from(cmA, r)
        rec = {k: float(cM[i]) for i, k in enumerate(RULE_KEYS)}
        rec["ZUFALL"] = float(cM[nr:].mean())
        d = cS[iSD] - cS[iSPT]
        rec["se_diff"] = float(d.std(ddof=1) / math.sqrt(DA.shape[0])) if DA.shape[0] > 1 else 0.0
        if enum:
            c = K.cost_from(cmAll, r)
            j = int(c.argmin())
            opt_perm, opt_cost = P[j], float(c[j])
        else:
            best = min(RULE_KEYS, key=lambda k: rec[k])
            opt_perm, opt_cost = R.local_improvement(ro[best], DA, m, T, lambda comp, r=r: K.cost_from(comp, r), cache=cache)
        rec["OPT"] = opt_cost
        rec["opt_perm"] = [int(x) for x in opt_perm]
        rec["opt_is_sd"] = int(np.array_equal(opt_perm, ro["SD_auf"]))
        rec["opt_is_spt"] = int(np.array_equal(opt_perm, ro["SPT"]))
        if cmB is not None:
            cB = K.cost_from(cmB, r)
            for i, k in enumerate(RULE_KEYS):
                rec[k + "_B"] = float(cB[i])
            rec["ZUFALL_B"] = float(cB[nr:].mean())
            rec["OPT_B"] = float(K.cost_from(K.comp_means(K.eval_perms(opt_perm[None, :], DB, m, T)), r)[0])
        out["r"][r] = rec
    if lead and enum:
        lead_rec = {}
        spt_sd = np.array([ro["SPT"], ro["SD_auf"]])
        for r in rs:
            base_all = K.cost_from(cmAll, r)
            baseA = K.cost_from(cmA, r)
            baseB = K.cost_from(cmB, r) if cmB is not None else None
            lead_rec[r] = {}
            for ekey in C.E_FROZEN:
                e = C.E_VALUES[ekey]
                c = base_all + K.lead_term(P, m, e)
                ls = K.lead_term(spt_sd, m, e)
                row = [float(baseA[iSPT] + ls[0]), float(baseA[iSD] + ls[1]), float(c.min())]
                if baseB is not None:
                    row += [float(baseB[iSPT] + ls[0]), float(baseB[iSD] + ls[1])]
                lead_rec[r][ekey] = row
        out["lead"] = lead_rec
    return out


# --- Zusammenfassung über Instanzen -------------------------------------------------------------------------------------------------------

def bootstrap_ci(values, seed: int = 271828, reps: int = 2000):
    """95-%-Intervall des Mittelwerts (Bootstrap mit SplitMix64, plattformfest)."""
    x = np.asarray(values, float)
    g = SplitMix64(seed)
    idx = np.array([[g.below(len(x)) for _ in range(len(x))] for _ in range(reps)])
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(recs: list, with_b: bool = True, ci: bool = True) -> dict:
    """Zusammenfassung einer Zelle (Liste von Instanz-Records eines r): mittlere relative Differenz, Zählungen, Lücken zum Optimum.

    Δ = (Streuung aufsteigend - SPT) / SPT, gemittelt über die Instanzen; Zählung: besser / gleich / schlechter als SPT (Instanzen)."""
    g = lambda k: np.array([x[k] for x in recs])
    spt, sd, opt = g("SPT"), g("SD_auf"), g("OPT")
    dif = np.where(spt > 0, (sd - spt) / np.where(spt > 0, spt, 1.0), 0.0)
    tol = 1e-9 * np.maximum(spt, 1.0)
    better = int(np.sum(sd < spt - tol))
    equal = int(np.sum(np.abs(sd - spt) <= tol))
    se = g("se_diff")
    res = {
        "n_inst": len(recs), "delta": float(dif.mean()), "better": better, "equal": equal, "worse": len(recs) - better - equal,
        "signif_better": int(np.sum(sd - spt < -2 * se)),
        "gap_sd": float(np.mean(sd / opt - 1)), "gap_spt": float(np.mean(spt / opt - 1)),
        "gap_cv": float(np.mean(g("CV_auf") / opt - 1)), "gap_nv": float(np.mean(g("NV") / opt - 1)),
        "rel": {k: float(np.mean(g(k) / np.where(spt > 0, spt, 1.0))) for k in ("LPT", "SD_auf", "SD_ab", "CV_auf", "NV", "ZUFALL", "OPT")},
        "opt_is_sd": int(g("opt_is_sd").sum()), "opt_is_spt": int(g("opt_is_spt").sum()),
    }
    if ci:
        lo, hi = bootstrap_ci(dif)
        res["delta_lo"], res["delta_hi"] = lo, hi
    if with_b and "SPT_B" in recs[0]:
        sdB, sptB = g("SD_auf_B"), g("SPT_B")
        res["better_B"] = int(np.sum(sdB < sptB - 1e-9 * np.maximum(sptB, 1.0)))
        res["opt_B_over_A"] = float(np.mean(g("OPT_B") / opt))
        res["gap_sd_B_vs_optA"] = float(np.mean(sdB / g("OPT_B") - 1))
    return res


# --- Reiter „Geplante Dauer“ -----------------------------------------------------------------------------------------------------------------

def tau_curve(m, cv, DA, perm, r: float, cw: float, taus=C.TAUS):
    """Mittlere Kosten einer festen Reihenfolge bei geplanter Dauer = Mittelwert (erste Zahl) und = Quantil tau (je Raster-Wert); T wie im übrigen Modell.
    Bei r = 0 zählt das Überziehen nicht."""
    m = np.asarray(m, float)
    T = K.day_end(m)
    plans = [m] + [K.quantile_plan(m, cv, t) for t in taus]
    costs = []
    perm = np.asarray(perm)[None, :]
    for pl in plans:
        cm = K.comp_means(K.eval_perms(perm, DA, pl, T))
        costs.append(float(K.cost_from(cm, r, cw=cw)[0]))
    return np.array(costs)


def newsvendor_ratio(cw: float) -> float:
    """Zeitungsverkäufer-Verhältnis cw / (cw + 1): bester Quantilswert für einen einzelnen Übergang bei Leerlauf-Gewicht 1."""
    return cw / (cw + 1.0)


def mean_quantile_level(cv) -> np.ndarray:
    """Das Quantil, bei dem der Mittelwert einer Lognormalverteilung liegt: Phi(sigma / 2) mit sigma^2 = ln(1 + c^2) (geschlossen)."""
    sig = np.sqrt(np.log1p(np.asarray(cv, float) ** 2))
    return np.array([K._ND.cdf(float(s) / 2.0) for s in sig])


# --- Verdikte (reine Funktionen: (Art, Text)) -------------------------------------------------------------------------------------------------

def _thousands(x: int) -> str:
    return f"{x:,}".replace(",", ".")          # nur die Zahl umformatieren, nie den ganzen Satz


def _cmp(delta: float, digits: int = 1) -> str:
    """„x % weniger als SPT“ / „x % mehr als SPT“ / „gleich teuer wie SPT“ für Δ = (Streuungsregel - SPT) / SPT."""
    if abs(delta) < 5 * 10 ** (-digits - 3):
        return "gleich teuer wie SPT"
    return f"{pct(abs(delta), digits)} {'weniger' if delta < 0 else 'mehr'} als SPT"


def nearest_frozen_n(n: int) -> int:
    return min(C.FROZEN_NS, key=lambda x: (abs(x - n), x))


def verdict_order(live: dict, cell: dict, info: dict):
    """Experiment 1: eine Live-Instanz und die vorgerechnete Zelle (30 Instanzen). live: seed, scen, delta, se_rel, gap_sd, gap_spt, enum, n; cell: summarize(...);
    info: n_cell, level, r."""
    if live["enum"]:
        opt = (f"die Streuungsregel liegt {pct(live['gap_sd'], 2)} über dem Stichproben-Optimum (alle {_thousands(math.factorial(live['n']))} Reihenfolgen, auf denselben "
               f"Szenarien gewählt), SPT {pct(live['gap_spt'], 2)}")
    else:
        opt = (f"gegen die beste lokale Verbesserung (kein Optimum; Start bei der besten Regel) liegt die Streuungsregel {pct(live['gap_sd'], 2)} darüber, SPT {pct(live['gap_spt'], 2)}")
    tag = "zum Stichproben-Optimum" if info["n_cell"] <= C.N_ENUM_MAX else "zur besten lokalen Verbesserung (kein Optimum)"
    live_txt = (f"Instanz Zufalls-Seed {live['seed']}, {live['scen']} Szenarien: Streuung aufsteigend kostet {_cmp(live['delta'])} (± {pct(live['se_rel'])}); {opt}. ")
    cell_txt = (f"In den 30 erfundenen Instanzen der vorgerechneten Zelle (n = {info['n_cell']}, Stufe {info['level']}, r = {fmt(info['r'], 1)}; 2 000 Szenarien): "
                f"besser in {cell['better']}, gleich in {cell['equal']}, schlechter in {cell['worse']}; mittlere Differenz {pct(cell['delta'], 1, True)}; "
                f"Lücke {tag}: SPT {pct(cell['gap_spt'], 2)}, Streuungsregel {pct(cell['gap_sd'], 2)}")
    if "better_B" in cell:
        cell_txt += f"; auf einem zweiten Szenariensatz besser in {cell['better_B']} von 30"
    if "opt_is_sd" in cell:
        cell_txt += (f"; die Streuungsregel ist in {cell['opt_is_sd']} von 30 Instanzen genau die Reihenfolge des {'Stichproben-Optimums' if info['n_cell'] <= C.N_ENUM_MAX else 'Ergebnisses der besten lokalen Verbesserung'} "
                     f"(nahe am Optimum heißt nicht: das Optimum)")
    cell_txt += ". Gilt für Maß A, Termine nach den Mittelwerten und diese erfundenen Dauern; die Faustregel selbst ist in der Literatur bekannt."
    kind = "success" if cell["better"] >= 20 else "info"
    return kind, live_txt + cell_txt


def lead_tipping(deltas: dict):
    """Erster Wert von e (Reihenfolge der Raster), bei dem Δ >= 0 ist; None, wenn Δ im ganzen Raster negativ bleibt. deltas: {ekey: Δ}."""
    for ekey in C.E_FROZEN:
        if deltas[ekey] >= 0:
            return ekey
    return None


def verdict_mass(level: str, r: float, lead: dict, mb: dict, mass: str):
    """Experiment 2: Δ je Vorlauf e (n = 7, 30 Instanzen, vorgerechnet) und Maß B. lead: {ekey: lead_summary}; mb: maass_b_summary; mass: gewählte Stufe des Reglers."""
    def part(ekey):
        s = lead[ekey]
        name = "∞ (Wartezeit ab Tagesbeginn zusätzlich)" if ekey == "∞" else f"{ekey} min"
        return f"Vorlauf {name}: Streuungsregel besser als SPT in {s['better']} von 30 (Δ {pct(s['delta'], 1, True)})"
    tip = lead_tipping({k: v["delta"] for k, v in lead.items()})
    if tip is None:
        tip_txt = "Im Raster bis e = 160 min (und ∞) wechselt das Vorzeichen von Δ nicht."
    elif tip == "0":
        tip_txt = "Schon bei e = 0 ist Δ nicht negativ."
    else:
        prev = C.E_FROZEN[C.E_FROZEN.index(tip) - 1]
        tip_txt = (f"Das Vorzeichen von Δ wechselt zwischen e = {prev} und e = {tip if tip != '∞' else '∞ (Wartezeit ab Tagesbeginn zusätzlich)'}"
                   f"{' min' if tip != '∞' else ''} (Klassenmittel 45 / 90 / 150 min); Δ liegt dort nahe 0, die genaue Lage der Kippstelle zwischen den Rasterwerten ist nicht belegt.")
    maß = (f"Maß B (alle Patienten von Tagesbeginn an da, keine Termine): SPT ist in {mb['spt_le_sd']} von {mb['n_inst']} Instanzen höchstens so teuer wie die Streuungsregel "
           f"(geschlossene Formel); die Streuungsregel hat im Mittel die {fmt(mb['ratio'], 3)}fache Wartezeit.")
    txt = (f"Stufe {level}, r = {fmt(r, 1)}, n = 7 (vorgerechnet, 30 Instanzen): {part('0')}; {part('80')}; {part('160')}; {part('∞')}. {tip_txt} {maß} "
           f"Das gilt für dieses erfundene Maß, nicht für ein Krankenhaus.")
    return "info", txt


def verdict_coupling(n_cell: int, r: float, per: dict):
    """Experiment 3: per = {'K': summarize, 'M': ..., 'U': ...} (30 Instanzen je Zelle)."""
    K_, M_, U_ = per["K"], per["M"], per["U"]
    txt = (f"n = {n_cell}, r = {fmt(r, 1)}, 30 Instanzen je Kopplung (vorgerechnet). Konstanter Variationskoeffizient: Streuungsregel und SPT sind in {K_['equal']} von 30 Instanzen "
           f"dieselbe Reihenfolge, Zufall {pct(K_['rel']['ZUFALL'] - 1)} über SPT; die beiden Regeln fallen nur bei gleichem Variationskoeffizienten zusammen. Stufe M (die absolute Streuung wächst mit dem Mittel nur im Mittel, die Regeln unterscheiden sich): Zufall {pct(M_['rel']['ZUFALL'] - 1)} über SPT, Streuungsregel {pct(M_['delta'], 1, True)}. "
           f"Streuung unabhängig vom Mittel: Zufall {pct(U_['rel']['ZUFALL'] - 1, 1, True)} gegen SPT, Streuungsregel {pct(U_['delta'], 1, True)} "
           f"(besser in {U_['better']} von 30); Lücke der Streuungsregel zur besten lokalen Verbesserung bzw. zum Stichproben-Optimum {pct(U_['gap_sd'], 1)}, in Stufe M {pct(M_['gap_sd'], 2)}. "
           f"Deutung, nicht gemessen als Ursache: SPT sieht nur dort gut aus, wo die Streuung mit dem Mittel wächst.")
    return "info", txt


def verdict_exact(src: str, r, lines: dict, stat: dict | None):
    """Experiment 4: lines = {opt, opt_name, opt_cost, sd_name, sd_cost, spt_name, spt_cost} (Brüche); stat = gefrorene Zählung zu (n, r) oder None."""
    o, sd, sp = lines["opt_cost"], lines["sd_cost"], lines["spt_cost"]
    txt = f"Exakte Erwartungswerte, kein Stichprobenfehler ({src}), r = {fmt(float(r), 1)}: Streuungsregel {lines['sd_name']} {fmt_fraction(sd)}, SPT {lines['spt_name']} {fmt_fraction(sp)}"
    if sp != sd:
        txt += f" (SPT kostet {pct(float(sp / sd - 1))} mehr; die Streuungsregel {pct(float(1 - sd / sp))} weniger als SPT)"
    if sd == o:
        txt += f"; die Streuungsregel ist hier exakt optimal ({lines['opt_name']})"
    else:
        txt += f"; das Optimum ist {lines['opt_name']} mit {fmt_fraction(o)}, die Streuungsregel liegt {pct(float(sd / o - 1))} darüber"
    if stat is not None:
        txt += (f". In {_thousands(stat['sd_opt'])} von {_thousands(stat['n_inst'])} zufälligen Zweipunkt-Instanzen mit n = {stat['n']} ist die Streuungsregel bei r = {fmt(float(r), 1)} exakt optimal "
                f"(SPT in {_thousands(stat['spt_opt'])}); das gilt für diese symmetrische Zweipunkt-Verteilung, bei schiefen Lognormal-Dauern (Experiment 1) ist die Regel nicht in jeder Instanz optimal")
    return ("success" if sd == o else "warning"), txt + "."


def verdict_tau(n: int, r: float, ratio: str, summ: dict, mean_level: dict):
    """Reiter Geplante Dauer: bestes Quantil tau* auf Satz A, Gewinn auf Satz B. summ = tau_summary; mean_level = tau_phi(n)."""
    star = "Mittelwert" if summ["tau"] is None else f"{fmt(summ['tau'], 2)}"
    gain = summ["rel_B"]
    txt = (f"Stufe M, n = {n}, Wartezeit : Leerlauf = {ratio}, r = {fmt(r, 0)} (vorgerechnet, 30 Instanzen): bestes τ* = {star} (Zeitungsverkäufer-Verhältnis für einen Übergang, ohne Überziehpreis: "
           f"{fmt(summ['nv'], 2)}; der Mittelwert der Lognormalverteilung liegt im Mittel beim Quantil {fmt(mean_level['mean'], 3)}). Auf dem unabhängigen Satz B kostet die Planung mit τ* das "
           f"{fmt(gain, 3)}fache der Mittelwert-Planung, die Planung mit dem Median (τ = 0,50) das {fmt(summ['rel_B_median'], 3)}fache. ")
    if summ["tau"] is None:
        txt += "Hier bleibt der Mittelwert die beste geplante Dauer. "
    elif summ["tau"] < mean_level["mean"] - 0.01:
        txt += "τ* liegt unter dem Quantil des Mittelwerts: hier lohnt eine knappere Planung als der Mittelwert. "
    elif gain > 0.99:
        txt += "Der Gewinn ist hier klein. "
    txt += "Das Verhältnis gilt für einen Übergang, nicht für einen Tag; die Abhängigkeit von n ist gemessen, die Ursache (Übertrag der Verspätung) eine Deutung."
    return "info", txt


def verdict_beds(level: str, r: float, s: dict, dec: dict):
    """Reiter Aufwachbett: s = o3_summary(level, r), dec = o3_decomposition(r) (30 Instanzen, n = 6, vorgerechnet)."""
    b1, b2, b3 = s["B1"], s["B2"], s["B3"]
    d = dec[level] if level in dec else None
    txt = (f"Stufe {level}, r = {fmt(r, 0)}, n = 6 (vorgerechnet, 30 Instanzen): Kosten des besten Tagesplans bei einem Bett das {fmt(b1['ratio'], 2)}fache von „ohne Blockade“, "
           f"bei zwei Betten {fmt(b2['ratio'], 3)}, bei drei {fmt(b3['ratio'], 3)}. Bei einem Bett liegt die Streuungsregel {pct(b1['gap_SD_auf'], 1)} über dem Optimum "
           f"(ohne Blockade {pct(s['ohne']['gap_SD_auf'], 2)}), bei zwei Betten {pct(b2['gap_SD_auf'], 2)}; die beste Reihenfolge ist bei einem Bett in {b1['other_best']} von 30 Instanzen "
           f"eine andere als ohne Blockade (Stichproben-Optimum). ")
    if d is not None:
        txt += (f"Die Blockade fügt bei streuenden Eingriffen {fmt(d['zusatz'], 1)} Kosteneinheiten hinzu, das {fmt(d['zusatz_vs_blockade_allein'], 1)}fache ihrer Kosten bei festen "
                f"Dauern ({fmt(dec['blockade_allein'], 1)}): eine Wechselwirkung. ")
    txt += (f"„Längste zuerst“ liegt bei einem Bett {pct(b1['gap_LPT'], 0)} über dem Optimum und war in {b1['lpt_better_sd']} von 30 Instanzen billiger als die Streuungsregel; "
            f"auch ohne Blockade liegt sie {pct(s['ohne']['gap_LPT'], 0)} darüber: das tritt in diesem Modell auch ohne Blockade auf.")
    return "info", txt

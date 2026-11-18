"""Live-Rechnungen der App und Zusammenfassungen der gefrorenen Tabellen (Reiter Reihenfolge, Geplante Dauer, Aufwachbett).

Baut auf `opr_evaluation.evaluate_instance` auf; alles reine Funktionen ohne Streamlit."""

import math

import numpy as np

import opr_constants as C
import opr_evaluation as ev
import opr_frozen as F
import opr_kern as K
import opr_rules as R

RULE_KEYS = R.RULES


def dekemp_condition(m, cv) -> bool:
    """Hinreichende Ordnungsbedingung für lognormale Dauern (de Kemp u. a. 2021, Prop. D.1, im Wortlaut gelesen): E ln D und Var ln D wachsen nicht fallend in der Reihenfolge
    „Streuung aufsteigend“. Nur eine hinreichende Bedingung; die Schranken dieser Arbeit werden in der Demo nicht übernommen."""
    order = R.rule_orders(m, cv, 1)["SD_auf"]
    mu, sg = K.ln_params(m, cv)
    mu, sg = mu[order], sg[order]
    return bool(np.all(np.diff(mu) >= -1e-12) and np.all(np.diff(sg) >= -1e-12))


def dekemp_share(n: int, level: str, count: int = 2000) -> float:
    """Anteil der ersten `count` Instanzen (gen_instance) einer Zelle, die die Bedingung erfüllen."""
    ok = 0
    for inst in range(count):
        m, cv = K.gen_instance(n, level, inst)
        ok += int(dekemp_condition(m, cv))
    return ok / count


# --- Live-Rechnung einer Instanz (Reiter Reihenfolge) ---------------------------------------------------------------------------------------

def live_order(m, cv, DA, DB=None, r: float = C.DEFAULT_R, rule_seed: int = 1) -> dict:
    """Eine Instanz live: Regeln und (Stichproben-)Optimum auf DA (optional Probe auf DB), dazu die Komponenten je Regel für die Balken und die Szenario-Kosten
    von SPT und Streuungsregel für Histogramm und Standardfehler."""
    m = np.asarray(m, float)
    cv = np.asarray(cv, float)
    res = ev.evaluate_instance(m, cv, DA, DB, rule_seed, rs=(r,))
    rec = res["r"][r]
    T = K.day_end(m)
    ro = R.rule_orders(m, cv, rule_seed)
    perms = np.array([ro[k] for k in RULE_KEYS] + [rec["opt_perm"]])
    cm = K.comp_means(K.eval_perms(perms, DA, m, T))
    comps = {name: {k: float(cm[k][i]) for k in ("wait", "idle", "over")} for i, name in enumerate(list(RULE_KEYS) + ["OPT"])}
    two = K.eval_perms(np.array([ro["SPT"], ro["SD_auf"]]), DA, m, T)
    scen_cost = K.cost_from(two, r)
    return {"rec": rec, "mB": res["mB"], "orders": {k: [int(x) for x in ro[k]] for k in RULE_KEYS}, "comps": comps,
            "scen_spt": scen_cost[0], "scen_sd": scen_cost[1], "n": len(m), "r": r}


def live_lead(m, cv, DA, ekeys=C.E_FROZEN) -> dict:
    """Δ(e) für eine Instanz live: SPT gegen Streuungsregel bei Vorlauf e (Kosten = Maß A + Summe min(Termin, e)), mit Standardfehler der gepaarten Differenz je Szenario."""
    m = np.asarray(m, float)
    cv = np.asarray(cv, float)
    T = K.day_end(m)
    ro = R.rule_orders(m, cv)
    two = np.array([ro["SPT"], ro["SD_auf"]])
    comp = K.eval_perms(two, DA, m, T)
    out = {}
    for r in C.RS:
        base = K.cost_from(comp, r)
        out[r] = {}
        for ekey in ekeys:
            lt = K.lead_term(two, m, C.E_VALUES[ekey])
            c = base + lt[:, None]
            d = c[1] - c[0]
            out[r][ekey] = {"spt": float(c[0].mean()), "sd": float(c[1].mean()), "se": float(d.std(ddof=1) / math.sqrt(len(d)))}
    return out


# --- Zusammenfassungen aus den gefrorenen Tabellen ----------------------------------------------------------------------------------------

def lead_summary(recs: list) -> dict:
    """Δ, Zählung besser/gleich/schlechter und Lücken zum Stichproben-Optimum für eine Zelle (Stufe, r, e) der Vorlauf-Tabelle (n = 7)."""
    spt = np.array([x["SPT"] for x in recs])
    sd = np.array([x["SD_auf"] for x in recs])
    opt = np.array([x["OPT"] for x in recs])
    tol = 1e-9 * np.maximum(spt, 1.0)
    better = int(np.sum(sd < spt - tol))
    equal = int(np.sum(np.abs(sd - spt) <= tol))
    sptB = np.array([x["SPT_B"] for x in recs])
    sdB = np.array([x["SD_auf_B"] for x in recs])
    return {"delta": float(((sd - spt) / spt).mean()), "better": better, "equal": equal, "worse": len(recs) - better - equal,
            "gap_sd": float(np.mean(sd / opt - 1)), "gap_spt": float(np.mean(spt / opt - 1)),
            "better_B": int(np.sum(sdB < sptB - 1e-9 * np.maximum(sptB, 1.0)))}


def maass_b_summary(recs: list) -> dict:
    """Maß B: Verhältnis der Wartezeit der Streuungsregel zu SPT (Mittel über Instanzen) und Zählung SPT <= Streuungsregel (geschlossene Formel, exakt)."""
    spt = np.array([x["SPT"] for x in recs])
    sd = np.array([x["SD_auf"] for x in recs])
    return {"ratio": float(np.mean(sd / spt)), "spt_le_sd": int(np.sum(spt <= sd + 1e-9)), "n_inst": len(recs),
            "ratio_min": float(np.min(sd / spt)), "ratio_max": float(np.max(sd / spt))}


def tau_summary(n: int, r: float, cw: float) -> dict:
    """Bestes Quantil tau* (auf Satz A gewählt) und Gewinn gegen „Mittelwert geplant“ auf dem unabhängigen Satz B (Mittel der Kostenverhältnisse über 30 Instanzen).

    Kosten = cw * Wartezeit + Leerlauf + r * Überziehen; Plan 0 = Mittelwert, Plan 1 bis 13 = Quantil tau des Rasters."""
    frozen = F.tau_data(n)

    def cost(arr):
        return cw * arr[:, :, 0] + arr[:, :, 1] + r * arr[:, :, 2]
    ca, cb = cost(frozen["A"]), cost(frozen["B"])
    j = int(ca.mean(axis=0).argmin())
    relB = (cb / cb[:, [0]]).mean(axis=0)
    k05 = 1 + C.TAUS.index(0.5)
    return {"j": j, "tau": None if j == 0 else C.TAUS[j - 1], "rel_B": float(relB[j]), "rel_B_median": float(relB[k05]),
            "curve_A": [float(x) for x in (ca / ca[:, [0]]).mean(axis=0)], "curve_B": [float(x) for x in relB], "nv": ev.newsvendor_ratio(cw)}


def o3_summary(level: str, r: float) -> dict:
    """Reiter Aufwachbett aus der gefrorenen Tabelle (n = 6, 30 Instanzen, Stufe L, M oder H): Kosten des Optimums relativ zu „ohne Blockade“ je Bettenzahl,
    Lücken der Regeln, Blockadezeit, Zählung „beste Reihenfolge anders als ohne Blockade“."""
    base = F.o3_cell(level, "ohne", r)
    o0 = np.array([x["opt"] for x in base])
    rule_keys = ("SPT", "SD_auf", "LPT", "SD_ab", "ALT", "ZUFALL")

    def block(recs):
        d = {"opt": float(np.mean([x["opt"] for x in recs]))}
        for k in rule_keys:
            d["gap_" + k] = float(np.mean([x[k] / x["opt"] - 1 for x in recs]))
        d["sd_better_spt"] = int(sum(1 for x in recs if x["SD_auf"] < x["SPT"] - 1e-9))
        d["lpt_better_sd"] = int(sum(1 for x in recs if x["LPT"] < x["SD_auf"] - 1e-9))
        d["alt_better_sd"] = int(sum(1 for x in recs if x["ALT"] < x["SD_auf"] - 1e-9))
        return d

    out = {"ohne": block(base)}
    for v in ("B3", "B2", "B1"):
        recs = F.o3_cell(level, v, r)
        d = block(recs)
        d["ratio"] = float(np.mean(np.array([x["opt"] for x in recs]) / o0))
        d["blk_spt"] = float(np.mean([x["blk_spt"] for x in recs]))
        d["other_best"] = int(sum(1 for a, b in zip(recs, base) if a["opt_idx"] != b["opt_idx"]))
        out[v] = d
    return out


def o3_decomposition(r: float) -> dict:
    """Zerlegung (Mittel über 30 Instanzen, Kosten des Optimums): Streuung der Eingriffe allein (ohne Blockade), Blockade allein (Dauern fest, ein Bett, Aufwachdauer
    streuend), beides (ein Bett), dazu Dauern und Aufwachdauer fest."""
    def mean_opt(level, v):
        return float(np.mean([x["opt"] for x in F.o3_cell(level, v, r)]))
    out = {"blockade_allein": mean_opt("D", "B1"), "fest_fest": mean_opt("D", "B1d")}
    for lv in C.LEVELS:
        stoch, both = mean_opt(lv, "ohne"), mean_opt(lv, "B1")
        out[lv] = {"streuung_allein": stoch, "beides": both, "zusatz": both - stoch, "zusatz_vs_blockade_allein": (both - stoch) / out["blockade_allein"]}
    return out


# --- Reiter Aufwachbett live -------------------------------------------------------------------------------------------------------------------

def aufwach_live(beds: str, recovery: str, r: float, m, cv_level, Z) -> dict:
    """n = 6, alle 720 Reihenfolgen auf den Szenarien Z (S, 12): Optimum mit und ohne Blockade, Blockade allein (Dauern fest), Regeln.

    `cv_level`: Variationskoeffizienten der gewählten Stufe (Nullen bei „fest“); `beds`: „1“, „2“, „3“ oder „unbegrenzt“."""
    m = np.asarray(m, float)
    n = len(m)
    T = K.day_end(m)
    B = None if beds == "unbegrenzt" else int(beds)
    rec = K.recovery_durations(n, Z, r_cv=0.0 if recovery == C.RECOVERY[0] else C.R_CV)
    P = K.all_perms(n)
    cvv = np.asarray(cv_level, float)
    Dv, Df = K.durations(m, cvv, Z), K.durations(m, np.zeros(n), Z)

    def run(D, beds_):
        cm = K.enum_means(P, D, m, T, rec if beds_ else None, beds_, chunk=120)
        c = K.cost_from(cm, r)
        return cm, c, int(c.argmin())

    cm_main, c_main, j_main = run(Dv, B)
    out = {"opt": float(c_main[j_main]), "best": [int(x) for x in P[j_main]]}
    ro = R.rule_orders(m, cvv if cvv.any() else np.full(n, 0.3), 1)
    index = {tuple(int(x) for x in p): i for i, p in enumerate(P)}
    out["rules"] = {k: float(c_main[index[tuple(int(x) for x in ro[k])]]) for k in ("SPT", "SD_auf", "LPT", "SD_ab", "ALT")}
    out["rules"]["ZUFALL"] = float(np.mean([c_main[index[tuple(int(x) for x in p)]] for p in ro["ZUFALL"]]))
    out["blk_spt"] = float(cm_main["blk"][index[tuple(int(x) for x in ro["SPT"])]] / m.sum())
    if B is not None:
        _, c0, j0 = run(Dv, None)
        out["opt_ohne"], out["best_ohne"] = float(c0[j0]), [int(x) for x in P[j0]]
        out["opt_ohne_unter_blockade"] = float(c_main[j0])
        out["opt_blockade_allein"] = float(run(Df, B)[1].min()) if cvv.any() else out["opt"]
    else:
        out["opt_ohne"], out["best_ohne"] = out["opt"], out["best"]
        out["opt_ohne_unter_blockade"] = out["opt"]
        out["opt_blockade_allein"] = None
    return out

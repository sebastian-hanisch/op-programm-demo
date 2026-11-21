"""Regeln für die Reihenfolge, Aufzählung (Stichproben-Optimum) und beste lokale Verbesserung.

Regeln: SPT (Mittel aufsteigend), LPT (Mittel absteigend), Streuung aufsteigend, Streuung absteigend, Variationskoeffizient aufsteigend,
Newsvendor-Index aufsteigend (siebte Regel), Zufall (Mittel über 20 Zufallsreihenfolgen), abwechselnd kurz-lang (eigene Regel, nur im Reiter Aufwachbett).
Gleichstände werden nach dem Index des Eingriffs gebrochen. Das „Optimum“ ist das Stichproben-Optimum: die beste Reihenfolge auf den Szenarien,
die zur Bewertung dienen (n bis 7 durch Aufzählung aller n! Reihenfolgen); für n ab 8 gibt es nur die beste lokale Verbesserung, kein Optimum."""

import numpy as np

import opr_constants as C
import opr_kern as K
from opr_rng import SplitMix64

RULE_NAMES = {
    "SPT": "kürzeste Dauer zuerst (SPT)",
    "SD_auf": "kleinste Streuung zuerst",
    "SD_ab": "größte Streuung zuerst",
    "LPT": "längste Dauer zuerst (LPT)",
    "CV_auf": "kleinster Variationskoeffizient zuerst",
    "NV": "Newsvendor-Index aufsteigend",
    "ZUFALL": "Zufallsreihenfolge (Mittel von 20)",
    "ALT": "abwechselnd kurz und lang",
}
RULES = ("SPT", "LPT", "SD_auf", "SD_ab", "CV_auf", "NV")


def newsvendor_index(m, cv) -> np.ndarray:
    """Newsvendor-Index je Eingriff (Gewichte 1 : 1): minimale Kosten, wenn die Dauer als Zielzeit geplant wird, E|D - Median|, geschlossen für Lognormal."""
    q = K.quantile_plan(m, cv, 0.5)
    return K.lognormal_over(m, cv, q) + K.lognormal_under(m, cv, q)


def _order(key, n):
    idx = np.arange(n)
    return np.lexsort((idx, key))


def alternating(m) -> np.ndarray:
    """Kürzester, längster, zweitkürzester, zweitlängster, ... nach dem Mittel."""
    order = list(_order(np.asarray(m, float), len(m)))
    out, lo, hi = [], 0, len(order) - 1
    while lo <= hi:
        out.append(order[lo])
        lo += 1
        if lo <= hi:
            out.append(order[hi])
            hi -= 1
    return np.array(out)


def random_orders(n: int, seed: int, count: int = C.N_RULE_RANDOM) -> np.ndarray:
    g = SplitMix64(seed)
    out = []
    for _ in range(count):
        items = list(range(n))
        g.shuffle(items)
        out.append(items)
    return np.array(out)


def rule_orders(m, cv, seed: int = 1) -> dict:
    """Reihenfolgen aller Regeln (außer ZUFALL: dort (20, n) Reihenfolgen)."""
    m = np.asarray(m, float)
    cv = np.asarray(cv, float)
    sd = m * cv
    n = len(m)
    return {
        "SPT": _order(m, n),
        "LPT": _order(-m, n),
        "SD_auf": _order(sd, n),
        "SD_ab": _order(-sd, n),
        "CV_auf": _order(cv, n),
        "NV": _order(newsvendor_index(m, cv), n),
        "ALT": alternating(m),
        "ZUFALL": random_orders(n, seed),
    }


def neighbors(perm) -> np.ndarray:
    """Alle Reihenfolgen, die durch Vertauschen zweier Eingriffe oder Verschieben eines Eingriffs entstehen."""
    n = len(perm)
    out = set()
    for i in range(n):
        for j in range(i + 1, n):
            q = list(perm)
            q[i], q[j] = q[j], q[i]
            out.add(tuple(int(x) for x in q))
    for i in range(n):
        for j in range(n):
            if j != i:
                q = list(perm)
                v = q.pop(i)
                q.insert(j, v)
                out.add(tuple(int(x) for x in q))
    out.discard(tuple(int(x) for x in perm))
    return np.array(sorted(out))


def local_improvement(start, D, planned, T, cost_fn, max_iter: int = 60, cache: dict | None = None):
    """Beste lokale Verbesserung: vom Start aus immer den besten Nachbarn nehmen, bis kein Zug mehr hilft (kein Optimum).

    `cost_fn(means: dict) -> Kosten je Reihenfolge`; `cache` (perm -> Komponenten-Mittel) darf über mehrere Kostengewichte geteilt werden."""
    cache = {} if cache is None else cache

    def costs(perms):
        need = [p for p in perms if tuple(int(x) for x in p) not in cache]
        if need:
            cm = K.comp_means(K.eval_perms(np.array(need), D, planned, T))
            for i, p in enumerate(need):
                cache[tuple(int(x) for x in p)] = {k: float(v[i]) for k, v in cm.items()}
        comp = {k: np.array([cache[tuple(int(x) for x in p)][k] for p in perms]) for k in ("wait", "idle", "over")}
        return cost_fn(comp)

    cur = np.array(start)
    cur_cost = float(costs(cur[None, :])[0])
    for _ in range(max_iter):
        nb = neighbors(cur)
        if len(nb) == 0:                       # n = 1: kein Nachbar
            break
        c = costs(nb)
        j = int(c.argmin())
        if c[j] < cur_cost - 1e-12:
            cur, cur_cost = nb[j], float(c[j])
        else:
            break
    return cur, cur_cost

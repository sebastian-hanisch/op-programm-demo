"""Kern: Instanzen, Dauern, Tageskosten einer Reihenfolge (Termine, Wartezeit, Leerlauf, Überziehen, Aufwachbett).

Herkunft: eigene Neufassung der Messlogik einer früheren Messreihe zum selben Modell; keine Importe aus anderen Demos.
Zufall nur über `opr_rng.SplitMix64` (ganzzahliger Generator, plattformfeste Normalzahlen); numpy rechnet nur mit den fertigen Zahlen.

Modell (alles erfunden, Minuten sind erfundene Zeiteinheiten): ein Saal, ein Tag, n Eingriffe in fester Reihenfolge pi.
Dauer D_i lognormal mit Mittel m_i und Variationskoeffizient c_i. Termin (geplanter Start) s_k = Summe der geplanten Dauern der Vorgänger
(Standard: der Mittelwerte). Tatsächlicher Start = max(s_k, Ende des Vorgängers). Wartezeit = Start - s_k, Leerlauf = s_k - Ende des Vorgängers (falls positiv),
Überziehen = max(0, Ende des letzten - T) mit T = 1,10 * Summe der Mittel. Kosten = cw * Wartezeit + Leerlauf + r * Überziehen.
Aufwachbett: B Betten; der Patient verlässt den Saal, sobald ein Bett frei ist (der Saal ist bis dahin blockiert)."""

import itertools
from statistics import NormalDist

import numpy as np

import opr_constants as C
from opr_rng import MASK, SplitMix64

_ND = NormalDist()


# --- Zufall -----------------------------------------------------------------------------------------------------------------------------

def seed_of(*parts: int) -> int:
    """Mischt ganze Zahlen zu einem Zufalls-Seed (SplitMix64, plattformfest)."""
    h = 0x243F6A8885A308D3
    for p in parts:
        h = SplitMix64((h ^ (int(p) & MASK)) & MASK).next_u64()
    return h


def normals(cols: int, scen: int, seed: int) -> np.ndarray:
    """Standardnormalzahlen, Zeilen = Szenarien, Spalten = Eingriffe (plattformfest, aber langsam: etwa 8 000 Zahlen je Sekunde)."""
    g = SplitMix64(seed)
    return np.array([[g.normal() for _ in range(cols)] for _ in range(scen)], dtype=float)


def gen_instance(n: int, level: str, inst: int, base: int = C.INSTANCE_BASE):
    """Mittel m (ganze Minuten) und Variationskoeffizienten cv der n Eingriffe. Klasse und Streuung werden unabhängig voneinander gezogen.

    Stufen L, M, H, K: Variationskoeffizient aus der Stufe. Stufe U: absolute Streuung aus {8, 20, 40} min, unabhängig vom Mittel."""
    g = SplitMix64(seed_of(base, n, ord(level), inst))
    m = np.empty(n)
    cv = np.empty(n)
    for i in range(n):
        cls = g.below(3)
        factor = C.FACTOR_LO + (C.FACTOR_HI - C.FACTOR_LO) * g.random()
        m[i] = float(int(C.CLASS_MEANS[cls] * factor + 0.5))
        if level == "U":
            cv[i] = C.U_SDS[g.below(len(C.U_SDS))] / m[i]
        else:
            levels = C.CV_LEVELS[level]
            cv[i] = levels[g.below(len(levels))]
    return m, cv


# --- Dauern -----------------------------------------------------------------------------------------------------------------------------

def ln_params(m, cv):
    """Parameter (mu, sigma) der Lognormalverteilung mit Mittel m und Variationskoeffizient cv."""
    s2 = np.log1p(np.asarray(cv, float) ** 2)
    return np.log(np.asarray(m, float)) - s2 / 2.0, np.sqrt(s2)


def durations(m, cv, Z) -> np.ndarray:
    """Dauern (S, n) aus Standardnormalzahlen Z (S, >= n)."""
    n = len(m)
    mu, sg = ln_params(m, cv)
    return np.exp(mu[None, :] + sg[None, :] * np.asarray(Z)[:, :n])


def recovery_durations(n: int, Z, r_mean: float = C.R_MEAN, r_cv: float = C.R_CV) -> np.ndarray:
    """Aufwachdauern (S, n) aus den Spalten n bis 2n von Z; r_cv = 0 gibt die feste Dauer r_mean."""
    if r_cv == 0:
        return np.full((np.asarray(Z).shape[0], n), float(r_mean))
    mu, sg = ln_params(np.full(n, r_mean), np.full(n, r_cv))
    return np.exp(mu[None, :] + sg[None, :] * np.asarray(Z)[:, n:2 * n])


def quantile_plan(m, cv, tau: float) -> np.ndarray:
    """Geplante Dauern = Quantil tau der Lognormalverteilung (tau = 0,5 ist der Median)."""
    mu, sg = ln_params(m, cv)
    return np.exp(mu + sg * _ND.inv_cdf(tau))


def lognormal_over(m, cv, q) -> np.ndarray:
    """E[(D - q)+] für lognormale D, geschlossen."""
    mu, sg = ln_params(m, cv)
    cdf = np.vectorize(_ND.cdf)
    safe = np.where(sg > 0, sg, 1.0)
    d1 = (mu + sg ** 2 - np.log(q)) / safe
    stochastic = np.exp(mu + sg ** 2 / 2) * cdf(d1) - q * cdf(d1 - sg)
    return np.where(sg > 0, stochastic, np.maximum(np.exp(mu) - q, 0.0))      # Variationskoeffizient 0: feste Dauer


def lognormal_under(m, cv, q) -> np.ndarray:
    """E[(q - D)+] = E[(D - q)+] + q - m."""
    return lognormal_over(m, cv, q) + q - np.asarray(m, float)


# --- Reihenfolgen und Kosten --------------------------------------------------------------------------------------------------------------

def all_perms(n: int) -> np.ndarray:
    return np.array(list(itertools.permutations(range(n))), dtype=np.int16)


def day_end(m) -> float:
    return (1.0 + C.SLACK) * float(np.sum(m))


def eval_perms(perms, D, planned, T, Rt=None, beds=None) -> dict:
    """Wartezeit, Leerlauf, Überziehen, Blockade und Endzeit je Reihenfolge und Szenario, alle (P, S).

    perms (P, n) Indizes; D (S, n) Dauern; planned (n,) geplante Dauern; Rt (S, n) Aufwachdauern; beds = Zahl der Betten (None = unbegrenzt, keine Blockade)."""
    perms = np.asarray(perms)
    P, n = perms.shape
    S = D.shape[0]
    DT = D.T
    pl = np.asarray(planned, float)[perms]
    s = np.concatenate([np.zeros((P, 1)), np.cumsum(pl, axis=1)[:, :-1]], axis=1)
    wait = np.zeros((P, S))
    idle = np.zeros((P, S))
    blk = np.zeros((P, S))
    f = np.zeros((P, S))
    if beds is not None:
        RT = Rt.T
        free = np.zeros((P, S, beds))
    for k in range(n):
        sk = s[:, k][:, None]
        wait += np.maximum(0.0, f - sk)
        idle += np.maximum(0.0, sk - f)
        end = np.maximum(sk, f) + DT[perms[:, k]]
        if beds is None:
            f = end
        else:
            bi = free.argmin(axis=2)
            tb = np.take_along_axis(free, bi[:, :, None], axis=2)[:, :, 0]
            dep = np.maximum(end, tb)
            blk += dep - end
            np.put_along_axis(free, bi[:, :, None], (dep + RT[perms[:, k]])[:, :, None], axis=2)
            f = dep
    over = np.maximum(0.0, f - T)
    return {"wait": wait, "idle": idle, "over": over, "blk": blk, "end": f}


def comp_means(comp: dict) -> dict:
    return {k: v.mean(axis=1) for k, v in comp.items()}


def cost_from(comp: dict, r: float, cw: float = 1.0, ci: float = 1.0):
    """Kosten aus den Komponenten (Szenarien oder Mittel): cw * Wartezeit + ci * Leerlauf + r * Überziehen."""
    return cw * comp["wait"] + ci * comp["idle"] + r * comp["over"]


def enum_means(perms, D, planned, T, Rt=None, beds=None, chunk: int = 500) -> dict:
    """Mittel der Komponenten je Reihenfolge, in Blöcken (begrenzt den Speicher)."""
    parts = []
    for idx in np.array_split(np.arange(len(perms)), max(1, len(perms) // chunk)):
        parts.append(comp_means(eval_perms(perms[idx], D, planned, T, Rt, beds)))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def planned_starts(perms, planned) -> np.ndarray:
    """Termine s_k je Reihenfolge (P, n): Summe der geplanten Dauern der Vorgänger."""
    perms = np.asarray(perms)
    pl = np.asarray(planned, float)[perms]
    return np.concatenate([np.zeros((len(perms), 1)), np.cumsum(pl, axis=1)[:, :-1]], axis=1)


def lead_term(perms, planned, e: float) -> np.ndarray:
    """Zusatz des Vorlaufs e: Summe über k von min(s_k, e) je Reihenfolge. Die Wartezeit ab Ankunft ist exakt (Start - Termin) + min(s_k, e);
    der Zusatz hängt nur von der Reihenfolge (über die Termine) ab, nicht von den Dauern."""
    return np.minimum(planned_starts(perms, planned), e).sum(axis=1)


def wait_from_day_start(perms, m) -> np.ndarray:
    """Maß B (alle Patienten von Tagesbeginn an da, keine Termine): erwartete Summe der Wartezeiten = Summe_k Summe_{j<k} m_pi(j), geschlossen."""
    perms = np.asarray(perms)
    return planned_starts(perms, m).sum(axis=1)

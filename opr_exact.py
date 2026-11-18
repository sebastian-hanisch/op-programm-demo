"""Exakte Rechnung für kleine Fälle: jeder Eingriff hat zwei gleich wahrscheinliche Dauern (lo, hi).

Alle 2^n Ausgänge und alle n! Reihenfolgen werden in ganzen Zahlen gerechnet (Zeiten mal 10); die Erwartungswerte sind damit exakt (Brüche), ohne Stichprobenfehler
und ohne Gleitkomma-Gleichstände. Kosten wie im übrigen Modell: Wartezeit + Leerlauf + r * Überziehen, Termine = Summe der Mittel der Vorgänger, T = 1,10 * Summe der Mittel.
Herkunft: eigene Neufassung der Messlogik einer früheren Messreihe; keine Importe aus anderen Demos."""

import itertools
from fractions import Fraction

import numpy as np

import opr_constants as C
from opr_rng import SplitMix64

A_CHOICES = ((1, 6), (1, 3), (1, 2), (2, 3))      # Streuung a = 1/6, 1/3, 1/2, 2/3 des Mittels (Dauern m(1 - a) und m(1 + a))
EXACT_RS = (Fraction(1, 2), Fraction(1), Fraction(2), Fraction(4), Fraction(8))

# Handbeispiel: drei Eingriffe, zwei Dauern je Eingriff, alle Zahlen erfunden
HAND_NAMES = ("A", "B", "C")
HAND_M = (60, 90, 120)
HAND_LO = (30, 75, 80)
HAND_HI = (90, 105, 160)
# Zwei Eingriffe mit gleichem Mittel (90 Minuten), verschiedene Streuung
PAIR_NAMES = ("P", "Q")
PAIR_M = (90, 90)
PAIR_LO = (75, 45)
PAIR_HI = (105, 135)


def two_point_instance(g: SplitMix64, n: int):
    """Zufallsinstanz: Mittel 6k (k = 7..25, also 42 bis 150 Minuten), a aus {1/6, 1/3, 1/2, 2/3} unabhängig vom Mittel; alle Mittel und alle absoluten
    Streuungen paarweise verschieden (sonst ist die Regel nicht eindeutig). Rückgabe: (m, lo, hi) als Listen ganzer Minuten."""
    while True:
        m = [6 * (7 + g.below(19)) for _ in range(n)]
        a = [A_CHOICES[g.below(4)] for _ in range(n)]
        sd = [mi * p // q for mi, (p, q) in zip(m, a)]
        if len(set(m)) == n and len(set(sd)) == n:
            return m, [mi - s for mi, s in zip(m, sd)], [mi + s for mi, s in zip(m, sd)]


def outcomes(lo, hi) -> np.ndarray:
    """Alle 2^n Dauernkombinationen (O, n), jede gleich wahrscheinlich."""
    n = len(lo)
    bits = np.array(list(itertools.product((0, 1), repeat=n)), dtype=np.int64)
    return np.where(bits == 0, np.array(lo, dtype=np.int64)[None, :], np.array(hi, dtype=np.int64)[None, :])


def all_perms_int(n: int) -> np.ndarray:
    return np.array(list(itertools.permutations(range(n))), dtype=np.int64)


def exact_parts(perms, D, m):
    """Summen über alle Ausgänge (Zeiten mal 10, ganze Zahlen): (wi, ov) je Reihenfolge mit wi = Summe von |Ende des Vorgängers - Termin| (Wartezeit plus Leerlauf),
    ov = Summe des Überziehens (Tagesende T = 1,10 * Summe der Mittel, also 11 * Summe der Mittel in Zehntelminuten)."""
    perms = np.asarray(perms, dtype=np.int64)
    P, n = perms.shape
    O = D.shape[0]
    D10 = D * 10
    m10 = np.array(m, dtype=np.int64) * 10
    T10 = 11 * int(sum(m))
    s = np.concatenate([np.zeros((P, 1), np.int64), np.cumsum(m10[perms], axis=1)[:, :-1]], axis=1)
    f = np.zeros((P, O), np.int64)
    wi = np.zeros((P, O), np.int64)
    for k in range(n):
        sk = s[:, k][:, None]
        wi += np.abs(f - sk)
        f = np.maximum(sk, f) + D10[:, perms[:, k]].T
    ov = np.maximum(0, f - T10)
    return wi.sum(axis=1), ov.sum(axis=1)


def exact_costs(wi, ov, n_outcomes: int, r: Fraction) -> list:
    """Erwartete Kosten als Brüche: (wi + r * ov) / (10 * Ausgänge)."""
    den = 10 * n_outcomes
    return [Fraction(int(a), den) + r * Fraction(int(b), den) for a, b in zip(wi, ov)]


def exact_table(lo, hi, m, r: Fraction):
    """Alle n! Reihenfolgen mit exakten Kosten: Liste (Reihenfolge, Kosten) in der Ordnung von itertools.permutations."""
    perms = all_perms_int(len(m))
    wi, ov = exact_parts(perms, outcomes(lo, hi), m)
    costs = exact_costs(wi, ov, 2 ** len(m), r)
    return [(tuple(int(x) for x in p), c) for p, c in zip(perms, costs)]


def rule_orders_exact(m, lo, hi):
    """SPT und Streuung aufsteigend (Streuung = (hi - lo) / 2), Gleichstand nach Index."""
    n = len(m)
    sd = [(h - l) / 2 for l, h in zip(lo, hi)]
    spt = tuple(sorted(range(n), key=lambda i: (m[i], i)))
    sda = tuple(sorted(range(n), key=lambda i: (sd[i], i)))
    return spt, sda


def exact_stats(n: int, count: int, seed: int = C.EXACT_SEED, rs=EXACT_RS) -> dict:
    """Zählungen über `count` Zufallsinstanzen mit n Eingriffen (Zähleinheit: Instanzen), je Überziehpreis r.

    Gleichstände der Kosten sind ganzzahlig exakt. Die Folge der Instanzen hängt nur von (seed + n) ab."""
    g = SplitMix64(seed + n)
    perms = all_perms_int(n)
    index = {tuple(int(x) for x in p): i for i, p in enumerate(perms)}
    out = {str(r): {"n_inst": count, "sd_opt": 0, "spt_opt": 0, "better": 0, "equal": 0, "worse": 0, "gap_sd_sum": 0.0, "gap_spt_sum": 0.0,
                    "dif_sum": 0.0, "gap_sd_max": 0.0, "worst_instance": None} for r in rs}
    for _ in range(count):
        m, lo, hi = two_point_instance(g, n)
        D = outcomes(lo, hi)
        wi, ov = exact_parts(perms, D, m)
        spt, sda = rule_orders_exact(m, lo, hi)
        for r in rs:
            r2 = int(2 * r)                                      # r ist ein Vielfaches von 1/2: 2r ist ganz
            tot = (2 * wi + r2 * ov).tolist()                    # Kosten mal 2 * 10 * Ausgänge, ganze Zahlen
            opt = min(tot)
            cs, cd = tot[index[spt]], tot[index[sda]]
            st = out[str(r)]
            st["sd_opt"] += int(cd == opt)
            st["spt_opt"] += int(cs == opt)
            st["better"] += int(cd < cs)
            st["equal"] += int(cd == cs)
            st["worse"] += int(cd > cs)
            gap = cd / opt - 1.0
            st["gap_sd_sum"] += gap
            st["gap_spt_sum"] += cs / opt - 1.0
            st["dif_sum"] += cd / cs - 1.0
            if gap > st["gap_sd_max"]:
                st["gap_sd_max"] = gap
                st["worst_instance"] = {"m": m, "lo": lo, "hi": hi}
    for st in out.values():
        for key in ("gap_sd_sum", "gap_spt_sum", "dif_sum"):
            st[key.replace("_sum", "_mean")] = st.pop(key) / count
    return out

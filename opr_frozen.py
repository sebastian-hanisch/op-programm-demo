"""Laden der vorgerechneten (gefrorenen) Tabellen aus data/*.json; erzeugt von tools/opr_make_*.py (je mit --check).

Alle Zahlen sind eigene Messungen im erfundenen Modell (30 Instanzen je Zelle, 2 000 Szenarien je Satz). In der App tragen sie das Wort „vorgerechnet“."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

import opr_constants as C

DATA = Path(__file__).resolve().parent / "data"


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _o1_index():
    d = _load("opr_o1_tabelle.json")
    cols = d["cols"]
    idx = {}
    for row in d["rows"]:
        rec = dict(zip(cols, row))
        idx.setdefault((rec["n"], rec["level"], float(rec["r"])), []).append(rec)
    for recs in idx.values():
        recs.sort(key=lambda x: x["inst"])
    mb = {}
    for row in d["mb_rows"]:
        rec = dict(zip(d["mb_cols"], row))
        mb.setdefault((rec["n"], rec["level"]), []).append(rec)
    for recs in mb.values():
        recs.sort(key=lambda x: x["inst"])
    return idx, mb


def o1_cell(n: int, level: str, r: float) -> list:
    """30 Instanz-Records einer Zelle (n, Stufe, Überziehpreis r)."""
    return _o1_index()[0][(n, level, float(r))]


def o1_maass_b(n: int, level: str) -> list:
    """Maß B (Wartezeit ab Tagesbeginn, geschlossene Formel): 30 Records je Zelle."""
    return _o1_index()[1][(n, level)]


@lru_cache(maxsize=None)
def _lead_index():
    d = _load("opr_o1_vorlauf.json")
    idx = {}
    for row in d["rows"]:
        rec = dict(zip(d["cols"], row))
        idx.setdefault((rec["level"], float(rec["r"]), rec["e"]), []).append(rec)
    for recs in idx.values():
        recs.sort(key=lambda x: x["inst"])
    return idx


def lead_cell(level: str, r: float, ekey: str) -> list:
    """Kosten von SPT, Streuungsregel und Optimum bei Vorlauf e (n = 7, 30 Instanzen)."""
    return _lead_index()[(level, float(r), ekey)]


def exact_stats(n: int) -> dict:
    """Exakte Zählungen über zufällige Zweipunkt-Instanzen, je Überziehpreis (Schlüssel '1/2', '1', '2', '4', '8')."""
    return _load("opr_o1_exakt.json")["stats"][str(n)]


def tau_data(n: int) -> dict:
    """Komponenten-Mittel je Plan: {'A': (30, 14, 3), 'B': (30, 14, 3)} als numpy-Felder; Plan 0 = Mittelwert, 1 bis 13 = Quantil tau."""
    d = _load("opr_o2_tau.json")["data"][str(n)]
    return {w: np.array(d[w], dtype=float) for w in ("A", "B")}


def tau_phi(n: int) -> dict:
    return _load("opr_o2_tau.json")["phi"][str(n)]


@lru_cache(maxsize=None)
def _o3_index():
    d = _load("opr_o3_betten.json")
    idx = {}
    for row in d["rows"]:
        rec = dict(zip(d["cols"], row))
        idx.setdefault((rec["level"], rec["variant"], float(rec["r"])), []).append(rec)
    for recs in idx.values():
        recs.sort(key=lambda x: x["inst"])
    return idx


def o3_cell(level: str, variant: str, r: float) -> list:
    """30 Records (Optimum, Regeln, Blockade) der Zelle: Stufe D, L, M, H; Variante ohne, B3, B2, B1 (und B1d, B2d bei D)."""
    return _o3_index()[(level, variant, float(r))]


def files_present() -> bool:
    return all((DATA / f).exists() for f in ("opr_o1_tabelle.json", "opr_o1_vorlauf.json", "opr_o1_exakt.json", "opr_o2_tau.json", "opr_o3_betten.json"))

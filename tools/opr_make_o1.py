"""Vorrechnen der gefrorenen Tabellen zu Reiter „Reihenfolge“: data/opr_o1_tabelle.json und data/opr_o1_vorlauf.json.

Aufruf (Test-venv, Stunden-Rechnung im Hintergrund erlaubt, etwa 15 bis 30 Minuten):
    python tools/opr_make_o1.py            # alles neu rechnen und schreiben
    python tools/opr_make_o1.py --ns 5     # nur n = 5 (Probelauf, schreibt nicht)
    python tools/opr_make_o1.py --check    # zwei zufällige Zellen neu rechnen und mit den Dateien im Band vergleichen

Je (n, Stufe) 30 Instanzen (gen_instance, Zufalls-Seed = Instanznummer), 2 000 Szenarien je Satz (Satz A wählt das Optimum und bewertet, Satz B bewertet nur).
Zur Laufzeitersparnis teilen sich die fünf Stufen derselben Instanznummer (verschiedene Instanzen) dieselben Standardnormalzahlen; Zufall nur über SplitMix64.
Zähleinheit überall: Instanzen. Das „Optimum“ ist das Stichproben-Optimum auf Satz A (n bis 7 Aufzählung, n ab 8 beste lokale Verbesserung, kein Optimum)."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import opr_constants as C  # noqa: E402
import opr_evaluation as ev  # noqa: E402
import opr_kern as K  # noqa: E402

DATA = ROOT / "data"
COLS = ["n", "level", "inst", "r", "SPT", "LPT", "SD_auf", "SD_ab", "CV_auf", "NV", "ZUFALL", "OPT", "se_diff", "opt_is_sd", "opt_is_spt",
        "SPT_B", "LPT_B", "SD_auf_B", "SD_ab_B", "CV_auf_B", "NV_B", "ZUFALL_B", "OPT_B"]
MB_COLS = ["n", "level", "inst", "SPT", "LPT", "SD_auf", "SD_ab", "CV_auf", "NV"]
LEAD_COLS = ["level", "inst", "r", "e", "SPT", "SD_auf", "OPT", "SPT_B", "SD_auf_B"]


def blocks(n: int, inst: int):
    """Standardnormalzahlen der Szenariensätze A und B der Instanznummer `inst`; die Stufen derselben Nummer (verschiedene Instanzen) teilen sie."""
    za = K.normals(n, C.N_SCEN_FROZEN, ev.scenario_seed("A", n, inst))
    zb = K.normals(n, C.N_SCEN_FROZEN, ev.scenario_seed("B", n, inst))
    return za, zb


def one_instance(n: int, level: str, inst: int, za, zb, lead: bool):
    m, cv = K.gen_instance(n, level, inst)
    DA, DB = K.durations(m, cv, za), K.durations(m, cv, zb)
    return ev.evaluate_instance(m, cv, DA, DB, rule_seed=K.seed_of(6000, n, ord(level), inst), lead=lead)


def rows_of(n, level, inst, res):
    rows = []
    for r in C.RS:
        rec = res["r"][r]
        rows.append([n, level, inst, r] + [round(rec[k], 4) if isinstance(rec[k], float) else rec[k] for k in COLS[4:]])
    mb = [n, level, inst] + [round(res["mB"][k], 3) for k in MB_COLS[3:]]
    return rows, mb


def compute(ns):
    table, mb_rows, lead_rows = [], [], []
    for n in ns:
        t0 = time.time()
        for inst in range(C.N_INST):
            za, zb = blocks(n, inst)
            for level in C.FROZEN_LEVELS:
                lead = n == 7 and level in C.LEVELS
                res = one_instance(n, level, inst, za, zb, lead)
                rows, mb = rows_of(n, level, inst, res)
                table += rows
                mb_rows.append(mb)
                if lead:
                    for r, d in res["lead"].items():
                        for ekey, vals in d.items():
                            lead_rows.append([level, inst, r, ekey] + [round(v, 4) for v in vals])
            print(f"n = {n}, Instanz {inst} fertig ({time.time() - t0:.0f} s)", flush=True)
    return table, mb_rows, lead_rows


def write(table, mb_rows, lead_rows):
    DATA.mkdir(exist_ok=True)
    meta = {"scenarios": C.N_SCEN_FROZEN, "instances": C.N_INST, "slack": C.SLACK, "base": C.INSTANCE_BASE,
            "hinweis": "Zufall SplitMix64; gemeinsame Szenarien je Instanznummer über die Stufen; Optimum = Stichproben-Optimum (n bis 7) bzw. beste lokale Verbesserung (n ab 8)"}
    with open(DATA / "opr_o1_tabelle.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "cols": COLS, "rows": table, "mb_cols": MB_COLS, "mb_rows": mb_rows}, f, ensure_ascii=False, separators=(",", ":"))
    with open(DATA / "opr_o1_vorlauf.json", "w", encoding="utf-8") as f:
        json.dump({"meta": {**meta, "n": 7}, "cols": LEAD_COLS, "rows": lead_rows}, f, ensure_ascii=False, separators=(",", ":"))


def check(seed_cells=((5, "M", 3), (8, "U", 2))) -> int:
    """Zwei Zellen neu rechnen und mit der Datei vergleichen: Regeln relativ 1e-6, Optimum (kann bei Gleichständen anders wählen) relativ 2e-3."""
    data = json.loads((DATA / "opr_o1_tabelle.json").read_text(encoding="utf-8"))
    bad = 0
    for n, level, inst in seed_cells:
        za, zb = blocks(n, inst)
        res = one_instance(n, level, inst, za, zb, False)
        rows, _ = rows_of(n, level, inst, res)
        stored = {tuple(r[:4]): r for r in data["rows"]}
        for row in rows:
            old = stored[tuple(row[:4])]
            for col, a, b in zip(COLS[4:], row[4:], old[4:]):
                tol = 2e-3 if col.startswith("OPT") else 1e-6
                if col in ("opt_is_sd", "opt_is_spt"):
                    continue
                if abs(a - b) > tol * max(1.0, abs(b)):
                    bad += 1
                    print("ABWEICHUNG", n, level, inst, row[3], col, a, b)
        print(f"Zelle n = {n}, Stufe {level}, Instanz {inst}: geprüft")
    print("check:", "ok" if not bad else f"{bad} Abweichungen")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="*")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        raise SystemExit(check())
    ns = a.ns or list(C.FROZEN_NS)
    table, mb_rows, lead_rows = compute(ns)
    if a.ns:
        print(f"Probelauf: {len(table)} Zeilen, nichts geschrieben")
        return
    write(table, mb_rows, lead_rows)
    print(f"geschrieben: {len(table)} Zeilen, {len(mb_rows)} Maß-B-Zeilen, {len(lead_rows)} Vorlauf-Zeilen")


if __name__ == "__main__":
    main()

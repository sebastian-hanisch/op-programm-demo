"""Vorrechnen des Reiters „Aufwachbett“: data/opr_o3_betten.json.

    python tools/opr_make_o3.py          # rechnen und schreiben (einige Minuten)
    python tools/opr_make_o3.py --check  # zwei Instanzen neu rechnen und im Band (relativ 1e-6) mit der Datei vergleichen

n = 6 Eingriffe (alle 720 Reihenfolgen aufgezählt), 30 Instanzen, 2 000 Szenarien. Die Mittel sind für alle Stufen dieselben (aus der Stufe M erzeugt), die Streuung der
Stufe kommt aus eigenen Zufalls-Seeds; Stufe „D“ = Dauern fest. Aufwachdauer lognormal (Mittel 60, Variationskoeffizient 0,5) oder fest 60 (Varianten „B1d“, „B2d“, nur Stufe D).
Betten: unbegrenzt („ohne“), 3, 2, 1. Kosten wie im Reiter Reihenfolge (Maß A, Termine nach den Mittelwerten, Wartezeit : Leerlauf = 1 : 1, Überziehpreis r)."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import opr_constants as C  # noqa: E402
import opr_evaluation as ev  # noqa: E402
import opr_kern as K  # noqa: E402
import opr_rules as R  # noqa: E402

FILE = ROOT / "data" / "opr_o3_betten.json"
N = C.N3
STUFEN = ("D", "L", "M", "H")
VARIANTS = (("ohne", None, True), ("B3", 3, True), ("B2", 2, True), ("B1", 1, True))
RS3 = (1.0, 2.0, 4.0)
COLS = ["level", "inst", "variant", "r", "opt", "opt_idx", "SPT", "SD_auf", "LPT", "SD_ab", "ALT", "ZUFALL", "blk_spt"]


def instance(level: str, inst: int):
    m, _ = K.gen_instance(N, "M", inst)
    cv = np.zeros(N) if level == "D" else K.gen_instance(N, level, inst)[1]
    return m, cv


def one(level: str, inst: int, P, index, Z):
    m, cv = instance(level, inst)
    T = K.day_end(m)
    D = K.durations(m, cv, Z)
    rec_log = K.recovery_durations(N, Z)
    rec_fix = K.recovery_durations(N, Z, r_cv=0)
    ro = R.rule_orders(m, cv if level != "D" else np.full(N, 0.3), K.seed_of(6001, inst))
    rule_idx = {k: index[tuple(int(x) for x in ro[k])] for k in ("SPT", "SD_auf", "LPT", "SD_ab", "ALT")}
    zuf_idx = [index[tuple(int(x) for x in p)] for p in ro["ZUFALL"]]
    variants = list(VARIANTS)
    if level == "D":
        variants += [("B1d", 1, False), ("B2d", 2, False)]
    rows = []
    for name, beds, lognormal in variants:
        cm = K.enum_means(P, D, m, T, rec_log if lognormal else rec_fix, beds, chunk=120)
        for r in RS3:
            c = K.cost_from(cm, r)
            j = int(c.argmin())
            rows.append([level, inst, name, r, round(float(c[j]), 4), j] + [round(float(c[rule_idx[k]]), 4) for k in ("SPT", "SD_auf", "LPT", "SD_ab", "ALT")]
                        + [round(float(np.mean(c[zuf_idx])), 4), round(float(cm["blk"][rule_idx["SPT"]] / m.sum()), 6)])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    P = K.all_perms(N)
    index = {tuple(int(x) for x in p): i for i, p in enumerate(P)}
    if a.check:
        old = json.loads(FILE.read_text(encoding="utf-8"))["rows"]
        stored = {(r[0], r[1], r[2], r[3]): r for r in old}
        bad = 0
        for inst in (0, 11):
            Z = K.normals(2 * N, C.N_SCEN_FROZEN, ev.scenario_seed("A", N, 200 + inst))
            for level in ("M",):
                for row in one(level, inst, P, index, Z):
                    o = stored[tuple(row[:4])]
                    for col, x, y in zip(COLS[4:], row[4:], o[4:]):
                        if col == "opt_idx":
                            continue
                        if abs(x - y) > 1e-6 * max(1.0, abs(y)):
                            bad += 1
                            print("ABWEICHUNG", row[:4], col, x, y)
        print("check:", "ok" if not bad else f"{bad} Abweichungen")
        raise SystemExit(1 if bad else 0)
    rows = []
    for inst in range(C.N_INST):
        Z = K.normals(2 * N, C.N_SCEN_FROZEN, ev.scenario_seed("A", N, 200 + inst))
        for level in STUFEN:
            rows += one(level, inst, P, index, Z)
        print("Instanz", inst, "fertig", flush=True)
    FILE.parent.mkdir(exist_ok=True)
    FILE.write_text(json.dumps({"meta": {"n": N, "scenarios": C.N_SCEN_FROZEN, "instances": C.N_INST, "recovery_mean": C.R_MEAN, "recovery_cv": C.R_CV},
                                "cols": COLS, "rows": rows}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()

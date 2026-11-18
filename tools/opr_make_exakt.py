"""Vorrechnen der exakten Zählungen (Zweipunkt-Dauern, ganze Zahlen): data/opr_o1_exakt.json.

    python tools/opr_make_exakt.py          # rechnen und schreiben
    python tools/opr_make_exakt.py --check  # n = 3 und n = 6 neu rechnen und exakt mit der Datei vergleichen (reine Ganzzahlarithmetik, plattformfest)

Je n zufällige Instanzen aus der SplitMix64-Folge (Zufalls-Seed 20261118 + n); alle 2^n Ausgänge und alle n! Reihenfolgen; Zähleinheit: Instanzen."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import opr_constants as C  # noqa: E402
import opr_exact as X  # noqa: E402

FILE = ROOT / "data" / "opr_o1_exakt.json"


def compute(ns):
    return {str(n): X.exact_stats(n, C.EXACT_COUNTS[n]) for n in ns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        old = json.loads(FILE.read_text(encoding="utf-8"))["stats"]
        new = compute((3, 6))
        bad = 0
        for n in new:
            for r, st in new[n].items():
                for k in ("sd_opt", "spt_opt", "better", "equal", "worse"):
                    if st[k] != old[n][r][k]:
                        bad += 1
                        print("ABWEICHUNG", n, r, k, st[k], old[n][r][k])
        print("check:", "ok" if not bad else f"{bad} Abweichungen")
        raise SystemExit(1 if bad else 0)
    stats = compute(C.EXACT_NS)
    FILE.parent.mkdir(exist_ok=True)
    FILE.write_text(json.dumps({"meta": {"seed": C.EXACT_SEED, "counts": {str(n): C.EXACT_COUNTS[n] for n in C.EXACT_NS}}, "stats": stats},
                               ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    for n, per_r in stats.items():
        print(n, {r: (st["sd_opt"], st["spt_opt"]) for r, st in per_r.items()})


if __name__ == "__main__":
    main()

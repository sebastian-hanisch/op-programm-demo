"""Vorrechnen der Quantil-Kurven zu Reiter „Geplante Dauer“: data/opr_o2_tau.json.

    python tools/opr_make_o2.py          # rechnen und schreiben
    python tools/opr_make_o2.py --check  # n = 4 neu rechnen und im Band (relativ 1e-6) mit der Datei vergleichen

Je n in {2, 3, 4, 6, 8} 30 Instanzen der Stufe M, Reihenfolge „kleinste Streuung zuerst“, geplante Dauer = Mittelwert (Index 0) oder Quantil tau (Raster 0,30 bis 0,90 in
0,05-Schritten, Index 1 bis 13). Gespeichert werden die mittleren Komponenten (Wartezeit, Leerlauf, Überziehen) je Plan auf Satz A und auf dem unabhängigen Satz B (2 000 Szenarien
je Satz); die Kosten für jedes Gewicht (Wartezeit : Leerlauf, Überziehpreis r) setzt die App daraus zusammen."""

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

FILE = ROOT / "data" / "opr_o2_tau.json"


def one(n: int, inst: int):
    """Komponenten-Mittel (Pläne, 3) auf Satz A und B sowie der Streuungs-Mittelwert Φ(σ/2) der Instanz."""
    m, cv = K.gen_instance(n, "M", inst)
    T = K.day_end(m)
    perm = R.rule_orders(m, cv)["SD_auf"][None, :]
    plans = [m] + [K.quantile_plan(m, cv, t) for t in C.TAUS]
    out = {}
    for which in ("A", "B"):
        D = K.durations(m, cv, K.normals(n, C.N_SCEN_FROZEN, ev.scenario_seed(which, n, 100 + inst)))
        rows = []
        for pl in plans:
            cm = K.comp_means(K.eval_perms(perm, D, pl, T))
            rows.append([round(float(cm[k][0]), 4) for k in ("wait", "idle", "over")])
        out[which] = rows
    return out, [float(x) for x in ev.mean_quantile_level(cv)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        old = json.loads(FILE.read_text(encoding="utf-8"))["data"]["4"]
        bad = 0
        for inst in (0, 17):
            new, _ = one(4, inst)
            for which in ("A", "B"):
                if not np.allclose(new[which], old[which][inst], rtol=1e-6, atol=1e-4):
                    bad += 1
                    print("ABWEICHUNG", inst, which)
        print("check:", "ok" if not bad else f"{bad} Abweichungen")
        raise SystemExit(1 if bad else 0)
    data, phi = {}, {}
    for n in C.TAU_NS:
        recs = [one(n, i) for i in range(C.N_INST)]
        data[str(n)] = {w: [r[0][w] for r in recs] for w in ("A", "B")}
        allphi = np.concatenate([r[1] for r in recs])
        phi[str(n)] = {"mean": round(float(allphi.mean()), 4), "min": round(float(allphi.min()), 4), "max": round(float(allphi.max()), 4)}
        print("n =", n, "fertig", flush=True)
    FILE.parent.mkdir(exist_ok=True)
    FILE.write_text(json.dumps({"meta": {"taus": list(C.TAUS), "scenarios": C.N_SCEN_FROZEN, "instances": C.N_INST,
                                        "plans": "Index 0 = Mittelwert, 1..13 = Quantil tau"}, "phi": phi, "data": data},
                               ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()

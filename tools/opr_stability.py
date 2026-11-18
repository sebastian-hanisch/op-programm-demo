"""Stabilität der Zählung: dieselbe Zelle (n = 5, Stufe M, r = 2, 2 000 Szenarien je Satz) mit drei verschiedenen Instanzströmen (Basis 7000, 7100, 7200).

    python tools/opr_stability.py    # rechnet etwa zwei Minuten und schreibt die Zeilen nach data/opr_stabilitaet.txt

Zeigt, wie stark „besser in k von 30“ und die mittlere Differenz von den gezogenen 30 Instanzen abhängen (die vorgerechneten Tabellen nutzen Basis 7000).
Das Ergebnis steht in der README als Messung dieses Werkzeugs, nicht in einem Test."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import opr_constants as C  # noqa: E402
import opr_evaluation as ev  # noqa: E402
import opr_kern as K  # noqa: E402

N, LEVEL, R_ = 5, "M", 2.0


def main():
    lines = [f"n = {N}, Stufe {LEVEL}, r = {R_:g}, 30 Instanzen, {C.N_SCEN_FROZEN} Szenarien je Satz; Instanzstrom = Basis des Instanz-Zufalls-Seeds"]
    for base in (7000, 7100, 7200):
        recs = []
        for inst in range(C.N_INST):
            m, cv = K.gen_instance(N, LEVEL, inst, base=base)
            za = K.normals(N, C.N_SCEN_FROZEN, ev.scenario_seed("A", N, 5000 + base + inst))
            zb = K.normals(N, C.N_SCEN_FROZEN, ev.scenario_seed("B", N, 5000 + base + inst))
            res = ev.evaluate_instance(m, cv, K.durations(m, cv, za), K.durations(m, cv, zb), rule_seed=K.seed_of(6000, N, ord(LEVEL), inst), rs=(R_,))
            recs.append(res["r"][R_])
        s = ev.summarize(recs, ci=False)
        lines.append(f"Basis {base}: Δ {s['delta'] * 100:+.2f} %, besser/gleich/schlechter {s['better']}/{s['equal']}/{s['worse']}, auf Satz B besser in {s['better_B']}, "
                     f"Lücke Streuung {s['gap_sd'] * 100:.2f} %, SPT {s['gap_spt'] * 100:.2f} %, Optimum auf B / auf A {s['opt_B_over_A']:.3f}")
        print(lines[-1], flush=True)
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "opr_stabilitaet.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

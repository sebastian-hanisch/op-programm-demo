"""OP-Programm: Reihenfolge bei unsicheren Dauern - interaktive Fall-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anwendungsbeispiel, keine Beratung: ein erfundener Saal, ein erfundener Tag, erfundene Dauern. Drei Ansichten (Reihenfolge, Geplante Dauer, Aufwachbett) mit Umschalter;
nur die gewählte Ansicht rechnet. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

from fractions import Fraction

import numpy as np
import pandas as pd
import streamlit as st

import opr_analysis as AN
import opr_constants as C
import opr_evaluation as ev
import opr_exact as X
import opr_frozen as F
import opr_kern as K
import opr_rules as R
import opr_texts as TX
from opr_layout import dunkles_design, kopf_css
from opr_presets import (
    SETTING_SPECS,
    bounds,
    load_permalink_settings,
    randomize_seed,
    seed_widget,
    sidebar_presets,
    stash,
    sync_query_params,
    value_of,
)
from opr_rng import SplitMix64
from opr_visualization import (
    build_beds_bars,
    build_coupling_bars,
    build_exact_bars,
    build_histogram,
    build_lead_curve,
    build_rule_bars,
    build_scatter,
    build_tau_curve,
)

st.set_page_config(page_title="OP-Programm – Sebastian Hanisch", layout="wide")


# --- zwischengespeicherte Rechnungen --------------------------------------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _normals(cols, scen, seed):
    return K.normals(cols, scen, seed)


@st.cache_data(show_spinner=False)
def _instance(n, level, seed):
    m, cv = K.gen_instance(n, level, seed)
    return m, cv


@st.cache_data(show_spinner=False)
def _live(n, level, seed, scen, r, with_b):
    m, cv = K.gen_instance(n, level, seed)
    DA = K.durations(m, cv, _normals(n, scen, ev.scenario_seed("A", n, seed)))
    DB = K.durations(m, cv, _normals(n, scen, ev.scenario_seed("B", n, seed))) if with_b else None
    return AN.live_order(m, cv, DA, DB, r, rule_seed=K.seed_of(6000, n, ord(level), seed))


@st.cache_data(show_spinner=False)
def _live_lead(n, level, seed, scen):
    m, cv = K.gen_instance(n, level, seed)
    DA = K.durations(m, cv, _normals(n, scen, ev.scenario_seed("A", n, seed)))
    return AN.live_lead(m, cv, DA)


@st.cache_data(show_spinner=False)
def _cell(n, level, r):
    return ev.summarize(F.o1_cell(n, level, r))


@st.cache_data(show_spinner=False)
def _lead_cells(level, r):
    return {ekey: AN.lead_summary(F.lead_cell(level, r, ekey)) for ekey in C.E_FROZEN}


@st.cache_data(show_spinner=False)
def _tau_live(n, seed, scen, r, cw):
    m, cv = K.gen_instance(n, "M", seed)
    DA = K.durations(m, cv, _normals(n, scen, ev.scenario_seed("A", n, 100000 + seed)))
    perm = R.rule_orders(m, cv)["SD_auf"]
    return ev.tau_curve(m, cv, DA, perm, r, cw)


@st.cache_data(show_spinner=False)
def _beds_live(beds, recovery, level3, r, seed, scen):
    m, _ = K.gen_instance(C.N3, "M", seed)
    cv = np.zeros(C.N3) if level3 == "fest" else K.gen_instance(C.N3, level3, seed)[1]
    Z = _normals(2 * C.N3, scen, ev.scenario_seed("A", C.N3, 200000 + seed))
    return AN.aufwach_live(beds, recovery, r, m, cv, Z)


def _order_text(order):
    return " → ".join(str(i + 1) for i in order)


def _r_label(r):
    return ev.fmt(float(r), 1)


# --- Seitenkopf -------------------------------------------------------------------------------------------------------------------------------

st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 OP-Programm: Reihenfolge bei unsicheren Dauern")
st.markdown(TX.DISCLAIMER)
st.markdown(TX.INTRO)
st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
stash()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Szenario ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Instanz**")
    seed_widget("seed_input")
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1,
                           help="Zahl der Instanz: Mittel, Streuungen und Szenarien folgen aus ihr (die vorgerechneten Tabellen nutzen die Instanzen 0 bis 29).")
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)
seed = int(seed)

seed_widget("view_select")
view = st.segmented_control("Ansicht", C.VIEWS, key="view_select", selection_mode="single", label_visibility="collapsed")
view = view or C.DEFAULT_VIEW


def _select(label, options, key, help_=None, fmt_func=str):
    seed_widget(key)
    return st.select_slider(label, options=options, key=key, help=help_, format_func=fmt_func)


def _scen_widget():
    seed_widget("scen_select")
    return int(st.radio("Szenarien (live)", C.SCEN_CHOICES, key="scen_select", horizontal=True,
                        help="Anzahl der Zufallstage, über die live gemittelt wird. Die vorgerechneten Tabellen nutzen 2 000 je Satz."))


# =============================================================================================================================================
if view == "Reihenfolge":
    st.markdown("## 🎯 Reihenfolge der Eingriffe")
    st.markdown("### 1 · Welche Reihenfolge ist billig, und wie nah am Besten?")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        seed_widget("n_slider")
        n = st.slider("Eingriffe n", *bounds("n_slider"), key="n_slider", help="Bis n = 7 gibt es ein Stichproben-Optimum durch Aufzählung, ab 8 nur die beste lokale Verbesserung.")
    with c2:
        level = _select("Streuung der Eingriffe", C.LEVELS, "level_select", "Variationskoeffizienten: L 0,1 bis 0,3, M 0,2 bis 0,5, H 0,3 bis 0,8 (erfunden).", lambda x: C.LEVEL_LABELS[x])
    with c3:
        r = float(_select("Überziehpreis r", C.RS, "r_select", "Kosten des Überziehens je Minute im Verhältnis zum Leerlauf.", _r_label))
    with c4:
        scen = _scen_widget()
    with st.spinner("Rechne die Reihenfolgen (Zufallszahlen aus dem ganzzahligen Generator, das dauert einen Moment) …"):
        live = _live(n, level, seed, scen, r, False)
    m, cv = _instance(n, level, seed)
    rec = live["rec"]
    names = list(R.RULE_NAMES)
    opt_name = "Stichproben-Optimum (Aufzählung)" if n <= C.N_ENUM_MAX else "beste lokale Verbesserung (kein Optimum)"
    m1, m2, m3 = st.columns(3)
    spt, sd, opt = rec["SPT"], rec["SD_auf"], rec["OPT"]
    m1.metric("Kürzeste Dauer zuerst (SPT)", ev.fmt(spt, 1), delta="Referenz", delta_color="off")
    m2.metric("Kleinste Streuung zuerst", ev.fmt(sd, 1), delta=f"{ev.fmt(sd - spt, 1)} gegen SPT", delta_color="inverse")
    m3.metric(opt_name, ev.fmt(opt, 1), delta=f"{ev.fmt(opt - spt, 1)} gegen SPT", delta_color="inverse")
    order_rows = [("SPT", "SPT"), ("SD_auf", "SD_auf"), ("SD_ab", "SD_ab"), ("LPT", "LPT"), ("CV_auf", "CV_auf"), ("NV", "NV")]
    bar_rows = [(R.RULE_NAMES[k], live["comps"][k]["wait"], live["comps"][k]["idle"], r * live["comps"][k]["over"]) for k, _ in order_rows]
    bar_rows.append((opt_name.split(" (")[0], live["comps"]["OPT"]["wait"], live["comps"]["OPT"]["idle"], r * live["comps"]["OPT"]["over"]))
    st.plotly_chart(build_rule_bars(bar_rows), width="stretch", key="chart_rules")
    table = []
    for k, _ in order_rows:
        table.append({"Regel": R.RULE_NAMES[k], "Reihenfolge": _order_text(live["orders"][k]), "Kosten": ev.fmt(rec[k], 1), "relativ zu SPT": ev.fmt(rec[k] / spt, 3),
                      "über dem Optimum bzw. der lokalen Verbesserung": ev.pct(rec[k] / opt - 1, 2)})
    table.append({"Regel": R.RULE_NAMES["ZUFALL"], "Reihenfolge": "20 Zufallsreihenfolgen", "Kosten": ev.fmt(rec["ZUFALL"], 1), "relativ zu SPT": ev.fmt(rec["ZUFALL"] / spt, 3),
                  "über dem Optimum bzw. der lokalen Verbesserung": ev.pct(rec["ZUFALL"] / opt - 1, 2)})
    table.append({"Regel": opt_name, "Reihenfolge": _order_text(rec["opt_perm"]), "Kosten": ev.fmt(opt, 1), "relativ zu SPT": ev.fmt(opt / spt, 3), "über dem Optimum bzw. der lokalen Verbesserung": "–"})
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    st.caption("Kosten = Wartezeit + Leerlauf + r mal Überziehen, Mittel über die Szenarien, erfundene Minuten. Die Reihenfolge nennt die Eingriffsnummern der Tabelle unten.")
    inst_df = pd.DataFrame({"Eingriff": [str(i + 1) for i in range(n)], "Mittel (min)": [int(x) for x in m], "Variationskoeffizient": [ev.fmt(x, 2) for x in cv],
                            "Streuung (min)": [ev.fmt(a * b, 1) for a, b in zip(m, cv)]})
    with st.expander("Die Eingriffe dieser Instanz"):
        st.dataframe(inst_df, hide_index=True, width="stretch")
    st.plotly_chart(build_histogram(live["scen_spt"], live["scen_sd"]), width="stretch", key="chart_hist")
    d_scen = live["scen_sd"] - live["scen_spt"]
    se_rel = float(d_scen.std(ddof=1) / np.sqrt(len(d_scen)) / spt) if spt > 0 else 0.0
    if st.button("Probe auf einem zweiten, unabhängigen Szenariensatz rechnen", key="probe_start"):
        st.session_state["probe_for"] = (n, level, seed, scen, r)
    if st.session_state.get("probe_for") == (n, level, seed, scen, r):
        pb = _live(n, level, seed, scen, r, True)["rec"]
        st.info(f"Probe auf {scen} neuen Szenarien: das {'Stichproben-Optimum' if n <= C.N_ENUM_MAX else 'Ergebnis der lokalen Verbesserung'} kostet dort das {ev.fmt(pb['OPT_B'] / pb['OPT'], 3)}fache "
                f"seines Wertes auf dem Satz, auf dem es gewählt wurde (unter 1: optimistisch gewählt). Streuungsregel gegen SPT auf dem neuen Satz: "
                f"{ev.pct((pb['SD_auf_B'] - pb['SPT_B']) / pb['SPT_B'], 1, True)}. Einzelne Kosten liegen bei {scen} Szenarien um einige Prozent neben dem zweiten Satz.")
    n_cell = ev.nearest_frozen_n(n)
    cell = _cell(n_cell, level, r)
    live_info = {"seed": seed, "scen": scen, "delta": (sd - spt) / spt if spt else 0.0, "se_rel": se_rel, "gap_sd": sd / opt - 1, "gap_spt": spt / opt - 1, "enum": n <= C.N_ENUM_MAX, "n": n}
    kind, text = ev.verdict_order(live_info, cell, {"n_cell": n_cell, "level": level, "r": r})
    getattr(st, kind)(text)
    st.markdown(f"**Vorgerechnet** (30 Instanzen, 2 000 Szenarien je Satz, n = {n_cell}, Stufe {level}; die Zelle ist die nächstliegende zu n = {n}):")
    rows = []
    for rr in C.RS:
        c = _cell(n_cell, level, rr)
        rows.append({"r": _r_label(rr), "Δ Streuung − SPT": ev.pct(c["delta"], 1, True), "95-%-Intervall": f"[{ev.pct(c['delta_lo'], 1, True)}; {ev.pct(c['delta_hi'], 1, True)}]",
                     "besser / gleich / schlechter": f"{c['better']} / {c['equal']} / {c['worse']}", "Lücke SPT": ev.pct(c["gap_spt"], 2), "Lücke Streuung": ev.pct(c["gap_sd"], 2),
                     "Optimum = Streuungsregel in": f"{c['opt_is_sd']} von 30", "besser auf Satz B": f"{c['better_B']} von 30"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.info(TX.LIT_ORDER)

    # --- Experiment 2 -----------------------------------------------------------------------------------------------------------------------
    st.markdown("### 2 · Das Maß entscheidet")
    st.markdown("Der **Vorlauf e** ist ein stetiger Maß-Regler: der Patient kommt e Minuten vor seinem Termin und zählt ab der Ankunft als wartend. Bei **B** gibt es keine Termine: alle Patienten sind von Tagesbeginn an da.")
    mass = _select("Maß: Vorlauf e in Minuten (∞: Wartezeit ab Tagesbeginn zusätzlich; B: Maß B ohne Termine)", C.MASS_OPTIONS, "mass_select", None, lambda x: {"∞": "e = ∞", "B": "Maß B"}.get(x, f"e = {x}"))
    lead_level = level
    lead = _lead_cells(lead_level, r)
    labels = list(C.E_FROZEN)
    live_lead = _live_lead(n, level, seed, scen)[r]
    live_d = [(live_lead[k]["sd"] - live_lead[k]["spt"]) / live_lead[k]["spt"] for k in labels]
    st.plotly_chart(build_lead_curve(labels, [lead[k]["delta"] for k in labels], [lead[k]["better"] for k in labels], live_d), width="stretch", key="chart_lead")
    st.caption("Die Linie „30 Instanzen“ ist vorgerechnet (n = 7, Stufe und Überziehpreis wie oben); die gestrichelte Linie ist die gewählte Instanz mit n wie oben, live.")
    if mass == "B":
        b_rows = [{"Regel": R.RULE_NAMES[k], "Wartezeit ab Tagesbeginn (Summe der Mittel)": ev.fmt(live["mB"][k], 1), "relativ zu SPT": ev.fmt(live["mB"][k] / live["mB"]["SPT"], 3)}
                  for k in ("SPT", "SD_auf", "SD_ab", "LPT", "CV_auf", "NV")]
        st.dataframe(pd.DataFrame(b_rows), hide_index=True, width="stretch")
        st.caption("Maß B in geschlossener Form: erwartete Summe der Wartezeiten = Σ_k Σ_(j<k) m; die Streuung kommt nicht vor, SPT ist exakt optimal (Vertauschungsargument).")
    else:
        ll = live_lead[mass]
        st.metric(f"Gewählte Instanz, Vorlauf {'∞' if mass == '∞' else mass + ' min'}: Streuungsregel gegen SPT", ev.pct((ll["sd"] - ll["spt"]) / ll["spt"], 1, True),
                  delta=f"± {ev.pct(ll['se'] / ll['spt'], 1)} (Standardfehler der gepaarten Differenz)", delta_color="off")
    mb = AN.maass_b_summary(F.o1_maass_b(7, lead_level))
    kind, text = ev.verdict_mass(lead_level, r, lead, mb, mass)
    getattr(st, kind)(text)
    st.caption("Das Wartezeit-Gewicht allein ist kein Maß-Regler: es kippt die Rangfolge nicht (Streuungsregel besser als SPT in 29 von 30 Instanzen für jedes Gewicht von 0,25 bis 8, Stufe M, n = 7; Nachrechnung mit dem heutigen Code, nicht eingefroren, die frühere Messreihe nannte 28); der Wegfall der Termine tut es.")
    st.info("**Bekannt:** bei Maß B minimiert die kürzeste erwartete Dauer zuerst die erwartete Summe der Fertigstellungszeiten; die Streuung kommt nicht vor (Pinedo, Kap. 15, arXiv-Fassung im Wortlaut gelesen; das Springer-Lehrbuch nicht).")

    # --- Experiment 3 -----------------------------------------------------------------------------------------------------------------------
    st.markdown("### 3 · Wer gewinnt gegen den Zufall, und warum?")
    seed_widget("coupling_select")
    coupling = st.radio("Kopplung von Streuung und Mittel (für die Live-Instanz)", C.COUPLINGS, key="coupling_select", horizontal=True, format_func=lambda x: C.COUPLING_LABELS[x])
    per = {k: _cell(n_cell, k, r) for k in C.COUPLINGS}
    labels3 = [C.COUPLING_LABELS[k] for k in C.COUPLINGS]
    rel = {R.RULE_NAMES[k]: [per[c]["rel"][k if k != "SD_auf" else "SD_auf"] for c in C.COUPLINGS] for k in ("SD_auf", "CV_auf", "ZUFALL", "LPT", "SD_ab")}
    st.plotly_chart(build_coupling_bars([lbl.split(" (")[0] for lbl in labels3], rel), width="stretch", key="chart_coupling")
    mc, cvc = _instance(n, coupling, seed)
    sdc = mc * cvc
    corr = float(np.corrcoef(mc, sdc)[0, 1]) if np.std(sdc) > 0 and np.std(mc) > 0 else 0.0
    st.plotly_chart(build_scatter(mc, sdc, corr), width="stretch", key="chart_scatter")
    kind, text = ev.verdict_coupling(n_cell, r, per)
    getattr(st, kind)(text)

    # --- Experiment 4 -----------------------------------------------------------------------------------------------------------------------
    st.markdown("### 4 · Exakt statt Stichprobe: kleine Fälle")
    st.markdown("Jeder Eingriff hat **zwei gleich wahrscheinliche Dauern**; alle Ausgänge und alle Reihenfolgen werden in ganzen Zahlen durchgerechnet: **exakte** Erwartungswerte, kein Stichprobenfehler. Der Überziehpreis r ist der von oben.")
    seed_widget("exsrc_select")
    src = st.radio("Eingriffe", C.EX_SOURCES, key="exsrc_select", horizontal=True)
    if src == C.EX_SOURCES[0]:
        ex_names, ex_m, ex_lo, ex_hi = X.HAND_NAMES, list(X.HAND_M), list(X.HAND_LO), list(X.HAND_HI)
    elif src == C.EX_SOURCES[1]:
        ex_names, ex_m, ex_lo, ex_hi = X.PAIR_NAMES, list(X.PAIR_M), list(X.PAIR_LO), list(X.PAIR_HI)
    else:
        seed_widget("exn_slider")
        ex_n = st.slider("Eingriffe n (Zufallsinstanz, Zufalls-Seed wie in der Seitenleiste)", *bounds("exn_slider"), key="exn_slider")
        g = SplitMix64(K.seed_of(C.EXACT_SEED, ex_n, seed))
        ex_m, ex_lo, ex_hi = X.two_point_instance(g, ex_n)
        ex_names = tuple(f"E{i + 1}" for i in range(ex_n))
    r_frac = Fraction(str(r))
    table_ex = X.exact_table(ex_lo, ex_hi, ex_m, r_frac)
    spt_o, sd_o = X.rule_orders_exact(ex_m, ex_lo, ex_hi)
    cost_of = {p: c for p, c in table_ex}
    best = min(c for _, c in table_ex)
    opt_o = next(p for p, c in table_ex if c == best)
    nm = lambda p: "–".join(ex_names[i] for i in p)
    colors = ["#e45756" if c == best else (C.COLOR_SD if p == sd_o else (C.COLOR_SPT if p == spt_o else "rgba(150,150,150,0.65)")) for p, c in table_ex]
    st.plotly_chart(build_exact_bars([nm(p) for p, _ in table_ex], [float(c) for _, c in table_ex], colors), width="stretch", key="chart_exact")
    ex_df = pd.DataFrame([{"Reihenfolge": nm(p), "erwartete Kosten": ev.fmt_fraction(c), "Rolle": ", ".join(t for t, cond in (("Optimum", c == best), ("Streuungsregel", p == sd_o), ("SPT", p == spt_o)) if cond)}
                          for p, c in sorted(table_ex, key=lambda x: x[1])])
    st.dataframe(ex_df, hide_index=True, width="stretch")
    st.caption("Dauern der Eingriffe (je zu ½): " + "; ".join(f"{nm_}: {lo} oder {hi} min" for nm_, lo, hi in zip(ex_names, ex_lo, ex_hi)) + ".")
    stat = None
    if len(ex_m) in C.EXACT_NS:
        stat = dict(F.exact_stats(len(ex_m))[str(r_frac)], n=len(ex_m))
    lines = {"opt_name": nm(opt_o), "opt_cost": best, "sd_name": nm(sd_o), "sd_cost": cost_of[sd_o], "spt_name": nm(spt_o), "spt_cost": cost_of[spt_o]}
    kind, text = ev.verdict_exact(src, r, lines, stat)
    getattr(st, kind)(text)
    if stat is not None and stat["worst_instance"] is not None:
        wi = stat["worst_instance"]
        st.caption(f"Größte Lücke der Streuungsregel in den {ev._thousands(stat['n_inst'])} Instanzen mit n = {stat['n']} bei r = {_r_label(r)}: {ev.pct(stat['gap_sd_max'], 2)} (Mittel {wi['m']}, Dauern {wi['lo']} / {wi['hi']}; vorgerechnet).")
    st.caption("Eigene Rechnung, kein Nachbau eines Gegenbeispiels aus der Literatur; die Aussage gilt für diese symmetrische Zweipunkt-Verteilung und das Modell oben.")

# =============================================================================================================================================
elif view == "Geplante Dauer":
    st.markdown("## 🎯 Geplante Dauer: Mittelwert oder Quantil?")
    st.markdown("Statt der Mittelwerte kann man jedem Eingriff ein **Quantil τ** seiner Dauer als geplante Dauer geben. Reihenfolge: kleinste Streuung zuerst, Stufe M. Für **einen** Übergang ist das beste Quantil cw/(cw + 1) (Zeitungsverkäufer); hier zeigt sich, was ein ganzer Tag daraus macht.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n2 = int(_select("Eingriffe n", C.N2_CHOICES, "n2_select"))
    with c2:
        ratio = _select("Wartezeit : Leerlauf", C.RATIOS, "ratio_select", "Gewicht der Wartezeit im Verhältnis zum Leerlauf.")
    with c3:
        r2 = float(_select("Überziehpreis r", C.R2_CHOICES, "r2_select", "0: Überziehen zählt nicht.", lambda x: ev.fmt(x, 0)))
    with c4:
        tau = _select("Marke: Quantil τ", C.TAU_OPTIONS, "tau_select", None, lambda x: x if isinstance(x, str) else ev.fmt(x, 2))
    scen = _scen_widget()
    cw = C.RATIO_CW[ratio]
    summ = AN.tau_summary(n2, r2, cw)
    with st.spinner("Rechne die Kostenkurve der gewählten Instanz …"):
        live_curve = _tau_live(n2, seed, scen, r2, cw)
    live_rel = [float(x) for x in live_curve / live_curve[0]]
    xs_idx = None if tau == "Mittelwert" else 1 + C.TAUS.index(tau)
    st.plotly_chart(build_tau_curve(C.TAUS, summ["curve_A"], summ["curve_B"], summ["nv"], summ["j"] if summ["j"] else None, live=live_rel, mark=xs_idx), width="stretch", key="chart_tau")
    st.caption("Die Linien „Satz A“ und „Satz B“ sind vorgerechnet (30 Instanzen, 2 000 Szenarien je Satz); die gepunktete Linie ist die gewählte Instanz live (Instanz = Zufalls-Seed der Seitenleiste).")
    mean_level = F.tau_phi(n2)
    kind, text = ev.verdict_tau(n2, r2, ratio, summ, mean_level)
    getattr(st, kind)(text)
    rows = []
    for nn in C.TAU_NS:
        s = AN.tau_summary(nn, r2, cw)
        rows.append({"n": nn, "bestes τ*": "Mittelwert" if s["tau"] is None else ev.fmt(s["tau"], 2), "Zeitungsverkäufer-Verhältnis": ev.fmt(s["nv"], 2),
                     "Kosten bei τ* (Satz B, relativ zum Mittelwert)": ev.fmt(s["rel_B"], 3), "Median τ = 0,5": ev.fmt(s["rel_B_median"], 3)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Raster 0,05; ein gröberes Raster kann den Gewinn unterschätzen. τ* wird auf Satz A gewählt, der Gewinn auf dem unabhängigen Satz B gemessen.")
    st.info(TX.LIT_TAU)

# =============================================================================================================================================
else:
    st.markdown("## 🎯 Ein Bett im Aufwachraum")
    st.markdown("Nach dem Eingriff braucht der Patient ein **Aufwachbett**; ist keines frei, bleibt er im Saal und **blockiert** ihn. Es werden immer **n = 6** Eingriffe gerechnet (alle 720 Reihenfolgen).")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        beds = _select("Aufwachbetten", C.BEDS, "beds_select")
    with c2:
        recovery = _select("Aufwachdauer (Mittel 60 min)", C.RECOVERY, "recovery_select")
    with c3:
        level3 = _select("Streuung der Eingriffe", C.LEVELS3, "level3_select", None, lambda x: x if x == "fest" else C.LEVEL_LABELS[x])
    with c4:
        r3 = float(_select("Überziehpreis r", C.R3_CHOICES, "r3_select", None, lambda x: ev.fmt(x, 0)))
    scen = _scen_widget()
    with st.spinner("Rechne alle 720 Reihenfolgen mit und ohne Blockade …"):
        bl = _beds_live(beds, recovery, level3, r3, seed, scen)
    a1, a2, a3 = st.columns(3)
    a1.metric("Bester Tag mit Blockade (Stichproben-Optimum)", ev.fmt(bl["opt"], 1))
    a2.metric("Ohne Blockade", ev.fmt(bl["opt_ohne"], 1),
              delta=(f"{ev.pct(bl['opt'] / bl['opt_ohne'] - 1, 1, True)} durch Blockade" if bl["opt_ohne"] > 0 else f"{ev.fmt(bl['opt'] - bl['opt_ohne'], 1)} durch Blockade (ohne Blockade kostet der Tag nichts)"),
              delta_color="inverse")
    a3.metric("Blockadezeit (SPT, Anteil der Eingriffsdauern)", ev.pct(bl["blk_spt"], 2))
    gap_rows = [{"Regel": R.RULE_NAMES[k], "Kosten": ev.fmt(bl["rules"][k], 1), "über dem Optimum": (ev.pct(bl["rules"][k] / bl["opt"] - 1, 1) if bl["opt"] > 0 else "–")} for k in ("SPT", "SD_auf", "LPT", "SD_ab", "ALT", "ZUFALL")]
    st.dataframe(pd.DataFrame(gap_rows), hide_index=True, width="stretch")
    st.caption(f"Gewählte Instanz live ({scen} Szenarien). Beste Reihenfolge mit Blockade: {_order_text(bl['best'])}; ohne Blockade: {_order_text(bl['best_ohne'])}. "
               "Die Regel „abwechselnd kurz und lang“ ist eine eigene, nicht die Regel von Marcon/Dexter.")
    lvl = level3 if level3 in C.LEVELS else "M"
    s3 = AN.o3_summary(lvl, r3)
    dec = AN.o3_decomposition(r3)
    st.plotly_chart(build_beds_bars(["1", "2", "3", "unbegrenzt"], [s3["B1"]["ratio"], s3["B2"]["ratio"], s3["B3"]["ratio"], 1.0]), width="stretch", key="chart_beds")
    kind, text = ev.verdict_beds(lvl, r3, s3, dec)
    getattr(st, kind)(text)
    st.caption("Vorgerechnet: n = 6, 30 Instanzen, 2 000 Szenarien, Aufwachdauer lognormal (Mittel 60, Variationskoeffizient 0,5), unabhängig vom Eingriff, kein Ausfall von Betten; "
               "die Zerlegung zählt die Streuung der Aufwachdauer mit (fest 60 statt streuend: Blockadekosten " f"{ev.fmt(dec['fest_fest'], 1)} statt {ev.fmt(dec['blockade_allein'], 1)}).")
    st.info(TX.LIT_BEDS)

sync_query_params({key: value_of(key) for key in SETTING_SPECS})

st.markdown("---")

# --- Kernfragen -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
st.markdown(TX.KERNFRAGEN)

# --- Methodenvergleich und Formulierung -----------------------------------------------------------------------------------------------------

with st.expander("🔧 Wie wir das erreichen – vollständiger Methodenvergleich"):
    st.markdown(TX.METHODS_MD)

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(TX.MATH_MD_INTRO)
    for caption, eq in TX.MATH_EQUATIONS:
        st.markdown(caption)
        st.latex(eq)
    st.markdown(TX.MATH_MD_OUTRO)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)

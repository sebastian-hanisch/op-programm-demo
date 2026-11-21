"""Plotly-Abbildungen (Achsen fest, damit Touch-Scrollen nicht am Diagramm hängen bleibt)."""

import plotly.graph_objects as go

import opr_constants as C

_LAYOUT = dict(margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0), height=380)


def _fix(fig, **kw):
    fig.update_layout(**_LAYOUT, **kw)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_rule_bars(rows, title="Kosten je Regel relativ zu SPT (Mittel über die Szenarien)"):
    """rows: [(Beschriftung, Wartezeit, Leerlauf, Überziehen mal r)] ; Balken relativ zu den Gesamtkosten der ersten Zeile (SPT)."""
    base = sum(rows[0][1:]) or 1.0
    fig = go.Figure()
    for i, (name, color) in enumerate((("Wartezeit", C.COLOR_WAIT), ("Leerlauf", C.COLOR_IDLE), ("Überziehen (mit Preis r)", C.COLOR_OVER))):
        fig.add_bar(x=[r[0] for r in rows], y=[r[1 + i] / base for r in rows], name=name, marker_color=color)
    fig.update_layout(barmode="stack", title=title, yaxis_title="Kosten relativ zu SPT")
    return _fix(fig)


def build_histogram(scen_spt, scen_sd):
    fig = go.Figure()
    fig.add_histogram(x=scen_spt, name="kürzeste Dauer zuerst (SPT)", marker_color=C.COLOR_SPT, opacity=0.6, nbinsx=40)
    fig.add_histogram(x=scen_sd, name="kleinste Streuung zuerst", marker_color=C.COLOR_SD, opacity=0.6, nbinsx=40)
    fig.update_layout(barmode="overlay", title="Verteilung der Tageskosten über die Szenarien", xaxis_title="Kosten eines Tages", yaxis_title="Szenarien")
    return _fix(fig)


def build_lead_curve(labels, deltas, better, live=None):
    """Δ(e) in Prozent (negativ: die Streuungsregel ist billiger) mit Zählung „besser in k von 30“; optional die Live-Instanz als zweite Linie."""
    fig = go.Figure()
    fig.add_scatter(x=labels, y=[d * 100 for d in deltas], mode="lines+markers+text", name="30 Instanzen (vorgerechnet)", line=dict(color=C.COLOR_SD),
                    text=[f"{b}/30" for b in better], textposition="top center")
    if live is not None:
        fig.add_scatter(x=labels, y=[d * 100 for d in live], mode="lines+markers", name="gewählte Instanz (live)", line=dict(color=C.COLOR_SPT, dash="dot"))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title="Streuungsregel gegen SPT bei Vorlauf e (negativ: Streuungsregel billiger)", xaxis_title="Vorlauf e in Minuten (∞: Wartezeit ab Tagesbeginn zusätzlich)",
                      yaxis_title="Differenz in % von SPT")
    return _fix(fig)


def build_coupling_bars(labels, rel_by_rule):
    """rel_by_rule: {Regelname: [Kosten relativ zu SPT je Kopplung]}."""
    fig = go.Figure()
    for name, vals in rel_by_rule.items():
        fig.add_bar(x=labels, y=vals, name=name)
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
    fig.update_layout(barmode="group", title="Kosten der Regeln relativ zu SPT, je Kopplung von Streuung und Mittel (Mittel über 30 Instanzen)", yaxis_title="relativ zu SPT")
    return _fix(fig)


def build_scatter(m, sd, corr):
    fig = go.Figure()
    fig.add_scatter(x=list(m), y=list(sd), mode="markers", marker=dict(color=C.COLOR_SD, size=10), name="Eingriffe")
    fig.update_layout(title=f"Mittel und Streuung der Eingriffe dieser Instanz (Korrelation {corr:+.2f})".replace(".", ","),
                      xaxis_title="Mittel der Dauer (Minuten)", yaxis_title="Streuung (Minuten)")
    return _fix(fig)


def build_exact_bars(labels, values, marks):
    """Alle n! Reihenfolgen mit exakten Kosten; `marks`: Farbe je Balken."""
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=marks))
    fig.update_layout(title="Exakte erwartete Kosten aller Reihenfolgen (grün: Streuungsregel, orange: SPT, rot: Optimum)", yaxis_title="erwartete Kosten", xaxis_title="Reihenfolge")
    return _fix(fig, showlegend=False)


def build_tau_curve(taus, curve_a, curve_b, nv, star, live=None, mark=None):
    xs = ["Mittelwert"] + [f"{t:.2f}".replace(".", ",") for t in taus]
    fig = go.Figure()
    fig.add_scatter(x=xs, y=curve_a, mode="lines+markers", name="Satz A (hier wird τ* gewählt)", line=dict(color=C.COLOR_LINE))
    fig.add_scatter(x=xs, y=curve_b, mode="lines+markers", name="unabhängiger Satz B", line=dict(color=C.COLOR_SD))
    if live is not None:
        fig.add_scatter(x=xs, y=live, mode="lines+markers", name="gewählte Instanz (live)", line=dict(color=C.COLOR_SPT, dash="dot"))
    if mark is not None:
        fig.add_vline(x=xs[mark], line_dash="dash", line_color="gray")
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
    fig.update_layout(title=f"Kosten relativ zu „Mittelwert geplant“ je geplantem Quantil (Newsvendor-Verhältnis {nv:.2f}; τ* markiert)".replace(".", ","),
                      xaxis_title="geplante Dauer = Quantil τ der Dauerverteilung", yaxis_title="Kosten relativ zu Mittelwert-Planung")
    if star is not None:
        fig.add_vline(x=xs[star], line_dash="dot", line_color=C.COLOR_OVER)
    return _fix(fig)


def build_beds_bars(labels, ratios, lows=None, highs=None):
    fig = go.Figure(go.Bar(x=labels, y=ratios, marker_color=C.COLOR_OVER, text=[f"{r:.3f}".replace(".", ",") for r in ratios], textposition="outside"))
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
    fig.update_layout(title="Kosten des besten Tagesplans relativ zu „ohne Blockade“ (Mittel über 30 Instanzen, vorgerechnet)", yaxis_title="relativ zu ohne Blockade",
                      xaxis_title="Aufwachbetten")
    return _fix(fig, showlegend=False)

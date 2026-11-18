"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


# --- Ergänzungen: Ansichten, Schnellstart, Regler-Optionen, Worst-Case, Permalink -----------------------------------------------------------------

import pytest  # noqa: E402

import opr_constants as C  # noqa: E402

ANSICHTEN = {"Reihenfolge": ("level_select", "r_select", "scen_select", "mass_select", "coupling_select", "exsrc_select"),
             "Geplante Dauer": ("n2_select", "ratio_select", "r2_select", "tau_select", "scen_select"),
             "Aufwachbett": ("beds_select", "recovery_select", "level3_select", "r3_select", "scen_select")}
OPTIONEN = {"level_select": C.LEVELS, "r_select": C.RS, "scen_select": C.SCEN_CHOICES, "mass_select": C.MASS_OPTIONS, "coupling_select": C.COUPLINGS, "exsrc_select": C.EX_SOURCES,
            "n2_select": C.N2_CHOICES, "ratio_select": C.RATIOS, "r2_select": C.R2_CHOICES, "tau_select": C.TAU_OPTIONS, "beds_select": C.BEDS, "recovery_select": C.RECOVERY,
            "level3_select": C.LEVELS3, "r3_select": C.R3_CHOICES}


def _widget(at, key):
    for kind in ("slider", "select_slider", "radio", "button_group"):
        try:
            return getattr(at, kind)(key=key)
        except KeyError:
            continue
    raise AssertionError(key)


@pytest.mark.parametrize("view", list(ANSICHTEN))
def test_jede_option_jedes_reglers_der_ansicht_laeuft_ohne_fehler(view):
    at = _fresh_app()
    _widget(at, "view_select").set_value(view).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    for key in ANSICHTEN[view]:
        for option in OPTIONEN[key]:
            _widget(at, key).set_value(option).run(timeout=TIMEOUT)
            assert not at.exception, (view, key, option, [str(e) for e in at.exception])


def test_jedes_preset_zeigt_seine_ansicht_und_rechnet():
    for name, preset in C.PRESETS.items():
        at = _fresh_app()
        at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert _widget(at, "view_select").value == preset["view"]
        assert len(at.info) + len(at.success) + len(at.warning) >= 2, name       # Verdikt und Hinweisblock sind da


def test_worst_case_am_reglermaximum_ohne_fehler():
    """Reihenfolge: n = 12, Stufe H, r = 8, 1 000 Szenarien (Timeout statt Zeit-Assertion); Aufwachbett: 1 000 Szenarien, drei Betten."""
    at = _fresh_app()
    at.slider(key="n_slider").set_value(12).run(timeout=120)
    _widget(at, "level_select").set_value("H").run(timeout=120)
    _widget(at, "r_select").set_value(8.0).run(timeout=120)
    _widget(at, "scen_select").set_value(1000).run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]
    _widget(at, "view_select").set_value("Aufwachbett").run(timeout=120)
    _widget(at, "beds_select").set_value("3").run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]


def test_permalink_mit_ungueltigem_wert_faellt_auf_den_standard_zurueck():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["view"] = "Unsinn"
    at.query_params["n"] = "99"
    at.query_params["r"] = "3"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert _widget(at, "view_select").value == C.DEFAULT_VIEW
    assert at.slider(key="n_slider").value == C.N_MAX                          # begrenzt auf die Reglergrenze
    assert _widget(at, "r_select").value == C.DEFAULT_R                        # 3 steht nicht in den Optionen


def test_sichtbar_anwendungsbeispiel_keine_beratung_und_expander_per_titel():
    at = _fresh_app()
    assert any("Anwendungsbeispiel, keine Beratung" in m.value for m in at.markdown)
    labels = {e.label for e in at.expander}
    assert {"💬 In einfachen Worten", "🔧 Wie wir das erreichen – vollständiger Methodenvergleich", "📐 Mathematische Formulierung"} <= labels
    texts = " ".join(m.value for m in at.markdown)
    assert "https://sebastianhanisch.net/op-programm-erklaert.html)" in texts and "op-programm-erklaert.html#fachleute" in texts
    assert texts.count("Lesehilfe.") == 1 and "| Symbol | Bedeutung |" in texts and texts.count("| Symbol | Bedeutung |") == 1

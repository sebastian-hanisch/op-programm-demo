"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart oben in der Seitenleiste, aktives Szenario hervorgehoben, jeder Regler hebt die Markierung auf.

Die Regler der drei Ansichten stehen im Hauptbereich und gibt es nur in ihrer Ansicht; der Merkspeicher (`stash`/`seed_widget`) hält ihre Werte beim Wechsel der Ansicht."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import opr_constants as C
import opr_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
WIDGET_KINDS = ("slider", "number_input", "select_slider", "selectbox", "radio", "button_group", "toggle", "checkbox", "text_input")

VIEW_OF = {
    "view_select": None, "seed_input": None,
    "n_slider": "Reihenfolge", "level_select": "Reihenfolge", "r_select": "Reihenfolge", "scen_select": None, "mass_select": "Reihenfolge", "coupling_select": "Reihenfolge",
    "exsrc_select": "Reihenfolge", "exn_slider": "Reihenfolge",
    "tau_select": "Geplante Dauer", "ratio_select": "Geplante Dauer", "r2_select": "Geplante Dauer", "n2_select": "Geplante Dauer",
    "beds_select": "Aufwachbett", "recovery_select": "Aufwachbett", "level3_select": "Aufwachbett", "r3_select": "Aufwachbett",
}
OPTIONS = {
    "view_select": C.VIEWS, "level_select": C.LEVELS, "r_select": C.RS, "scen_select": C.SCEN_CHOICES, "mass_select": C.MASS_OPTIONS, "coupling_select": C.COUPLINGS,
    "exsrc_select": C.EX_SOURCES, "tau_select": C.TAU_OPTIONS, "ratio_select": C.RATIOS, "r2_select": C.R2_CHOICES, "n2_select": C.N2_CHOICES,
    "beds_select": C.BEDS, "recovery_select": C.RECOVERY, "level3_select": C.LEVELS3, "r3_select": C.R3_CHOICES,
}


def _app(timeout=300) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=timeout)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _widget(at, key):
    for kind in WIDGET_KINDS:
        try:
            return getattr(at, kind)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} wird nicht gezeichnet")


def _show_view(at, view):
    if view is not None and _widget(at, "view_select").value != view:
        _widget(at, "view_select").set_value(view).run()
        assert not at.exception, [str(e) for e in at.exception]


def _types(at) -> dict:
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _alt_value(at, key):
    """Ein gültiger Wert, der vom aktuellen verschieden ist."""
    w = _widget(at, key)
    if key == "exsrc_select":
        return C.EX_SOURCES[2]      # „Zwei Eingriffe“ ist selbst ein Schnellstart-Wert und würde dann (zu Recht) hervorgehoben
    if key in OPTIONS:
        return next(o for o in OPTIONS[key] if o != w.value)
    spec = P.SETTING_SPECS[key]
    return spec.hi if w.value != spec.hi else spec.lo


def test_presets_stehen_in_der_seitenleiste_und_nicht_im_hauptbereich():
    at = _app()
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)
    assert not [b for b in at.main.button if b.label in C.PRESETS]


def test_titel_steht_im_festen_kopf():
    at = _app()
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))
    assert at.title[0].value


def test_beim_start_ist_genau_das_standard_preset_hervorgehoben():
    assert [n for n, t in _types(_app()).items() if t == "primary"] == [next(iter(C.PRESETS))]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_preset_ist_hervorgehoben_und_bleibt_es_im_naechsten_lauf(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    assert [n for n, t in _types(at).items() if t == "primary"] == [name]
    at.run()
    assert [n for n, t in _types(at).items() if t == "primary"] == [name]
    assert _widget(at, "view_select").value == C.PRESETS[name]["view"]


def test_jeder_regler_der_seitenleiste_steht_in_setting_specs():
    at = _app()
    keys = {w.key for kind in WIDGET_KINDS for w in getattr(at.sidebar, kind) if w.key}
    assert keys <= set(P.SETTING_SPECS), f"in SETTING_SPECS fehlt: {sorted(keys - set(P.SETTING_SPECS))}"


def test_jeder_regler_der_drei_ansichten_steht_in_setting_specs_und_im_testplan():
    """Die Regler der Ansichten sind im Hauptbereich; ihre Schlüssel müssen in SETTING_SPECS (Hervorhebung, Permalink) und in VIEW_OF dieses Tests stehen."""
    assert set(VIEW_OF) == set(P.SETTING_SPECS)
    for view in C.VIEWS:
        at = _app()
        _show_view(at, view)
        keys = {w.key for kind in WIDGET_KINDS for w in getattr(at.main, kind) if w.key}
        assert keys <= set(P.SETTING_SPECS), (view, sorted(keys - set(P.SETTING_SPECS)))
        assert {k for k in keys if VIEW_OF[k] not in (None, view)} == set()


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_jede_aenderung_eines_reglers_hebt_die_hervorhebung_auf(state_key):
    at = _app()
    _show_view(at, VIEW_OF[state_key])
    if state_key == "exn_slider":                          # der Regler wird nur bei „Zufallsinstanz“ gezeichnet (Zustand bleibt im Merkspeicher)
        _widget(at, "exsrc_select").set_value(C.EX_SOURCES[2]).run()
        assert not at.exception, [str(e) for e in at.exception]
    value = _alt_value(at, state_key)
    _widget(at, state_key).set_value(value).run()
    assert not at.exception, (state_key, [str(e) for e in at.exception])
    assert "primary" not in _types(at).values(), state_key
    name = next(iter(C.PRESETS))
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert _types(at)[name] == "primary"


def test_ansichtswechsel_behaelt_die_werte_der_regler():
    at = _app()
    _widget(at, "r_select").set_value(8.0).run()
    _show_view(at, "Aufwachbett")
    _widget(at, "r3_select").set_value(4.0).run()
    _show_view(at, "Reihenfolge")
    assert _widget(at, "r_select").value == 8.0
    _show_view(at, "Aufwachbett")
    assert _widget(at, "r3_select").value == 4.0
    assert not at.exception

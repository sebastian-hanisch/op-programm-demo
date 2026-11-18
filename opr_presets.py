"""Permalink (Begrenzen und Einrasten), Schnellstart-Szenarien und Zufalls-Seed-Button.

Die Regler des gewählten Reiters stehen im Hauptbereich und verschwinden mit dem Reiter. Streamlit löscht den Zustand eines Reglers, sobald er nicht gezeichnet wird:
darum merkt sich `stash()` in jedem Lauf den Wert jedes gezeichneten Reglers unter `_kept_<Schlüssel>`, und `seed_widget(key)` setzt unmittelbar vor dem Zeichnen den
zuletzt gewählten (oder Standard-)Wert. Preset und Permalink schreiben nur in diesen Speicher (und, falls der Regler gerade gezeichnet ist, in seinen Zustand, im Callback)."""

import random
from dataclasses import dataclass

import streamlit as st

import opr_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: object
    default: object
    lo: float | None = None
    hi: float | None = None


def _choice(options):
    def cast(value):
        for o in options:
            if str(o) == str(value):
                return o
            if isinstance(o, (int, float)) and not isinstance(o, bool):
                try:
                    if float(value) == float(o):
                        return o
                except (TypeError, ValueError):
                    pass
        raise ValueError(value)
    return cast


SETTING_SPECS = {
    "view_select": SettingSpec("view", _choice(C.VIEWS), C.DEFAULT_VIEW),
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "level_select": SettingSpec("level", _choice(C.LEVELS), C.DEFAULT_LEVEL),
    "r_select": SettingSpec("r", _choice(C.RS), C.DEFAULT_R),
    "scen_select": SettingSpec("scen", _choice(C.SCEN_CHOICES), C.DEFAULT_SCEN),
    "mass_select": SettingSpec("mass", _choice(C.MASS_OPTIONS), C.DEFAULT_MASS),
    "coupling_select": SettingSpec("coupling", _choice(C.COUPLINGS), C.DEFAULT_COUPLING),
    "exsrc_select": SettingSpec("exsrc", _choice(C.EX_SOURCES), C.DEFAULT_EX_SOURCE),
    "exn_slider": SettingSpec("exn", int, C.DEFAULT_EX_N, C.EX_N_MIN, C.EX_N_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "tau_select": SettingSpec("tau", _choice(C.TAU_OPTIONS), C.DEFAULT_TAU),
    "ratio_select": SettingSpec("ratio", _choice(C.RATIOS), C.DEFAULT_RATIO),
    "r2_select": SettingSpec("r2", _choice(C.R2_CHOICES), C.DEFAULT_R2),
    "n2_select": SettingSpec("n2", _choice(C.N2_CHOICES), C.DEFAULT_N2),
    "beds_select": SettingSpec("beds", _choice(C.BEDS), C.DEFAULT_BEDS),
    "recovery_select": SettingSpec("rec", _choice(C.RECOVERY), C.DEFAULT_RECOVERY),
    "level3_select": SettingSpec("level3", _choice(C.LEVELS3), C.DEFAULT_LEVEL3),
    "r3_select": SettingSpec("r3", _choice(C.R3_CHOICES), C.DEFAULT_R3),
}
PRESET_KEYS = {"view": "view_select", "n": "n_slider", "level": "level_select", "r": "r_select", "scen": "scen_select", "mass": "mass_select",
               "coupling": "coupling_select", "ex_src": "exsrc_select", "ex_n": "exn_slider", "seed": "seed_input", "tau": "tau_select",
               "ratio": "ratio_select", "r2": "r2_select", "n2": "n2_select", "beds": "beds_select", "recovery": "recovery_select",
               "level3": "level3_select", "r3": "r3_select"}
KEPT = {key: f"_kept_{key}" for key in SETTING_SPECS}


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def value_of(state_key):
    """Aktueller Wert eines Reglers: gezeichneter Zustand, sonst gemerkter Wert, sonst Standard."""
    return st.session_state.get(state_key, st.session_state.get(KEPT[state_key], SETTING_SPECS[state_key].default))


def stash():
    """Am Anfang jedes Laufs: den Wert jedes gezeichneten Reglers merken."""
    for key, kept in KEPT.items():
        if key in st.session_state:
            st.session_state[kept] = st.session_state[key]


def seed_widget(state_key):
    """Unmittelbar vor dem Zeichnen eines Reglers: fehlt sein Zustand, kommt der gemerkte (oder der Standard-)Wert."""
    if state_key not in st.session_state:
        st.session_state[state_key] = st.session_state.get(KEPT[state_key], SETTING_SPECS[state_key].default)


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf Reglergrenzen begrenzt, in den Merkspeicher."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if spec.lo is not None:
                value = max(spec.lo, value)
            if spec.hi is not None:
                value = min(spec.hi, value)
            st.session_state[KEPT[state_key]] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[KEPT[state_key]] = C.PRESETS[name][key]
            if state_key in st.session_state:
                st.session_state[state_key] = C.PRESETS[name][key]


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None (zustandslos, über alle Regler aus SETTING_SPECS)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(value_of(state_key) == preset.get(preset_key_of.get(state_key), spec.default) for state_key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Szenario ist hervorgehoben (`type="primary"`)."""
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state[KEPT["seed_input"]] = random.randint(0, C.SEED_MAX)
    if "seed_input" in st.session_state:
        st.session_state["seed_input"] = st.session_state[KEPT["seed_input"]]

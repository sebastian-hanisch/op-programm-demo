"""Presets (Schnellstart): Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink begrenzt und rastet ein."""

import pytest

import opr_constants as C
import opr_presets as P


def test_jedes_preset_hat_hilfetext_und_alle_schluessel():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_werte_liegen_in_den_reglergrenzen_und_optionen():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            value = preset[key]
            assert spec.caster(value) == value, (name, key, value)         # Wert steht in den Optionen bzw. ist eine ganze Zahl
            if spec.lo is not None:
                assert spec.lo <= value <= spec.hi, (name, key)


def test_standard_preset_ist_die_standardeinstellung():
    first = C.PRESETS[next(iter(C.PRESETS))]
    assert all(first[k] == P.SETTING_SPECS[s].default for k, s in P.PRESET_KEYS.items())


def test_setting_specs_decken_jeden_preset_schluessel_und_kept_jeden_regler():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS) == set(P.KEPT)


def test_jedes_preset_wirkt_auf_mindestens_einen_regler_gegenueber_dem_standard():
    base = C.PRESETS[next(iter(C.PRESETS))]
    for name, preset in list(C.PRESETS.items())[1:]:
        assert preset != base, f"Preset {name} hat keine Wirkung"


def test_caster_wirft_bei_ungueltigem_wert_und_matcht_zahlen_und_texte():
    caster = P.SETTING_SPECS["r_select"].caster
    assert caster("0.5") == 0.5 and caster("8") == 8.0
    assert P.SETTING_SPECS["tau_select"].caster("0.8") == 0.8 and P.SETTING_SPECS["tau_select"].caster("Mittelwert") == "Mittelwert"
    with pytest.raises(ValueError):
        caster("3")
    with pytest.raises(ValueError):
        P.SETTING_SPECS["view_select"].caster("Unbekannt")

"""Konstanten: Annahmen des erfundenen Modells, Regler-Grenzen, Farben und Schnellstart-Szenarien.

Alles ist erfunden: Minuten sind erfundene Zeiteinheiten, Klassenmittel, Variationskoeffizienten, Kostengewichte, Aufwachdauern und Bettenzahlen
sind Annahmen, keine Aussage über ein Krankenhaus (Anwendungsbeispiel, keine Beratung)."""

# --- Modell (erfundene Annahmen) ---------------------------------------------------------------------------------------------------------
CLASS_NAMES = ("klein", "mittel", "groß")
CLASS_MEANS = (45.0, 90.0, 150.0)            # Klassenmittel der Eingriffe in erfundenen Minuten; je Eingriff mit Faktor 0,85 bis 1,15 gezogen und gerundet
FACTOR_LO, FACTOR_HI = 0.85, 1.15
CV_LEVELS = {                                # Variationskoeffizienten je Streuungsstufe (unabhängig vom Mittel gezogen)
    "L": (0.10, 0.20, 0.30),
    "M": (0.20, 0.35, 0.50),
    "H": (0.30, 0.50, 0.80),
    "K": (0.40,),                            # konstanter Variationskoeffizient: der Gegenfall zur Kopplung
}
U_SDS = (8.0, 20.0, 40.0)                    # Stufe "U": absolute Streuung in Minuten, unabhängig vom Mittel
LEVEL_LABELS = {"L": "niedrig (L)", "M": "mittel (M)", "H": "hoch (H)", "K": "konstant 0,4 (K)", "U": "unabhängig vom Mittel (U)"}
SLACK = 0.10                                 # Tagesende T = (1 + SLACK) mal Summe der Mittel
R_MEAN, R_CV = 60.0, 0.5                     # Aufwachdauer (erfunden): Mittel 60 min, Variationskoeffizient 0,5
INSTANCE_BASE = 7000                         # Basis der Zufalls-Seeds der Instanzen (gefrorene Tabellen und Live-Rechnung nutzen dieselben Instanzen)
N_RULE_RANDOM = 20                           # Zufallsreihenfolgen für die Regel "Zufall"

# --- Regler ------------------------------------------------------------------------------------------------------------------------------
VIEWS = ("Reihenfolge", "Geplante Dauer", "Aufwachbett")
DEFAULT_VIEW = "Reihenfolge"
N_MIN, N_MAX, DEFAULT_N = 4, 12, 7
N_ENUM_MAX = 7                               # bis hier wird das Stichproben-Optimum durch Aufzählung aller n! Reihenfolgen bestimmt
LEVELS = ("L", "M", "H")
DEFAULT_LEVEL = "M"
RS = (0.5, 1.0, 2.0, 4.0, 8.0)               # Überziehen : Leerlauf
DEFAULT_R = 2.0
SCEN_CHOICES = (500, 1000)
DEFAULT_SCEN = 500
SEED_MAX, DEFAULT_SEED = 999999, 7
MASS_OPTIONS = ("0", "10", "20", "40", "80", "160", "∞", "B")   # Vorlauf e in Minuten, "∞" = Wartezeit ab Tagesbeginn zusätzlich, "B" = Maß B
DEFAULT_MASS = "0"
E_VALUES = {"0": 0.0, "10": 10.0, "20": 20.0, "40": 40.0, "80": 80.0, "160": 160.0, "∞": 1e9}
E_FROZEN = ("0", "10", "20", "40", "80", "160", "∞")
COUPLINGS = ("K", "M", "U")
COUPLING_LABELS = {"K": "konstanter Variationskoeffizient", "M": "Stufe M (Streuung wächst mit dem Mittel nur im Mittel)", "U": "Streuung unabhängig vom Mittel"}
DEFAULT_COUPLING = "M"
EX_SOURCES = ("Handbeispiel A, B, C", "Zwei Eingriffe, gleiches Mittel", "Zufallsinstanz")
DEFAULT_EX_SOURCE = EX_SOURCES[0]
EX_N_MIN, EX_N_MAX, DEFAULT_EX_N = 3, 5, 4
# Reiter "Geplante Dauer"
TAU_OPTIONS = ("Mittelwert",) + tuple(round(0.30 + 0.05 * i, 2) for i in range(13))
DEFAULT_TAU = "Mittelwert"
RATIOS = ("1 : 2", "1 : 1", "2 : 1", "4 : 1")
RATIO_CW = {"1 : 2": 0.5, "1 : 1": 1.0, "2 : 1": 2.0, "4 : 1": 4.0}   # Wartezeit-Gewicht bei Leerlauf-Gewicht 1
DEFAULT_RATIO = "1 : 1"
R2_CHOICES = (0.0, 1.0)
DEFAULT_R2 = 0.0
N2_CHOICES = (2, 3, 4, 6, 8)
DEFAULT_N2 = 8
# Reiter "Aufwachbett"
BEDS = ("1", "2", "3", "unbegrenzt")
DEFAULT_BEDS = "1"
RECOVERY = ("fest 60", "lognormal, Variationskoeffizient 0,5")
DEFAULT_RECOVERY = RECOVERY[1]
LEVELS3 = ("fest", "L", "M", "H")
DEFAULT_LEVEL3 = "M"
R3_CHOICES = (1.0, 2.0, 4.0)
DEFAULT_R3 = 2.0
N3 = 6                                       # der Reiter Aufwachbett rechnet immer n = 6 (alle 720 Reihenfolgen)

# --- gefrorene Tabellen ------------------------------------------------------------------------------------------------------------------
N_INST = 30                                  # Instanzen je Zelle
N_SCEN_FROZEN = 2000                         # Szenarien je Satz (A wählt das Optimum und bewertet, B bewertet nur)
FROZEN_NS = (5, 7, 8, 12)
FROZEN_LEVELS = ("L", "M", "H", "K", "U")
TAUS = tuple(round(0.30 + 0.05 * i, 2) for i in range(13))
TAU_NS = (2, 3, 4, 6, 8)
EXACT_NS = (3, 4, 5, 6)
EXACT_COUNTS = {3: 2000, 4: 2000, 5: 1000, 6: 300}
EXACT_SEED = 20261118
SCEN_SEED = {"A": 4001, "B": 8001}           # Zufalls-Seeds der Szenariensätze (gefroren); live: Zufalls-Seed der Instanz

# --- Farben ------------------------------------------------------------------------------------------------------------------------------
COLOR_WAIT = "#4c78a8"
COLOR_IDLE = "#9ecae9"
COLOR_OVER = "#e45756"
COLOR_SD = "#2F6B65"
COLOR_SPT = "#f58518"
COLOR_OTHER = "rgba(150,150,150,0.65)"
COLOR_LINE = "#4c78a8"

# --- Schnellstart ------------------------------------------------------------------------------------------------------------------------
_BASE = {
    "view": DEFAULT_VIEW, "n": DEFAULT_N, "level": DEFAULT_LEVEL, "r": DEFAULT_R, "scen": DEFAULT_SCEN, "mass": DEFAULT_MASS,
    "coupling": DEFAULT_COUPLING, "ex_src": DEFAULT_EX_SOURCE, "ex_n": DEFAULT_EX_N, "seed": DEFAULT_SEED,
    "tau": DEFAULT_TAU, "ratio": DEFAULT_RATIO, "r2": DEFAULT_R2, "n2": DEFAULT_N2,
    "beds": DEFAULT_BEDS, "recovery": DEFAULT_RECOVERY, "level3": DEFAULT_LEVEL3, "r3": DEFAULT_R3,
}
_OVERRIDES = {
    "Standardtag (n = 7, Stufe M, r = 2, Maß A)": {},
    "Maß B: Wartezeit ab Tagesbeginn": {"mass": "B"},
    "Kipppunkt Vorlauf": {"mass": "160", "r": 1.0},
    "Konstanter Variationskoeffizient": {"coupling": "K", "n": 8},
    "Streuung unabhängig vom Mittel": {"coupling": "U", "n": 8},
    "Zwei Eingriffe, gleiches Mittel": {"ex_src": EX_SOURCES[1]},
    "Überziehen sehr teuer (r = 8)": {"r": 8.0},
    "Quantil-Planung 2 : 1": {"view": "Geplante Dauer", "ratio": "2 : 1", "tau": 0.8, "n2": 8},
    "Ein Aufwachbett": {"view": "Aufwachbett", "beds": "1"},
}
PRESETS = {name: {**_BASE, **o} for name, o in _OVERRIDES.items()}
PRESET_HELP = {
    "Standardtag (n = 7, Stufe M, r = 2, Maß A)": "Sieben Eingriffe, mittlere Streuung, Überziehen doppelt so teuer wie Leerlauf, Termine nach den Mittelwerten, Patienten kommen zum Termin.",
    "Maß B: Wartezeit ab Tagesbeginn": "Alle Patienten sind von Tagesbeginn an da, es gibt keine Termine: dann ist die Reihenfolge nach der kürzesten Dauer die beste, die Streuung zählt nicht.",
    "Kipppunkt Vorlauf": "Patienten kommen 160 Minuten vor ihrem Termin: die Rangfolge von Streuungsregel und kürzester Dauer ist in diesem erfundenen Modell etwa an der Kippstelle.",
    "Konstanter Variationskoeffizient": "Alle Eingriffe haben denselben Variationskoeffizienten: die Streuungsregel und die Regel nach der kürzesten Dauer geben dieselbe Reihenfolge.",
    "Streuung unabhängig vom Mittel": "Die absolute Streuung wird unabhängig vom Mittel gezogen: die Regel nach der kürzesten Dauer ist nur wenig besser als eine Zufallsreihenfolge (bei n = 8 liegt Zufall 8,9 % über SPT).",
    "Zwei Eingriffe, gleiches Mittel": "Zwei Eingriffe mit demselben Mittel (90 Minuten) und verschiedener Streuung, exakt gerechnet: wer zuerst?",
    "Überziehen sehr teuer (r = 8)": "Überziehen kostet das Achtfache des Leerlaufs: im exakt gerechneten Handbeispiel ist die Streuungsregel nicht mehr optimal.",
    "Quantil-Planung 2 : 1": "Wartezeit kostet doppelt so viel wie Leerlauf, n = 8: ein höheres Quantil als der Mittelwert als geplante Dauer senkt die Kosten.",
    "Ein Aufwachbett": "n = 6 Eingriffe, ein Aufwachbett: die Aufwachdauer blockiert den Saal.",
}

"""Texte der App: Einleitung, Hinweisblöcke „Bekannt aus der Literatur“, Methodenvergleich und Formulierung des Modells.

Alle Zahlen und Dauern der App sind erfunden (Anwendungsbeispiel, keine Beratung). Quellen nur so, wie sie gelesen sind: „im Wortlaut gelesen“ oder „nach Abstract bzw.
Sekundärangabe“; nichts davon ist hier als Neuheit dargestellt. Nur kurze eigene Paraphrasen."""

DISCLAIMER = (
    "**Anwendungsbeispiel, keine Beratung.** Ein erfundener Operationssaal an einem erfundenen Tag: alle Dauern, Kosten, Aufwachzeiten und Bettenzahlen sind Annahmen, "
    "keine Daten und keine Aussage über ein Krankenhaus; nichts hier ist eine Empfehlung für die Behandlung oder die Terminplanung echter Patienten."
)

INTRO = """
Ein Saal, ein Tag, **n Eingriffe** mit erfundenen, **lognormal** streuenden Dauern; die Termine werden nach den **Mittelwerten** vergeben. Kosten entstehen aus **Wartezeit**,
**Leerlauf** und **Überziehen**. Gefragt ist nur die **Reihenfolge**: Wie gut ist die Faustregel „kleinste Streuung zuerst“ gegen „kürzeste Dauer zuerst“ (SPT) und gegen das beste
gefundene Tagesprogramm, und woran hängt das? Die Antwort hängt an drei Annahmen, die hier **Regler** sind, keine Randbemerkung: am **Maß** (kommen die Patienten vor ihrem Termin, oder
zählt die Wartezeit ab Tagesbeginn?), an der **Kopplung** von Streuung und Mittel und am **Überziehpreis**. Zwei weitere Reiter zeigen, was die **geplante Dauer** (Mittelwert oder Quantil)
und ein knappes **Aufwachbett** verändern.

Weiter unten: die **Kernfragen** und die Expander **🔧 Wie wir das erreichen – vollständiger Methodenvergleich** und **📐 Mathematische Formulierung**.
"""

KERNFRAGEN = """
1. **Welche Reihenfolge ist billig, und wie nah am Besten?** Zählung über 30 erfundene Instanzen je Zelle, nicht eine einzelne Instanz.
2. **Woran hängt das?** Am Maß (Experiment 2), an der Kopplung von Streuung und Mittel (Experiment 3), am Überziehpreis r (Experiment 4, exakt gerechnet).
3. **Wie sicher ist die Zahl?** Standardfehler je Instanz, Zählung über Instanzen, ein zweiter Szenariensatz: das Stichproben-Optimum ist optimistisch (es wird auf denselben Szenarien gewählt, auf denen es bewertet wird).
"""

LIT_ORDER = (
    "**Bekannt aus der Literatur.** Die Faustregel „kleinste Streuung zuerst“ ist in der Literatur bekannt (nach Abstracts und Sekundärangaben: E. N. Weiss 1990, Denton u. a. 2007, "
    "Kong u. a. 2016; im Wortlaut gelesen: de Kemp u. a. 2021, Jafarnia-Jahromi/Jain 2020). Bekannt ist auch, dass sie nicht allgemein optimal ist; dass sie bei Terminen nach den "
    "Mittelwerten und wachsender Patientenzahl unter einer Momentbedingung gegen das Optimum geht, bei optimierten Terminen nicht; dass bei „Wartezeit ab Tagesbeginn“ die kürzeste erwartete "
    "Dauer zuerst optimal ist (Lenstra/Shmoys, „Elements of Scheduling“, arXiv 2001.06005, im Wortlaut gelesen; ob die Aussage dort Pinedos Kap. 15 entspricht, ist nicht belegt; Pinedos Lehrbuch nicht gelesen); dass ein hoher Überziehpreis die Regel verwundbarer macht und dass das Zeitungsverkäufer-Verhältnis nur für zwei "
    "Eingriffe gilt. **Nichts davon ist hier neu.** Diese Demo lässt es in einem erfundenen Modell ausprobieren: mit Zählung über Instanzen, mit einem stetigen Maß-Regler (Vorlauf) "
    "und mit einem Aufwachbett. Alle Zahlen sind eigene Messungen in diesem erfundenen Modell; die Schranken von de Kemp u. a. werden hier nicht angewendet (ob sie trotzdem gelten, ist nicht geprüft), weil sie eine "
    "Ordnungsannahme verlangen, die die Instanzen hier selten erfüllen (README, Tabelle „Befunde“, Zeile 11)."
)

LIT_TAU = (
    "**Bekannt aus der Literatur.** Das Zeitungsverkäufer-Verhältnis cw/(cw + 1) gilt für einen einzelnen Übergang (zwei Eingriffe, ohne Überziehpreis, Weiss nach Sekundärangabe); für Reihen braucht es eine "
    "Heuristik oder Simulation (Jafarnia-Jahromi/Jain, Bemerkung 16; de Kemp u. a., Abschn. 5: nur eine untere Schranke). Mittelwert-Termine sind nach Jafarnia-Jahromi/Jain (Sekundärangabe "
    "aus Denton/Gupta 2003) nur nahe am Optimum, solange die Wartezeitkosten etwa 10 bis 50 % der Leerlaufkosten betragen. Unsere Messung passt zur Richtung (bei 1 : 2 liegt das beste τ* "
    "für r = 0 und r = 1 bei höchstens 0,60 und wächst auch dort mit n; bei wenigen Eingriffen lohnt dort eine knappere Planung als der Mittelwert, bei n = 8 und r = 1 ist der Mittelwert selbst das Beste), die Zahl „10 bis 50 %“ ist hier nicht gemessen. Eigene Messung: wie das beste τ* mit n wächst. Die Herleitung des Quantils steht in den Nachbar-Demos "
    "[newsvendor-direkt-demo](https://github.com/sebastian-hanisch/newsvendor-direkt-demo) und [quantil-ueberdeckung-demo](https://github.com/sebastian-hanisch/quantil-ueberdeckung-demo)."
)

LIT_BEDS = (
    "**Bekannt aus der Literatur (qualitativ).** Dass der Aufwachraum den Saal blockieren kann und ein Bett je Saal der kritische Fall ist, ist belegt (Fairley u. a. 2019 und Marcon/Dexter 2006 "
    "nach dem Abstract; Bam u. a. 2015 im Preprint gelesen). Die Zahlen hier sind erfunden und stehen in keiner Quelle; die Aussage von Marcon/Dexter über „längste zuerst“ wird hier "
    "nicht geprüft (anderes Maß: Spitzenbelegung im Aufwachraum). Den Blockier-Mechanismus als Markow-Kette zeigt "
    "[jackson-network-demo](https://github.com/sebastian-hanisch/jackson-network-demo); hier ist es ein Tagesprogramm mit fester Reihenfolge."
)

METHODS_MD = """
**Regeln für die Reihenfolge** (Gleichstände nach dem Index des Eingriffs):

| Regel | Reihenfolge |
|---|---|
| kürzeste Dauer zuerst (SPT) | Mittel aufsteigend |
| kleinste Streuung zuerst | absolute Streuung $s_i = m_i c_i$ aufsteigend (die Hauptregel) |
| größte Streuung zuerst | Streuung absteigend |
| längste Dauer zuerst (LPT) | Mittel absteigend |
| kleinster Variationskoeffizient zuerst | $c_i$ aufsteigend |
| Zeitungsverkäufer-Index | minimale Kosten einer Zielzeit (bei Gewichten 1 : 1 die mittlere absolute Abweichung vom Median), aufsteigend; nur eine Zeile der Tabelle |
| Zufall | Mittel über 20 Zufallsreihenfolgen |
| abwechselnd kurz und lang | kürzester, längster, zweitkürzester, … (eine eigene Regel, nur im Reiter Aufwachbett) |

**Optimum.** Alle Kosten sind Mittel über Szenarien (gleiche Szenarien für alle Regeln, gepaarter Vergleich). Für **n bis 7** werden alle $n!$ Reihenfolgen aufgezählt; das beste Ergebnis ist das
**Stichproben-Optimum**: die beste Reihenfolge auf den Szenarien, auf denen auch bewertet wird. Es ist optimistisch (ein unabhängiger zweiter Satz zeigt das) und deshalb nur als Band zu lesen:
„nahe am Optimum“, nie „genau optimal“ oder „nicht optimal“. Für **n ab 8** gibt es nur die **beste lokale Verbesserung** (Vertauschen und Verschieben, Start bei der besten Regel, bis kein Zug
mehr hilft): **kein Optimum**. „Nicht optimal“ wird nur aus der **exakten Rechnung** (Experiment 4) belegt, die alle Ausgänge in ganzen Zahlen durchrechnet.

**Zufall und Zählung.** Alle Zufallszahlen kommen aus einem ganzzahligen Generator (SplitMix64) und sind auf jeder Plattform gleich. Die Tabellen „vorgerechnet“ stammen aus
`tools/opr_make_*.py` (je mit `--check`); die Zähleinheit ist die Instanz (30 je Zelle, 2 000 Szenarien je Satz). Live rechnet die App mit 500 oder 1 000 Szenarien.

**Verfahren im Vergleich:**

| Frage | Verfahren | Rechenweg |
|---|---|---|
| Regeln und Optimum, n bis 7 | Aufzählung | alle $n!$ Reihenfolgen auf denselben Szenarien |
| Regeln, n ab 8 | beste lokale Verbesserung | kein Optimum |
| Maß B | geschlossene Formel | $\\sum_k \\sum_{j<k} m$, keine Simulation |
| kleine Fälle | exakt | alle $2^n$ Ausgänge in ganzen Zahlen |
| Quantil | Raster $\\tau$ | Auswahl auf Satz A, Bewertung auf Satz B |
| Aufwachbett | Aufzählung | alle 720 Reihenfolgen, $n = 6$ |
"""

MATH_MD_INTRO = """
Die Lesehilfe oben erklärt die Symbole. **Zusätzlich** gilt: $n$ ist 4 bis 12 (Reiter Aufwachbett: 6); $s_i = m_i c_i$ ist die absolute Streuung des Eingriffs $i$; $\\mu_i, \\sigma_i$ sind die Parameter der
Lognormalverteilung (erste Gleichung); das Tagesende ist $T = 1{,}10 \\sum_i m_i$, $f_0 = 0$; das Gewicht der Wartezeit ist $c_w = 1$, nur im Reiter Geplante Dauer wählbar ($c_w$ = 0,5 / 1 / 2 / 4, also Wartezeit : Leerlauf = 1 : 2, 1 : 1, 2 : 1, 4 : 1).
"""

MATH_EQUATIONS = [
    ("Lognormale Dauer (Mittel $m_i$, Variationskoeffizient $c_i$):", r"\sigma_i^2=\ln(1+c_i^2),\qquad \mu_i=\ln m_i-\tfrac12\sigma_i^2,\qquad D_i=e^{\mu_i+\sigma_i Z},\ Z\sim\mathcal N(0,1)"),
    ("Termine nach den Mittelwerten, Start und Ende:", r"t_k=\sum_{j<k} m_{\pi(j)},\qquad x_k=\max(t_k,f_{k-1}),\qquad f_k=x_k+D_{\pi(k)}"),
    ("Wartezeit und Leerlauf:", r"W_k=(f_{k-1}-t_k)^+,\qquad I_k=(t_k-f_{k-1})^+"),
    ("Erwartete Kosten einer Reihenfolge (Maß A):", r"C(\pi)=\mathbb E\Big[\sum_{k=2}^{n}\big(c_w W_k+I_k\big)+r\,(f_n-T)^+\Big]"),
    ("Vorlauf $e$ (Patient kommt $e$ Minuten vor dem Termin, frühestens zum Tagesbeginn); die Identität ist exakt:", r"C^{(e)}(\pi)=C(\pi)+c_w\sum_{k=1}^{n}\min(t_k,e)"),
    ("Maß B (alle Patienten von Tagesbeginn an da, keine Termine, kein Leerlauf, kein Überziehen):", r"C^{B}(\pi)=\sum_{k=1}^{n}\sum_{j<k} m_{\pi(j)}"),
    ("Geplante Dauer als Quantil (Reiter Geplante Dauer; $\\tau$ ist für alle Eingriffe gleich; das letzte Verhältnis, das Zeitungsverkäufer-Verhältnis, gilt nur für zwei Eingriffe und ohne Überziehpreis, $r = 0$):", r"q_i(\tau)=e^{\mu_i+\sigma_i\Phi^{-1}(\tau)},\qquad t_k=\sum_{j<k} q_{\pi(j)}(\tau),\qquad \tau^\star_{n=2}=\tfrac{c_w}{c_w+1}"),
    ("Aufwachbett ($B$ Betten; der Saal ist bis zum Abgang des Patienten blockiert):", r"g_k=\max\big(f_k,\ \text{früheste Zeit, zu der ein Bett frei ist}\big),\qquad x_{k+1}=\max(t_{k+1},g_k),\qquad \text{Bett frei ab } g_k+R_k"),
    ("Stichproben-Optimum auf $S$ Szenarien:", r"\hat\pi=\arg\min_\pi\ \frac1S\sum_{s=1}^{S}C_s(\pi)"),
]

MATH_MD_OUTRO = """
**Was das Modell nicht enthält und wie die Demo damit umgeht:** abhängige Dauern (Chirurg, Tagesform), Notfälle, mehrere Säle, andere Dauerverteilungen, optimierte Termine. Der Überziehpreis $r$
und das Tagesende $T$ sind Annahmen; ein Optimierungsmodell mit den Terminen als Entscheidung ist hier **nicht** umgesetzt (die Termine sind die Summe der Mittel oder eines Quantils).
Die Gleichungen sind das vollständige Modell dieser Demo; alle Zahlenwerte sind erfunden.
"""

# OP-Programm: Reihenfolge bei unsicheren Dauern – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-op-programm-demo.streamlit.app/)** -->

Fall-Demo für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (`demos.html`). **Anwendungsbeispiel, keine Beratung:** ein erfundener Operationssaal an einem
erfundenen Tag. Alle Dauern, Kosten, Aufwachzeiten und Bettenzahlen sind Annahmen, keine Daten und keine Aussage über ein Krankenhaus; nichts hier ist eine Empfehlung für die
Behandlung oder die Terminplanung echter Patienten.

Gefragt ist nur die **Reihenfolge** der Eingriffe eines Tages bei lognormal streuenden Dauern, wenn die Termine nach den Mittelwerten vergeben werden. Die Demo vergleicht die
Faustregel „kleinste Streuung zuerst“ mit „kürzeste Dauer zuerst“ (SPT) und mit dem besten gefundenen Tagesprogramm und zeigt, woran die Rangfolge hängt: am **Maß**, an der
**Kopplung** von Streuung und Mittel und am **Überziehpreis**. Zwei weitere Ansichten zeigen die geplante Dauer (Mittelwert oder Quantil) und ein knappes Aufwachbett. Bedienung:
Schnellstart oben in der Seitenleiste, Umschalter der drei Ansichten (Reihenfolge, Geplante Dauer, Aufwachbett) über dem Inhalt, die Regler der Ansicht beim Experiment. Der Expander „💬 In einfachen Worten“ unter dem Hinweis fasst die Fragestellung ohne Formeln zusammen und verweist auf die Erklärseite „verständlich erklärt“ der Website (`op-programm-erklaert.html`).

## Warum dieses Problem

Wer Termine nach den Mittelwerten der Dauern vergibt, trägt die Streuung in die Wartezeit der Folgepatienten, in Leerlauf und in Überziehen. Eine einfache Faustregel ordnet nach der
Streuung. Ob sie gut ist, hängt davon ab, was man zählt: Eine Regel ohne ihr Maß und ohne ihre Kopplung ist keine Aussage. Die Demo zeigt das als Experimente mit Zählung über
30 erfundene Instanzen je Zelle, nicht als einzelnes Beispiel.

## Was in der Literatur schon steht (bekannt, nicht neu)

Quellenstufen: „im Wortlaut gelesen“ oder „nach Abstract bzw. Sekundärangabe“ (Original nicht gelesen). Nichts davon ist hier neu; die Demo lässt es in einem erfundenen Modell ausprobieren.

| Aussage | Quelle (Stufe) |
|---|---|
| „Kleinste Streuung zuerst“ ist eine bekannte Faustregel und schlägt in Zahlenbeispielen die Reihenfolge nach Mittel und nach Variationskoeffizient | E. N. Weiss 1990, Denton u. a. 2007, Kong u. a. 2016 (nach Abstract bzw. Sekundärangabe) |
| Sie ist nicht allgemein optimal; es gibt Gegenbeispiele | de Kemp u. a. 2021 (im Wortlaut gelesen), Jafarnia-Jahromi/Jain 2020 (im Wortlaut gelesen, optimierte Termine ohne Überziehen), Kong u. a. 2016 (nach Sekundärangabe) |
| Bei Terminen nach den Mittelwerten geht sie mit wachsender Patientenzahl unter einer Momentbedingung gegen das Optimum; bei optimierten Terminen nicht | de Kemp u. a. 2021 (im Wortlaut gelesen) |
| Maß B (Wartezeit ab Tagesbeginn, keine Termine): kürzeste erwartete Dauer zuerst ist optimal, die Streuung zählt nicht | Lenstra/Shmoys, „Elements of Scheduling“ (arXiv 2001.06005) im Wortlaut gelesen; ob die Aussage dort Kap. 15 von Pinedo entspricht, ist nicht belegt; Pinedos Springer-Lehrbuch nicht gelesen |
| Ein hoher Überziehpreis macht die Regel verwundbarer; die Rolle von Patientenzahl und Schiefe | Kong u. a. 2016 (nach Abstract), de Kemp u. a. 2021 (im Wortlaut gelesen) |
| Das Zeitungsverkäufer-Verhältnis gilt nur für zwei Eingriffe; für Reihen braucht es eine Heuristik oder Simulation | Weiss (nach Sekundärangabe), Jafarnia-Jahromi/Jain, de Kemp u. a. (im Wortlaut gelesen) |
| Das Stichproben-Optimum ist nach unten verzerrt | Jafarnia-Jahromi/Jain 2020, Abschn. 5 (im Wortlaut gelesen) |
| Lognormal als Dauerverteilung | de Kemp u. a. 2021, Sun/Li 2011 (im Wortlaut gelesen) |
| Aufwachraum-Blockade wirkt auf Reihenfolge und Wartezeit im Saal; ein Bett je Saal ist der kritische Fall | Fairley u. a. 2019, Marcon/Dexter 2006 (nach Abstract), Bam u. a. 2015 (Preprint, im Wortlaut gelesen) |

In den gelesenen Quellen **nicht gefunden** (das ist kein Neuheitsbeleg, denn Weiss, Denton und Kong sind im Original ungelesen): der Vorlauf e als stetiger Maß-Regler mit Kippstelle, das
beste Quantil τ\* als Funktion von n, die Wechselwirkung von Streuung und Aufwachblockade, der Verlust der einfachen Regel bei einem Bett und die Zählung in Instanzen. Diese Zahlen
sind **eigene Messung in einem erfundenen Modell**; neu ist höchstens in diesem Portfolio (soweit die lokalen Demo-Ordner reichen; der GitHub-Bestand ist nicht durchsucht) die interaktive Vorführung mit Zähleinheit. Nicht übernommen werden die Schranken von de Kemp u. a.:
sie verlangen eine Ordnungsannahme (hinreichend für Lognormal: Mittel und Streuung des Logarithmus wachsen nicht fallend in der Streuungsreihenfolge), die die Instanzen dieser Demo
selten erfüllen; ob die Schranken trotzdem gelten, ist nicht geprüft (Tabelle „Befunde“, Zeile 11).

## Modell

Alles erfunden; Minuten sind erfundene Zeiteinheiten. Der 📐-Expander der App enthält das vollständige Modell mit Symboltabelle.

* **Eingriffe** $i = 1 \dots n$ (n = 4 bis 12), Dauer lognormal mit Mittel $m_i$ und Variationskoeffizient $c_i$. Klassenmittel 45 / 90 / 150 min, je mit Faktor 0,85 bis 1,15 gezogen und gerundet; $c_i$ aus den
  Stufen L {0,1; 0,2; 0,3}, M {0,2; 0,35; 0,5}, H {0,3; 0,5; 0,8}, K {0,4 für alle}, **unabhängig vom Mittel** gezogen; Stufe U: absolute Streuung aus {8, 20, 40} min, unabhängig vom Mittel.
* **Termine** $t_k$ = Summe der Mittel der Vorgänger (Reiter Geplante Dauer: Summe eines Quantils $\tau$). Start = max(Termin, Ende des Vorgängers).
* **Kosten (Maß A)** = $c_w$ mal Wartezeit + Leerlauf + $r$ mal Überziehen; das Gewicht der Wartezeit ist $c_w = 1$ in allen Tabellen und Experimenten, nur im Reiter Geplante Dauer wählbar (Wartezeit : Leerlauf = 1 : 2, 1 : 1, 2 : 1, 4 : 1, also $c_w$ = 0,5 / 1 / 2 / 4); Tagesende $T = 1{,}10 \sum m_i$; $r \in \{0{,}5; 1; 2; 4; 8\}$ (Überziehen : Leerlauf).
* **Vorlauf e:** der Patient kommt $e$ Minuten vor seinem Termin; die Wartezeit ab Ankunft ist exakt (Start − Termin) + min(Termin, e).
* **Maß B:** alle Patienten von Tagesbeginn an da, keine Termine; erwartete Summe der Wartezeiten $= \sum_k \sum_{j<k} m_{\pi(j)}$, geschlossen.
* **Aufwachbett:** $B$ Betten; der Patient verlässt den Saal, sobald ein Bett frei ist; Aufwachdauer lognormal (Mittel 60, Variationskoeffizient 0,5) oder fest 60; $n = 6$.
* **Zählung:** die Zähleinheit ist die **Instanz** (30 je Zelle). Das „Optimum“ ist das **Stichproben-Optimum** auf den Szenarien, auf denen auch bewertet wird (n bis 7 Aufzählung aller $n!$
  Reihenfolgen); ab n = 8 gibt es nur die **beste lokale Verbesserung** (kein Optimum). „Lücke“ = Kosten der Regel / Kosten des Stichproben-Optimums − 1.

## Methodik

Regeln: SPT, Streuung aufsteigend (Hauptregel), Streuung absteigend, LPT, Variationskoeffizient aufsteigend, Zeitungsverkäufer-Index aufsteigend (siebte Regel, nur als Zeile der Tabelle),
Zufall (Mittel über 20 Zufallsreihenfolgen), im Reiter Aufwachbett zusätzlich „abwechselnd kurz und lang“ (eigene Regel, nicht die von Marcon/Dexter). Alle Regeln sehen **dieselben Szenarien**
(gepaarter Vergleich); ein unabhängiger zweiter Satz („Satz B“) bewertet nur. Zufall nur über den ganzzahligen Generator SplitMix64 (`opr_rng.py`, plattformfeste Normalzahlen). Der **Instanzstrom** ist die Basiszahl, aus der die Instanzen gezogen werden (7000; ein zweiter Strom 7100, ein dritter 7200); **Satz A** ist der Szenariensatz, auf dem Regeln bewertet und das Stichproben-Optimum gewählt wird, **Satz B** der unabhängige zweite Satz, der nur bewertet.
Live rechnet die App mit 500 oder 1 000 Szenarien, die Tabellen „vorgerechnet“ stammen aus `tools/opr_make_*.py` mit 2 000 Szenarien je Satz (je mit `--check`, das einzelne Zellen neu rechnet und
im Band vergleicht). Je Instanznummer teilen sich die fünf Streuungsstufen derselben Zelle dieselben Standardnormalzahlen (es sind verschiedene Instanzen), das spart Rechenzeit.

Experimente der App: (1) Welche Reihenfolge ist billig? (2) Das Maß entscheidet (Vorlauf e, Maß B). (3) Kopplung von Streuung und Mittel (konstanter Variationskoeffizient, Stufe M, unabhängige Streuung).
(4) Exakt statt Stichprobe: Handbeispiel, zwei Eingriffe mit gleichem Mittel, Zufallsinstanzen mit n = 3 bis 5. Reiter Geplante Dauer: Quantil statt Mittelwert. Reiter Aufwachbett: ein Bett im Aufwachraum.

## Befunde (gemessen, keine Behauptungen)

Voraussetzung, wenn nichts anderes steht: das Modell oben, Maß A, Termine nach den Mittelwerten, Mittel über 30 Instanzen je Zelle, 2 000 Szenarien, Zähleinheit Instanzen, Δ = mittlere relative
Differenz (Streuung aufsteigend − SPT) / SPT. Die Spalte „Test“ nennt den Test, der die Zahl als **Band** (Zählungen ± 3, Anteile ± 0,01, Gleitkommazahlen relativ) oder **exakt** (Ganzzahlen, Brüche) prüft
(`tests/test_claims.py`, Test `test_claim_<Nr>`).

| Nr. | Befund | Test |
|---|---|---|
| 1 | In den 30 Instanzen je Zelle (Instanzstrom 7000) liegt Δ in allen 60 Zellen (n = 5, 7, 8, 12; Stufen L, M, H; r = 0,5 bis 8; die Zellen teilen Instanzen und Szenarien, sind also keine 60 unabhängigen Beobachtungen) zwischen **−12,9 %** (n = 8, Stufe L, r = 0,5) und **−3,1 %** (n = 5, Stufe L, r = 8). Stufe M, r = 2: −7,3 % [−9,7; −5,1] (n = 5), −7,3 % [−8,9; −5,7] (n = 7), −7,1 % [−8,8; −5,5] (n = 8), −8,2 % [−9,4; −7,1] (n = 12; Bootstrap-Intervalle). Die Streuungsregel ist in **20 bis 30 von 30** Instanzen je Zelle besser als SPT und in **höchstens 3** schlechter; auf dem zweiten Szenariensatz in 19 bis 30. Ein anderer Instanzstrom verschiebt eine Zelle um mehrere Prozentpunkte (Grenzen, Szenarienzahl); alle Bootstrap-Intervalle gelten nur für den Strom 7000. Der Vorsprung (\|Δ\|) ist bei r = 8 in allen 12 Kombinationen aus n und Stufe L, M, H am kleinsten (gezählt in den 60 Zellen). Diese Zählung gilt nur für die Stufen L, M, H; Stufe K und Stufe U stehen in Befund 5. | `test_claim_01` |
| 2 | **Lücke zum Stichproben-Optimum** (n = 5, 7, Aufzählung): Streuungsregel 0,05 bis 0,27 %, SPT 3,5 bis 14,1 %. Ab n = 8 (beste lokale Verbesserung, kein Optimum): Streuungsregel 0,12 bis 0,38 %, SPT 5,0 bis 15,7 %. Die Streuungsregel **ist** die Reihenfolge des Stichproben-Optimums (n = 5, 7) bzw. der besten lokalen Verbesserung (n = 8) in 10 bis 25 von 30 Instanzen (Stufen L, M, H), bei n = 12 (beste lokale Verbesserung) in 0 bis 6. | `test_claim_02` |
| 3 | Rangfolge der Regeln in **allen 60 Zellen** gleich: Optimum bzw. lokale Verbesserung ≤ Streuung aufsteigend < SPT < Variationskoeffizient aufsteigend < Zufall < LPT < Streuung absteigend (Mittel der Kosten relativ zu SPT). Zufall liegt 9 bis 45 %, LPT 17 bis 84 %, Streuung absteigend 20 bis 90 % über SPT. | `test_claim_03` |
| 4 | **Maß B (Gegenfall):** SPT ist in **30 von 30** Instanzen in jeder der 20 Zellen (n = 5, 7, 8, 12; Stufen L, M, H, K, U) höchstens so teuer wie die Streuungsregel (das folgt aus der geschlossenen Formel für Maß B, einer Umordnungsungleichung, und ist keine Entdeckung der Messung: die Zählung prüft den Code gegen die Formel); deren Wartezeit ist in den 12 Zellen der Stufen L, M, H im Mittel das **1,057- bis 1,085fache** (Instanzstrom 7000; ein zweiter Strom, nur n = 5 und 7 nachgerechnet, gibt 1,07 bis 1,10). | `test_claim_04` |
| 5 | **Kopplung** (n = 8, r = 2): konstanter Variationskoeffizient: Streuungsregel = SPT in 30 von 30 Instanzen (alle 20 Zellen der Stufe K: Δ = 0; das folgt aus der Konstruktion, denn mit $s_i = c\,m_i$ ordnen beide Regeln gleich: die Zählung prüft den Code, sie ist keine Messung am Modell). Streuung unabhängig vom Mittel (Stufe U, mittlere Korrelation Mittel/Streuung +0,15, Spanne −0,39 bis +0,88): Zufall nur 8,9 % über SPT (Stufe M: 30,5 %), Streuungsregel **−24,1 %** gegen SPT (29 von 30; Stufe M: −7,1 %), Lücke der Streuungsregel 1,3 % (Stufe M: 0,13 %). Bei n = 5 und n = 12 liegt Zufall in Stufe U im Mittel über 30 Instanzen gleich auf mit SPT (0,997 bis 1,004), bei n = 7 höchstens 1,1 % darüber: n = 8 ist ein Ausreißer, SPT ist dort ein schwacher Maßstab, und die 8,9 % hängen stark von der Instanzstichprobe ab; derselbe schwache Maßstab steckt auch in den −24,1 % (ebenfalls n = 8) (Abschnitt „Befunde und Korrekturen“: früher 4,9 %). | `test_claim_05` |
| 6 | **Vorlauf e** (n = 7, Stufe M, r = 1; Δ, besser in k von 30): e = 0: −8,1 % (29); 80 min: −1,8 % (25); 160 min: **+0,3 %** (16 besser, 1 gleich, 13 schlechter; nahe bei 0); e → ∞ (Wartezeit ab Tagesbeginn zusätzlich zu Leerlauf und Überziehen): +4,3 % (4). Das Vorzeichen von Δ wechselt in den Stufen L und M (r = 1 und r = 4) **zwischen e = 80 und e = 160 min**, in Stufe H erst zwischen 160 min und ∞ (bei 160 min Δ = −0,5 % bei r = 1, 19 von 30, und −0,4 % bei r = 4). Δ liegt dort nahe 0: die Lage der Kippstelle innerhalb von 80 bis 160 min ist nicht belegt; ein zweiter Instanzstrom (Basis 7100, n = 7) zeigt dieselbe Aufteilung der Stufen. | `test_claim_06` |
| 7 | **Exakt, Zweipunkt-Dauern** (zwei gleich wahrscheinliche Dauern je Eingriff, alle Ausgänge und Reihenfolgen in ganzen Zahlen, zufällige Instanzen): für r ≤ 4 ist die Streuungsregel in **jeder** der gezogenen 2 000 (n = 3), 2 000 (n = 4), 1 000 (n = 5) und 300 (n = 6) Instanzen exakt optimal; bei r = 8 nur in 1 004, 1 076, 405 und 142 (SPT: 654, 344, 71, 2). Das ist ein Befund dieses Modells (symmetrische Zweipunkt-Verteilung); bei den schiefen Lognormal-Dauern (Befund 2) ist die Regel nicht in jeder Instanz optimal. | `test_claim_07` |
| 8 | **Handbeispiel** (drei Eingriffe, exakte Brüche): B–A–C 307/4 = 76,75, SPT A–B–C 337/4 = 84,25 bei r = 2 (Streuungsregel 30/337 = 8,9 % billiger als SPT; einziges Optimum bei r = 1 und 2, bei r = 4 gleichauf mit B–C–A, beide 108,5); bei r = 8 Optimum B–C–A mit 162, Streuungsregel 172 (5/81 = 6,2 % darüber), SPT 179,5. Maß B: 210 (SPT) gegen 240 (Streuung); zwei Eingriffe mit gleichem Mittel 90 min: 23,625 / 51,75 bei r = 0,5, bei r = 8 beide 153. | `test_claim_08`, `tests/test_oracle_exact.py` |
| 9 | **Probe auf dem zweiten Szenariensatz:** das Stichproben-Optimum kostet dort im Mittel über 30 Instanzen das 0,997- bis 1,017fache seines Wertes auf dem Satz, auf dem es gewählt wurde (alle 100 Zellen; je Instanz weicht das Verhältnis gemessen um −13 bis +17 % von 1 ab, Werte unter 1 sind Streuung, keine Verzerrung). Lücken von 0,05 bis 0,38 % liegen innerhalb der Verzerrung des Zellenmittels: „nahe am Stichproben-Optimum“ ist belegt, „genau optimal“ nicht. | `test_claim_09` |
| 10 | **Quantil statt Mittelwert** (Stufe M, Reihenfolge Streuung aufsteigend, n = 2, 3, 4, 6, 8, Raster 0,05; τ\* auf Satz A gewählt, Gewinn auf Satz B): Wartezeit : Leerlauf = 1 : 1, r = 0: τ\* = 0,50 / Mittelwert / 0,60 / 0,65 / 0,70 (Zeitungsverkäufer-Verhältnis 0,50), Kosten bei τ\* 0,986 / 1,000 / 0,980 / 0,931 / **0,883** der Mittelwert-Planung; 2 : 1, r = 0: τ\* = 0,65 / 0,70 / 0,75 / 0,75 / 0,80 (0,67), 0,963 / 0,902 / 0,829 / 0,748 / **0,686**. Die Planung mit dem Median (τ = 0,5) kostet bei 2 : 1 für n = 3 bis 8 das 1,06- bis 1,20fache (über r = 0 und r = 1 gemessen; r = 0 allein liegt bei 1,09 bis 1,20, r = 1 bei 1,06 bis 1,18). Bei 1 : 1 und r = 1 ist der Gewinn kleiner (0,937 bei n = 8; 0,974 bis 0,998 bei n = 2 bis 6). Bei 1 : 2 (Leerlauf teurer als Warten) liegt τ\* für r = 0 und r = 1 bei höchstens 0,60 und damit bei kleinem n unter dem Quantil, bei dem der Mittelwert der Dauer liegt (im Mittel 0,56 bis 0,57); bei n = 2 und r = 0 kostet die Planung mit τ\* = 0,35 das 0,842fache der Mittelwert-Planung, also lohnt dort eine knappere Planung als der Mittelwert. Bei n = 8, 1 : 2 und r = 1 bleibt der Mittelwert die beste geplante Dauer. | `test_claim_10` |
| 11 | **Ordnungsbedingung von de Kemp u. a. (hinreichend, Lognormal):** von den ersten 2 000 Instanzen je Zelle erfüllen sie in den Stufen L, M, H bei n = 5 5,3 bis 6,3 %, n = 7 0,6 bis 1,0 %, n = 8 0,4 bis 0,5 %, n = 12 keine (0 von 2 000); in Stufe K (konstanter Variationskoeffizient) 100 %. Die Schranken dieser Arbeit gelten für die Instanzen hier in der Regel nicht und erklären die Befunde 1 bis 3 nicht. | `test_claim_11` |
| 12 | **Zeitungsverkäufer-Index als siebte Regel:** gegen die Streuungsregel liegt die Lücke zum Optimum in allen 60 Zellen höchstens 0,08 Prozentpunkte daneben (−0,08 bis +0,07); er bekommt eine Zeile der Tabelle, keinen eigenen Balken. | `test_claim_12` |
| 13 | **Aufwachbett** (Reiter Aufwachbett: n = 6, alle 720 Reihenfolgen, Aufwachdauer lognormal mit Mittel 60 und Variationskoeffizient 0,5, r = 2, Stufen L / M / H, 30 Instanzen): die Kosten des besten Tagesprogramms sind bei **einem Bett das 1,88- / 1,50- / 1,28fache** von „ohne Blockade“, bei zwei Betten 1,012 / 1,005 / 1,002, bei drei Betten 1,000. Blockadezeit der SPT-Reihenfolge: 6,2 bis 8,4 % der Summe der Eingriffsdauern bei einem Bett, 0,12 bis 0,25 % bei zwei, höchstens 0,004 % bei drei Betten. | `test_claim_13` |
| 14 | **Wechselwirkung** (r = 2): Streuung der Eingriffe allein 87,1 / 174,5 / 272,5 (L / M / H), Blockade allein (Dauern fest, ein Bett, Aufwachdauer streuend) 43,1, beides 150,4 / 243,5 / 339,6. Die Blockade fügt bei streuenden Eingriffen 63,3 / 68,9 / 67,0 Kosteneinheiten hinzu, das **1,47- / 1,60- / 1,56fache** ihrer Kosten bei festen Dauern. „Blockade stärker als Streuung allein“ bestätigt sich nicht (Blockade allein 25 % der Streuung allein in Stufe M). Sind auch die Aufwachdauern fest, kostet die Blockade 8,3 statt 43,1: die Streuung der Aufwachdauer verfünffacht sie. | `test_claim_14` |
| 15 | **Bei einem Bett verliert die einfache Regel deutlich (Stufe L), in Stufe M und H weniger** (r = 2): Lücke der Streuungsregel zum Optimum **17,0 / 5,6 / 1,9 %** (L / M / H) statt 0,09 / 0,15 / 0,08 % ohne Blockade und 0,34 / 0,21 / 0,09 % bei zwei Betten; SPT 20,5 / 8,7 / 5,8 % bei einem Bett. Die Streuungsregel ist bei einem Bett in 19 / 22 / 23 von 30 Instanzen besser als SPT (ohne Blockade 30 / 27 / 26). Die beste Reihenfolge ist bei einem Bett in 26 / 25 / 22 von 30 Instanzen (73 von 90) eine andere als ohne Blockade (gemessen am Stichproben-Optimum, also nur so scharf wie dieses; die Kontrolle mit drei Betten findet keine abweichende beste Reihenfolge). | `test_claim_15` |
| 16 | **Weitere Regeln im Aufwachbett** (r = 2): „längste zuerst“ liegt bei einem Bett 58,8 / 50,8 / 57,9 % über dem Optimum (zwei Betten 69,9 / 56,6 / 58,4 %, ohne Blockade 69,6 / 55,6 / 57,1 %) und ist in 0 von 30 Instanzen billiger als die Streuungsregel, außer in 1 Instanz der Stufe L bei einem Bett; der Nachteil tritt in diesem Modell auch ohne Blockade auf (Termine nach den Mittelwerten, Stufen L, M, H, n = 6). Die eigene Regel „abwechselnd kurz und lang“ liegt bei einem Bett 21,3 / 19,3 / 24,1 % über dem Optimum und ist in 11 / 4 / 0 von 30 Instanzen billiger als die Streuungsregel (ohne Blockade in keiner). Das prüft die Aussage von Marcon/Dexter nicht (andere Zielgröße). | `test_claim_16` |

## Befunde und Korrekturen gegenüber der ersten Rechnung

Die Zahlen einer früheren Messreihe zum selben Modell (anderer Zufallsgenerator, numpy-Zufall, je Instanz eigene Szenarien) waren Erwartungen; der Bau misst mit dem ganzzahligen Generator neu.
Die Aussagen **bestätigen** sich; die Zahlen weichen ab. Die „früheren“ Werte stammen aus einer nicht mehr vorhandenen Messreihe und sind hier nicht nachprüfbar; der Sprung in Stufe U, n = 8, zeigt selbst, wie stark Zahlen an der Instanzstichprobe hängen:

* Die Streuungsregel ist in einzelnen Zellen in bis zu 3 statt höchstens 1 von 30 Instanzen schlechter als SPT (Befund 1); Δ liegt zwischen −12,9 und −3,1 % (früher −14,6 bis −2,8 %), die Lücke der
  Streuungsregel zwischen 0,05 und 0,38 % (früher 0,03 bis 0,31 %), die der SPT zwischen 3,5 und 15,7 % (früher 3,1 bis 17,8 %).
* Stufe U, n = 8: Zufall liegt 8,9 % über SPT (früher 4,9 %); die Streuungsregel −24,1 % (früher −25,9 %).
* Maß B: das 1,057- bis 1,085fache (früher 1,051 bis 1,105fach), in einer Instanz bis zum 1,41fachen.
* Quantil: τ\* und Gewinne stimmen innerhalb von 0,05 auf dem Raster überein (n = 8, 1 : 1, r = 0: 0,70 und 0,883, früher ebenfalls).
* Handbeispiel bei r = 4: die Streuungsregel B–A–C ist nicht das einzige Optimum, B–C–A kostet ebenfalls genau 108,5 (die erste Rechnung nannte es „einziges Optimum“); die Aussage „Streuungsregel
  exakt optimal“ bleibt richtig.
* **Eigenschaft, die der Bau widerlegt:** „zusätzliche Betten verschlechtern die Kosten nie“ gilt **nicht** für Kosten mit Leerlauf (es gilt für Blockadezeit und Tagesende): wer wegen der Blockade später
  fertig wird, hat weniger Leerlauf vor dem nächsten Termin (`tests/test_eigenschaften_op_programm.py`).
* Das Stichproben-Optimum auf dem zweiten Satz weicht je Instanz von seinem Wert auf dem Wahlsatz um −13 bis +17 % ab (gemessen; im Zellenmittel 0,997 bis 1,017); deshalb stehen Lücken nur als Band und nie als „genau optimal“ oder „nicht optimal“ (Letzteres nur aus der exakten Rechnung).
* Die Aussage „Weiss nennt ein Gegenbeispiel für drei Patienten“ steht nicht in der Demo (in keiner gelesenen Quelle belegt); das Gegenbeispiel bei r = 8 ist eigene Rechnung.

## Ehrliche Grenzen

* **Erfundenes Modell:** lognormale Dauern, unabhängig von Eingriff zu Eingriff, keine abhängigen Dauern (Chirurg, Tagesform), keine Notfälle, ein Saal, ein Tag, n höchstens 12; Parameter sind Annahmen.
* **Nur Termine nach den Mittelwerten oder nach einem Quantil.** Optimierte Terminzeiten (Denton u. a., Jafarnia-Jahromi/Jain) sind **nicht gemessen**; nach de Kemp u. a. ist die Regel dort nicht
  asymptotisch optimal. Tagesende nur 1,10 · Σ m in der App; 1,00 und 1,25 nur in einer Nachrechnung der Aussagenprüfung mit dem heutigen Code (n = 6, nicht eingefroren, nicht in der App): Streuungsregel besser in 24 bis 27 von 30 (die früher genannten 26 bis 29 stammen aus der früheren Messreihe).
* **Stichproben-Optimum:** optimistisch; „nahe am Optimum“ nur als Band (Befund 9). Ab n = 8 kein Optimum, nur die beste lokale Verbesserung.
* **Vorlauf e** nur für n = 7 (Aufzählung) und als Modellierungsmittel (kein Satz über Krankenhäuser); das Gewicht der Wartezeit allein kippt die Rangfolge nicht (Streuungsregel besser in 29 von 30 für jedes
  Gewicht von 0,25 bis 8, Stufe M, n = 7, r = 1, 2, 4; Nachrechnung der Aussagenprüfung mit dem heutigen Code, nicht eingefroren, nicht in der App; die frühere Messreihe nannte 28 von 30).
* **Exakte Rechnung** nur für symmetrische Zweipunkt-Verteilungen und Maß A.
* **Quantil** nur Stufe M, eine Reihenfolge, τ für alle Eingriffe gleich; die Ursache des Wachsens von τ\* mit n (Übertrag der Verspätung auf die Folgeeingriffe) ist eine Deutung, kein Beweis. Ein gröberes Raster kann den
  Gewinn unterschätzen.
* **Aufwachbett:** nur n = 6, 1 / 2 / 3 Betten, ein Aufwachprofil (Mittel 60, Variationskoeffizient 0 oder 0,5, unabhängig vom Eingriff), kein Ausfall von Betten.
* **Szenarienzahl:** nur 2 000 (Tabellen) und 500 oder 1 000 (live); je Instanznummer gemeinsame Normalzahlen der fünf Stufen. Stabilität der Zählung über Instanzströme: `data/opr_stabilitaet.txt`
  (`tools/opr_stability.py`, Messung dieses Werkzeugs, nicht im Test): in der Zelle n = 5, Stufe M, r = 2 geben drei Instanzströme (Basis 7000, 7100, 7200, je 30 Instanzen, neue Szenarien) Δ = −7,3 / −6,4 /
  −4,6 %, besser in 25 / 24 / 22 von 30 Instanzen und eine Lücke der Streuungsregel von 0,02 / 0,05 / 0,03 %. Eine einzelne Zelle schwankt also um mehrere Prozentpunkte in Δ und um einige Instanzen
  in der Zählung; die Spannen der Befunde 1 bis 3 sind deshalb Bänder, keine genauen Werte.
* **Literatur:** Weiss 1990, Denton u. a. 2007, Kong u. a. 2016 und das Springer-Lehrbuch von Pinedo sind im Original **nicht gelesen**; ihre Aussagen stehen nur nach Abstract oder Sekundärangabe. Gelesen ist nur die arXiv-Fassung von Lenstra/Shmoys („Elements of Scheduling“, arXiv 2001.06005); dass die Maß-B-Aussage dort Pinedos Kap. 15 entspricht, ist nicht belegt. Ob das Gegenbeispiel bei
  r = 8 (n = 3) mit den hinreichenden Bedingungen von Kong u. a. für drei Patienten vereinbar ist, ist nicht geprüft. Literaturwert-Tests gibt es nicht (die Beispiele der gelesenen Arbeiten verlangen
  Verteilungen, Termine und Kostenformen, die dieses Modell nicht hat): offen.

## Bewusst nicht umgesetzt

* Optimierte Termine als Entscheidung, Mehrstufen- und Zwei-Stufen-Modelle, Stichprobenmittelwert-Näherung (SAA) mit Konvergenzanalyse.
* Abhängige Dauern, Notfälle, mehrere Säle, andere Dauerverteilungen (Gamma, Mischungen), n über 12.
* Ein PDF-Export.
* Nachbau der Quantil-Herleitung ([newsvendor-direkt-demo](https://github.com/sebastian-hanisch/newsvendor-direkt-demo), [quantil-ueberdeckung-demo](https://github.com/sebastian-hanisch/quantil-ueberdeckung-demo))
  und des Beweises von SPT ([spt-scheduling-demo](https://github.com/sebastian-hanisch/spt-scheduling-demo), dessen Annahme „feste Dauern“ hier fällt); die Blockier-Kette als Markow-Kette
  ([jackson-network-demo](https://github.com/sebastian-hanisch/jackson-network-demo)); Streuung kostet Wartezeit: [mg1-kingman-demo](https://github.com/sebastian-hanisch/mg1-kingman-demo).

## Verifikation

- **Orakel:** `tests/test_oracle_kern.py` (reine Python-Schleife je Szenario ohne und mit 1, 2, 3 Betten, Bilanz Ende = Dauern + Leerlauf + Blockade, Tandem-Blockade gegen alle 64 Ausgänge, Handbeispiel-Brüche, Vorlauf-Identität,
  Maß B gegen Simulation, Lognormal gegen Eingaben), `tests/test_oracle_exact.py` (drei Rechenwege: Bruchschleife, Verteilung der Endzeit, ganzzahlige Vektorrechnung; alle sechs Reihenfolgen des Handbeispiels),
  `tests/test_oracle_rules.py` (Aufzählung n = 5 und 7 gegen Schleife, lokale Verbesserung, Quantil gegen `scipy.stats.lognorm`, Zeitungsverkäufer-Verhältnis bei zwei Eingriffen, Index gegen `scipy.integrate.quad`).
- **Eigenschaften:** `tests/test_eigenschaften_op_programm.py` (feste Dauern kosten nichts, gleicher Variationskoeffizient gibt dieselbe Reihenfolge, Skalierung, Umnummerieren, Betten, Überziehpreis, Szenarienreihenfolge, Maß B ohne Streuung).
- **Gefrorene Tabellen:** `tests/test_frozen.py` (Form, Optimum ≤ jede Regel, Zweig-Tests ohne tote Spalte, Vorlauf bei e = 0 gleich Haupttabelle, Neu-Rechnen einer Instanz im Band); `tools/opr_make_*.py --check`.
- **Claims:** `tests/test_claims.py` prüft die Zahlen der Tabelle „Befunde“ (Test `test_claim_<Nr>`) im Band, nicht scharf, Rundungen fängt das nicht; Verdikte: `tests/test_verdicts.py` (Schwellen, jede Zahl im Text).
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an den Grenzen, jede Option jeder Ansicht, jeder Schnellstart, Worst-Case n = 12 mit 1 000 Szenarien, Permalink), Footer-Test, Layout-Test (jeder Regler hebt die Markierung auf).
- **Randfälle:** n = 1, n = 2, Gleichstände, Variationskoeffizient 0, r = 0, ein Szenario (`tests/test_edge_cases.py`).
- **Testzahl:** 142 Tests (`pytest --collect-only -q tests`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: drei Ansichten mit Umschalter, Experimente, Kernfragen, Expander |
| `opr_constants.py` | Modellannahmen, Regler-Grenzen, Schnellstart-Szenarien |
| `opr_kern.py` | Instanzen, Dauern, Tageskosten (Termine, Wartezeit, Leerlauf, Überziehen, Aufwachbett), Vorlauf, Maß B |
| `opr_rules.py` | Regeln, Zeitungsverkäufer-Index, Aufzählung, beste lokale Verbesserung |
| `opr_exact.py` | exakte Rechnung für Zweipunkt-Dauern in ganzen Zahlen und Brüchen |
| `opr_evaluation.py` | Auswertung einer Instanz, Zusammenfassung über Instanzen, Verdikte (reine Funktionen) |
| `opr_analysis.py` | Live-Rechnungen der Ansichten, Zusammenfassungen der gefrorenen Tabellen, Ordnungsbedingung |
| `opr_frozen.py` | Laden der vorgerechneten Tabellen |
| `opr_rng.py` | SplitMix64 mit plattformfesten Normalzahlen (Kopie der geprüften Vorlage) |
| `opr_presets.py` | Permalink, Schnellstart, Merkspeicher der Regler |
| `opr_layout.py` | fester Kopf |
| `opr_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `opr_texts.py` | Texte: Einleitung, Hinweisblöcke zur Literatur, Methodik, Formulierung des Modells |
| `data/*.json` | vorgerechnete Tabellen (`opr_o1_tabelle`, `opr_o1_vorlauf`, `opr_o1_exakt`, `opr_o2_tau`, `opr_o3_betten`), `opr_stabilitaet.txt` |
| `tools/opr_make_o1.py`, `opr_make_exakt.py`, `opr_make_o2.py`, `opr_make_o3.py`, `opr_stability.py` | Werkzeuge zum Vorrechnen (je mit `--check`) |
| `tests/` | Orakel-, Eigenschafts-, Claims-, Verdikt-, Layout-, Preset-, Frozen-, Rand-, Smoke- und Footer-Tests |
| `requirements.txt` / `requirements-dev.txt` | Laufzeit (streamlit, numpy, pandas, plotly) / plus pytest, scipy für die Gegenproben |
| `.github/workflows/tests.yml`, `.gitignore`, `.streamlit/config.toml` | CI, Ignorierliste, Streamlit-Einstellungen |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).

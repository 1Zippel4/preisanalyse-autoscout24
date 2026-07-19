# Preisanalyse von Gebrauchtwagen

Untersucht werden die Preisstruktur und die Preistreiber von Gebrauchtwagen anhand eines realen Datensatzes von AutoScout24. Das Projekt durchläuft einen vollständigen Data-Science-Zyklus von der Datenbereinigung über die explorative Analyse bis zu zwei Vorhersagemodellen und ist so aufgebaut, dass sich jedes Ergebnis von der Rohdatei bis zur Kennzahl nachvollziehen lässt.

## Forschungsfragen

1. Welche Fahrzeugmerkmale hängen mit dem Angebotspreis zusammen und wie ist der Preis verteilt?
2. Wie gut lässt sich der Angebotspreis anhand der Fahrzeugmerkmale vorhersagen?

## Daten

Grundlage ist der Germany Used Cars Dataset 2023 von AutoScout24, veröffentlicht auf Kaggle. Die Rohdaten umfassen 251.079 Fahrzeuganzeigen mit 15 Merkmalen wie Marke, Modell, Baujahr, Laufleistung, Leistung sowie Kraftstoff- und Getriebeart. Nach der Bereinigung in Notebook 01 verbleiben 243.060 Anzeigen mit 10 Merkmalen.

Die Bereinigung umfasst das Entfernen unplausibler und fehlender Werte, die Vereinheitlichung der Datentypen sowie die Zerlegung des Zulassungsdatums in Monat und Jahr. Die extrem hochpreisigen Fahrzeuge wurden als echte Ausreißer erkannt und dokumentiert, aber bewusst nicht entfernt.

Die bereinigten Daten liegen nicht im Repository, da sie aus den Rohdaten reproduzierbar erzeugt werden. Die Rohdatei `data.csv` gehört in den Ordner `data/raw/`.

## Vorgehen

Das Projekt folgt dem CRISP-DM-Prozess und ist auf drei aufeinander aufbauende Notebooks verteilt.

- **01_eda.ipynb** bereitet die Daten auf und untersucht Verteilungen, Zusammenhänge und Ausreißer mit Kennzahlen und Grafiken.
- **02_regressionsbaum.ipynb** sagt den Preis mit einem Regressionsbaum vorher und stellt die Merkmalswichtigkeiten dar.
- **03_multiple_regression.ipynb** nutzt eine multiple lineare Regression mit logarithmierter Zielgröße und interpretiert die Koeffizienten.

In beiden Modell-Notebooks werden die Daten vor jeder datenabhängigen Verarbeitung in Trainings- und Testdaten getrennt, um Data Leakage zu vermeiden. Jedes Modell wird gegen ein einfaches Baseline-Modell geprüft und über eine Kreuzvalidierung auf Robustheit untersucht.

## Struktur

```
Autoscout24_Preisanalyse/
├── data/
│   ├── raw/            Rohdaten (data.csv)
│   └── processed/      bereinigte Daten (aus Notebook 01 erzeugt)
├── notebooks/
│   ├── 01_eda.ipynb                  explorative Datenanalyse
│   ├── 02_regressionsbaum.ipynb      Regressionsbaum
│   └── 03_multiple_regression.ipynb  multiple lineare Regression
├── images/            erzeugte Grafiken
├── requirements.txt
└── README.md
```

## Installation

```
pip install -r requirements.txt
```

## Nutzung

Die Notebooks in der Reihenfolge 01 bis 03 in Jupyter öffnen und über Restart und Run All ausführen. Notebook 01 erzeugt die bereinigten Daten, auf denen die Modelle in Notebook 02 und 03 aufbauen.

## Ergebnisse

Die Motorleistung, das Baujahr und die Laufleistung sind die wichtigsten Preistreiber. Beide Modelle übertreffen die Baseline deutlich, setzen aber unterschiedliche Schwerpunkte.

| Modell | R² (Testfeld) | MAE | Eigenschaft |
|---|---|---|---|
| Baseline (Mittelwert) | ~0,00 | ~15.500 € | Vergleichsmaßstab |
| Regressionsbaum | 0,79 | ~5.150 € | genau, in der Kreuzvalidierung aber instabil |
| Multiple Regression | 0,60 | ~6.200 € | etwas ungenauer, dafür robust und interpretierbar |

Der Regressionsbaum trifft typische Fahrzeuge am genauesten. Sein hohes R² erweist sich in der Kreuzvalidierung wegen der wenigen Extrempreise jedoch als instabil. Die lineare Regression verallgemeinert dank der logarithmischen Skala stabiler und macht den Einfluss jedes Merkmals als prozentualen Effekt sichtbar.

## Grenzen der Analyse

Untersucht werden Angebotspreise und nicht die tatsächlichen Verkaufspreise. Die Modelle zeigen Zusammenhänge, aber keine kausalen Wirkungen. Sehr teure Fahrzeuge werden von beiden Modellen mit größeren Fehlern geschätzt.

## Credits

Der zugrunde liegende Datensatz Germany Used Cars Dataset 2023 stammt von AutoScout24 und wurde auf Kaggle veröffentlicht.

Die Auswertung nutzt die Open-Source-Bibliotheken pandas, NumPy, Matplotlib, seaborn, scikit-learn und SciPy in JupyterLab.
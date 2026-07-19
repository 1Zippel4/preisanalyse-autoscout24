"""Gemeinsame Bewertungsfunktionen für die Modell-Notebooks.

Die Kennzahlen sind hier zentral definiert, damit Regressionsbaum und lineare
Regression nachweislich am identischen Maßstab gemessen werden. Jedes Notebook
legt sein Ergebnis in results/ ab; die Vergleichstabelle wird daraus erzeugt,
ohne dass Zahlen zwischen Notebooks übertragen werden müssen.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RESULTS = Path(__file__).resolve().parent.parent / "results"


def kennzahlen(y_wahr, y_vorhersage):
    """Gibt R², RMSE und MAE einer Vorhersage zurück."""
    r2 = r2_score(y_wahr, y_vorhersage)
    rmse = mean_squared_error(y_wahr, y_vorhersage) ** 0.5
    mae = mean_absolute_error(y_wahr, y_vorhersage)
    return r2, rmse, mae


def euro(x, decimals=0):
    """Formatiert eine Zahl als Eurobetrag in deutscher Schreibweise."""
    return f"{x:,.{decimals}f} €".replace(",", "X").replace(".", ",").replace("X", ".")


def speichere_ergebnis(modell, rang, r2_test, rmse_test, mae_test,
                       r2_cv=None, mae_cv=None):
    """Schreibt die Kennzahlen eines Modells als CSV nach results/.

    r2_cv und mae_cv sind die Ergebnisarrays aus cross_val_score. Fehlen sie
    (etwa beim Baseline-Modell), bleiben die Kreuzvalidierungsspalten leer.
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    zeile = {
        "Rang": rang,
        "Modell": modell,
        "R2_Test": r2_test,
        "RMSE_Test": rmse_test,
        "MAE_Test": mae_test,
        "R2_CV": None if r2_cv is None else r2_cv.mean(),
        "R2_CV_Std": None if r2_cv is None else r2_cv.std(),
        "MAE_CV": None if mae_cv is None else mae_cv.mean(),
        "MAE_CV_Std": None if mae_cv is None else mae_cv.std(),
    }
    datei = RESULTS / f"{rang}_{modell.lower().replace(' ', '_')}.csv"
    pd.DataFrame([zeile]).to_csv(datei, index=False)
    return zeile


def vergleichstabelle():
    """Liest alle Einzelergebnisse aus results/ und fügt sie zusammen."""
    dateien = sorted(RESULTS.glob("[0-9]_*.csv"))
    if not dateien:
        raise FileNotFoundError(
            "Keine Ergebnisse in results/. Bitte zuerst Notebook 02 und 03 ausführen."
        )
    tab = pd.concat([pd.read_csv(f) for f in dateien], ignore_index=True)
    return tab.sort_values("Rang").reset_index(drop=True)


def formatierte_tabelle():
    """Vergleichstabelle in deutscher Schreibweise für die schriftliche Arbeit."""
    tab = vergleichstabelle()

    def zahl(x, stellen=2):
        if pd.isna(x):
            return "—"
        if abs(x) < 0.005:      # verhindert die Ausgabe "-0,00"
            x = 0.0
        return f"{x:.{stellen}f}".replace(".", ",")

    return pd.DataFrame({
        "Modell": tab["Modell"],
        "R² (Test)": tab["R2_Test"].map(lambda x: zahl(x)),
        "R² (Kreuzvalidierung)": tab.apply(
            lambda r: "—" if pd.isna(r["R2_CV"])
            else f'{zahl(r["R2_CV"])} ± {zahl(r["R2_CV_Std"])}', axis=1),
        "MAE (Test)": tab["MAE_Test"].map(euro),
        "MAE (Kreuzvalidierung)": tab["MAE_CV"].map(
            lambda x: "—" if pd.isna(x) else euro(x)),
    })


def als_markdown(tabelle):
    """Gibt einen DataFrame als Markdown-Tabelle aus, ohne Zusatzpakete.

    Das Ergebnis lässt sich direkt in Word einfügen; Word erkennt die
    Pipe-Schreibweise beim Einfügen als Tabelle.
    """
    spalten = list(tabelle.columns)
    zeilen = [[str(w) for w in reihe] for reihe in tabelle.values]
    breiten = [max(len(spalten[i]), *(len(z[i]) for z in zeilen)) for i in range(len(spalten))]

    def formatiere(werte):
        return "| " + " | ".join(w.ljust(breiten[i]) for i, w in enumerate(werte)) + " |"

    kopf = formatiere(spalten)
    trenn = "|" + "|".join("-" * (b + 2) for b in breiten) + "|"
    return "\n".join([kopf, trenn] + [formatiere(z) for z in zeilen])


"""
Analisi della griglia di valutazione delle rune
=================================================
Legge il CSV "griglia_valutazione_rune_compilata.csv" (che ha un'intestazione
su due righe, con i gruppi LESSICO / CONTENUTO) e produce tre grafici:

1. Bar chart dei criteri mediamente più rispettati (media su tutte le rune)
2. Scatter plot Similarità lessicale vs Coerenza di contenuto, con
   correlazione (per capire se l'imitazione lessicale è accompagnata da vera
   coerenza strutturale, o se è solo imitazione superficiale)
3. Bar chart raggruppato: medie per criterio, Zero-shot (prime 5 rune) vs
   New-shot (ultime 5 rune)

Uso:
    python analizza_rune.py [percorso_csv]

Se non viene passato un percorso, viene usato di default
"griglia_valutazione_rune_compilata.csv" nella cartella corrente.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. CARICAMENTO E PULIZIA DEI DATI
# ---------------------------------------------------------------------------

# Nomi "puliti" delle 9 colonne di valutazione, nell'ordine in cui compaiono
# nel CSV originale (la prima colonna, non elencata qui, è il nome della runa)
COLONNE = [
    "Lunghezza",
    "Conta latinismi",
    "Similarita' (lessico)",
    "Similarita' effetto",
    "Contesto",
    "Coerenza N/E",
    "Chiarezza",
    "Contraddizioni",
    "Originalita'",
]

# Colonne che appartengono al gruppo LESSICO e al gruppo CONTENUTO
# (secondo la struttura a due righe del CSV originale)
COLONNE_LESSICO = ["Conta latinismi", "Similarita' (lessico)", "Similarita' effetto", "Contesto"]
COLONNE_CONTENUTO = ["Coerenza N/E", "Chiarezza", "Contraddizioni", "Originalita'"]


def carica_dati(percorso_csv: str) -> pd.DataFrame:
    """Legge il CSV con doppia intestazione e restituisce un DataFrame pulito
    con una riga per runa e una colonna per criterio (numerico)."""

    grezzo = pd.read_csv(percorso_csv, sep=";", encoding="utf-8-sig", header=None)

    # Le righe dati vere e proprie sono dalla 3 (indice 2) in poi, tranne
    # l'ultima riga che contiene i totali/medie già calcolati (TOT:)
    righe_dati = grezzo.iloc[2:-1].copy()
    righe_dati = righe_dati.reset_index(drop=True)

    righe_dati.columns = ["Runa"] + COLONNE
    righe_dati["Runa"] = righe_dati["Runa"].astype(str).str.strip()

    for col in COLONNE:
        righe_dati[col] = pd.to_numeric(righe_dati[col], errors="coerce")

    return righe_dati


def aggiungi_gruppo_shot(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge una colonna 'Gruppo' che vale 'Zero-shot' per le prime 5 righe
    e 'New-shot' per le ultime 5 righe."""

    n = len(df)
    meta = n // 2
    df = df.copy()
    df["Gruppo"] = ["Zero-shot"] * meta + ["New-shot"] * (n - meta)
    return df


# ---------------------------------------------------------------------------
# 2. GRAFICO 1 - Criterio maggiormente rispettato (media generale)
# ---------------------------------------------------------------------------

def grafico_criteri_medi(df: pd.DataFrame, output_path: str):
    medie = df[COLONNE].mean().sort_values(ascending=False)

    colori = [
        "#2E7D32" if c in COLONNE_CONTENUTO else "#1565C0" if c in COLONNE_LESSICO else "#6A1B9A"
        for c in medie.index
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    barre = ax.bar(medie.index, medie.values, color=colori)

    ax.set_title("Media dei punteggi per criterio (tutte le rune)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Punteggio medio")
    ax.set_ylim(0, max(medie.values) * 1.15)
    ax.tick_params(axis="x", rotation=35)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")

    for barra, valore in zip(barre, medie.values):
        ax.text(barra.get_x() + barra.get_width() / 2, valore + 0.05, f"{valore:.2f}",
                ha="center", va="bottom", fontsize=9)

    # Legenda manuale per i gruppi di colore
    from matplotlib.patches import Patch
    legenda = [
        Patch(facecolor="#1565C0", label="Lessico"),
        Patch(facecolor="#2E7D32", label="Contenuto"),
        Patch(facecolor="#6A1B9A", label="Altro"),
    ]
    ax.legend(handles=legenda, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Grafico 1 salvato in: {output_path}")
    print(medie.round(2).to_string())
    print()


# ---------------------------------------------------------------------------
# 3. GRAFICO 2 - Scatter Similarità lessicale vs Coerenza di contenuto
# ---------------------------------------------------------------------------

def grafico_scatter_lessico_contenuto(df: pd.DataFrame, output_path: str):
    x = df["Similarita' (lessico)"]
    y = df[COLONNE_CONTENUTO].mean(axis=1)  # punteggio composito "contenuto"

    # Correlazione di Pearson (se c'è varianza sufficiente)
    if x.std() > 0 and y.std() > 0:
        corr = np.corrcoef(x, y)[0, 1]
    else:
        corr = float("nan")

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.scatter(x, y, s=140, color="#C62828", edgecolor="black", zorder=3)

    for nome, xi, yi in zip(df["Runa"], x, y):
        ax.annotate(nome, (xi, yi), textcoords="offset points", xytext=(6, 6), fontsize=8)

    # Retta di regressione, solo se ha senso calcolarla
    if x.std() > 0 and not np.isnan(corr):
        coeff = np.polyfit(x, y, 1)
        xs = np.linspace(x.min() - 0.2, x.max() + 0.2, 50)
        ax.plot(xs, np.polyval(coeff, xs), "--", color="gray", zorder=1)

    titolo_corr = f"r = {corr:.2f}" if not np.isnan(corr) else "r = n/d (varianza nulla)"
    ax.set_title(f"Similarità lessicale vs Coerenza di contenuto ({titolo_corr})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Similarità lessicale (imitazione della lingua)")
    ax.set_ylabel("Coerenza di contenuto (media Coerenza N/E, Chiarezza,\nContraddizioni, Originalità)")
    ax.grid(alpha=0.3)

    # Interpretazione testuale in basso al grafico
    if np.isnan(corr):
        interpretazione = "Dati insufficienti a variare per stimare la correlazione."
    elif abs(corr) >= 0.5:
        interpretazione = "Correlazione presente → possibile vera generalizzazione."
    else:
        interpretazione = "Correlazione debole/assente → possibile imitazione superficiale."

    fig.text(0.5, -0.02, interpretazione, ha="center", fontsize=9, style="italic")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Grafico 2 salvato in: {output_path}")
    print(f"Correlazione (Pearson r) similarità lessicale vs contenuto: {corr:.3f}" if not np.isnan(corr) else "Correlazione non calcolabile (varianza nulla in una delle due variabili)")
    print()


# ---------------------------------------------------------------------------
# 4. GRAFICO 3 - Barre raggruppate Zero-shot vs New-shot
# ---------------------------------------------------------------------------

def grafico_zero_vs_new_shot(df: pd.DataFrame, output_path: str):
    df = aggiungi_gruppo_shot(df)

    medie_zero = df[df["Gruppo"] == "Zero-shot"][COLONNE].mean()
    medie_new = df[df["Gruppo"] == "New-shot"][COLONNE].mean()

    x = np.arange(len(COLONNE))
    larghezza = 0.38

    fig, ax = plt.subplots(figsize=(12, 6.5))
    b1 = ax.bar(x - larghezza / 2, medie_zero.values, larghezza, label="Zero-shot (prime 5 rune)", color="#1565C0")
    b2 = ax.bar(x + larghezza / 2, medie_new.values, larghezza, label="New-shot (ultime 5 rune)", color="#EF6C00")

    ax.set_title("Confronto medie per criterio: Zero-shot vs New-shot", fontsize=14, fontweight="bold")
    ax.set_ylabel("Punteggio medio")
    ax.set_xticks(x)
    ax.set_xticklabels(COLONNE, rotation=35, ha="right")
    ax.legend()
    ax.set_ylim(0, max(medie_zero.max(), medie_new.max()) * 1.2)

    for barre in (b1, b2):
        for barra in barre:
            h = barra.get_height()
            ax.text(barra.get_x() + barra.get_width() / 2, h + 0.05, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Grafico 3 salvato in: {output_path}")

    confronto = pd.DataFrame({"Zero-shot": medie_zero, "New-shot": medie_new})
    confronto["Differenza (New-Zero)"] = confronto["New-shot"] - confronto["Zero-shot"]
    print(confronto.round(2).to_string())
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    # Cartella dove si trova questo script (non quella da cui viene lanciato)
    cartella_script = Path(__file__).resolve().parent

    if len(sys.argv) > 1:
        percorso_csv = Path(sys.argv[1])
    else:
        percorso_csv = cartella_script / "griglia_valutazione_rune_compilata.csv"

    if not percorso_csv.exists():
        print(f"[ERRORE] Non trovo il file CSV in: {percorso_csv}")
        print("Soluzioni possibili:")
        print(f"  1) Metti il file 'griglia_valutazione_rune_compilata.csv' nella cartella: {cartella_script}")
        print("  2) Oppure lancia lo script passando il percorso del CSV come argomento, es.:")
        print(f'     python "{Path(__file__).name}" "/percorso/completo/al/tuo/file.csv"')
        sys.exit(1)

    cartella_output = cartella_script / "grafici_rune"
    cartella_output.mkdir(exist_ok=True)

    df = carica_dati(str(percorso_csv))
    print(f"Rune caricate: {len(df)}")
    print(df[["Runa"] + COLONNE].to_string(index=False))
    print()

    grafico_criteri_medi(df, str(cartella_output / "1_criteri_medi.png"))
    grafico_scatter_lessico_contenuto(df, str(cartella_output / "2_scatter_lessico_contenuto.png"))
    grafico_zero_vs_new_shot(df, str(cartella_output / "3_zero_vs_new_shot.png"))

    print(f"Tutti i grafici sono stati salvati nella cartella: {cartella_output.resolve()}")


if __name__ == "__main__":
    main()
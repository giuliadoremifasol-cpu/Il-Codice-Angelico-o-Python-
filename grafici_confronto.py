"""
Script di confronto tra Valutazione Media Umana e Griglia di Valutazione (AI)
tramite boxplot affiancati per ciascuna metrica.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# 0. PERCORSI
# ---------------------------------------------------------------------------

# Cartella in cui si trova questo script: i CSV devono stare qui accanto.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PATH_UMANA = os.path.join(BASE_DIR, "Valutazione_Media_umana.csv")
PATH_GRIGLIA = os.path.join(BASE_DIR, "griglia_valutazione_rune_compilata.csv")
PATH_OUTPUT = os.path.join(BASE_DIR, "confronto_boxplot.png")

# ---------------------------------------------------------------------------
# 1. CARICAMENTO E PULIZIA DATI
# ---------------------------------------------------------------------------

def carica_umana(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File non trovato: {path}\n"
            "Controlla che 'Valutazione_Media_umana.csv' si trovi nella stessa "
            "cartella di questo script."
        )
    df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8")
    df.columns = [c.strip().upper().replace("’", "'") for c in df.columns]
    df = df.rename(columns={df.columns[0]: "NOME RUNA"})
    # rimuove la riga di totale/riepilogo
    df = df[~df["NOME RUNA"].astype(str).str.lower().str.startswith("total")]
    return df


def carica_griglia(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File non trovato: {path}\n"
            "Controlla che 'griglia_valutazione_rune_compilata.csv' si trovi nella "
            "stessa cartella di questo script."
        )
    # le prime due righe sono header multi-livello (LESSICO/CONTENUTO + nomi colonna)
    raw = pd.read_csv(path, sep=";", header=None, encoding="utf-8")
    header = raw.iloc[1].tolist()
    header[0] = "NOME RUNA"
    df = raw.iloc[2:].copy()
    df.columns = [str(c).strip().upper().replace("’", "'") for c in header]
    df = df.reset_index(drop=True)
    # rimuove la riga di totale (colonna 1 == 'TOT:')
    col1 = df.columns[1]
    df = df[~df[col1].astype(str).str.upper().str.strip().eq("TOT:")]
    # converte le colonne numeriche (usa la virgola o il punto come decimale)
    for c in df.columns[1:]:
        df[c] = (
            df[c].astype(str)
            .str.replace(",", ".", regex=False)
            .astype(float)
        )
    return df


umana = carica_umana(PATH_UMANA)
griglia = carica_griglia(PATH_GRIGLIA)

# ---------------------------------------------------------------------------
# 2. METRICHE COMUNI
# ---------------------------------------------------------------------------

metriche_comuni = [c for c in umana.columns if c in griglia.columns and c != "NOME RUNA"]
print("Metriche confrontate:", metriche_comuni)

if not metriche_comuni:
    raise ValueError(
        "Nessuna metrica comune trovata tra i due file. "
        "Controlla i nomi delle colonne nei CSV."
    )

# ---------------------------------------------------------------------------
# 3. BOXPLOT AFFIANCATI
# ---------------------------------------------------------------------------

n = len(metriche_comuni)
ncols = 3
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows))
axes = np.array(axes).reshape(-1)

colori = ["#4C72B0", "#DD8452"]  # umana / griglia

for i, metrica in enumerate(metriche_comuni):
    ax = axes[i]
    dati = [umana[metrica].dropna().values, griglia[metrica].dropna().values]
    bp = ax.boxplot(
        dati,
        tick_labels=["Umana", "Griglia (AI)"],
        patch_artist=True,
        widths=0.55,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=6),
    )
    for patch, colore in zip(bp["boxes"], colori):
        patch.set_facecolor(colore)
        patch.set_alpha(0.75)
    ax.set_title(metrica, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 5.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

# nasconde eventuali assi vuoti in eccesso
for j in range(n, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    "Confronto Valutazione Umana vs Griglia di Valutazione (AI)\nper ciascuna metrica",
    fontsize=15,
    fontweight="bold",
    y=1.02,
)
fig.tight_layout()

fig.savefig(PATH_OUTPUT, dpi=200, bbox_inches="tight")
print("Grafico salvato in:", PATH_OUTPUT)
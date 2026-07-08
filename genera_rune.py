#!/usr/bin/env python3
"""
Script per generare nuove rune "shadowhunter-style" usando Google Generative AI
(Gemini), tramite chiamate REST dirette con `requests` (niente SDK ufficiale).

Flusso:
1. Legge il file rune.csv originale (colonne: nome;descrizione) per ricavare
   alcuni esempi da usare come few-shot.
2. Per ogni runa da generare, interroga il modello DUE volte:
   - una prima richiesta SENZA esempi (solo istruzioni sul mondo di Shadowhunter)
   - una seconda richiesta CON esempi presi dal csv originale (few-shot)
3. Il testo di risposta del modello viene ripulito con BeautifulSoup (rimuove
   eventuali tag html/markdown residui che il modello a volte include).
4. Aggiunge una pausa di 1 secondo tra una richiesta e l'altra per non
   sovraccaricare l'API.
5. Se una richiesta fallisce (errore di rete, risposta malformata, ecc.) i
   campi vengono impostati a None ("null" nel csv) e lo script prosegue.
6. Scrive un file rune_nuove.csv con le stesse colonne dell'originale
   (nome;descrizione), usando lo stesso delimitatore ';'.

Uso:
    export GOOGLE_API_KEY="la_tua_api_key"
    python genera_rune.py

Requisiti:
    pip install requests beautifulsoup4
"""

import csv
import json
import os
import time
import random
import requests
from bs4 import BeautifulSoup

try:
    from dotenv import load_dotenv
    # Carica automaticamente il file .env che si trova nella stessa cartella
    # di questo script (se presente), impostando le variabili d'ambiente.
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    print(
        "[AVVISO] Libreria 'python-dotenv' non installata: il file .env "
        "non verrà letto automaticamente. Installa con:\n"
        "    pip install python-dotenv\n"
        "In alternativa, imposta la variabile con 'export GOOGLE_API_KEY=...' "
        "nello stesso terminale prima di lanciare lo script."
    )

# ------------------------------------------------------------------
# Configurazione
# ------------------------------------------------------------------

API_KEY = os.environ.get("GOOGLE_API_KEY")
MODEL = "gemini-3.1-flash-lite"  # in tabella: "Gemini 3.1 Flash Lite" -> 15 RPM, 500 RPD
# ATTENZIONE: non ho modo di verificare con certezza che questo sia l'ID esatto
# usato dall'API (i nomi "commerciali" mostrati in AI Studio a volte differiscono
# leggermente dall'ID tecnico del modello). Se lo script fallisce con un errore
# 404 "model not found", elenca i modelli disponibili per la tua chiave con:
#
#   curl "https://generativelanguage.googleapis.com/v1beta/models?key=LA_TUA_CHIAVE"
#
# e copia qui l'ID esatto che trovi (campo "name", es. "models/gemini-3.1-flash-lite"
# -> in tal caso usa solo la parte dopo "models/").
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

# Percorsi calcolati rispetto alla cartella dove si trova QUESTO file .py,
# così lo script funziona indipendentemente da dove lo lanci da terminale.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "rune.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "rune_nuove.csv")

DELIMITER = ";"
PAUSE_SECONDS = 1  # pausa fra una richiesta e l'altra

# Quante nuove rune vogliamo generare in totale
NUM_RUNE_DA_GENERARE = 5

# Quanti esempi (few-shot) pescare dal csv originale per ogni richiesta
NUM_ESEMPI = 5


# ------------------------------------------------------------------
# Lettura del csv originale (per ricavare gli esempi few-shot)
# ------------------------------------------------------------------

def leggi_rune_originali(path):
    """Legge il csv originale e restituisce una lista di dict {nome, descrizione}."""
    rune = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=DELIMITER)
        for row in reader:
            rune.append({
                "nome": row.get("nome", "").strip(),
                "descrizione": row.get("descrizione", "").strip(),
            })
    return rune


# ------------------------------------------------------------------
# Costruzione dei prompt
# ------------------------------------------------------------------

def costruisci_prompt_senza_esempi():
    """Prompt che chiede una nuova runa SENZA fornire esempi concreti."""
    return (
        "Sei un esperto del mondo narrativo di Shadowhunters (Cacciatori di Ombre, "
        "da 'Shadowhunters: Chronicles' / 'The Mortal Instruments'). "
        "In questo mondo gli Shadowhunter usano delle rune magiche (chiamate anche Marchi) "
        "incise sulla pelle per ottenere poteri speciali: forza, velocità, guarigione, "
        "invisibilità, visione notturna, tracciamento e molto altro.\n\n"
        "Inventa UNA nuova runa originale, coerente con questo universo narrativo.\n\n"
        "Rispondi ESCLUSIVAMENTE in formato JSON valido, senza testo aggiuntivo, "
        "senza markdown e senza tag html, con questa struttura esatta:\n"
        '{"nome": "<nome della runa>", "descrizione": "<descrizione dettagliata del suo potere>"}'
    )


def costruisci_prompt_con_esempi(esempi):
    """Prompt few-shot che fornisce alcuni esempi reali presi dal csv originale."""
    blocco_esempi = "\n".join(
        f'- {e["nome"]}: {e["descrizione"]}' for e in esempi
    )
    return (
        "Sei un esperto del mondo narrativo di Shadowhunters (Cacciatori di Ombre, "
        "da 'Shadowhunters: Chronicles' / 'The Mortal Instruments'). "
        "Gli Shadowhunter usano rune magiche (Marchi) incise sulla pelle per ottenere "
        "poteri speciali.\n\n"
        "Ecco alcuni esempi di rune realmente esistenti in questo universo narrativo:\n"
        f"{blocco_esempi}\n\n"
        "Prendendo ispirazione dallo STILE, dal TONO e dal LIVELLO DI DETTAGLIO di questi "
        "esempi, inventa UNA nuova runa originale (diversa da quelle elencate), coerente "
        "con questo universo.\n\n"
        "Rispondi ESCLUSIVAMENTE in formato JSON valido, senza testo aggiuntivo, "
        "senza markdown e senza tag html, con questa struttura esatta:\n"
        '{"nome": "<nome della runa>", "descrizione": "<descrizione dettagliata del suo potere>"}'
    )


# ------------------------------------------------------------------
# Chiamata all'API di Google Generative AI (Gemini) via requests
# ------------------------------------------------------------------

def chiama_gemini(prompt, max_tentativi=4):
    """
    Invia il prompt a Google Generative AI usando requests.
    In caso di errore 429 (rate limit) riprova con attesa progressiva
    (exponential backoff): 5s, 10s, 20s, 40s.
    Restituisce il testo grezzo della risposta, oppure None se fallisce
    dopo tutti i tentativi.
    """
    if not API_KEY:
        raise RuntimeError(
            "Variabile d'ambiente GOOGLE_API_KEY non impostata. "
            "Esegui: export GOOGLE_API_KEY='la_tua_chiave'"
        )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.9,
            "maxOutputTokens": 512,
        },
    }

    headers = {"Content-Type": "application/json"}
    params = {"key": API_KEY}

    attesa = 5  # secondi, raddoppia ad ogni tentativo

    for tentativo in range(1, max_tentativi + 1):
        try:
            response = requests.post(
                API_URL,
                headers=headers,
                params=params,
                data=json.dumps(payload),
                timeout=30,
            )

            if response.status_code in (429, 503):
                motivo = "Rate limit" if response.status_code == 429 else "Servizio non disponibile"
                if tentativo < max_tentativi:
                    print(
                        f"  [AVVISO] {response.status_code} {motivo} (tentativo {tentativo}/"
                        f"{max_tentativi}). Riprovo tra {attesa}s..."
                    )
                    time.sleep(attesa)
                    attesa *= 2
                    continue
                else:
                    print(
                        f"  [ERRORE] {response.status_code} {motivo} persistente anche dopo "
                        "i tentativi. Riprova più tardi."
                    )
                    return None

            response.raise_for_status()
            data = response.json()

            testo = (
                data["candidates"][0]["content"]["parts"][0]["text"]
            )
            return testo

        except (requests.RequestException, KeyError, IndexError, ValueError) as e:
            print(f"  [ERRORE] Chiamata API fallita: {e}")
            return None

    return None


def pulisci_testo_con_bs4(testo_grezzo):
    """
    Usa BeautifulSoup per ripulire il testo restituito dal modello da eventuali
    tag html o markup residuo (es. ```json ... ``` oppure tag <b>, <i>, ecc.).
    Restituisce solo il testo pulito.
    """
    if testo_grezzo is None:
        return None

    # rimuove eventuali fence markdown tipo ```json ... ```
    testo = testo_grezzo.strip()
    if testo.startswith("```"):
        testo = testo.strip("`")
        if testo.lower().startswith("json"):
            testo = testo[4:]

    soup = BeautifulSoup(testo, "html.parser")
    testo_pulito = soup.get_text().strip()
    return testo_pulito


def estrai_nome_descrizione(testo_pulito):
    """
    Prova a interpretare il testo pulito come JSON con chiavi 'nome' e 'descrizione'.
    Se il parsing fallisce, restituisce (None, None).
    """
    if not testo_pulito:
        return None, None
    try:
        obj = json.loads(testo_pulito)
        nome = obj.get("nome")
        descrizione = obj.get("descrizione")
        if not nome or not descrizione:
            return None, None
        return nome.strip(), descrizione.strip()
    except (json.JSONDecodeError, AttributeError):
        print("  [ERRORE] Impossibile interpretare la risposta come JSON valido.")
        return None, None


def genera_una_runa(prompt):
    """
    Esegue l'intero ciclo: chiamata API -> pulizia con BeautifulSoup -> parsing.
    In caso di qualunque fallimento restituisce (None, None).
    """
    testo_grezzo = chiama_gemini(prompt)
    testo_pulito = pulisci_testo_con_bs4(testo_grezzo)
    nome, descrizione = estrai_nome_descrizione(testo_pulito)
    return nome, descrizione


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    if not API_KEY:
        print(
            "\n[ERRORE] Variabile d'ambiente GOOGLE_API_KEY non impostata.\n"
            "Prima di lanciare lo script, esegui nel terminale:\n\n"
            "    export GOOGLE_API_KEY='la_tua_chiave'\n\n"
            "Puoi ottenere una chiave gratuita su: https://aistudio.google.com/apikey\n"
        )
        return

    rune_originali = leggi_rune_originali(INPUT_CSV)
    print(f"Lette {len(rune_originali)} rune dal file originale '{INPUT_CSV}'.")

    risultati = []

    for i in range(NUM_RUNE_DA_GENERARE):
        print(f"\n--- Generazione runa {i + 1}/{NUM_RUNE_DA_GENERARE} ---")

        # 1) Richiesta SENZA esempi
        print("Richiesta senza esempi...")
        prompt_no_esempi = costruisci_prompt_senza_esempi()
        nome_1, descrizione_1 = genera_una_runa(prompt_no_esempi)

        if nome_1 is None:
            print("  -> fallimento, salvo 'null'.")
        else:
            print(f"  -> ottenuto: {nome_1}")

        risultati.append({
            "nome": nome_1 if nome_1 is not None else "null",
            "descrizione": descrizione_1 if descrizione_1 is not None else "null",
            "metodo": "zero_shot",
        })

        time.sleep(PAUSE_SECONDS)

        # 2) Richiesta CON esempi (few-shot dal csv originale)
        print("Richiesta con esempi (few-shot)...")
        esempi = random.sample(
            rune_originali, k=min(NUM_ESEMPI, len(rune_originali))
        )
        prompt_con_esempi = costruisci_prompt_con_esempi(esempi)
        nome_2, descrizione_2 = genera_una_runa(prompt_con_esempi)

        if nome_2 is None:
            print("  -> fallimento, salvo 'null'.")
        else:
            print(f"  -> ottenuto: {nome_2}")

        risultati.append({
            "nome": nome_2 if nome_2 is not None else "null",
            "descrizione": descrizione_2 if descrizione_2 is not None else "null",
            "metodo": "few_shot",
        })

        time.sleep(PAUSE_SECONDS)

    # Scrittura del csv finale, stesse colonne dell'originale
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["nome", "descrizione", "metodo"], delimiter=DELIMITER
        )
        writer.writeheader()
        writer.writerows(risultati)

    print(f"\nFatto! Risultati salvati in '{OUTPUT_CSV}'.")


if __name__ == "__main__":
    main()
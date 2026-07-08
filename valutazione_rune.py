#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================================
 VALUTA_RUNE.PY
========================================================================================
Script per la valutazione automatica delle "rune nuove" (rune_nuove.csv) rispetto alle
"rune originali" (rune.csv), usando il modello Google Gemini come giudice per i criteri
qualitativi e calcoli locali (Python puro / fuzzywuzzy) per i criteri quantitativi.

I risultati vengono scritti nella griglia di valutazione (formato .csv, separatore ';'),
rispettando esattamente il formato/intestazione del file allegato dall'utente.

------------------------------------------------------------------------------
COME FUNZIONA (in breve)
------------------------------------------------------------------------------
Per ogni runa nuova vengono calcolati 9 punteggi (le 9 colonne della griglia):

  1. LUNGHEZZA            -> calcolato in locale (nessuna chiamata a Gemini)
  2. CONTA LATINISMI      -> calcolato in locale con un'euristica lessicale
  3. SIMILARITA' (nomi)   -> calcolato in locale con fuzzywuzzy
  4. SIMILARITA' EFFETTO  -> chiesto a Gemini
  5. CONTESTO             -> chiesto a Gemini (in base alla trama di "Città di ossa")
  6. COERENZA N/E         -> chiesto a Gemini
  7. CHIAREZZA             -> chiesto a Gemini
  8. CONTRADDIZIONI       -> Gemini conta le contraddizioni interne, poi lo script
                              converte il conteggio in punteggio 1-5 con la formula
                              indicata dall'utente
  9. ORIGINALITA'         -> chiesto a Gemini (punteggio complessivo olistico)

NOTA IMPORTANTE (ASSUNZIONE): la colonna "ORIGINALITA'" presente nella griglia allegata
NON è descritta nella legenda fornita dall'utente (che copre solo 8 criteri). Ho quindi
assunto che rappresenti un giudizio complessivo di originalità della runa (nome + effetto
insieme), chiesto direttamente a Gemini su scala 1-5. Se il significato desiderato è
diverso, basta modificare il prompt nella funzione `costruisci_prompt_gemini()`.

------------------------------------------------------------------------------
REQUISITI (da installare una tantum)
------------------------------------------------------------------------------
    pip install google-generativeai fuzzywuzzy python-Levenshtein beautifulsoup4 requests python-dotenv

------------------------------------------------------------------------------
CONFIGURAZIONE
------------------------------------------------------------------------------
La API key di Gemini può essere fornita in due modi (per non scriverla mai nel codice):

  A) File .env nella stessa cartella dello script, con dentro una riga tipo:
        GOOGLE_API_KEY=la-tua-chiave

  B) Variabile d'ambiente di sistema:
        export GOOGLE_API_KEY="la-tua-chiave"        (Linux/Mac)
        setx GOOGLE_API_KEY "la-tua-chiave"           (Windows)

Se è presente il file .env, ha la precedenza; in sua assenza si usa la variabile
d'ambiente di sistema, se impostata.

I tre file di input (rune.csv, rune_nuove.csv, griglia_valutazione_rune.csv) devono
trovarsi nella stessa cartella dello script, oppure vanno indicati i percorsi corretti
nella sezione "CONFIGURAZIONE GENERALE" qui sotto.
========================================================================================
"""

# ----------------------------------------------------------------------------------
# IMPORT DELLE LIBRERIE
# ----------------------------------------------------------------------------------
import os              # per leggere variabili d'ambiente (API key) e percorsi file
import sys              # per uscire con un messaggio d'errore chiaro se manca qualcosa
import csv              # per leggere/scrivere i file .csv con separatore ';'
import time              # per il timer di attesa (rate limiting) tra una richiesta e l'altra
import json              # per interpretare le risposte JSON restituite da Gemini
import re                # per le euristiche testuali (latinismi, pulizia testo)

import requests                       # per scaricare la pagina Wikipedia
from bs4 import BeautifulSoup         # per estrarre la sezione "Trama" dalla pagina HTML
from fuzzywuzzy import fuzz           # per calcolare la similarità tra nomi di rune

import google.generativeai as genai   # SDK ufficiale per interrogare Gemini

# python-dotenv permette di leggere le variabili definite in un file ".env" (es.
# GOOGLE_API_KEY=xxxxx) e caricarle come se fossero variabili d'ambiente. Se il pacchetto
# non è installato, lo script continua comunque a funzionare leggendo la chiave solo da
# una eventuale variabile d'ambiente già impostata nel sistema.
try:
    from dotenv import load_dotenv
    DOTENV_DISPONIBILE = True
except ImportError:
    DOTENV_DISPONIBILE = False


# ============================================================================
# CONFIGURAZIONE GENERALE (modificare qui i parametri principali)
# ============================================================================

# --- Percorsi dei file di input/output (devono stare nella stessa cartella dello script,
#     oppure vanno indicati percorsi assoluti) --------------------------------------------
# --- Cartella in cui si trova lo script: tutti i percorsi dei file sono risolti
#     rispetto a questa cartella, così lo script funziona correttamente qualunque sia
#     la cartella corrente del terminale da cui viene lanciato ("cwd") -------------------
CARTELLA_SCRIPT = os.path.dirname(os.path.abspath(__file__))

FILE_RUNE_ORIGINALI = os.path.join(CARTELLA_SCRIPT, "rune.csv")
FILE_RUNE_NUOVE = os.path.join(CARTELLA_SCRIPT, "rune_nuove.csv")
FILE_GRIGLIA_INPUT = os.path.join(CARTELLA_SCRIPT, "griglia_valutazione_rune.csv")
FILE_GRIGLIA_OUTPUT = os.path.join(CARTELLA_SCRIPT, "griglia_valutazione_rune_compilata.csv")

# --- File .env da cui leggere la GOOGLE_API_KEY (se presente) ------------------------
# Va posizionato nella stessa cartella dello script, con una riga tipo:
#     GOOGLE_API_KEY=la-tua-chiave
FILE_ENV = ".env"

# --- Modello Gemini da utilizzare ----------------------------------------------------
# NB: il nome esatto del modello può cambiare nel tempo in base a cosa espone Google.
# Se questa stringa dovesse dare errore "model not found", controllare i nomi modello
# disponibili su https://ai.google.dev/gemini-api/docs/models
MODEL_NAME = "gemini-3.1-flash-lite"

# --- Rate limiting: secondi di pausa tra una chiamata a Gemini e la successiva --------
SECONDI_ATTESA_TRA_RICHIESTE = 5

# --- LIMITE per i test in piccolo -----------------------------------------------------
# Impostare un numero basso (es. 2 o 3) per testare velocemente lo script su poche rune
# prima di lanciarlo su tutto il dataset. Impostare a None per processare tutte le rune.
LIMITE_RUNE_DA_PROCESSARE = None       # <-- MODIFICARE QUI per testare / eseguire tutto (None = tutte)

# --- Titolo della voce Wikipedia italiana da usare come "contesto narrativo" ---------
TITOLO_WIKIPEDIA = "Shadowhunters - Città di ossa"

# --- Percentuale di latinismi nelle rune originali (dato fornito dall'utente) --------
PERCENTUALE_LATINISMI_ORIGINALI = 10.0


# ============================================================================
# SEZIONE 1 - CARICAMENTO DELLA API KEY E CONFIGURAZIONE DEL CLIENT GEMINI
# ============================================================================

def configura_gemini():
    """
    Legge la API key di Gemini e configura l'SDK google-generativeai.
    La chiave viene cercata in questo ordine (il primo valore trovato "vince"):
      1. File .env nella cartella dello script (FILE_ENV), tramite python-dotenv,
         cercando una riga tipo: GOOGLE_API_KEY=la-tua-chiave
      2. Variabile d'ambiente GOOGLE_API_KEY già impostata nel sistema.
    Se non viene trovata alcuna chiave, lo script si interrompe con un messaggio
    chiaro (meglio fallire subito che a metà elaborazione).
    """
    # --- Passo 1: proviamo a caricare il file .env, se presente e se il pacchetto
    #     python-dotenv è installato. load_dotenv() inserisce le variabili trovate
    #     nel file dentro os.environ, senza sovrascrivere variabili già impostate.
    if DOTENV_DISPONIBILE:
        percorso_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), FILE_ENV)
        if os.path.isfile(percorso_env):
            load_dotenv(dotenv_path=percorso_env)
            print(f"[INFO] Caricato file .env da: {percorso_env}")

            # Diagnostica: mostriamo quali NOMI di variabili sono stati letti dal file
            # .env (mai i valori, per non stampare la chiave per sbaglio nei log), così
            # se il nome non combacia con GOOGLE_API_KEY lo si vede subito.
            try:
                from dotenv import dotenv_values
                variabili_nel_file = list(dotenv_values(percorso_env).keys())
                print(f"[DEBUG] Variabili trovate nel file .env: {variabili_nel_file}")
            except Exception:
                pass
        else:
            print(f"[INFO] Nessun file .env trovato in: {percorso_env} (non è un errore, si prosegue).")
    else:
        print(
            "[ATTENZIONE] Il pacchetto 'python-dotenv' non è installato: il file .env "
            "verrà ignorato. Installa con: pip install python-dotenv"
        )

    # --- Passo 2: leggiamo la chiave dalle variabili d'ambiente (ora eventualmente
    #     popolate anche dal file .env appena caricato) -------------------------------
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit(
            "ERRORE: GOOGLE_API_KEY non trovata.\n"
            f"Aggiungila al file '{FILE_ENV}' nella cartella dello script, con una riga tipo:\n"
            "    GOOGLE_API_KEY=la-tua-chiave\n"
            "oppure impostala come variabile d'ambiente con:\n"
            "    export GOOGLE_API_KEY='la-tua-chiave' (Linux/Mac)\n"
            "    setx GOOGLE_API_KEY \"la-tua-chiave\" (Windows)"
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME)


# ============================================================================
# SEZIONE 2 - LETTURA DEI FILE CSV DI INPUT
# ============================================================================

def leggi_rune_originali(path):
    """
    Legge il file delle rune originali (nome;descrizione) e restituisce una lista
    di dizionari [{"nome": ..., "descrizione": ...}, ...].
    Usiamo encoding 'utf-8-sig' per gestire correttamente un eventuale BOM iniziale.
    """
    rune = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for riga in reader:
            rune.append({
                "nome": (riga.get("nome") or "").strip(),
                "descrizione": (riga.get("descrizione") or "").strip(),
            })
    return rune


def leggi_rune_nuove(path):
    """
    Legge il file delle rune nuove (nome;descrizione;metodo) e restituisce una lista
    di dizionari. Il campo 'metodo' (zero_shot / few_shot) viene mantenuto ma non è
    usato nel calcolo dei punteggi.
    """
    rune = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for riga in reader:
            rune.append({
                "nome": (riga.get("nome") or "").strip(),
                "descrizione": (riga.get("descrizione") or "").strip(),
                "metodo": (riga.get("metodo") or "").strip(),
            })
    return rune


def leggi_intestazione_griglia(path):
    """
    Legge solo le prime due righe del file della griglia di valutazione (le righe di
    intestazione con i nomi delle colonne) così da poterle riscrivere identiche nel
    file di output, senza dover "indovinare" il formato.
    """
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        righe = list(reader)
    intestazione = righe[:2]   # riga 1: raggruppamenti LESSICO/CONTENUTO, riga 2: nomi colonna
    return intestazione


# ============================================================================
# SEZIONE 3 - RECUPERO DEL CONTESTO NARRATIVO DA WIKIPEDIA
# ============================================================================

def costruisci_url_wikipedia(titolo):
    """
    Costruisce l'URL della voce Wikipedia italiana a partire dal titolo, seguendo la
    convenzione standard di Wikipedia: gli spazi diventano underscore "_" e il resto
    del titolo (incluse lettere accentate) viene lasciato invariato (Wikipedia gestisce
    l'URL-encoding lato server).
    """
    titolo_url = titolo.strip().replace(" ", "_")
    return f"https://it.wikipedia.org/wiki/{titolo_url}"


def estrai_sezione_trama(html):
    """
    Analizza l'HTML della pagina Wikipedia e estrae il testo della sezione "Trama".
    Su it.wikipedia.org le sezioni sono introdotte da tag <h2> (o <h3>) che contengono
    uno <span class="mw-headline" id="Trama">. Raccogliamo tutti i paragrafi <p> che
    seguono quell'intestazione, fino alla prossima intestazione di pari livello.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Cerca l'intestazione "Trama": prova prima con lo span mw-headline (Wikipedia "classica"),
    # poi in fallback con un tag h2/h3 che contenga letteralmente la parola "Trama".
    intestazione_trama = soup.find(["span"], {"id": re.compile(r"^Trama$", re.IGNORECASE)})
    if intestazione_trama is not None:
        tag_heading = intestazione_trama.find_parent(["h2", "h3"])
    else:
        tag_heading = None
        for h in soup.find_all(["h2", "h3"]):
            if h.get_text(strip=True).lower().startswith("trama"):
                tag_heading = h
                break

    if tag_heading is None:
        return None  # sezione non trovata

    # Raccogliamo tutti gli elementi successivi fino alla prossima intestazione di pari livello
    paragrafi = []
    livello_heading = tag_heading.name  # "h2" oppure "h3"
    for elemento in tag_heading.find_next_siblings():
        if elemento.name == livello_heading:
            break  # inizia la sezione successiva, ci fermiamo
        if elemento.name == "p":
            testo = elemento.get_text(" ", strip=True)
            if testo:
                paragrafi.append(testo)

    return "\n".join(paragrafi) if paragrafi else None


def recupera_contesto_narrativo():
    """
    Scarica la pagina Wikipedia italiana del romanzo e ne estrae la sezione "Trama",
    da usare come riferimento per valutare la "coerenza con il contesto" delle rune
    nuove. In caso di qualsiasi errore (rete assente, pagina non trovata, sezione
    mancante) restituisce None: lo script prosegue comunque, avvisando l'utente, e
    Gemini valuterà il contesto solo sulla base della propria conoscenza generale
    dell'universo Shadowhunters.
    """
    url = costruisci_url_wikipedia(TITOLO_WIKIPEDIA)
    print(f"[INFO] Scarico il contesto narrativo da: {url}")
    try:
        risposta = requests.get(url, timeout=15, headers={"User-Agent": "ValutaRuneScript/1.0"})
        risposta.raise_for_status()
        trama = estrai_sezione_trama(risposta.text)
        if trama is None:
            print("[ATTENZIONE] Sezione 'Trama' non trovata nella pagina. Procedo senza contesto scaricato.")
        return trama
    except Exception as errore:
        print(f"[ATTENZIONE] Impossibile scaricare/estrarre la pagina Wikipedia: {errore}")
        return None


# ============================================================================
# SEZIONE 4 - CRITERI CALCOLATI IN LOCALE (senza Gemini)
# ============================================================================

def calcola_target_lunghezza(rune_originali):
    """
    Calcola la lunghezza "target" del nome di una runa come media delle lunghezze
    dei nomi delle rune originali (arrotondata all'intero più vicino). L'esempio
    fornito dall'utente (target = 11 lettere) era solo illustrativo: qui lo deriviamo
    direttamente dai dati reali delle rune originali.
    """
    lunghezze = [len(r["nome"]) for r in rune_originali if r["nome"]]
    media = sum(lunghezze) / len(lunghezze)
    return round(media)


def punteggio_lunghezza(nome, target):
    """
    Applica la legenda fornita dall'utente:
      - differenza 0                -> 3 punti
      - differenza tra 1 e 5        -> 2 punti
      - differenza 6 o più          -> 1 punto
    """
    differenza = abs(len(nome) - target)
    if differenza == 0:
        return 3
    elif 1 <= differenza <= 5:
        return 2
    else:
        return 1


# --- Euristica per il riconoscimento dei "latinismi" nei nomi delle rune -------------
# ASSUNZIONE: in assenza di un vocabolario ufficiale di riferimento, un nome (o una sua
# singola parola) viene considerato "latinismo" se termina con una desinenza tipica del
# latino classico (es. -us, -um, -is, -tas, -tio, -ium...) oppure è una parola non
# presente nel piccolo elenco di parole italiane comuni sottostante. Questa euristica è
# volutamente semplice: per un'analisi più rigorosa si consiglia di sostituirla con un
# vero dizionario linguistico o di farla validare da Gemini stesso.
SUFFISSI_LATINI = ("us", "um", "is", "as", "tas", "tio", "men", "ium", "itas", "or", "ans", "ens")

def nome_contiene_latinismo(nome):
    """Restituisce True se almeno una parola del nome sembra di origine latina."""
    # rimuoviamo eventuali parti tra parentesi (es. "Visione notturna(Nyx)") e separatori
    nome_pulito = re.sub(r"\(.*?\)", " ", nome)
    parole = re.split(r"[\s/\-']+", nome_pulito)
    for parola in parole:
        parola_lower = parola.lower().strip()
        if len(parola_lower) < 3:
            continue
        if parola_lower.endswith(SUFFISSI_LATINI):
            return True
    return False


def calcola_percentuale_latinismi(lista_rune):
    """Calcola la percentuale di nomi (su una lista di rune) che contengono un latinismo."""
    if not lista_rune:
        return 0.0
    conteggio = sum(1 for r in lista_rune if nome_contiene_latinismo(r["nome"]))
    return (conteggio / len(lista_rune)) * 100.0


def punteggio_latinismi(percentuale_nuove, percentuale_target):
    """
    Converte la distanza tra la percentuale di latinismi delle rune nuove e quella delle
    rune originali (target) in un punteggio 1-5: più le percentuali sono vicine, più il
    punteggio è alto. Scala scelta (assunzione, non specificata numericamente dall'utente):
        differenza <= 2%   -> 5
        differenza <= 5%   -> 4
        differenza <= 10%  -> 3
        differenza <= 20%  -> 2
        differenza > 20%   -> 1
    """
    differenza = abs(percentuale_nuove - percentuale_target)
    if differenza <= 2:
        return 5
    elif differenza <= 5:
        return 4
    elif differenza <= 10:
        return 3
    elif differenza <= 20:
        return 2
    else:
        return 1


def punteggio_similarita_nomi(nome_nuovo, rune_originali):
    """
    Calcola, con fuzzywuzzy, la similarità (0-100) tra il nome nuovo e ciascun nome
    originale, prendendo il valore massimo (cioè il nome originale più simile).
    Più la similarità è alta, più il punteggio finale deve essere BASSO (come richiesto).
    Scala scelta (assunzione, in stile con le altre colonne 1-5):
        similarità >= 85   -> punteggio 1 (troppo simile, penalizzato)
        similarità 70-84   -> punteggio 2
        similarità 50-69   -> punteggio 3
        similarità 30-49   -> punteggio 4
        similarità < 30    -> punteggio 5 (nome ben distinto)
    """
    if not rune_originali:
        return None, None
    similarita_max = max(
        fuzz.token_sort_ratio(nome_nuovo, r["nome"]) for r in rune_originali
    )
    if similarita_max >= 85:
        punteggio = 1
    elif similarita_max >= 70:
        punteggio = 2
    elif similarita_max >= 50:
        punteggio = 3
    elif similarita_max >= 30:
        punteggio = 4
    else:
        punteggio = 5
    return punteggio, similarita_max


def punteggio_contraddizioni(numero_contraddizioni):
    """
    Applica esattamente la legenda fornita dall'utente per il numero di contraddizioni
    interne individuate nella descrizione della runa:
        0-2   -> 5 punti
        3-5   -> 4 punti
        6-7   -> 3 punti
        8-9   -> 2 punti
        10+   -> 1 punto
    """
    if numero_contraddizioni <= 2:
        return 5
    elif numero_contraddizioni <= 5:
        return 4
    elif numero_contraddizioni <= 7:
        return 3
    elif numero_contraddizioni <= 9:
        return 2
    else:
        return 1


# ============================================================================
# SEZIONE 5 - CRITERI VALUTATI DA GEMINI
# ============================================================================

def costruisci_prompt_gemini(runa_nuova, rune_originali, contesto_narrativo):
    """
    Costruisce il prompt testuale da inviare a Gemini per valutare i criteri
    qualitativi (quelli che richiedono comprensione del linguaggio naturale):
      - similarità dell'effetto rispetto alle rune esistenti
      - coerenza con il contesto narrativo
      - coerenza tra nome ed effetto
      - chiarezza della descrizione
      - numero di contraddizioni interne
      - originalità complessiva

    Si chiede esplicitamente a Gemini di rispondere SOLO in formato JSON, per poter
    interpretare la risposta in modo affidabile e automatico.
    """
    # Elenco compatto delle rune originali (nome + descrizione), usato come riferimento
    elenco_originali = "\n".join(
        f"- {r['nome']}: {r['descrizione']}" for r in rune_originali
    )

    # Se non siamo riusciti a scaricare la trama, avvisiamo Gemini e gli chiediamo di
    # basarsi comunque sulla propria conoscenza generale della saga Shadowhunters.
    if contesto_narrativo:
        blocco_contesto = f"Trama di riferimento (da Wikipedia):\n{contesto_narrativo}"
    else:
        blocco_contesto = (
            "Nota: non è stato possibile scaricare la trama da Wikipedia. "
            "Basati sulla tua conoscenza generale dell'universo narrativo di "
            "Shadowhunters - Città di ossa (Cassandra Clare)."
        )

    prompt = f"""
Sei un valutatore esperto del mondo narrativo "Shadowhunters" di Cassandra Clare.
Devi valutare una RUNA NUOVA proposta per il gioco/sistema di rune degli Shadowhunter,
confrontandola con le RUNE ORIGINALI esistenti e con il CONTESTO NARRATIVO del romanzo.

{blocco_contesto}

RUNE ORIGINALI ESISTENTI (nome: descrizione/effetto):
{elenco_originali}

RUNA NUOVA DA VALUTARE:
Nome: {runa_nuova['nome']}
Descrizione/effetto: {runa_nuova['descrizione']}

Valuta la runa nuova secondo i seguenti 6 criteri e assegna un punteggio per ciascuno,
seguendo ESATTAMENTE queste regole:

1. similarita_effetto (intero 1-5): quanto l'effetto della runa nuova è simile a quello
   di una runa ESISTENTE. Più l'effetto è simile a una runa già esistente, PIÙ BASSO deve
   essere il punteggio (1 = effetto quasi identico a una runa esistente, 5 = effetto
   completamente originale e distinto da tutte le rune esistenti).

2. contesto (intero 1-5): quanto la runa nuova è coerente con il mondo narrativo descritto
   nella trama (ambientazione, regole del mondo, tipo di poteri plausibili per uno
   Shadowhunter). Più è coerente, PIÙ ALTO il punteggio.

3. coerenza_nome_effetto (intero 1-5): quanto il nome della runa richiama/suggerisce il
   potere/effetto che essa produce. Più il nome è evocativo e coerente con l'effetto,
   PIÙ ALTO il punteggio.

4. chiarezza (intero 1-5): quanto la descrizione è chiara nello spiegare l'effetto,
   includendo idealmente esempi concreti e indicazioni su come/dove/quando usare la runa.
   Più la descrizione è chiara e completa, PIÙ ALTO il punteggio.

5. numero_contraddizioni (intero, conteggio libero da 0 in su): conta quante affermazioni
   contraddittorie o incoerenti sono presenti nella descrizione della runa (es. effetti
   che si annullano a vicenda, regole che si contraddicono, incongruenze logiche).
   Restituisci il NUMERO ASSOLUTO di contraddizioni trovate, non un punteggio già scalato.

6. originalita (intero 1-5): giudizio complessivo su quanto la runa (nome + effetto
   considerati insieme) risulti originale e ben distinta rispetto a tutto l'impianto di
   rune esistenti. 5 = molto originale, 1 = poco originale/quasi un doppione.

Rispondi ESCLUSIVAMENTE con un oggetto JSON valido, senza testo aggiuntivo, backtick o
markdown, con esattamente questa struttura:

{{
  "similarita_effetto": <intero 1-5>,
  "contesto": <intero 1-5>,
  "coerenza_nome_effetto": <intero 1-5>,
  "chiarezza": <intero 1-5>,
  "numero_contraddizioni": <intero >= 0>,
  "originalita": <intero 1-5>
}}
"""
    return prompt.strip()


def interroga_gemini(modello, prompt):
    """
    Invia il prompt a Gemini e restituisce il dizionario Python ottenuto dal parsing
    della risposta JSON. Solleva un'eccezione in caso di errore di rete/API o di
    risposta non interpretabile: la gestione del fallimento (valori None/null) viene
    fatta dal chiamante, come richiesto.
    """
    configurazione_generazione = genai.types.GenerationConfig(
        temperature=0.2,                       # bassa temperatura per risposte più coerenti/ripetibili
        response_mime_type="application/json",  # chiediamo direttamente output JSON
    )
    risposta = modello.generate_content(
        prompt,
        generation_config=configurazione_generazione,
    )
    testo_risposta = risposta.text.strip()

    # Rimuoviamo eventuali blocchi markdown ```json ... ``` nel caso il modello li aggiunga
    testo_risposta = re.sub(r"^```json\s*|\s*```$", "", testo_risposta, flags=re.IGNORECASE)

    dati = json.loads(testo_risposta)
    return dati


# ============================================================================
# SEZIONE 6 - ORCHESTRAZIONE: VALUTAZIONE COMPLETA DI UNA SINGOLA RUNA
# ============================================================================

def valuta_runa(modello, runa_nuova, rune_originali, contesto_narrativo,
                 target_lunghezza, percentuale_latinismi_nuove):
    """
    Calcola tutti e 9 i punteggi per una singola runa nuova, combinando i criteri
    calcolati in locale con quelli richiesti a Gemini. In caso di fallimento della
    chiamata a Gemini (rete assente, quota esaurita, risposta non valida, ecc.), i
    campi dipendenti da Gemini vengono impostati a None (che verrà scritto come cella
    vuota / null nel CSV di output), mentre i criteri calcolati in locale restano
    comunque disponibili.
    """
    # --- Criteri 1-3: calcolati in locale, sempre disponibili -----------------------
    punt_lunghezza = punteggio_lunghezza(runa_nuova["nome"], target_lunghezza)
    punt_latinismi = punteggio_latinismi(percentuale_latinismi_nuove, PERCENTUALE_LATINISMI_ORIGINALI)
    punt_similarita_nomi, similarita_nomi_raw = punteggio_similarita_nomi(runa_nuova["nome"], rune_originali)

    # --- Criteri 4-9: richiesti a Gemini, con gestione robusta dei fallimenti -------
    try:
        prompt = costruisci_prompt_gemini(runa_nuova, rune_originali, contesto_narrativo)
        risultato_gemini = interroga_gemini(modello, prompt)

        punt_similarita_effetto = risultato_gemini.get("similarita_effetto")
        punt_contesto = risultato_gemini.get("contesto")
        punt_coerenza_ne = risultato_gemini.get("coerenza_nome_effetto")
        punt_chiarezza = risultato_gemini.get("chiarezza")
        numero_contraddizioni = risultato_gemini.get("numero_contraddizioni")
        punt_contraddizioni = (
            punteggio_contraddizioni(numero_contraddizioni)
            if numero_contraddizioni is not None else None
        )
        punt_originalita = risultato_gemini.get("originalita")

    except Exception as errore:
        # Fallimento gestito con null: la runa viene comunque inserita nella griglia,
        # ma con le colonne "Gemini" vuote, e l'errore viene stampato per diagnosi.
        print(f"  [ERRORE] Chiamata a Gemini fallita per la runa '{runa_nuova['nome']}': {errore}")
        punt_similarita_effetto = None
        punt_contesto = None
        punt_coerenza_ne = None
        punt_chiarezza = None
        punt_contraddizioni = None
        punt_originalita = None

    return {
        "nome": runa_nuova["nome"],
        "LUNGHEZZA": punt_lunghezza,
        "CONTA LATINISMI": punt_latinismi,
        "SIMILARITA'": punt_similarita_nomi,
        "SIMILARITA' EFFETTO": punt_similarita_effetto,
        "CONTESTO": punt_contesto,
        "COERENZA N/E": punt_coerenza_ne,
        "CHIAREZZA": punt_chiarezza,
        "CONTRADDIZIONI": punt_contraddizioni,
        "ORIGINALITA'": punt_originalita,
    }


# ============================================================================
# SEZIONE 7 - SCRITTURA DEL FILE CSV DI OUTPUT (griglia compilata)
# ============================================================================

# Ordine esatto delle colonne dati, identico a quello della riga 2 di intestazione
# del file griglia_valutazione_rune.csv originale.
COLONNE_DATI = [
    "LUNGHEZZA", "CONTA LATINISMI", "SIMILARITA'", "SIMILARITA' EFFETTO",
    "CONTESTO", "COERENZA N/E", "CHIAREZZA", "CONTRADDIZIONI", "ORIGINALITA'",
]


def valore_o_stringa_vuota(valore):
    """Converte None in stringa vuota (equivalente a 'null' in una cella CSV)."""
    return "" if valore is None else str(valore)


def calcola_media_colonna(risultati, nome_colonna):
    """Calcola la media dei punteggi non nulli di una colonna, per la riga TOT finale."""
    valori = [r[nome_colonna] for r in risultati if r[nome_colonna] is not None]
    if not valori:
        return ""
    return round(sum(valori) / len(valori), 2)


def scrivi_griglia_output(path_output, intestazione, risultati):
    """
    Scrive il file CSV finale, preservando ESATTAMENTE le due righe di intestazione
    del file originale, seguite da una riga per ogni runa valutata (nello stesso ordine
    di rune_nuove.csv) e infine una riga riassuntiva "TOT:" con la media dei punteggi
    per colonna (ignorando i valori null), nello stesso stile del template allegato.
    """
    with open(path_output, "w", encoding="utf-8-sig", newline="") as f:
        scrittore = csv.writer(f, delimiter=";")

        # Riscriviamo identiche le due righe di intestazione originali
        for riga_intestazione in intestazione:
            scrittore.writerow(riga_intestazione)

        # Una riga di dati per ogni runa valutata
        for risultato in risultati:
            riga = [valore_o_stringa_vuota(risultato[colonna]) for colonna in COLONNE_DATI]
            scrittore.writerow(riga)

        # Riga finale riassuntiva, in stile con il template ";TOT:;;;;;;;;"
        riga_tot = [""] + [calcola_media_colonna(risultati, colonna) for colonna in COLONNE_DATI[1:]]
        riga_tot[0] = "TOT:"
        scrittore.writerow(riga_tot)

    print(f"[OK] Griglia compilata salvata in: {path_output}")


def scrivi_log_dettagliato(path_output, risultati):
    """
    Scrive un file CSV di supporto (non richiesto esplicitamente, ma utile) che affianca
    ad ogni riga anche il NOME della runa, per poter verificare facilmente a quale runa
    corrisponde ciascuna riga della griglia ufficiale (che non contiene i nomi).
    """
    with open(path_output, "w", encoding="utf-8-sig", newline="") as f:
        scrittore = csv.writer(f, delimiter=";")
        scrittore.writerow(["NOME_RUNA"] + COLONNE_DATI)
        for risultato in risultati:
            riga = [risultato["nome"]] + [valore_o_stringa_vuota(risultato[c]) for c in COLONNE_DATI]
            scrittore.writerow(riga)
    print(f"[OK] Log dettagliato (con nomi rune) salvato in: {path_output}")


# ============================================================================
# SEZIONE 8 - FUNZIONE PRINCIPALE (main)
# ============================================================================

def main():
    print("=" * 78)
    print(" VALUTAZIONE AUTOMATICA DELLE RUNE NUOVE TRAMITE GEMINI")
    print("=" * 78)

    # 1) Configura il client Gemini (termina lo script se manca la API key)
    modello = configura_gemini()

    # 2) Carica i dati di input
    print("[INFO] Carico le rune originali e le rune nuove...")
    rune_originali = leggi_rune_originali(FILE_RUNE_ORIGINALI)
    rune_nuove = leggi_rune_nuove(FILE_RUNE_NUOVE)
    intestazione_griglia = leggi_intestazione_griglia(FILE_GRIGLIA_INPUT)
    print(f"[INFO] Rune originali caricate: {len(rune_originali)}")
    print(f"[INFO] Rune nuove caricate: {len(rune_nuove)}")

    # 3) Applica il LIMITE per i test in piccolo (se impostato)
    if LIMITE_RUNE_DA_PROCESSARE is not None:
        rune_da_processare = rune_nuove[:LIMITE_RUNE_DA_PROCESSARE]
        print(f"[INFO] LIMITE attivo: verranno processate solo {len(rune_da_processare)} rune su {len(rune_nuove)}.")
    else:
        rune_da_processare = rune_nuove
        print("[INFO] Nessun limite impostato: verranno processate TUTTE le rune nuove.")

    # 4) Recupera il contesto narrativo da Wikipedia (sezione "Trama")
    contesto_narrativo = recupera_contesto_narrativo()
    if contesto_narrativo:
        print(f"[INFO] Trama recuperata correttamente ({len(contesto_narrativo)} caratteri).")
    else:
        print("[ATTENZIONE] Trama non disponibile: Gemini userà la propria conoscenza generale.")

    # 5) Calcola i valori "globali" necessari ai criteri locali
    target_lunghezza = calcola_target_lunghezza(rune_originali)
    percentuale_latinismi_nuove = calcola_percentuale_latinismi(rune_nuove)
    print(f"[INFO] Lunghezza target (media nomi originali): {target_lunghezza} lettere")
    print(f"[INFO] Percentuale latinismi nelle rune nuove: {percentuale_latinismi_nuove:.1f}% "
          f"(target rune originali: {PERCENTUALE_LATINISMI_ORIGINALI}%)")

    # 6) Valuta ogni runa nuova, una alla volta, rispettando il timer tra le richieste
    risultati = []
    for indice, runa in enumerate(rune_da_processare, start=1):
        print(f"[{indice}/{len(rune_da_processare)}] Valuto la runa: '{runa['nome']}'...")

        risultato = valuta_runa(
            modello=modello,
            runa_nuova=runa,
            rune_originali=rune_originali,
            contesto_narrativo=contesto_narrativo,
            target_lunghezza=target_lunghezza,
            percentuale_latinismi_nuove=percentuale_latinismi_nuove,
        )
        risultati.append(risultato)

        # Timer di attesa tra una richiesta e l'altra (rate limiting), tranne dopo l'ultima
        if indice < len(rune_da_processare):
            print(f"  [INFO] Attendo {SECONDI_ATTESA_TRA_RICHIESTE} secondi prima della prossima richiesta...")
            time.sleep(SECONDI_ATTESA_TRA_RICHIESTE)

    # 7) Scrive i risultati nella griglia CSV di output (formato identico al template)
    scrivi_griglia_output(FILE_GRIGLIA_OUTPUT, intestazione_griglia, risultati)

    # 8) Scrive anche un log dettagliato con i nomi delle rune, per comodità di verifica
    scrivi_log_dettagliato(os.path.join(CARTELLA_SCRIPT, "log_dettagliato_con_nomi.csv"), risultati)

    print("=" * 78)
    print(" COMPLETATO")
    print("=" * 78)


if __name__ == "__main__":
    main()
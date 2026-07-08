#!/usr/bin/env python3
"""
Script per estrarre la lista delle rune (nome e descrizione)
dalla pagina Wiki Fandom di Shadowhunters e salvarle in un file CSV.

URL sorgente: https://shadowhunters.fandom.com/it/wiki/Rune

Dipendenze:
    pip install requests beautifulsoup4
"""

import csv
import os
import sys

import requests
from bs4 import BeautifulSoup

# URL della pagina da cui estrarre la tabella delle rune
URL = "https://shadowhunters.fandom.com/it/wiki/Rune"

# URL della homepage dello stesso wiki: la visitiamo prima della pagina
# vera e propria per ottenere gli stessi cookie che otterrebbe un
# browser normale (alcuni sistemi anti-bot lo richiedono)
HOME_URL = "https://shadowhunters.fandom.com/it/"

# Cartella in cui si trova questo script: la usiamo come base per i
# percorsi dei file, così lo script funziona correttamente anche se lo
# si lancia da una cartella diversa (es. una shell posizionata altrove)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Nome del file CSV di output (salvato nella stessa cartella dello script)
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "rune.csv")

# Se il download dalla rete fallisce (es. 403 per protezione anti-bot),
# lo script prova a leggere un file HTML salvato manualmente a questo
# percorso, come soluzione di ripiego (anche questo nella cartella dello script)
FALLBACK_HTML_FILE = os.path.join(SCRIPT_DIR, "rune_page.html")

# Header HTTP completi "da browser": non basta lo User-Agent, servono
# anche Accept/Accept-Language/Accept-Encoding perché alcuni sistemi
# anti-bot controllano la coerenza dell'intero set di header
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://shadowhunters.fandom.com/it/",
    "Upgrade-Insecure-Requests": "1",
}


def fetch_page_html(url):
    """
    Scarica il contenuto HTML della pagina indicata, usando una sessione
    che visita prima la homepage del wiki per ottenere i cookie, in modo
    simile a come farebbe un browser reale.

    Args:
        url (str): l'indirizzo della pagina da scaricare.

    Returns:
        str: il contenuto HTML della pagina come testo.

    Solleva requests.RequestException se la richiesta fallisce
    (status code >= 400 o problemi di rete).
    """
    with requests.Session() as session:
        session.headers.update(HEADERS)

        # Prima richiesta "di riscaldamento" verso la homepage: serve a
        # raccogliere eventuali cookie di sessione/anti-bot. Ignoriamo
        # errori qui: se fallisce, proviamo comunque la pagina target.
        try:
            session.get(HOME_URL, timeout=15)
        except requests.RequestException:
            pass

        response = session.get(url, timeout=15)
        response.raise_for_status()
        return response.text


def load_fallback_html(filename):
    """
    Legge l'HTML da un file locale, usato come fallback quando il
    download diretto dalla rete non è possibile (es. blocco anti-bot).

    Args:
        filename (str): percorso del file HTML salvato manualmente
            (es. con "Salva pagina con nome" dal browser).

    Returns:
        str | None: il contenuto del file, oppure None se il file
            non esiste.
    """
    if not os.path.isfile(filename):
        return None
    with open(filename, encoding="utf-8") as f:
        return f.read()


def parse_rune_table(html):
    """
    Analizza l'HTML e restituisce la lista delle rune trovate.

    La ricerca della tabella corretta è "tollerante": invece di pretendere
    che le intestazioni siano esattamente "Runa" e "Descrizione" (il
    testo reale sulla pagina wiki potrebbe usare parole diverse, maiuscole
    diverse, o spazi/accenti), consideriamo valida una tabella se una
    delle intestazioni contiene "runa" o "nome" e un'altra contiene
    "descriz" (case-insensitive).

    Args:
        html (str): l'HTML della pagina.

    Returns:
        list[tuple[str, str]]: lista di coppie (nome, descrizione).
    """
    soup = BeautifulSoup(html, "html.parser")

    all_tables = soup.find_all("table")
    if not all_tables:
        raise ValueError("Nessuna tabella <table> trovata nella pagina.")

    table = None
    debug_headers = []  # per stampare le intestazioni trovate in caso di errore

    for candidate in all_tables:
        # Le intestazioni possono stare in <th> oppure, in alcune tabelle
        # wiki, nella prima riga fatta di <td> in grassetto: prendiamo
        # comunque tutti i <th> della tabella, ovunque si trovino
        header_cells = [th.get_text(strip=True) for th in candidate.find_all("th")]
        header_cells_lower = [h.lower() for h in header_cells]
        debug_headers.append(header_cells)

        has_name_col = any(
            "runa" in h or "rune" in h or "nome" in h for h in header_cells_lower
        )
        has_desc_col = any("descriz" in h for h in header_cells_lower)

        if has_name_col and has_desc_col:
            table = candidate
            break

    if table is None:
        # Nessuna corrispondenza: stampiamo le intestazioni di tutte le
        # tabelle trovate, così è facile capire come si chiamano davvero
        # le colonne e aggiustare la ricerca sopra
        details = "\n".join(
            f"  Tabella {i+1}: {headers}" for i, headers in enumerate(debug_headers)
        )
        raise ValueError(
            "Nessuna tabella con colonne tipo 'Runa/Nome' + 'Descrizione' "
            f"trovata. Intestazioni delle tabelle presenti nella pagina:\n{details}"
        )

    runes = []

    # Il corpo della tabella (tbody) contiene le righe con i dati;
    # se manca il tag tbody esplicito, BeautifulSoup lo aggiunge
    # comunque durante il parsing, quindi possiamo usarlo in sicurezza.
    tbody = table.find("tbody")
    rows = tbody.find_all("tr") if tbody else table.find_all("tr")

    for row in rows:
        cells = row.find_all("td")

        # Saltiamo righe non valide (es. eventuale riga di intestazione
        # ripetuta o righe vuote/malformate)
        if len(cells) < 2:
            continue

        nome = cells[0].get_text(strip=True)
        descrizione = cells[1].get_text(strip=True)

        if nome:  # evitiamo di salvare righe con nome vuoto
            runes.append((nome, descrizione))

    return runes


def save_to_csv(runes, filename):
    """
    Salva la lista di rune in un file CSV con colonne "nome" e "descrizione".

    Usiamo il punto e virgola (;) come separatore invece della virgola:
    Excel (soprattutto nelle versioni con impostazioni regionali italiane,
    che usano la virgola come separatore decimale) interpreta meglio i
    CSV con ; come delimitatore di colonna.

    Args:
        runes (list[tuple[str, str]]): lista di coppie (nome, descrizione).
        filename (str): percorso del file CSV di output.
    """
    with open(filename, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow(["nome", "descrizione"])  # intestazione colonne
        writer.writerows(runes)


def main():
    """Funzione principale: orchestra download, parsing e salvataggio."""
    print(f"Scarico la pagina: {URL}")
    html = None
    try:
        html = fetch_page_html(URL)
    except requests.RequestException as exc:
        print(f"Errore durante il download della pagina: {exc}", file=sys.stderr)
        print(
            "Il sito potrebbe bloccare le richieste automatiche (protezione "
            "anti-bot). Provo a usare un file HTML salvato manualmente come "
            f"fallback: '{FALLBACK_HTML_FILE}'",
            file=sys.stderr,
        )
        html = load_fallback_html(FALLBACK_HTML_FILE)
        if html is None:
            print(
                "Nessun file di fallback trovato. Per risolvere: apri la "
                f"pagina '{URL}' nel browser, salvala come pagina HTML "
                f"completa con il nome '{FALLBACK_HTML_FILE}' nella stessa "
                "cartella dello script, e rilancia lo script.",
                file=sys.stderr,
            )
            sys.exit(1)
        print("File di fallback trovato, procedo con quello.")

    print("Estraggo la tabella delle rune...")
    try:
        runes = parse_rune_table(html)
    except ValueError as exc:
        print(f"Errore durante il parsing: {exc}", file=sys.stderr)
        sys.exit(1)

    if not runes:
        print("Nessuna runa trovata: controlla la struttura della pagina.", file=sys.stderr)
        sys.exit(1)

    print(f"Trovate {len(runes)} rune. Salvo in '{OUTPUT_CSV}'...")
    save_to_csv(runes, OUTPUT_CSV)
    print("Fatto.")


if __name__ == "__main__":
    main()
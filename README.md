# Il Codice: Angelico o Python? #

*Domanda di ricerca*:
Un LLM, esposto al linguaggio delle rune del mondo narrativo della saga di Shadowhunters, riesce a generalizzare le regole strutturali che legano nome, effetto e contesto narrativo abbastanza da produrre nuove rune sistematicamente coerenti, o si limita a un'imitazione stilistica superficiale degli esempi esistenti?

*Obiettivi*
1. Analizzare le rune originali come base di riferimento sistematica;
2.  Verificare se il LLM sia in grado di generare nuove rune coerenti;
3.  Osservare il risultato in base al tipo di esempi forniti nel prompt (zero-shot/few-shot);
4.  Valutare i risultati in maniera qualitativa e confrontare le risposte del LLM con le risposte umane;
6.   Riflettere sulla capacità del LLM di generalizzare regole di un "linguaggio impossibile" rispetto al semplice riconoscimento di pattern.

*Metodologia di ricerca* 
1. Raccolta dati: scraping della pagina web di wikifandom dedicata alle rune di shadowhunters, tramite script Python;
2. Generazione guidata: creazione di uno script Python che ha chiesto ad un LLM di generare delle nuove rune in due condizioni (zero-shot/few-shot);
3. Analisi qualitativa: creazione di una griglia di valutazione con criteri raggruppati in lessicali e contenutistici;
4. Valutazione qualitativa: valutazione tramite l'utilizzo di uno script Python che ha chiesto al LLM di attribuire dei punteggi ed inserirli all'interno della griglia di valutazione seguendo una legenda fornitagli;
5. Analisi qualitativa manuale: condivisione della griglia di valutazione a dei fan della saga, creazione di una tabella con la media dei risultati "umani" da confrontare con quelli del LLM;
6. Estrazione dati finale: creazione di grafici di diverso tipo, tramite script Python, che riassume i risultati ottenuti.

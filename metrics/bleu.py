"""
Dieses Scriptfile beinhaltet Funktionen zur Berechnung des BLEU-Scores. Es bietet zusaetzlich einen Einstiegspunkt um
den BLEU-Score fuer Hypothesen zuberechnen, die in einer Datei gespeichert sind.
"""

import argparse
import numpy as np
import data


def _match(n_gram_1: list, n_gram_2: list) -> bool:
    """
    Dies ist eine Hilfsfunktion welche dazu verwendet wird, zwei n-grams auf Uebereinstimmung zu pruefen. Die Vergleiche
    sind dabei Case-sensitive.

    :param n_gram_1:
    :param n_gram_2:
    :return: True, falls n_gram_1 und n_gram_2 uebereinstimmen, sonst False.
    """
    n_gram_equal = True
    for token_1, token_2 in zip(n_gram_1, n_gram_2):
        if token_1 != token_2:
            n_gram_equal = False
            break

    return n_gram_equal


def get_n_gram_matches(n: int, ref: str, hyp: str) -> int:
    """
    Diese Funktion zaehlt die Uebereinstimmungen von n-grams zwischen der uebergebenen Referenz und Hypothese.

    (siehe Aufgabe 1.3)

    1)  Spalte die Saetze wortweise auf und uebertrage die Woerter jeweils in ein Array.
    2)  Für jedes n-gram aus dem Hypothesensatz:
      2.1)  Falls ein solches n-gram schon gezaehlt wurde, dann fahre mit dem naechsten n-gram fort. Sonst vermerke
            dieses n-gram als verbraucht.
      2.2)  Zaehle wie oft dieses n-gram in Referenz und Hypothese vorkommt.
      2.3)  Addiere das Minimum der in 2.2) gezaehlten Anzahl Vorkommnisse zu der Anzahl matches.
    3)  Gib die Anzahl an matches zurueck.

    :param n: Anzahl der Woerter eines n-grams
    :param ref: Referenzsatz
    :param hyp: Hypothesensatz
    :return: Anzahl der n-gram-Uebereinstimmungen
    """

    # 1)
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()

    matches = 0
    dirty_n_grams = []

    # 2)
    for h in range(len(hyp_tokens) + 1 - n):
        n_gram = hyp_tokens[h:h + n]

        # 2.1)
        if n_gram in dirty_n_grams:
            continue
        else:
            dirty_n_grams.append(n_gram)

        hyp_match = 0
        ref_match = 0

        # 2.2)
        for i in range(len(hyp_tokens) + 1 - n):
            hyp_gram = hyp_tokens[i:i + n]
            if _match(hyp_gram, n_gram):
                hyp_match += 1

        for j in range(len(ref_tokens) + 1 - n):
            ref_gram = ref_tokens[j:j + n]
            if _match(ref_gram, n_gram):
                ref_match += 1

        # 2.3)
        matches += min(ref_match, hyp_match)

    # 3)
    return matches


def get_modified_n_gram_precision(n: int, ref: list, hyp: list) -> float:
    """
    Diese Funktion berechnet die modifizierte n-gram Praezision fuer die uebergebene Menge von Referenzen und Hypothesen.

    (siehe Aufgabe 1.3)

    1)  Es ist notwendig, dass die Laenge der Liste der Referenzen mit der der Hypothesen uebereinstimmt, damit die
        Referenzen und Hypothesen paarweise zugeordnet werden koennen.
    2)  Fuer jedes Referenz-Hypothesen-Paar:
      2.1)  Update die Summe im Zaehler des Bruchs (siehe Formel): In der Formel wird fuer jedes n-gram der Hypothese
            das Minimum, entweder die Haeufigkeit des n-grams in der Referenz oder in der Hypothese zur Summe addiert.
            Die Teilformel der Summe ueber alle n-grams der Hypothese ist gleich bedeutend mit der Funktion
            get_n_gram_matches.
      2.2)  Update den Nenner des Bruchs: Addiere die Anzahl n-grams der Hypothese zur Summe des Nenners.
    3)  Berechne die modifizierte n-gram Praezidion und gebe sie zurueck.

    :param n: Anzahl der Woerter eines n-grams
    :param ref: Liste der Referenzsaetze
    :param hyp: Liste der Hypothesensaetze
    :return: modifizierte n-gram Praezision P_n
    """

    # 1)
    assert len(ref) == len(hyp), "Die Anzahl der Hypothesen und Referenzen stimmt nicht ueberein!"

    numerator = 0
    denominator = 0

    # 2)
    for ref_line, hyp_line in zip(ref, hyp):
        # 2.1)
        numerator += get_n_gram_matches(n, ref_line, hyp_line)

        # 2.2)
        denominator += len(hyp_line.split()) - n + 1

    # 3)
    return numerator / denominator


def get_brevity_penalty(c: int, r: int) -> float:
    """
    Diese Funktion berechnet die Brevity Penalty (BP) aus der Laenge der Hypothese und Laenge der Referenz.

    (siehe Aufgabe 1.3)

    :param c: Laenge der Hypothese
    :param r: Laenge der Referenz
    :return: Brevity Penalty (BP)
    """
    if c > r:
        return 1
    else:
        return np.exp(1 - (r / c))


def get_bleu_score(N: int, ref: list, hyp: list) -> float:
    """
    Diese Funktion berechnet den BLEU-Score fuer ein N, einen Referenztext und einen Hypothesentext, beide in form einer
    Liste aus Saetzen.

    (siehe Aufgabe 1.3)

    1)  Bestimme die Anzahl der Woerter/Tokens von Referenz- und Hypothesentext (oder auch "die Laenge") um weiter die
        Brevity Penalty (BP) zu bestimmen.
    2)  Ermittle fuer jedes n in [1, N] die modifizierte n-gram Praezision P_n, und summiere deren Logarithmen.
    3)  Die Summe wird gemittelt und der Exponentialfunktion uebergeben.
    4)  Schliesslich wird aus BP und dem exponentiellen der Summe der BLEU-Score berechnet und zurueckgegeben.

    :param N: Groesse der maximalen zu ueberpruefenen n-grams
    :param ref: Referenztext in Form einer Liste aus Saetzen
    :param hyp: Hypothesentext in Form einer Liste aus Saetzen
    :return: BLEU-Score
    """

    # 1)
    ref_length = 0
    hyp_length = 0

    for line in ref:
        ref_length += len(line.split())

    for line in hyp:
        hyp_length += len(line.split())

    BP = get_brevity_penalty(hyp_length, ref_length)

    # 2)
    log_sum = 0
    for n in range(1, N + 1):
        log_sum += np.log(get_modified_n_gram_precision(n, ref, hyp))

    # 3)
    log_sum *= 1 / N
    exp = np.exp(log_sum)

    # 4)
    bleu = BP * exp
    return bleu


if __name__ == '__main__':
    """
    Dieses Skript berechnet den BLEU-Score fuer eine oder mehrere Hypothesen anhand einer Referenz.
    
    (siehe Aufgabe 1.3)
    
    1)  Parse die uebergebenen Argumente.
    2)  Initialisiere eine Liste in der die Ermittlungsergebnisse gespeichert werden koennen.
    3)  Lade den Referenztext vom uebergebenen Dateipfad.
    4)  Fuer jeden uebergebenen Pfad zu einem Hypothesentext: Lade den Hypothesentext vom uebergebenen Dateipfad,
        ermittle den zugehoerigen BLEU-Score und speichere das Ergebnis in der Liste.
    5)  Falls ein Pfad zu einer Ausgabedatei definiert ist, dann schreibe das Ergebnis in diese Datei.
    
    Beispielhafter Aufruf:
    
    python bleu.py -ref "<Pfad zum Referenztext>" -hyp "<Pfad zum Hypothesentext 1>" "<Pfad zum Hypothesentext 2>"
    
    """

    # 1)
    parser = argparse.ArgumentParser(prog="bleu")
    parser.add_argument("-ref", "--reference", nargs=1, required=True,
                        help="Pfad zur Datei die den Referenztext beinhaltet.")
    parser.add_argument("-hyp", "--hypothesis", nargs='+', required=True,
                        help="Ein oder mehrere Pfade zu Dateien die Hypothesentexte beinhalten.")
    parser.add_argument("-out", "--outfile", nargs="?", default=None,
                        help="Optionaler Pfad zu einer Datei in welche das Ergebnis geschrieben werden soll.")
    args = parser.parse_args()

    # 2)
    lines = [None] * len(args.hypothesis)

    # 3)
    ref = data.read_txt(args.reference[0])

    # 4)
    for i, hyp_arg in enumerate(args.hypothesis):
        hyp_name = hyp_arg.split('/')[-1]

        hyp = data.read_txt(hyp_arg)

        print("\rCalculating BLEU score for '{0}' ...".format(hyp_name), end='')
        bleu_score = get_bleu_score(4, ref, hyp)

        print("\rBLEU score of '{0}' is {1}".format(hyp_name, bleu_score))

        lines[i] = "{0}: {1}\n".format(hyp_name, bleu_score)

    # 5)
    if args.outfile is not None:
        data.to_txt(args.outfile, lines)

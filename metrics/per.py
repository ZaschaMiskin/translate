"""
Dieses Scriptfile beinhaltet Funktionen zur Berechnung der PER. Es bietet zusaetzlich einen Einstiegspunkt um
die PER fuer Hypothesen zuberechnen, die in einer Datei gespeichert sind.
"""

import argparse
import data


def get_matches(ref: str, hyp: str) -> int:
    """
    Diese Funktion ermittelt die Uebereinstimmungen eines Referenzsatzes und einer Hypothese.

    Uebereinstimmungen sind definiert als die Anzahl Woerter aus der Hypothese, die auch in der Referenz gefunden werden
    koennen. (Siehe slides_metrics.pdf Folie 7)

    1)  Aus der Definition von Uebereinstimmungen folgt, dass wir fuer Woerter die im Referenzsatz doppelt vorkommen die
        Anzahl dieser Woerter in der Hypothese nicht doppelt zaehlen duerfen. Um Duplikate aus der Token-List der Referenz
        zu streichen wandeln wir die List mit set() in eine Menge um welche wir mittels list() dann wieder zu einer
        Liste konvertieren.

    :param ref: Referenzsatz
    :param hyp: Hypothesensatz
    :return: (Anzahl) Uebereinstimmungen
    """
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()

    # 1)
    uniq_ref_tokens = list(set(ref_tokens))

    matches = 0

    for ref_token in uniq_ref_tokens:
        for hyp_token in hyp_tokens:
            if ref_token == hyp_token:
                matches += 1

    return matches


def get_per(ref: list, hyp: list) -> float:
    """
    Diese Funktion berechnet die Position-independent Error Rate (PER) fuer einen Refenz- und einen Hypothesentext,
    welche beide in Form einer Liste aus Saetzen vorliegen.

    1)  Bestimme die Anzahl der Woerter/Tokens von Referenz- und Hypothesentext (oder auch "die Laenge").
    2)  Bestimme die (Anzahl) Uebereinstimmungen pro Satzpaar.
    3)  Bestimme die PER und gebe diese zurueck.

    :param ref: Referenztext in Form einer Liste aus Saetzen
    :param hyp: Hypothesentext in Form einer Liste aus Saetzen
    :return: PER
    """

    # 1)
    ref_length = 0
    hyp_length = 0

    for line in ref:
        ref_length += len(line.split())

    for line in hyp:
        hyp_length += len(line.split())

    # 2)
    matches = 0

    for r, h in zip(ref, hyp):
        matches += get_matches(r, h)

    # 3)
    per = 1 - ((matches - max(0, hyp_length - ref_length)) / ref_length)

    return per


if __name__ == '__main__':
    """
    Dieses Skript berechnet die PER fuer eine oder mehrere Hypothesen anhand einer Referenz.

    (siehe Aufgabe 1.4)

    1)  Parse die uebergebenen Argumente.
    2)  Initialisiere eine Liste in der die Ermittlungsergebnisse gespeichert werden koennen.
    3)  Lade den Referenztext vom uebergebenen Dateipfad.
    4)  Fuer jeden uebergebenen Pfad zu einem Hypothesentext: Lade den Hypothesentext vom uebergebenen Dateipfad,
        ermittle die zugehoerige PER und speichere das Ergebnis in der Liste.
    5)  Falls ein Pfad zu einer Ausgabedatei definiert ist, dann schreibe das Ergebnis in diese Datei.

    Beispielhafter Aufruf:

    python per.py -ref "<Pfad zum Referenztext>" -hyp "<Pfad zum Hypothesentext 1>" "<Pfad zum Hypothesentext 2>"

    """

    # 1)
    parser = argparse.ArgumentParser(prog="per")
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

        print("\rCalculating PER for '{0}' ...".format(hyp_name), end='')
        per = get_per(ref, hyp)

        print("\rPER of '{0}' is {1}".format(hyp_name, per))

        lines[i] = "{0}: {1}\n".format(hyp_name, per)

    # 5)
    if args.outfile is not None:
        data.to_txt(args.outfile, lines)

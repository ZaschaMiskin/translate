"""
Dieses Scriptfile beinhaltet Funktionen zur Berechnung der WER. Es bietet zusaetzlich einen Einstiegspunkt um
die WER fuer Hypothesen zuberechnen, die in einer Datei gespeichert sind.
"""

import argparse
import data
import levenshtein as lev


def get_wer(ref: list, hyp: list, debugmode=False) -> float:
    lev_dis = lev.levenshteindistanz(ref, hyp, debugmode)
    ref_length = 0
    for line in ref:
        ref_length += len(line.split())

    return lev_dis / ref_length


if __name__ == '__main__':
    """
    Dieses Skript berechnet die WER fuer eine oder mehrere Hypothesen anhand einer Referenz.

    (siehe Aufgabe 1.4)

    1)  Parse die uebergebenen Argumente.
    2)  Initialisiere eine Liste in der die Ermittlungsergebnisse gespeichert werden koennen.
    3)  Lade den Referenztext vom uebergebenen Dateipfad.
    4)  Fuer jeden uebergebenen Pfad zu einem Hypothesentext: Lade den Hypothesentext vom uebergebenen Dateipfad,
        ermittle die zugehoerige WER und speichere das Ergebnis in der Liste.
    5)  Falls ein Pfad zu einer Ausgabedatei definiert ist, dann schreibe das Ergebnis in diese Datei.

    Beispielhafter Aufruf:

    python wer.py -ref "<Pfad zum Referenztext>" -hyp "<Pfad zum Hypothesentext 1>" "<Pfad zum Hypothesentext 2>"

    """

    # 1)
    parser = argparse.ArgumentParser(prog="wer")
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

        print("\rCalculating WER for '{0}' ...".format(hyp_name), end='')
        wer = get_wer(ref, hyp)

        print("\rWER of '{0}' is {1}".format(hyp_name, wer))

        lines[i] = "{0}: {1}\n".format(hyp_name, wer)

    # 5)
    if args.outfile is not None:
        data.to_txt(args.outfile, lines)

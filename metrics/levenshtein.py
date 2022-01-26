import numpy as np
import data
import argparse

"""

Bestimmt Levenshtein-Distanz zweier Sätze

"""


def backtrack(LevMatrix, j, k) -> str:
    """
    Backtracking-Algorithmus, der die Matrix durchgeht und entscheidet, wann welche Operation zum optimalen Ergebnis geführt hat
    1) Bei Auslassung war der Wert links der kleinste. Die Wörter waren nicht gleich
    2) Bei Einfügung war der Wert oben der kleinste. Die Wörter waren nicht gleich
    3) Bei Ersetzung war der Wert links oben der kleinste. Die Wörter waren nicht gleich
    4) Bei Gleichheit waren die verglichenen Wörter gleich. Der Wert linksoben wurde übernommen

    :param LevMatrix: Die übertragene Levenshteinmatrix
    :param j: Index
    :param k: Index
    :return: String mit aktueller Anweisung
    """

    # 1
    if j > 0 and LevMatrix[j - 1][k] + 1 == LevMatrix[j][k]:
        return backtrack(LevMatrix, j - 1, k) + "Auslassung "
    # 2
    if k > 0 and LevMatrix[j][k - 1] + 1 == LevMatrix[j][k]:
        return backtrack(LevMatrix, j, k - 1) + "Einfügung "
    # 3
    if j > 0 and k > 0 and LevMatrix[j - 1][k - 1] + 1 == LevMatrix[j][k]:
        return backtrack(LevMatrix, j - 1, k - 1) + "Ersetzung "
    # 4
    if j > 0 and k > 0 and LevMatrix[j - 1][k - 1] == LevMatrix[j][k]:
        return backtrack(LevMatrix, j - 1, k - 1) + "Gleich "

    return ""


def levenshteindistanz(ref: list, hyp: list, showDebug=False):
    """
    Berechnet zunächst Levenshteinmatrix. Dann wird der backtracking-Algorithmus auf dieser Matrix angewandt

    0)Lese .txt-Dateien und teile auf nach Wörtern
    1)Matrix erstellen
    2)Matrix initialisieren auf (j=0,k) sowie (j,k=0)
    3)Restliche Matrix füllen nach der Levenshteinfunktion
    4)Lese Levenshteindistanz beider Texte aus(LevMatrix[J][K])

    :param ref: Liste aller Sätze aus RefText
    :param hyp: Liste aller Sätze aus Hyptext
    :param showDebug: Optionaler Parameter zu Debugzwecken
    :return: Levenshteindistanz
    """

    # 0
    lev_dis_sentence = []

    for i in range(len(ref)):
        data1 = hyp[i].split()
        data2 = ref[i].split()
        if showDebug:
            print(data1)
            print(data2)

        # 1
        J = len(data1)
        K = len(data2)
        LevMatrix = [[0 for x in range(K + 1)] for y in range(J + 1)]
        # LevMatrix[J][K]
        # 2
        LevMatrix[0][0] = 0
        for i in range(J + 1):
            LevMatrix[i][0] = i
        for i in range(K + 1):
            LevMatrix[0][i] = i

        # 3
        for i in range(1, J + 1):
            for j in range(1, K + 1):
                w1 = data1[i - 1]
                w2 = data2[j - 1]
                if w1 == w2:
                    LevMatrix[i][j] = LevMatrix[i - 1][j - 1]
                else:
                    LevMatrix[i][j] = min(LevMatrix[i - 1][j - 1] + 1, LevMatrix[i][j - 1] + 1, LevMatrix[i - 1][j] + 1)

        if showDebug:
            print(np.matrix(LevMatrix))  # Debug
            print()

        # 5
        """
        Um festzustellen, in welcher Reihenfolge die Operationen ausgeführt werden, muss backtracking genutzt werden.
        Wenn man das während der Matrixberechnung implementieren würde, würden alle Operationen ausgegeben, nicht nur
        "der optimale Weg"
        """

        if showDebug:
            print(backtrack(LevMatrix, J, K))
            print("----------")

        lev_dis_sentence.append(LevMatrix[J][K])

    lev = 0
    for i in lev_dis_sentence:
        lev += i
    # 4
    return lev


if __name__ == '__main__':
    """
    
    Dieses Skript berechnet die Levenshteindistanz fuer eine Hypothese anhand einer Referenz.
    
    """

    # 1)
    parser = argparse.ArgumentParser(prog="bleu")
    parser.add_argument("-ref", "--reference", nargs=1, required=True,
                        help="Pfad zur Datei die den Referenztext beinhaltet.")
    parser.add_argument("-hyp", "--hypothesis", nargs='+', required=True,
                        help="Ein oder mehrere Pfade zu Dateien die Hypothesentexte beinhalten.")
    parser.add_argument("-out", "--outfile", nargs="?", default=None,
                        help="Optionaler Pfad zu einer Datei in welche das Ergebnis geschrieben werden soll.")
    parser.add_argument("-db", "--debug", nargs="?", default=False,
                        help="Optionales Argument, welches angibt ob die Levenshtein-Matrizen ausgegeben werden sollen.")
    args = parser.parse_args()

    # 2)
    lines = [None] * len(args.hypothesis)

    # 3)
    ref = data.read_txt(args.reference[0])

    # 4)
    for i, hyp_arg in enumerate(args.hypothesis):
        hyp_name = hyp_arg.split('/')[-1]

        hyp = data.read_txt(hyp_arg)

        print("\rCalculating Lev-distance for '{0}' ...".format(hyp_name), end='')
        lev_dis = levenshteindistanz(ref, hyp, args.debug)

        print("\rLevenshtein-distance of '{0}' is {1}".format(hyp_name, lev_dis))

        lines[i] = "{0}: {1}\n".format(hyp_name, lev_dis)

    # 5)
    if args.outfile is not None:
        data.to_txt(args.outfile, lines)

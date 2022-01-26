import argparse
import data
from dictionary import END_OF_SEQUENCE, START_OF_SEQUENCE, Dictionary
from alignment import Alignment, SimpleAlignment, Alignment2
import numpy as np
import pandas as pd
import tensorflow as tf


class BatchHandler:
    """
    Diese Klasse bietet einige Methoden zur Handhabung mehrerer Batches. Fuer zukuenftige Use-Cases sollte sie ggf.
    ausgebaut werden.
    """

    def __init__(self, batches: list):
        for batch in batches:
            assert isinstance(batch, Batch), "{} is not a Batch.".format(batch)

        self._batches = batches

    def get_batches_of_lines(self, lines: list) -> list:
        """
        Diese Methode gibt die Batches (der lokalen Batchliste) zurueck, welche mindestens eine Batch-Zeile besitzen,
        die mindestens einer der uebergebenen Text-Zeilen (Text-Zeile = Text-Satz) zugehoerig sind.

        :param lines: Liste mit Indizes der gesuchten Text-Zeilen
        :return: Liste der zugehoerigen Batches
        """
        matching_batches = []

        for batch in self._batches:
            for line in lines:
                if line in batch.get_lines():
                    matching_batches.append(batch)
                    break

        return matching_batches

    def lines_to_dataframe(self, lines: list, tokenize=False,
                           src_dict: Dictionary = None, trg_dict: Dictionary = None) -> pd.DataFrame:
        """
        Diese Methode synthetisiert ein pandas-DataFrame welches die Batch-Zeilen die den uebergebenen Text-Zeilen
        (Text-Zeile = Text-Satz) zugehoerig sind darstellt.

        :param lines: Liste mit Indizes der gesuchten Text-Zeilen
        :param tokenize: Boolean, der angibt ob die Tokens im DataFrame durch ihr Dictionary-Label oder durch ihren
                         tatsaechlichen String-Wert repraesentiert werden sollen.
        :return: DataFrame welches die zugehoerigen Batch-Zeilen darstellt.
        """
        matching_batches = self.get_batches_of_lines(lines)

        line_indices = []
        src_windows = []
        trg_windows = []
        trg_labels = []

        for batch in matching_batches:
            for line, src, trg, lab in zip(*batch.get_batch(include_indices=True)):
                if line in lines:
                    line_indices.append(line)
                    src_windows.append(src.tolist())
                    trg_windows.append(trg.tolist())
                    trg_labels.append(lab)

        return _to_dataframe(line_indices, src_windows, trg_windows, trg_labels,
                             tokenize=tokenize, src_dict=src_dict, trg_dict=trg_dict)


class Batch:
    """
    Diese Klasse implementiert Batches. Die in der Vorlesung vorgestellten Batches koennten einfach als 3-Tupel
    implementiert werden, doch diese Klasse bietet zusaetzlich eine komfortable Handhabung der Daten durch Hilfsmethoden.
    """

    def __init__(self, B: int, w: int, S: np.ndarray, T: np.ndarray, L: np.ndarray, line_indices: list):
        """
        :param B: Batch Size
        :param w: Window Size
        :param S: B x 2*w+1 - source-window Matrix
        :param T: B x w - target-window Matrix
        :param L: B - target-label Vektor
        :param line_indices: Indizes der Text-Zeilen aus denen die Batch-Daten ihren Ursprung haben.
        """

        assert isinstance(S, np.ndarray), "Es wird erwartet, dass die source-window Matrix ein NumPy-Array ist."
        assert isinstance(T, np.ndarray), "Es wird erwartet, dass die target-window Matrix ein NumPy-Array ist."
        assert isinstance(L, np.ndarray), "Es wird erwartet, dass der target-labels Vektor ein NumPy-Array ist."

        assert S.shape == (B, 2 * w + 1), "Die Dimensionen der source-window Matrix sind ungueltig."
        assert T.shape == (B, w), "Die Dimensionen der target-window Matrix sind ungueltig."
        assert L.shape == (B,), "Die Dimensionen des target-label Vektors sind ungueltig."

        self._batch_size = B
        self._window_size = w
        self._source_windows = S
        self._target_windows = T
        self._target_labels = L
        self._line_indices = line_indices

    def __getitem__(self, item):
        if item == 'S' or item == 0:
            return self._source_windows
        elif item == 'T' or item == 1:
            return self._target_windows
        elif item == 'L' or item == 2:
            return self._target_labels

    def get_batch(self, include_indices=False) -> tuple:
        if include_indices:
            return self._line_indices, self._source_windows, self._target_windows, self._target_labels
        else:
            return self._source_windows, self._target_windows, self._target_labels

    def get_window_size(self) -> int:
        return self._window_size

    def get_lines(self) -> list:
        return list(set(self._line_indices))

    def to_dataframe(self, tokenize=False, src_dict: Dictionary = None, trg_dict: Dictionary = None) -> pd.DataFrame:
        """
        Diese Methode synthetisiert ein pandas-DataFrame, welches die Batch-Daten darstellt.
        :param tokenize: Boolean, der angibt ob die Tokens im DataFrame durch ihr Dictionary-Label oder durch ihren
                         tatsaechlichen String-Wert repraesentiert werden sollen.
        :return: DataFrame welches die Batch-Daten darstellt.
        """

        return _to_dataframe(self._line_indices, self._source_windows.tolist(), self._target_windows.tolist(),
                             self._target_labels.tolist(), tokenize=tokenize, src_dict=src_dict, trg_dict=trg_dict)


def _to_dataframe(line_indices: list, src_windows: list, trg_windows: list, trg_labels: list,
                  tokenize=False, src_dict: Dictionary = None, trg_dict: Dictionary = None) -> pd.DataFrame:
    """
    Dies ist eine Hilfsfunktion, welche von Batch und BatchHandler zur synthese von pandas-DataFrames aufgerufen wird.

    1)  Falls die Labels in Tokens uebersaetzt werden sollen, dann werden die uebergebenen Listen uebersaetzt.
    2)  Anschliessend werden die Daten in ein fuer DataFrames geeignetes Format gebracht.
    Im folgenden wird die Tabelle schrittweise zusammengefuegt:
    3)  Die erste Spalte gibt den Index der Ursprungs-Text-Zeile der jeweiligen Batch-Zeile an, gefolgt von einer leeren
        Spalte fuer verschoenerte Darstellung.
    4)  Darauf folgen die Spalten der source-window Eintraege, auch gefolgt von einer leeren Spalte.
    5)  Darauf die Spalten der target-window Eintraege.
    6)  Schliesslich folgt die Spalte fuer die target-labels, voran wieder eine leere Spalte.
    7)  Zum Schluss wird aus den formatierten Daten und dem erstellten Layout ein DataFrane erstellt und zurueckgegeben.

    :param line_indices:
    :param src_windows: Liste von source-window Matrix-Zeilen in Form von Listen
    :param trg_windows: Liste von target-window Matrix-Zeilen in Form von Listen
    :param trg_labels: Liste von traget-label Vektor Eintraegen
    :param tokenize: Boolean, der angibt ob die Tokens im DataFrame durch ihr Dictionary-Label oder durch ihren
                     tatsaechlichen String-Wert repraesentiert werden sollen.
    :return: DataFrame welches die Batch-Daten darstellt.
    """

    # 1)
    if tokenize:
        assert src_dict is not None, "Wenn die Tokens als String dargestellt werden sollen werden Dictionaries benoetigt."
        assert trg_dict is not None, "Wenn die Tokens als String dargestellt werden sollen werden Dictionaries benoetigt."

        src_windows = [[src_dict.get_token(label) for label in src_window] for src_window in src_windows]
        trg_windows = [[trg_dict.get_token(label) for label in trg_window] for trg_window in trg_windows]
        trg_labels = [trg_dict.get_token(label) for label in trg_labels]

    # 2)
    data = [[line_index, ''] + src_window + [''] + trg_window + ['', trg_label]
            for line_index, src_window, trg_window, trg_label
            in zip(line_indices, src_windows, trg_windows, trg_labels)]

    w = int((len(src_windows[0]) - 1) / 2)

    assert w == ((len(src_windows[0]) - 1) / 2), "Die Groesse der source-windows ist ungueltig."
    assert len(trg_windows[0]) == w, "Die Groesse der target-windows ist ungueltig."

    # 3)
    columns = ['line', '    ']
    # 4)
    for i in range(2 * w + 1):
        if i < w:
            columns.append("f_b_i-{0}".format(w - i))
        elif i == w:
            columns.append("  f_b_i")
        else:
            columns.append("f_b_i+{0}".format(i - w))

    columns.append('    ')
    # 5)
    for i in range(w):
        columns.append("e_i-{0}".format(w - i))
    # 6)
    columns += ['    ', 'e_i']
    # 7)
    df = pd.DataFrame(data=data, columns=columns)
    return df


def _get_target_window(i: int, w: int, trg_tokens: list) -> list:
    """
    Dies ist eine Hilfsfunktion von batchify. Sie gibt zu einem uebergebenen Index i in einem tokenisierten Satz der
    Target-Language das zugehoerige Target-Window zurueck.

    :param i: Index im Target-Satz
    :param w: Window Size
    :param trg_tokens: tokenisierter Target-Satz
    :return: Target-Window
    """

    proc_trg_tokens = [START_OF_SEQUENCE] * w + trg_tokens + [END_OF_SEQUENCE]
    return proc_trg_tokens[i:i + w]


def _get_source_window(i: int, w: int, alignment: Alignment, src_tokens: list) -> list:
    """
    Dies ist eine Hilfsfunktion von batchify. Sie gibt zu einem uebergebenen Index i in einem tokenisierten Satz der
    Target-Language durch Ausfuehrung eines Alignments ein Source-Window in einem entsprechenden Source-Satz zurueck.

    :param i: Index im Target-Satz
    :param w: Window Size
    :param alignment: Instanz eines Alignments ueber dem Source-Target-Satzpaar
    :param src_tokens: tokenisierter Source-Satz
    :return: Source-Window
    """

    proc_src_tokens = [START_OF_SEQUENCE] * w + src_tokens + [END_OF_SEQUENCE] * (w + 1)
    b_i = alignment[i]
    window = proc_src_tokens[b_i:b_i + 2 * w + 1]

    assert len(window) == 2 * w + 1, "Ungueltige Fenstergroesse! Ueberpruefe ob das verwendete Alignment wirklich auf " \
                                     "die gewuenschte Zielmenge abbildet!"
    return window


def _get_target_label(i: int, trg_tokens: list) -> str:
    """
    Dies ist eine Hilfsfunktion von batchify. Sie gibt zu einem uebergebenen Index i in einem tokenisierten Satz, den
    entsprechenden Token (Target-Label) zurueck.

    :param i: Token Index
    :param trg_tokens: tokensísieter Satz
    :return: Token Label (string)
    """
    if i == len(trg_tokens):
        return END_OF_SEQUENCE
    else:
        return trg_tokens[i]


def batch_format(source: list, target: list, w: int, alignment_class_ref, src_dict: Dictionary, trg_dict: Dictionary,
                 include_lines: bool = False) -> tuple:
    """
    Diese Funktion erhaelt eine Liste von Source- und Target-Saetzen und wandelt diese in eine Liste mit Batchdaten mit
    den uebergebenen Parametern um.

    1)  Initialisiere Listen in denen die Source- und Target-Windows, sowie die Target-Labels und die Indizes der
        Ursprungs-Text-Zeilen gespeichert werden sollen.
    2)  Fuer jedes Source-Target-Satzpaar:
      2.1)  Zerlege die Saetze in Tokens
      2.2)  Instanziiere ein Alignment auf dem tokenisierten Satzpaar
      2.3)  Fuer jeden Target-Token im Target-Satz:
        2.3.1)  Ermittle das zum Target-Token zugehoerige Source-Window im Source-Satz, das zum Target-Token zugehoerige
                Target-Window, sowie das zum Target-Token zugehoerigen Target-Label.
        2.3.2)  Speichere die ermittelten Werte in den jeweiligen Listen.
    3)  Gib die ermittelten Batchdaten in einem Tuple verpackt zurueck. Falls die Text-Zeilenindizes eingeschlossen werden
        sollen, so fuege diese hinzu.

    :param source: Liste von Saetzen der Source-Language
    :param target: Liste von Saetzen der Target-Language
    :param w: Window-Size
    :param alignment_class_ref: Klasse des Alignments welches verwendet werden soll
    :return: Liste von Batches
    """

    # 1)
    src_windows = []
    trg_windows = []
    trg_labels = []
    line_indices = []

    # 2)
    for line_index, (src_line, trg_line) in enumerate(zip(source, target)):
        # 2.1)
        src_tokens = src_line.split()
        trg_tokens = trg_line.split()
        # 2.2)
        alignment = alignment_class_ref(src_tokens, trg_tokens)
        # 2.3)
        for i in range(len(trg_tokens) + 1):
            # 2.3.1)
            src_window = [src_dict.get_index(token) for token in _get_source_window(i, w, alignment, src_tokens)]
            trg_window = [trg_dict.get_index(token) for token in _get_target_window(i, w, trg_tokens)]
            trg_label = trg_dict.get_index(_get_target_label(i, trg_tokens))
            # 2.3.2)
            src_windows.append(src_window)
            trg_windows.append(trg_window)
            trg_labels.append(trg_label)
            line_indices.append(line_index + 1)

    if include_lines:
        return line_indices, src_windows, trg_windows, trg_labels
    else:
        return src_windows, trg_windows, trg_labels


def source_windows_format(source: list, trg_src_ratio: float, w: int, alignment_class_ref, src_dict: Dictionary) -> list:
    """
    Diese Funktion dient zum Erzeugen des Inputs fuer die Suche. Sie erhaelt eine Liste von Source-Saetzen und wandelt
    diese in eine verschachtelte Liste um: Jeder Listen-Eintrag entspricht einem Source-Satz und besteht aus einer
    Liste pro Source-Window (Tokens kodiert als Indizes).

    Die Berechnung des Alignments erfolgt basierend auf einem als Objektdatei uebergebenem Target-Source-Ratio
    (Verhaeltnis von mittlere Target-Satz-Laenge zu mittlerer Source-Satz-Laenge). Es dient zur Vorhersage der
    wahrscheinlichen Target-Satz-Laenge.

    1)  Initialisiere Listen in der die Source-Windows, verpackt in einer Liste pro Source-Satz, gespeichert werden.
    2)  Fuer jeden Source-Satz:
      2.1)  Lege eine Liste fuer die Source-Windows eines Source-Satzes an
      2.2)  Zerlege den Source-Satz in Tokens. Um die korrekte Eingabe fuer die Alignment-Funktionen zu gewaehrleisten:
            Erzeuge eine Liste, deren Laenge der geschaetzen Target-Satz-Laenge entspricht (Werte irrelevant).
            Runde dabei.
      2.3)  Instanziiere ein Alignment.
      2.4)  Fuer jeden hypothetischen Target-Token im Target-Satz:
        2.4.1)  Ermittle das zum hypothetischen Target-Token zugehoerige Source-Window im Source-Satz.
        2.4.2)  Speichere die Source-Windows eines Satzes in einer Liste.
      2.5) Speichere die Listen der Source-Saetze in einer Liste.

    :param source: Liste von Saetzen der Source-Language
    :param trg_src_ratio: Target-Source-Ratio der Trainingsdaten
    :param w: Window-Size
    :param alignment_class_ref: Klasse des Alignments welches verwendet werden soll
    :param src_dict: Dictionary der Source-Sprache
    :return: Liste von Listen der Source-Windows
    """

    # 1)
    src_sentences_as_windows = []

    # 2)
    for src_line in source:
        # 2.1)
        src_windows = []
        # 2.2)
        src_tokens = src_line.split()
        trg_tokens = list(range(round(len(src_tokens) * trg_src_ratio)))
        # 2.3)
        alignment = alignment_class_ref(src_tokens, trg_tokens)
        # 2.4)
        for i in range(len(trg_tokens) + 1):
            # 2.4.1)
            src_window = [src_dict.get_index(token) for token in _get_source_window(i, w, alignment, src_tokens)]
            # 2.4.2)
            src_windows.append(src_window)
        # 2.5)
        src_sentences_as_windows.append(src_windows)

    return src_sentences_as_windows


def batchify(line_indices: list, src_windows: list, trg_windows: list, trg_labels: list, B: int) -> list:
    """
    Diese Funktion erhaelt eine Liste von Source- und Target-Saetzen und wandelt diese in Batches mit den uebergebenen
    Parametern um.
    
    ACHTUNG:    Diese Funktion wird ausschliesslich zur Erfuellung des Uebungsblatts 2 verwendet. Fuer alle folgenden
                fuer alle folgenden Uebungsblaetter werden TensorFlow's batch-Handhabungsfunktionalitaeten genutzt.
                Um die Texte in (S,T,L)-Batch-Format zu ueberfuehren verwende batch_format.

    1)  Wandle die Daten in Batchdaten um.
    2)  Als naechstes werden die Elemente der Listen in NumPy Arrays (Matrizen, Vektoren) ueberfuehrt:
        Fuer jeden Index i der Eintrags-Vierlinge (line_index, source_window, target_window, target_label) in den zuvor
        ermittleten Listen:
      2.1)  Falls i % B == 0, dann erstelle neue Source- und Target-Window, sowie Target-Label Matrizen, welche in den
            kommenden B Schleifendurchlaeufen gefuellt werden sollen.
      2.2)  Fuelle die Matrizen zeilenweise mit den Source- bzw. Target-Windows bzw. Target-Labels auf.
      2.3)  Sobald die Matrizen mit B Zeilen befuellt sind, werden sie in einer Liste abgespeichert.
    3)  Diese Liste wird letztlich zurueckgegeben.

    TODO:   In vielen Faellen wird die Anzahl der Source-Target-Satzpaare nicht durch B teilbar sein, d.h. der letzte
            Batch wird nicht komplett gefuellt und die uebrigen Batch-Zeilen bleiben (in der aktuellen Implementierung)
            mit 0 belegt. Entscheide was hier getan werden soll!

    :param source: Liste von Saetzen der Source-Language
    :param target: Liste von Saetzen der Target-Language
    :param B: Groesse der Batches
    :param w: Window-Size
    :param alignment_class_ref: Klasse des Alignments welches verwendet werden soll
    :return: Liste von Batches
    """

    batches = []
    # 2)
    for i in range(len(src_windows)):
        # 2.1)
        if i % B == 0:
            S = np.zeros((B, 2 * w + 1), dtype=int)
            T = np.zeros((B, w), dtype=int)
            L = np.zeros(B, dtype=int)

        # 2.2)
        S[i % B] = src_windows[i]
        T[i % B] = trg_windows[i]
        L[i % B] = trg_labels[i]

        # 2.3)
        if i % B == B - 1 or i == len(src_windows) - 1:
            batches.append(Batch(B, w, S, T, L, line_indices[i - (i % B): i + 1]))

    # 3)
    return batches


def batch_format_lines(source: list, target: list, w: int, alignment_class_ref, src_dict: Dictionary,
                       trg_dict: Dictionary, include_lines: bool = False) -> tuple:
    """
    Diese Funktion ueberfuehrt den uebergebenen Source- und Targettext in das (S,T,L)-Batchformat und gruppiert die
    Source- und Targetwindows, sowie die Targetlabels nach ihren Ursprungssatz (line) in Listen/Batches.

    :param source: Liste von Saetzen der Source-Language
    :param target: Liste von Saetzen der Target-Language
    :param B: Groesse der Batches
    :param w: Window-Size
    :param alignment_class_ref: Klasse des Alignments welches verwendet werden soll
    :param include_lines: Ob die Liste mit den Lineindizes im Rueckgabetuple enthalten sein soll
    :return: (S,T,L)-Batchdaten gruppiert nach Saetzen/Lines
    """
    batch_format_data = batch_format(source, target, w, alignment_class_ref, src_dict, trg_dict, include_lines=True)

    lines = []
    src_windows = []
    trg_windows = []
    trg_labels = []

    current_line = 1

    src_window_temp = []
    trg_window_temp = []
    trg_label_temp = []

    for line, src_win, trg_win, trg_lab in zip(*batch_format_data):
        if line == current_line:
            src_window_temp.append(src_win)
            trg_window_temp.append(trg_win)
            trg_label_temp.append(trg_lab)
        else:
            lines.append(current_line)

            src_windows.append(src_window_temp)
            trg_windows.append(trg_window_temp)
            trg_labels.append(trg_label_temp)

            current_line = line
            src_window_temp = []
            trg_window_temp = []
            trg_label_temp = []

    lines.append(current_line)

    src_windows.append(src_window_temp)
    trg_windows.append(trg_window_temp)
    trg_labels.append(trg_label_temp)

    if include_lines:
        return lines, src_windows, trg_windows, trg_labels
    else:
        return src_windows, trg_windows, trg_labels


def get_trg_src_ratio(train_source: list, train_target: list):
    """
    TODO: Assoziation von Target-Source-Ratio mit einem trainierten Modell.

    Berechnet das Verhaeltnis von mittlerer Source-Satz-Laenge zu mittlerer Target-Satz-Laenge.

    :param train_source: Liste der Source-Saetze, die im Training verwendet wurden.
    :param train_target: Liste der Target-Saetze, die im Training verwendet wurden.
    :return: Target-Source-ratio
    """
    token_sum_src = 0
    token_sum_trg = 0
    for src_line in train_source:
        token_sum_src += len(src_line.split())
    for trg_line in train_target:
        token_sum_trg += len(trg_line.split())
    trg_src_ratio = (token_sum_trg / len(train_target)) / (token_sum_src / len(train_source))

    return trg_src_ratio


if __name__ == '__main__':
    """
    Dieses Skript ueberfuehrt einen gegebenen Source- und Target-Text in das (S,T,L)-Batch-Format.
    Alternativ ueberfuehrt es einen gegebenen Source-Text in Source-Windows (eine Liste pro Source-Satz) fuer die Suche.
    
    Optional speichert es die Batchdaten in einer Ausgabedatei.
    Optional unterteilt es die Batchdaten in Batches einer festgelegten Groesse und gibt die Batchdaten angegebener
    Zeilen grafisch aufbereitet aus.
    
    0)  Lade den Source-Text von dem gegebenen Pfad und initialisiere weitere Variablen.
    
    Im Fall der Erzeugung von Batches fuer das Training:
    
        1)  Lade den Target-Text von dem gegebenen Pfad.
        2)  Falls Pfade zu Files gegeben sind, in welchen Dictionaries gespeichert sind, dann lade diese, sonst benutze
            Source- und Target-Text um jeweils neue Dictionaries zu instanzieren.
        3)  Konvertiere die Daten in das Batch-Format.
        4)  Falls outfile definiert ist, dann schreibe die Batchdaten in das File.
        5)  Falls batchify definiert ist, dann...
          5.1) Ueberfuehre die Batchdaten in Batches der Groesse B.
          5.2) Fuer weitere Handhabung der Batches, wird ein BatchHandler verwendet.
          5.3) Benutze den BatchHandler um ein pandas-DataFrame zu erzeugen welches die Batch-Zeilen zu den angegebenen
               Text-Zeilen in Label-Form darstellt. Gib dieses DataFrame anschliessend in der Konsole aus.
          5.4) Benutze den BatchHandler um ein pandas-DataFrame zu erzeugen welches die Batch-Zeilen zu den angegebenen
               Text-Zeilen in String-Form darstellt. Gib dieses DataFrame anschliessend in der Konsole aus.
          5.5) Ueberfuehre die Batchdaten der Angegebenen Textzeilen in DataFrames und schreibe diese in die angegebenen
               Ausgabepfade.
    
    Im Fall der Erzeugung von Source-Windows fuer die Suche:
    
        6)  Lade das Target-Source-Ratio aus einer Objektdatei.
        7)  Falls ein Pfad zum File gegeben ist, in welchem ein Dictionary gespeichert sind, dann lade dieses, sonst 
            benutze den Source-Text um ein neues Dictionary zu instanzieren.
        8)  Konvertiere die Daten in Listen von Source-Windows.
        9)  Falls outfile definiert ist, dann schreibe die Source-Windows in das File.
            
    """
    parser = argparse.ArgumentParser(prog="batch")
    parser.add_argument("window_size", nargs=1, type=int, help="Fenstergroesse (w)")
    parser.add_argument("source", nargs=1, help="Pfad zur Datei die den Sourcetext beinhaltet.")
    parser.add_argument("-trg", "--target", nargs=1, default=None, help="Optionaler Pfad zur Datei die den Targettext beinhaltet.")
    parser.add_argument("-dic", "--dictionaries", nargs="+", default=None,
                        help="Optionaler Pfad zu den Dateien, in denen Dictionaries zu Source- und Target-Language gespeichert sind.")
    parser.add_argument("-out", "--outfile", nargs=1, default=None,
                        help="Optionaler Pfad zu der Datei, in die Batchdaten geschrieben werden sollen.")
    parser.add_argument("-bfy", "--batchify", nargs=5, default=None,
                        help="Unterteile die Batchdaten in Batches der Groesse [0] und gebe die Batchdaten der Zeilen"
                             "[1] bis [2] aus. Schreibe die Labeldarstellung dieser Zeilen in das File [3] und die"
                             "tokenisierte Darstellung in das File [4].")
    parser.add_argument("-jtx", "--join", nargs=1, default=False, type=bool,
                        help="Optionales Argument, falls kein Dictionary angegeben ist und join=True ist, dann wird"
                             "fuer Source- und Targettext ein gemeinsames Dictionary erstellt.")
    parser.add_argument("-dou", "--dic_out", nargs="+", default=None,
                        help="Optionaler Pfad zu Dateien in die die verwendeten Dictionaries in uebersichtlicher"
                             "Form gespeichert werden.")
    parser.add_argument("-ratio", "--src_trg_ratio", nargs=1, default=None,
                        help="Optionale Pfad zu Objektdatei, die source-target-ratio enthaelt (Verhaeltniss von "
                             "mittlerer Source-Satz-Laenge zu mittlerer Target-Satz-Laenge der Trainingsdaten zur "
                             "Vorhersage der Target-Satz-Laenge bei der Suche)")
    args = parser.parse_args()

    # 0)
    source = data.read_gzip_text(args.source[0])
    w = args.window_size[0]

    # 1)
    if args.target:
        target = data.read_gzip_text(args.target[0])

        # 2)
        if args.dictionaries:
            src_dict_path = trg_dict_path = args.dictionaries[0]

            if len(args.dictionaries) > 1:
                trg_dict_path = args.dictionaries[1]

            print("Loading src-dict='{0}' and trg-dict='{1}' ...".format(src_dict_path, trg_dict_path), end='')
            src_dict = data.load_obj(src_dict_path)
            trg_dict = data.load_obj(trg_dict_path)
            print("\rLoaded src-dict='{0}' and trg-dict='{1}'.".format(src_dict_path, trg_dict_path))
        else:
            if args.join:
                print("Creating dictionary for src='{0}' and trg='{1}' ...".format(args.source[0], args.target[0]), end='')
                unite = source + target
                src_dict = trg_dict = Dictionary(unite)
                print("\rCreated dictionary for src='{0}' and trg='{1}'.".format(args.source[0], args.target[0]))
            else:
                print("Creating dictionaries for src='{0}' and trg='{1}' ...".format(args.source[0], args.target[0]),
                      end='')
                src_dict = Dictionary(source)
                trg_dict = Dictionary(target)
                print("\rCreated dictionaries for src='{0}' and trg='{1}'.".format(args.source[0], args.target[0]))

        if args.dic_out:
            data.save_obj(args.dic_out[0], src_dict)
            data.to_txt(args.dic_out[0] + ".txt", src_dict.to_dataframe().to_string(index=False))
            if len(args.dic_out) > 1:
                data.save_obj(args.dic_out[1], trg_dict)
                data.to_txt(args.dic_out[1] + ".txt", trg_dict.to_dataframe().to_string(index=False))

        # 3)
        print("Formatting data of src='{0}' and trg='{1}' to batch-data with window-size {2} ..."
              .format(args.source[0], args.target[0], w), end='')
        batch_data = batch_format(source, target, w, SimpleAlignment, src_dict, trg_dict, include_lines=True)
        print("\rFormatted data of src='{0}' and trg='{1}' to batch-data with window-size {2}."
              .format(args.source[0], args.target[0], w))

        # 4)
        if args.outfile:
            batch_out = args.outfile[0]
            print("Saving batch-data to '{0}'.".format(batch_out))
            data.save_obj(batch_out, batch_data[1:])

        # 5)
        if args.batchify:
            B = int(args.batchify[0])
            line_start = int(args.batchify[1])
            line_end = int(args.batchify[2])
            lab_out = args.batchify[3]
            tok_out = args.batchify[4]

            # 5.1)
            print("\rPartitioning data of src='{0}' and trg='{1}' into batches of size B={2} ..."
                  .format(args.source[0], args.target[0], B), end='')
            batches = batchify(*batch_data, B)
            print("\rPartitioned data of src='{0}' and trg='{1}' into batches of size B={2}."
                  .format(args.source[0], args.target[0], B))

            # 5.2)
            batch_handler = BatchHandler(batches)

            # 5.3)
            print("\rGathering batch data of lines {0} to {1} ...".format(line_start, line_end), end='')
            df_labels = batch_handler.lines_to_dataframe(range(line_start, line_end + 1), tokenize=False)
            print("\rBatch data of lines {0} to {1} as labels:".format(line_start, line_end))
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr',
                                   False):
                print(df_labels)

            # 5.4)
            print("\rGathering batch data of lines {0} to {1} ...".format(line_start, line_end), end='')
            df_tokens = batch_handler.lines_to_dataframe(range(line_start, line_end + 1),
                                                         tokenize=True, src_dict=src_dict, trg_dict=trg_dict)
            print("\rBatch data of lines {0} to {1} as tokens:".format(line_start, line_end))
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr',
                                   False):
                print(df_tokens)

            # 5.5)
            label_out = df_labels.to_string()
            token_out = df_tokens.to_string()
            data.to_txt(lab_out, label_out)
            data.to_txt(tok_out, token_out)

    # 6)
    elif args.src_trg_ratio:
        src_trg_ratio = float(data.load_obj(args.src_trg_ratio[0]))

        # 7)
        if args.dictionaries:
            src_dict_path = args.dictionaries[0]
            print("Loading src-dict='{0}' ...".format(src_dict_path, end=''))
            src_dict = data.load_obj(src_dict_path)
            print("\rLoaded src-dict='{0}'.".format(src_dict_path))
        else:
            print("Creating dictionary for src='{0}' ...".format(args.source[0], end=''))
            src_dict = Dictionary(source)
            print("\rCreated dictionary for src='{0}'.".format(args.source[0]))

        # 8)
        print("Formatting data of src='{0}' to source windows with window-size {1} ..."
              .format(args.source[0], w), end='')
        src_windows = source_windows_format(source, src_trg_ratio, w, SimpleAlignment, src_dict)
        print("\rFormatted data of src='{0}' to source windows with window-size {1}.".format(args.source[0], w))

        # 9)
        if args.outfile:
            windows_out = args.outfile[0]
            print("Saving source windows to '{0}'.".format(windows_out))
            data.save_obj(windows_out, src_windows)


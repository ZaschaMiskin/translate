class Alignment:
    """
    Dies ist eine abstrakte Oberklasse, welche grundlegende Funktionalitäten einer Alignment-Implementierung vorgibt.
    """

    def __init__(self, src_tokens: list, trg_tokens: list):
        """
        Eine Instanz einer Alignment-Implementierung benoetigt die Tokens des Source- und Targetsatzes zur Erstellung
        eines Mappings.

        Beachte:

        Das Mapping beginnt bei 0, d.h. das erste Token eines Satzes hat den Index 0 und nicht wie in der Vorlesung den
        Index 1. Das bedeutet auch, das letzte Token eines Satzes S hat den Index |S|-1.

        In dieser Implementierung wird erwartet, dass die Satze keine '<s>' oder '</s>' Tokens beinhalten.
        Dennoch, den Anforderungen an Alignments nach (siehe Slides 2, Folie 27), sollen fuer die End-of-Sequence '</s>'
        Tokens (spezielle) Abbildungsvorschriften existieren, daher wird hier so getan, als waeren sie Teil der Saetze:
        Ein '</s>' befindet sich immer am Ende eines Satzes, daher wird jedem Satz ein weiterer (virtueller) Index
        hinzugefuegt, welcher im Mapping des Alignments beruecksichtigt werden muss. Das letzte Token eines Satzes S
        hat den Index |S|-1, d.h. das '</s>' dieses Satzes hat den Index |S|. Diese Indizes werden hier im Konstruktor
        jeweils unter einem Attribut der Klasse gespeichert.

        :param src_tokens: Tokenliste des Source-Satzes
        :param trg_tokens: Tokenliste des Target-Satzes
        """
        self._src_tokens = src_tokens
        self._trg_tokens = trg_tokens
        self._src_end = len(src_tokens)
        self._trg_end = len(trg_tokens)

    def __getitem__(self, i):
        """
        Diese Methode wird von Python zur Auswertung von 'self[key]'-Abfragen aufgerufen
        (siehe https://docs.python.org/3/reference/datamodel.html#object.__getitem__).

        Von einer Alignment-Implementierung wird erwartet, dass diese Methode einen Index i zu einem Token aus dem
        Targetsatz erhaehlt und einen Index b_i zu einem Token aus dem Sourcesatz zurueckgibt.
        Dazu wird die Methode _map() der Alignment-Implementierung aufgerufen, welche die Abbildungsvorschrift des
        Alignments vorgibt.

        :param i: i - Index eines Tokens im Targetsatz
        :return: b_i - Index eines Tokens im Sourcesatz
        """
        assert self._trg_end >= i and i >= 0, "Der Index liegt nicht in der Definitionsmenge des Alignments."
        return self._map(i)

    def _map(self, i: int) -> int:
        """
        Diese Methode implementiert die Abbildungsvorschrift des Alignments. Sie muss von jeder Subklasse implementiert
        sein.

        :param i: i - Index eines Tokens im Targetsatz
        :return: b_i - Index eines Tokens im Sourcesatz
        """
        pass


class SimpleAlignment(Alignment):
    """
    Dies ist eine einfache Implementierung eines Alignments:

    Sofern i nicht der Index des '</s>' des Targetsatzes ist und i kleiner als der Index des '</s>' des Sourcesatzes ist,
    mapt das Alignment i auf i, sonst mapt das Alignment i auf den Index des '</s>' des Sourcesatzes.
    """
    def __init__(self, src_tokens: list, trg_tokens: list):
        Alignment.__init__(self, src_tokens, trg_tokens)

    def _map(self, i: int) -> int:
        if i == self._trg_end or i >= self._src_end:
            return self._src_end
        else:
            return i


class Alignment2(Alignment):
    """
    Sofern i nicht der Index des '</s>' des Targetsatzes ist, wird i wie folgt gemapt:

    1) Sind Target- und Sourcesatz gleich lang, so wird i auf i abgebildet
    2)  Wenn der Targetsatz länger ist als der Sourcesatz:
      2.1)  Für den Fall, dass der Sourcesatz mehr als doppelt so lang wie der Targetsatz ist, wird jeder Index vom
            Targetsatz auf jeden zweiten Index des Sourcesatzes gemapt.
      2.2) Der Index i wird auf sich selbst gemapt, falls der i kleiner als die Differenz vom Target- und Sourcesatz ist.
           Die restlichen Indizes des Targetsatzes werden jedem zweiten Index (ab dem letzten Index, der auf sich selbst
           abgebildet wurde) des Sourcesatzes zugewiesen.
           Dabei ist self._trg_length - self._length_diff der Index j, des zuletzt gemapten Index, welcher auf sich selbst
           abgebildet wird. Durch i - (self._trg_length - self._length_diff) + 1 wird die Position von i nach dem Index j
           beschrieben. Diese Position wird mit 2 multipliziert, um die Zuordnung auf jeden zweiten Index des Sourcesatzes
           zu realisieren.
           Da die Position mit 1 beginnt und nicht mit 0, muss am Ende 1 subtrahiert werden, um die richtige Position im
           Sourcesatz zu bekommen.
    3) Wenn der Targetsatz kürzer ist als der Sourcesatz:
      3.1) Ist der Targetsatz mehr als doppelt so lang wie der Sourcesatz, so werden zwei Indizes des Targetsatzes auf
           einen Index des Sourcesatzes abgebildet.
           Indizes, die keinem Index des Sourcesatzes zugeordnet werden, werden auf den Index des '</s>' des Targetsatzes
           abgebildet.
      3.2) Ansonsten wird zunächst i auf i abgebildet und die letzten 2 * self._length_diff Indizes des Targetsatzes
           werden auf die letzten self._length_diff Indizes des Sourcesatzes gemapt.
           Dabei wird jedem Index des Sourcesatzes zwei Indizes des Targetsatzes zugeordnet.
           Es beschreibt (self._src_length - self._length_diff) den Index j des Sourcesatzes, ab dem jeder Index zwei
           Indizes des Targetsatzes zugewiesen werden.
           Der Term i - (self._trg_length - 2 * self._length_diff) / 2 beschreibt die Position des Index nach j.
           Durch die Division durch 2 wird realisiert, dass zwei Indizes des Targetsatzes auf ein Index des Sourcesatzes
           abgebildet wird, ohne einen Index des Sourcesatzes zu überspringen.

    Der Index des </s> des Targetsatzes wird auf den Index des </s> des Sourcesatzes abgebildet.
    """
    def __init__(self, src_tokens: list, trg_tokens: list):
        Alignment.__init__(self, src_tokens, trg_tokens)
        self._src_length = len(self._src_tokens)
        self._trg_length = len(self._trg_tokens)
        self._length_diff = abs(self._src_length - self._trg_length)

    def _map(self, i: int) -> int:

        if i == self._trg_end:
            return self._src_end

        # 1)
        if self._src_length == self._trg_length:
            return i

        # 2)
        elif self._src_length > self._trg_length:
            # 2.1)
            if self._src_length > 2 * self._trg_length:
                return i * 2
            # 2.2)
            if i < self._trg_length - self._length_diff:
                return i
            else:
                return int((self._trg_length - self._length_diff) + (i - (self._trg_length - self._length_diff) + 1) * 2 - 1)

        # 3)
        elif self._src_length < self._trg_length:
            # 3.1)
            if 2 * self._src_length < self._trg_length:
                if i >= 2 * self._src_length:
                    return self._src_end

                if (self._trg_length - i) % 2 == 0:
                    return int(i / 2)
                else:
                    return int((i - 1) / 2)
            # 3.2)
            else:
                if i >= self._trg_length - 2 * self._length_diff:
                    if (self._trg_length - i) % 2 == 0:
                        return int((self._src_length - self._length_diff) + (i - (self._trg_length - 2 * self._length_diff)) / 2)
                    else:
                        return int((self._src_length - self._length_diff) + ((i - 1) - (self._trg_length - 2 * self._length_diff)) / 2)

                return i

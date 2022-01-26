import argparse
import data
import pandas as pd

UNKNOWN = "<UNK>"
START_OF_SEQUENCE = "<s>"
END_OF_SEQUENCE = "</s>"


class Dictionary:
    """
    Die Klasse realisiert die bidirektionale Zuweisung von Tokens und Indizes. Sie repraesentiert das Vokabular, das aus
    Trainingsdaten gewonnen wird.
    """

    def __init__(self, sentences: list = None):
        """
        Es wird ein Dictionary erzeugt, das zunaechts nur drei Eintraege fuer Spezialtokens enthaelt:
            UNKNOWN (Index 0)
            START_OF_SEQUENCE (Index 1)
            END_OF_SEQUENCE (Index 2)

        Weitere Eintraege koennen durch Uebergabe einer Liste an Saetzen hinzugefuegt werden, aus denen die Tokens
        extrahiert werden.

        Die Bidirektionalitaet wird durch Vorhaltung zweier redundanter dict-Objekte realisiert:
            _token_dict ordnet einem Token (str) einen Index (int) zu (Vergabe in der Reihenfolge der Einfuegung)
            _index_dict ordnet einem Index (int) ein Token (str) zu (Zuordnung zu _token_dict identisch)

        :param sentences: Liste der Saetze, deren Tokons initial hinzugefuegt werden sollen
        """
        self._token_dict = {}
        self._index_dict = {}

        self.add_token(UNKNOWN)
        self.add_token(START_OF_SEQUENCE)
        self.add_token(END_OF_SEQUENCE)

        if sentences is not None:
            for sentence in sentences:
                for token in sentence.split():
                    self.add_token(token)

    def add_token(self, token: str):
        """
        Diese Methode fuegt ein neues Token zum Vokabular hinzu. Ist es bereits vorhanden, geschieht nichts.

        Beim Einfuegen eines neuen Tokens wird diesem der naechste freie Index (len(_token_dict)) zugeordnet. Dieser
        Index wird zum Wert in _token_dict und zum Schluessel in index_dict.

        :param token: einzufuegendes Token
        """
        if token not in self._token_dict:
            index = len(self._token_dict)
            self._token_dict[token] = index
            self._index_dict[index] = token

    def get_index(self, token: str) -> int:
        """
        Diese Methode liefert den zu einem Token gehoerenden Index. Ist das Token nicht vorhanden, wird 0 (Index von
        UNKOWN) zurueckgegeben.

        :param token: Token, dessen Index bestimmt werden soll
        :return: Index des Tokens
        """
        return self._token_dict.get(token, 0)

    def get_token(self, index: int) -> str:
        """
         Diese Methode liefert das zu einem Index gehoerende Token. Ist der Index groesser als die Laenge des
         Dictionary, wird UNKNOWN zurueckgegeben.

         :param index: Index, dessen Token bestimmt werden soll
         :return: Token des Indexes
         """
        return self._index_dict.get(index, UNKNOWN)

    def get_size(self) -> int:
        """
         Diese Methode gibt die Groesse des Dictionary, also die Anzahl der gespeicherten Tokens, zurueck.

         :return: Groesse des Dictionary
         """
        return len(self._token_dict)

    def to_dataframe(self):
        x = list(self._index_dict.keys())
        y = list(self._index_dict.values())

        data = {'index': x, 'token': y}
        return pd.DataFrame(data=data)


if __name__ == '__main__':
    """
    Dieses Skript erzeugt ein Dictionary-Objekt und speichert es in einer Datei.

    (siehe Aufgabe 2.2)

    1)  Parse die uebergebenen Argumente.
    2)  Lade die Trainingsdaten vom uebergebenen Dateipfad und entfernen newline am Ende jeder Zeile.
    3)  Erzeuge das Dictionary-Objekt und speichere es in der uebergebenen Output-Datei.

    Beispielhafter Aufruf:

    python dictionary.py --data "<Pfad zu den Trainingsdaten>" --out "<Pfad zur Output-Datei>"

    """
    # 1)
    parser = argparse.ArgumentParser(prog="dictionary")
    parser.add_argument("--data", nargs=1, required=True, help="Pfad zur Datei, die Trainingsdaten beinhaltet.")
    parser.add_argument("--out", nargs=1, required=True, help="Pfad zur Output-Datei, die Dictionary-Objekt speichern soll.")
    args = parser.parse_args()

    # 2)
    if args.data[0].endswith(".gz"):
        text = data.read_gzip_text(args.data[0])
    else:
        text = data.read_txt(args.data[0])

    text = [sentence.strip() for sentence in text]

    # 3)
    dictionary = Dictionary(text)
    leng = dictionary.get_size()
    data.save_obj(args.out[0], dictionary)

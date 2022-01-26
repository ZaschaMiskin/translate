"""
In diesem Scriptfile sollen Funktionen definiert werden, die zum Laden und dem evtl. zugehoerigen Aufbereiten von
Daten verwendet werden. Die Funktionen sollen dabei allgemein gehalten werden, d.h. spezielle Parsings die nicht
dafuer gedacht sind evtl. von mehr als einem Programm verwendet zu werden, sollten dann in den jeweiligen Programm-Files
definiert sein. 
"""

import gzip
import pickle
import json


def read_json(path: str) -> object:
    with open(path, 'r') as file:
        json_obj = json.load(file)

    return json_obj


def read_gzip_text(path: str) -> list:
    """
    Diese Funktion liest eine gnuzip-komprimierte Textdatei ein und gibt die gelesenen Zeilen in einer Liste zurueck.

    :param path: Pfad zur Textdatei
    :return: Liste der gelesenen Zeilen
    """
    with gzip.open(path, 'rt', encoding="utf-8") as file:
        lines = file.readlines()
        file.close()

    lines = [x.strip() for x in lines]
    return lines


def read_txt(path: str) -> list:
    """
    Diese Funktion liest eine Textdatei ein und gibt die gelesenen Zeilen in einer Liste zurueck.

    :param path: Pfad zur Textdatei
    :return: Liste der gelesenen Zeilen
    """
    with open(path, 'r') as file:
        lines = file.readlines()
        file.close()

    lines = [x.strip() for x in lines]
    return lines


def load_obj(path: str) -> object:
    """
    Diese Funktion laedt ein Objekt aus einer Datei.

    :param path: Pfad zur Datei
    :return: Geladenes Objekt
    """
    with open(path, 'rb') as file:
        obj = pickle.load(file)
        file.close()

    return obj


def to_gzip_txt(path: str, lines: list):
    """
    Diese Funktion schreibt eine Liste von Strings als Zeilen in eine gzip-komprimierte Textdatei.
    Falls die Datei nicht existiert, dann wird sie erstellt, sonst wird sie ueberschrieben.

    :param path: Pfad zur Textdatei
    :param lines: Liste von Strings (anstatt Liste funktioniert auch String der '\n' beinhaltet)
    :return:
    """
    with gzip.open(path, 'wt', encoding="utf-8") as file:
        file.writelines(lines)
        file.close()


def to_txt(path: str, lines: list):
    """
    Diese Funktion schreibt eine Liste von Strings als Zeilen in eine Textdatei.
    Falls die Datei nicht existiert, dann wird sie erstellt, sonst wird sie ueberschrieben.

    :param path: Pfad zur Textdatei
    :param lines: Liste von Strings (anstatt Liste funktioniert auch String der '\n' beinhaltet)
    """
    with open(path, 'w', encoding="utf-8") as file:
        file.writelines(lines)
        file.close()


def save_obj(path: str, obj: object):
    """
    Diese Funktion speichert ein Objekt in einer Datei.

    :param path: Pfad zur Datei
    :param obj: Objekt welches gespeichert werden soll
    """
    with open(path, 'wb') as file:
        pickle.dump(obj, file)
        file.close()

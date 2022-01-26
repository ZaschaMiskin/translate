import argparse
import os
import tensorflow as tf
import numpy as np
import pandas as pd

import data
from dictionary import Dictionary
from batch import batch_format_lines, batch_format
from alignment import SimpleAlignment
from ffnn import ff_neural_network

# Fehlermeldungen entfernen
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)


def determine_model_score(network_function, net_config: dict, net_name: str, checkpoint: str,
                          src_text: list, trg_text: list, src_dict: Dictionary, trg_dict: Dictionary):
    """
    Diese Funktion ermittelt die scores und log(score)s eines Netzwerks auf den Saetzen von zwei Listen mit korrespondierenden
    Source- und Targetsaetzen.

    1)  Ueberfuehre Source- und Targettext ins (S,T,L)-Batchformat und gruppiere jeweils die Batchlines eines Satzes
        zu einer Liste/Batch.
    2)  Definiere den TensorFlow Computational Graph.
    3)  Beginne eine Session:
    4)  Lade die Modellparameter vom angegebenen Checkpoint-Pfad.
    5)  Fuer jeden Satz-Batch:
      5.1)  Uebergebe dem Netzwerk den Satz-Batch und fetche die Ausgaben des Softmaxlayers zum Batch.
      5.2)  Ermittle den log(score) des Satzes anhand der Ausgaben des Softmaxlayers.
      5.3)  Berechne score aus log(score) und fuege beide Werte in jeweilige Listen ein.
    6)  Gib die Listen mit score und log(score) zurueck

    :param network_function: netzwerk-erzeugende Funktion
    :param net_config: config-dict, welches die Netzwerkkonfiguration vorschreibt
    :param net_name: Name des Netzwerks
    :param checkpoint: Checkpoint-Pfad
    :param src_text: Liste mit Sourcesaetzen
    :param trg_text: Liste mit Targetsaetzen
    :param src_dict: Source-Dictionary
    :param trg_dict: Target-Dictionary
    :return (scores: list, log(scores): list): Tupel mit Listen die die scores und log(score)s der Uebersetzungen beinhalten.
    """
    window_size = net_config['window_size']

    # 1)
    print("[Preprocessing]: batch-formatting lines...", end='')
    batched_lines = batch_format_lines(src_text, trg_text, window_size, SimpleAlignment, src_dict, trg_dict)
    print("\r[Preprocessing]: batch-formatted lines.")

    # 2)
    src_windows_ph = tf.placeholder(dtype=tf.int32)
    trg_windows_ph = tf.placeholder(dtype=tf.int32)
    trg_labels_ph = tf.placeholder(dtype=tf.int32)

    projection, softmax = network_function(src_windows_ph, trg_windows_ph, False, net_config, net_name)

    saver = tf.train.Saver()

    log_scores = []
    scores = []

    # 3)
    with tf.Session() as sess:
        # 4)
        saver.restore(sess, checkpoint)
        print("[Checkpoint] Successfully restored model from path {0}.".format(checkpoint))

        # 5)
        for i, (src_windows, trg_windows, trg_labels) in enumerate(zip(*batched_lines)):
            # 5.1)
            sm = sess.run(
                fetches=softmax,
                feed_dict={src_windows_ph: src_windows, trg_windows_ph: trg_windows, trg_labels_ph: trg_labels}
            )

            # 5.2)
            log_score = 0
            for prob_dist, trg_label in zip(sm, trg_labels):
                log_score += np.log(prob_dist[trg_label])

            # 5.3)
            score = np.exp(log_score)
            log_scores.append(log_score)
            scores.append(score)

            print(f"[Line {i + 1}] score={score}, log(score)={log_score}")

    # 6)
    return scores, log_scores


def determine_additional_statistics(log_scores: list):
    """
    Diese Funktion erhaelt eine Liste von log(score)s und ermittelt dazu weitere Statistiken.

    :param log_scores: Liste von log(score)s die jeweils zu einem uebersaetzten Satz korrespondieren.
    :return: avg(score), avg(log(score)), sum(score), sum(log(score))
    """

    avg_log_score = np.mean(log_scores)
    avg_score = np.exp(avg_log_score)

    tot_log_score = np.sum(log_scores)
    tot_score = np.exp(tot_log_score)

    print(f"[Mean]: score={avg_score}, log(score)={avg_log_score}")
    print(f"[Total]: score={tot_score}, log(score)={tot_log_score}")

    return avg_score, avg_log_score, tot_score, tot_log_score


def _to_dataframe(scores: list, log_scores: list, general: bool = False):
    """
    Diese Funktion dient dazu ermittelte Daten tabellarisch darzustellen.

    :param scores: Liste von scores
    :param log_scores: Liste von log(scores)
    :param general: Ob die Tabelle avg und tot Wert anzeigt oder einzelne
    :return: Dataframe
    """
    if general:
        data = {
            'line': ['avg', 'tot'],
            'score': scores,
            'log(score)': log_scores
        }
    else:
        data = {
            'line': list(range(1, len(scores) + 1)),
            'score': scores,
            'log(score)': log_scores
        }

    dataframe = pd.DataFrame(data=data)
    return dataframe


if __name__ == '__main__':
    """
    Dieses Skript implementiert das Programm, welches in Aufgabe 4.1 gefordert wird.
    """

    parser = argparse.ArgumentParser(prog='score')
    parser.add_argument('model_checkpoint', nargs=1,
                        help='Pfad zum Checkpoint des Modells welches geladen werden soll.')
    parser.add_argument('net_config', nargs=2, help='Pfad zur Konfigurationsdatei des Netzwerks gefolgt vom Namen des '
                                                    'Netzwerks.')
    parser.add_argument('texts', nargs=2, help='Pfade zu Source- und Targettext (BPE-zerlegt).')
    parser.add_argument('dicts', nargs=2, help='Pfade zu Source- und Targetdictionary.')
    parser.add_argument('-out', '--outfile', nargs=1,
                        help='Pfade zu der Datei in die die Ergebnisse geschrieben werden sollen')
    args = parser.parse_args()

    checkpoint = args.model_checkpoint[0]
    net_config = data.read_json(args.net_config[0])[args.net_config[1]]
    net_name = args.net_config[1]
    src_text = data.read_gzip_text(args.texts[0])
    trg_text = data.read_gzip_text(args.texts[1])
    src_dict = data.load_obj(args.dicts[0])
    trg_dict = data.load_obj(args.dicts[1])

    network_function = ff_neural_network

    scores, log_scores = determine_model_score(network_function, net_config, net_name, checkpoint,
                                               src_text, trg_text, src_dict, trg_dict)

    avg_score, avg_log_score, tot_score, tot_log_score = determine_additional_statistics(log_scores)

    if args.outfile:
        dataframe_lines = _to_dataframe(scores, log_scores)
        dataframe_general = _to_dataframe([avg_score, tot_score], [avg_log_score, tot_log_score], general=True)

        string = dataframe_general.to_string(index=False) + "\n\n" + dataframe_lines.to_string(index=False)

        data.to_txt(args.outfile[0], string)
        print(f"[Score] wrote results into file at {args.outfile[0]}.")

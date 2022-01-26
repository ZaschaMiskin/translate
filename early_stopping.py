import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

import data

from metrics import bleu
from preproc import Dictionary
from ffnn import ff_neural_network
from search import greedy_search, beam_search

# Fehlermeldungen entfernen
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)


def dummy_search(src_sentences: list, trg_dict: Dictionary, window_size: int, checkpoint_path: str, net_config: dict,
                 net_name: str, network_function, reverse_bpe: bool, search_options: dict = {}) -> str:
    """
    Dies ist eine Dummy-Funktion. Alle richtigen Suchfunktionen müssen sich an die von ihr vorgegebene Signatur halten.

    :param src_sentences: Liste, die pro Satz des Sourcetexts eine Liste von Sourcewindows des Satzes beinhaltet
    :param trg_dict: Target-Dictionary
    :param window_size: Window-Size des Netzwerks
    :param checkpoint_path: Pfad zum zuladenden Checkpoint
    :param net_config: config-dict, welches die Netzwerkkonfiguration vorschreibt
    :param net_name: Name des Netzwerks
    :param network_function: Netzwerk-erzeugende Funktion
    :param reverse_bpe: Gibt an, ob die BPE-Zerlegung der Hypothese rueckgaengig gemacht werden soll
    :param search_options: dict mit weiteren Suchfunktions-spezifische Parameter
    :return: Liste von Hypothesen-Texten sortiert nach der Wahrscheinlichkeit der Texte, beginnend mit dem Wahrscheinlichsten
    """
    return [data.read_gzip_text(target)]


def determine_bleu_scores(checkpoint_dir, net_config, net_name, network_function, src_sentence_windows, target,
                          trg_dict, bleu_n, search_function, beam_size):
    """
    Diese Funktion berechnet die BLEU-Scores verschiedener Checkpoints eines Modells.

    1)  Fetche die Pfade zu allen Checkpoints die in dem checkpoint_dir Verzeichnis vorhanden sind.
    2)  Für jeden Checkpoint-Pfad:
      2.1)  Rufe die uebergebene Suchfunktion auf und erhalte von ihr eine Liste von Hypothesen-Texten sortiert nach
            der Wahrscheinlichkeit der Texte, beginnend mit dem Wahrscheinlichsten. Waehle daraufhin den
            wahrscheinlichsten Text aus.
      2.2)  Setze den TensorFlow Computational Graph zurueck, damit der Graph beim naechsten Schleifendurchlauf neu
            erstellt werden kann. (Dies wird nämlich von der Suchfunktion bei jedem Aufruf getan)
      2.3)  Ermittle den BLEU-Score des Hypothesentexts und speichere das Ergebnis in einer Liste
    3)  Gib die Liste der Checkpoint-Pfade mit der dazu korrespondierenden Liste der BLEU-Scores zurueck.

    :param checkpoint_dir: Pfad zu einem Verzeichnis, welches Checkpoints enthaelt.
    :param net_config: config-dict, welches die Netzwerkkonfiguration vorschreibt
    :param net_name: Name des Netzwerks
    :param network_function: netzwerk-erzeugende Funktion
    :param src_sentence_windows: Liste, die pro Satz des Sourcetexts eine Liste von Sourcewindows des Satzes beinhaltet
    :param target: Liste die die unbehandelten Targettextsaetze beinhaltet
    :param trg_dict: Target-Dictionary
    :param bleu_n: Groesse der maximalen zu ueberpruefenen n-grams bei Berechnung des BLEU-Scores
    :param search_function: Suchfunktion (sie muss die Signatur von dummy_search haben)
    :param beam_size: Bei Beam-Search verwendete Beam-Size.
    :return (checkpoints: list, bleu: list): Liste der Checkpoint-Pfade mit der dazu korrespondierenden Liste der BLEU-Scores
    """

    # TODO: Debug
    # src_sentence_windows = src_sentence_windows[:1000]
    # target = target[:1000]

    # 1)
    checkpoints = tf.train.get_checkpoint_state(checkpoint_dir).all_model_checkpoint_paths

    window_size = net_config['window_size']
    bleu_scores = []

    # 2)
    for checkpoint_path in checkpoints:
        checkpoint_name = checkpoint_path.split('\\')[-1]

        # 2.1)
        print(f"[Checkpoint {checkpoint_name}] Determining Hypotheses...", end="")
        hypotheses = search_function(src_sentence_windows, trg_dict, window_size, checkpoint_path, net_config,
                                     net_name, network_function, reverse_bpe=True,
                                     search_options={'output_num': 1, 'beam_size': beam_size})
        best_hypothesis = hypotheses[0]
        print(f"\r[Checkpoint {checkpoint_name}] Hypothesis text found.")
        # 2.2)
        tf.reset_default_graph()

        # 2.3)
        print(f"[Checkpoint {checkpoint_name}] Determining BLEU-Score...", end="")
        bleu_score = bleu(bleu_n, target, best_hypothesis)
        bleu_scores.append(bleu_score)
        print(f"\r[Checkpoint {checkpoint_name}] BLEU-Score is {bleu_score}.\n")

    # 3)
    return checkpoints, bleu_scores


def plot_bleu_perplexity(checkpoint_dir, net_config, net_name, network_function, src_sentence_windows, target,
                         trg_dict, bleu_n, search_function, summary_path, plot_out_path):

    checkpoints, bleu_scores = determine_bleu_scores(checkpoint_dir, net_config, net_name, network_function,
                                                     src_sentence_windows, target, trg_dict, bleu_n, search_function)

    # bleu_scores = [0.5, 0.5, 0.5, 0.5, 0.5]
    #
    # checkpoints = ['\\2019-05-30T12-27-58-4000', '2019-05-30T12-27-58-8000', '2019-05-30T12-27-58-12000',
    #                '2019-05-30T12-27-58-16000', '2019-05-30T12-27-58-17740']

    steps = [int(chk.split('-')[-1]) for chk in checkpoints]

    # summary_path = './approved-models/next-net-beta/logs/2019-05-30T12-27-58/valid/events.out.tfevents.1559212083.PHILIPPS-BEAST'

    perplexities = []

    for evt in tf.train.summary_iterator(summary_path):
        for val in evt.summary.value:
            step = evt.step
            if val.tag == 'perplexity/epoch' and step in steps:
                print(f"[{step}] {val.simple_value}")
                perplexities.append(val.simple_value)

    fig, ax1 = plt.subplots()

    ax1.plot(steps, bleu_scores)
    ax1.set_xlabel('step')
    ax1.set_ylabel('BLEU')

    ax2 = ax1.twinx()
    ax2.plot(steps, perplexities, color='red')
    ax2.set_ylabel('perplexity')

    time_stamp = checkpoints[0].split('\\')[-1].replace(f"-{steps[0]}", "")
    plt.title(f"{net_name} ({time_stamp})")

    if plot_out_path:
        plt.savefig(plot_out_path)

    plt.show()


if __name__ == '__main__':
    """
    Dieses Skript implementiert die automatisierte Methode die in Aufgabe 4.3 beschrieben ist.
    """

    parser = argparse.ArgumentParser(prog='early-stopping')
    parser.add_argument('checkpoint_dir', nargs=1,
                        help='Pfad zum Verzeichnis in dem die Checkpoints des Modells gespeichert sind.')
    parser.add_argument('net_config', nargs=2, help='Pfad zur Konfigurationsdatei des Netzwerks gefolgt vom Namen des '
                                                    'Netzwerks.')
    parser.add_argument('source', nargs=1,
                        help='Pfad zur Datei, welche den Sourcetext satzweise unterteilt in Source-Windows beinhaltet '
                             '(output von batch.py/source_windows_format)')
    parser.add_argument('target', nargs=1, help='Pfad zum Targettext (NICHT BPE-zerlegt).')
    parser.add_argument('dict', nargs=1, help='Pfad zum Targetdictionary.')
    parser.add_argument('-n', '--bleu_n', nargs=1, default=[4], type=int,
                        help='N fuer BLEU-Score: Groesse der maximalen zu ueberpruefenen n-grams.')
    parser.add_argument('-out', '--outfile', nargs=1,
                        help='Pfade zu der Datei in die die Ergebnisse geschrieben werden sollen.')
    parser.add_argument('-sea', '--search', nargs=1, default=["beam"],
                        help='Name des zuverwendenden Suchalgorithmus. (default="beam")')
    parser.add_argument('-beam', '--beam_size', nargs=1, type=int,
                        help='Beam-Size, sofern Beam-Search verwendet wird.')
    parser.add_argument('-plt', '--plot', nargs=1, help="Gibt an, ob die BLEU-Perplexity Beziehung geplottet werden soll."
                                                        " In diesem Fall muss ein Pfad zu einer tf.Summary Datei angegeben"
                                                        " werden.")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir[0]
    net_config = data.read_json(args.net_config[0])[args.net_config[1]]
    net_name = args.net_config[1]
    src_sentence_windows = data.load_obj(args.source[0])
    target = data.read_gzip_text(args.target[0])
    trg_dict = data.load_obj(args.dict[0])
    bleu_n = args.bleu_n[0]
    beam_size = args.beam_size[0]

    if args.search[0] == "beam":
        search_function = beam_search
    elif args.search[0] == "greedy":
        search_function = greedy_search
    elif args.search[0] == "dummy":
        search_function = dummy_search
    else:
        raise NameError(f"Es existiert kein Suchalgorithmus, der mit dem Namen "
                        f"'{args.search[0]}' in Verbindung gebracht werden kann.")

    if net_config['architecture'] == "exercise-3":
        network_function = ff_neural_network
    else:
        raise NameError(f"Es existiert keine Netzwerkarchitektur mit dem Namen {net_config['architecture']}.")

    if not args.plot:
        checkpoints, bleu_scores = determine_bleu_scores(checkpoint_dir, net_config, net_name, network_function,
                                                         src_sentence_windows, target, trg_dict, bleu_n,
                                                         search_function, beam_size)

        if args.outfile:
            dat = {
                'checkpoint': checkpoints,
                'BLEU': bleu_scores,
                'Beam_size': beam_size
            }
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr',
                                   False, 'display.max_colwidth', 1000):
                dataframe = pd.DataFrame(data=dat)
                data.to_txt(args.outfile[0], dataframe.to_string(index=False))

    else:
        summary_path = args.plot[0]
        plot_path = args.outfile[0] if args.outfile else None

        plot_bleu_perplexity(checkpoint_dir, net_config, net_name, network_function, src_sentence_windows, target,
                             trg_dict, bleu_n, search_function, summary_path, plot_path)

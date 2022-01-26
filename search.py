import argparse
from ffnn import ff_neural_network
from dictionary import Dictionary, END_OF_SEQUENCE, START_OF_SEQUENCE
from bpe import rev_bpe
import data
import tensorflow as tf
import numpy as np
import os

# Fehlermeldungen entfernen
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)


def to_words(sentence, dict: Dictionary):
    translation = ''

    for index in sentence:
        translation = translation + dict.get_token(index) + ' '

    return translation.strip()


def greedy_search(src_sentences: list, trg_dict: Dictionary, window_size: int, checkpoint_path, net_config, net_name,
                  network_function, reverse_bpe=1, search_options={}, src_trg_ratio=2):
    """
    Uebersetzt einen Text anhand von Greedy-Search. Dies entspricht einem Aufruf von Beam-Search mit Beam-Size=1.

    :param src_sentences: Liste der Source-Saetze. Jeder Source-Satz entspricht einer Liste, die die Source-Windows
                          enthaelt.
    :param trg_dict: Dictionary des Targetvokabulars
    :param window_size: Window Size
    :param checkpoint_path: Pfad zu einem Checkpoint des Modells
    :param net_config: Python-dict, welches die Netzwerk-Config Infos enthaelt
    :param net_name: Name des Netzwerks
    :param network_function: Netzwerkerzeugende Funktion
    :param reverse_bpe: Wenn True, wird die BPE-Zerlegung fuer die Uebersetzung rueckgaengig gemacht.
    :param search_options: Dict, dass die Anzahl der auszugebenden Hypothesen und die Beam-Size spezifiert.
    :return: Uebersetzten Text
    """

    return beam_search(src_sentences, trg_dict, window_size, checkpoint_path, net_config, net_name, network_function,
                reverse_bpe, {'output_num': 1, 'beam_size': 1}, src_trg_ratio)


def beam_search(src_sentences: list, trg_dict: Dictionary, window_size: int, checkpoint_path, net_config, net_name,
                network_function, reverse_bpe=1, search_options={'output_num': 4, 'beam_size': 5}, src_trg_ratio=2):
    """
    Uebersetzt einen Text anhand von Beam-Search.
    1) Lese Such-Optionen aus dem uebergebenem dict ein.
    2) Stelle sicher, dass nicht mehr Hypothesen ausgegeben werden sollen, als gespeichert werden.
    3) Erzeuge eine Liste fuer die uebersetzen Saetze. Die Saetze werden entsprechend der Rangfolge ihrer Scores in
       Unterlisten gruppiert. Daher gibt es output_num Unterlisten.
    4) Lade Start-of-Sequence- und End-of-Sequence-Symbole aus dem Target-Dictionary.
    5) Lege die maximale Laenge eines Target-Satzes fest.
    6) Erzeuge Tensoren, ueber die Windows in das Neuronale Netz eingebracht werden.
    7) Erzeuge den Graphen des Neuronales Netzes. Dabei muss spaeter nur auf die Ausgabe des Softmax-Layers, nicht des
       Projection-Layers zurueckgegriffen werden. An die Softmax-Ausgabe wird eine Operation angefuegt, die die k
       groessten (bei uns: beam-size groessten) Wahrscheinlichkeiten zuruckliefert. Prediction enthaelt dann eine Liste
       bestehend aus einem np-Array fuer die groessten Wahrscheinlichkeiten der Tokens und einem np-Array mit den
       zugehoerigen Indizes.
    8) Erzeuge ein Saver-Objekt, das zum Laden eines checkpoints dient.
    9) Erzeuge eine Operation, die die Variablen initialisiert.

    Innerhalb der Tensorflow-Session:
    10) Initialisiere die Variablen und stelle das trainierte Modell anhand des uebergebenen Checkpoints wieder her.
    11) Fuer jeden Satz der Source-Saetze (waehle jeweils eine Liste der Source-Windows aus):

        11.1) Setze die aktuelle Satzlaenge auf 0. Erzeuge Listen fuer die abgeschlossenen Hypothesen
        (finished_hypotheses) und die wachsenden Hypothesen (current_hypotheses). Initialisiere die wachsenden Hypothesen
         mit der Hypothese, die nur aus SOS besteht. Sie erhaelt den Score (laengennormalisierte log-probability) 0.
        11.2) Solange es noch wachsende Hypothesen gibt, und die maximale Satzlaenge nicht ueberschritten wurde:

            11.2.1) Lade das aktuelle Source-Window. Falls die Laenge des Satzes ueber die Anzahl der verfuegbaren
                    Source-Windows hinauswaechst: fuelle das letze Source-Window schrittweise mit EOS-Symbolen auf.
            11.2.2) Erzeuge eine leere Kandidaten-Liste. Als Kandidaten werden Sequenzen bezeichnet, die bei
                    ausreichend hohem Score zu den wachsende Hypothesen hinzugefuegt werden.
            11.2.3) Fuer alle wachsenden Hypothesen (halte Index vor):

                    11.2.3.1) Lade die Token-Sequenz und den Score der aktuellen Hypothese.
                    11.2.3.2) Erzeuge das Target-Window. Sollte die Hypothesen-Laenge noch geringer als die Window-Size
                              sein, dann Stelle an die Positionen vor der aktuellen Hypothese SOS-Symbole. Ansonsten
                              waehle die letzten window-size Indizes der Hypothese.
                    11.2.3.3) Uebergebe dem Modell die aktuellen Windows. Zurueckgegeben werden die beam-size groessten
                              durch das NN berechneten Wahrscheinlichkeiten fuer einzelne Tokensund die zugehoerigen
                              Indizes.
                    11.2.3.4) Berechne alle Kandidaten fuer diese Iteration. Iteriere dabei ueber die beam-size
                              berechneten top-Indizes. Haenge dabei an jede vorhandene Hypothese den neuen Index an.
                              Da im log-Space gerechnet wird, wird die neue logarithmierte Wahrscheinlichkeit zur
                              logarithmierten Wahrscheinlichkeit der Hypothese (score) addiert. Dabei wird bezueglich
                              der Satzlaenge normalisiert. Da die Wahrscheinlichkeiten <= 1 sind,
                              handelt es sich bei den Scores um Werte <= 0.

            11.2.4) Sortiere die berechneten Kandidaten aufsteigend bezueglich ihrer Scores. Ersetze die bisher
                    wachsenden Hypothesen durch die Kandidaten mit den groessten Scores. Waehle in der ersten Iteration
                    beam_size Kandidaten aus, darauffolgend nur noch so viele Kandidaten wie wachsende Hypothesen.
            11.2.5) Lege eine Liste fuer Hypothesen an, die in dieser Iteration beendet wurden, bei denen also ein
                    EOS-Token ausgewaehlt wurde. Speichere diese Hypothesen in der Liste und fuege sie ausserdem zur
                    globalen Liste finished_hypotheses hinzu.
            11.2.6) Entferne alle in dieser Iteration beendeten Hypothesen aus den aktuell wachsenden Hypothesen.
            11.2.7) Erhoehe den Zaehler fuer die aktuelle Satzlaenge um 1.

        11.3) Gebe den aktuell uebersetzen Satz aus.
        11.4) Iteriere ueber die sortierten beendeten Hypothesen des aktuellen Satzes. Waehle dabei nur die output_num
              besten aus.

              11.4.1) Entferne das SOS-Symbol vom Anfang des Satzes. Ueberfuehre den Satz von der Index- in die
                      Token-Repraesentation.
              11.4.2) Falls die Option reverse_bpe ausgewaehlt wurde: Mache die BPE-Zerlegung rueckgaengig.
              11.4.3) Entferne moegliche newline-Zeichen, die durch rev_bpe() hinzugefuegt wurden.
              11.4.4) Fuege die Hypothese zur i-ten Liste (i entspricht dem Hypothesen Rang entsprechend Score)
                      der trg_texts hinzu.
              11.4.5) Gebe die aktuelle Hypothese in ihrer Token-Repraesentation aus.


    :param src_sentences: Liste der Source-Saetze. Jeder Source-Satz entspricht einer Liste, die die Source-Windows
                          enthaelt.
    :param trg_dict: Dictionary des Targetvokabulars
    :param window_size: Window Size
    :param checkpoint_path: Pfad zu einem Checkpoint des Modells
    :param net_config: Python-dict, welches die Netzwerk-Config Infos enthaelt
    :param net_name: Name des Netzwerks
    :param network_function: Netzwerkerzeugende Funktion
    :param reverse_bpe: Wenn True, wird die BPE-Zerlegung fuer die Uebersetzung rueckgaengig gemacht.
    :param search_options: Dict, dass die Anzahl der auszugebenden Hypothesen und die Beam-Size spezifiert.
    :return: Uebersetzten Text
    """
    # 1)
    output_num = search_options['output_num']
    beam_size = search_options['beam_size']

    # 2)
    assert beam_size >= output_num, "Beam-size must be greater than or equal to output_num"

    # 3)
    trg_texts = [[] for i in range(output_num)]

    # 4)
    sos_index = trg_dict.get_index(START_OF_SEQUENCE)
    eos_index = trg_dict.get_index(END_OF_SEQUENCE)

    # 5)
    ratio = src_trg_ratio

    # 6)
    next_src_window = tf.placeholder(dtype='int32', name='source')
    next_trg_window = tf.placeholder(dtype='int32', name='target')

    # 7)
    _, softmax = network_function(next_src_window, next_trg_window, False, net_config, net_name)
    prediction = tf.math.top_k(softmax, k=beam_size, name='prediction')

    # 8)
    saver = tf.train.Saver()

    # 9)
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        # 10)
        sess.run(init_op)
        saver.restore(sess, checkpoint_path)

        # 11)
        for num, src_sentence in enumerate(src_sentences, 1):

            # 11.1)
            i = 0
            i_max = np.ceil(len(src_sentence) * ratio * 1.5)
            finished_hypotheses = []
            current_hypotheses = [[[sos_index], 0]]

            # 11.2)
            while len(current_hypotheses) > 0 and i <= i_max:

                # 11.2.1)
                if i < len(src_sentence) - 1:
                    src_window = src_sentence[i]
                else:
                    src_window = np.delete(src_window, 0)
                    src_window = np.append(src_window, eos_index)

                # 11.2.2)
                candidates = []

                # 11.2.3)
                for hyp_nr in range(len(current_hypotheses)):

                    # 11.2.3.1)
                    hypothesis, score = current_hypotheses[hyp_nr]

                    # 11.2.3.2)
                    if len(hypothesis) < window_size:
                        target_window = np.full(window_size - len(hypothesis), sos_index)
                        target_window = np.append(target_window, hypothesis)
                    else:
                        target_window = hypothesis[-window_size:]

                    # 11.2.3.3)
                    top_values, top_indices = sess.run(
                        fetches=prediction,
                        feed_dict={next_src_window: src_window, next_trg_window: target_window})

                    # 11.2.3.4)
                    for j in range(beam_size):
                        candidate = [hypothesis + [top_indices[0][j]], (score * i + np.log(top_values[0][j])) / (i + 1)]
                        candidates.append(candidate)

                # 11.2.4)
                ordered_candidates = sorted(candidates, key=lambda el: el[1])
                current_hypotheses = ordered_candidates[-(beam_size if i == 0 else len(current_hypotheses)):]

                # 11.2.5)
                recently_finished = []
                for l, hyp in enumerate(current_hypotheses):
                    if hyp[0][-1] == eos_index:
                        recently_finished.append(l)
                        finished_hypotheses.append([hyp[0][:-1], hyp[1]])

                # 11.2.6)
                current_hypotheses = [el for i, el in enumerate(current_hypotheses) if i not in recently_finished]

                # 11.2.7)
                i += 1

            # 11.3)
            if beam_size == 1:
                print(f"\r[Greedy Search] Translated sentence {num}/{len(src_sentences)}.", end='\n')
            else:
                print(f"\r[Beam Search] Translated sentence {num}/{len(src_sentences)}.", end='\n')

            # 11.4)
            for i, (hypothesis, score) in enumerate(
                    sorted(finished_hypotheses, key=lambda el: el[1], reverse=True)[0:output_num]):

                # 11.4.1)
                hypothesis = hypothesis[1:]
                tokenized_hypothesis = to_words(hypothesis, trg_dict)

                # 11.4.2)
                if reverse_bpe:
                    tokenized_hypothesis = rev_bpe([tokenized_hypothesis])[0]

                # 11.4.3)
                tokenized_hypothesis = tokenized_hypothesis.strip()

                # 11.4.4)
                trg_texts[i].append(tokenized_hypothesis)

                # 11.4.5)
                print('   {0}    (Score: {1})'.format(tokenized_hypothesis, score), end='\n')

    return trg_texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='search')

    parser.add_argument('-s', '--source', nargs=1, default=None, help='Pfad zur Objectdatei, die Source-Windows '
                                                                      'enthaelt')

    parser.add_argument('-c', '--checkpoint', nargs=1, default=None, help='Pfad zum Checkpoint-file')

    parser.add_argument('-d', '--dict', nargs=1, default=None, help='Pfad zum Target-Dictionary')

    parser.add_argument('-norev', '--reverse', default=True, action='store_false',
                        help='Gibt an, ob die BPE Zerlegung rückgängig gemacht werden soll')

    parser.add_argument('net_config', nargs=2, help='Pfad zur Konfigurationsdatei des Netzwerks gefolgt vom Namen des '
                                                    'Netzwerks.')

    parser.add_argument('-beam', '--beam_size', nargs=1, type=int,
                        help='Gibt an, ob die BPE Zerlegung rückgängig gemacht werden soll')

    parser.add_argument('-out', '--output_num', nargs=1, type=int,
                        help='Gibt an, ob die BPE Zerlegung rückgängig gemacht werden soll')

    parser.add_argument('-ratio', '--src_trg_ratio', nargs=1, default=None,
                        help='Optionale Pfad zu Objektdatei, die source-target-ratio enthaelt (Verhaeltniss von '
                             'mittlerer Source-Satz-Laenge zu mittlerer Target-Satz-Laenge der Trainingsdaten zur '
                             'Vorhersage der Target-Satz-Laenge bei der Suche)')

    parser.add_argument("-trans", "--output_file", nargs=1, default=None, help='Pfad, in dem der uebersetzte Text '
                                                                             'gespeichert werden soll' )

    args = parser.parse_args()

    src_sentences = data.load_obj(args.source[0])

    checkpoint_path = args.checkpoint[0]

    trg_dict = data.load_obj(args.dict[0])

    reverse_bpe = args.reverse

    net_config = data.read_json(args.net_config[0])[args.net_config[1]]
    net_name = args.net_config[1]

    if net_config['architecture'] == 'exercise-3':
        network_function = ff_neural_network
        window_size = net_config['window_size']

    beam_size = args.beam_size[0]

    output_num = args.output_num[0]

    if args.src_trg_ratio:
        src_trg_ratio = float(data.load_obj(args.src_trg_ratio[0]))
    else:
        src_trg_ratio = 2

    trg_text = beam_search(src_sentences, trg_dict, window_size, checkpoint_path, net_config, net_name, network_function,
                reverse_bpe, {'output_num': output_num, 'beam_size': beam_size}, src_trg_ratio)

    if args.output_file:
        output_path = args.output_file[0]
        output_path = output_path.split('/')

        output_end = output_path[-1]

        if beam_size == 1:
            if reverse_bpe:
                output_end = 'rev_bpe-greedy-' + output_end
            else:
                output_end = 'greedy-' + output_end
        else:
            if reverse_bpe:
                output_end = 'rev_bpe-beam-{0}-out-{1}-'.format(beam_size, output_num) + output_end
            else:
                output_end = 'beam-{0}-out-{1}-'.format(beam_size, output_num) + output_end

        output_path = '/'.join(output_path[:-1] + [output_end])

        translation = [(sentence + '\n') for sentence in trg_text[0]]

        data.to_gzip_txt(output_path, translation)

        print('\nTranslanted text saved in {0}'.format(output_path))

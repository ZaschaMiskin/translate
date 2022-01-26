import data
import argparse
from dictionary import Dictionary
from collections import defaultdict
from pandas import DataFrame
import pandas as pd

END_OF_TOKEN = '</w>'


def _get_token_dic(text: list) -> dict:
    """
    Diese Funktion initialisiert das Token-Dicionary. Sie zerlegt jeden Token des Textes in einzelne Buchstaben und
    haengt dem letzten Zeichen eines jeden Tokens ein END_OF_TOKEN an. Im Token-Dictionary werden dann zum einen die
    zerlegten Tokens als Schluessel gespeichert und zum anderen die Haeufigleit des jeweiligen Tokens als Wert zum
    Schluessel.

    :param text: Liste mit Saetzen des Trainingstexts
    :return: Token-Dictionary
    """
    token_dic = defaultdict(int)

    for line in text:
        for token in line.split():
            split_token = ' '.join(list(token)) + END_OF_TOKEN
            token_dic[split_token] += 1

    return token_dic


def _get_pair_frequencies(token_dic: dict) -> dict:
    """
    Diese Funktion erstellt ein Dictionary in dem zu jedem nebeneinanderstehenden Subtoken-Paar der aktuellen
    Tokenzerlegung die Haeufigkeit des jeweiligen Paars gespeichert wird.

    1)  Erstelle ein defaultdict, dessen Werte Integer sind. Besonderheit des defaultdicts ist, dass es keine KeyError
        wirft, wenn auf Werte von bisher unbenutzen Schluesseln zugegriffen wird.
    2)  Fuer jeden Token im Token-Dictionary und dessen Hauefigkeit im Text:
      2.1)  Zaehle die Haeufigkeit jedes Subtoken-Paars des Tokens in seiner aktuellen Zerlegung um die Hauefigkeit des
            Tokens im Text hoch.

    :param token_dic: Token-Dict mit Tokens in aktueller Zerlegung und deren Haeufigkeit im Text
    :return:
    """
    # 1)
    pair_freqs = defaultdict(int)

    # 2)
    for token, freq in token_dic.items():
        subwords = token.split()
        for i in range(len(subwords) - 1):
            # 2.1)
            pair_freqs[subwords[i], subwords[i + 1]] += freq

    return pair_freqs


def _update_token_dic(token_dic: dict, merge_pair: tuple) -> dict:
    """
    Diese Funktion erstellt ein aktualisiertes Token-Dictionary auf basis eines bereits bestehenden Token-Dictionarys
    und einem Subtoken-Paar was in jedem Token - in dessen aktuellen Zerlegung - zusammengefuehrt werden soll.

    1)  Zunaechst werden hier Muster-Strings erstellt: old_pat representiert das zuzusammenfuehrenden Token-Paar in
        seiner aktuellen Form und new_pat representiert das Token-Paar in seiner zusammengefuehrten Form.
    2)  Fuer jeden Token im Token-Dictionary wird nun der Substring, welcher auf old_pat matched mit new_pat ersaetzt.
        Dem Token wird vorher aber noch vorne und hinten ein Leerzeichen angehaengt, damit die Muster-Strings auch mit
        Subtokens matchen, welche am Rand des Token-String liegen und somit vorne bzw. hinten kein Leerzeichen haben.

    :param token_dic: Token-Dict mit Tokens in aktueller Zerlegung und deren Haeufigkeit im Text
    :param merge_pair: Subtoken-Paar, welches zusammengefuehrt werden soll
    :return: Aktualisiertes Token-Dictionary
    """
    new_token_dic = defaultdict(int)

    # 1)
    old_pat = " {0} {1} ".format(merge_pair[0], merge_pair[1])
    new_pat = " {0}{1} ".format(merge_pair[0], merge_pair[1])

    # 2)
    for token in token_dic:
        new_token = ((" " + token + " ").replace(old_pat, new_pat)).strip()
        new_token_dic[new_token] = token_dic[token]

    return new_token_dic


def learn_bpe(text: list, ops: int):
    """
    Diese Funktion die BPE Zerlegung eines Traingstexts anhand einer gegebenen Anzahl an Zusammenzugs-Operationen.

    1)  Initialisiere ein auf dem Trainingstext basierendes Token-Dictionary, welches als Schluessel die Tokens des
        Textes in Zeichen zerlegt und als Wert die Anzahl der jeweiligen Tokens im Text hat.
    2)  Fuer jede auszufuehrende Zusammenfuehrungsoperation:
      2.1)  Zaehle die Anzahl der jeweiligen nebeneinanderstehenden Subtoken-Paare in der aktuellen Zerlegung der Tokens
      2.2)  Waehle das Subtoken-Paar aus, welches am Haeufigsten vorkommt
      2.3)  Aktualisiere das Token-Dictionary indem das Subtoken-Paar in jeder aktuellen Tokenzerlegung zusammengefuehrt
            wird.

    :param text: Liste der Saetze des Trainingstexts
    :param ops: Anzahl der Zusammenzugs-Operationen
    :return: Liste der Zusammenzugs-Operationen (vgl. Slides 2, Folie 17)
    """
    # 1)
    token_dic = _get_token_dic(text)
    bpe_pairs = []
    # 2)
    for i in range(ops):
        # 2.1)
        pair_freqs = _get_pair_frequencies(token_dic)
        # 2.2)
        most_frequent_pair = max(pair_freqs, key=pair_freqs.get)
        # 2.3)
        token_dic = _update_token_dic(token_dic, most_frequent_pair)

        print("\rBPE Operation: {}/{}".format(i + 1, ops), end='')

        bpe_pairs.append(most_frequent_pair)

    return bpe_pairs


def bpe(text: list, bpe_ops: list):
    """

    Diese Funktion wendet die erlernte BPE Zerlegung an einem Text an.

    1)  Initialisiere Text: Teile ihn dazu in Token auf der Größe 1 und füge am Ende jeden Wortes ein END_OF_TOKEN hinzu
        bpe_text[i] ist eine Liste von Wörtern im Satz i, die tokenisiert wurden
    2)  Für jeden Satz aus bpe_text:
        2.1)  Satz wird zusammengefügt
        2.2)  Für jedes gelernte Paar:
            2.2.1)  Prüfe, ob das Paar im Satz vorkommt
            2.2.2)  Falls ja, merge alle gleichen Token-Paare im Satz zu jeweils einem
    3)  Falls kein Ausgabepfad angegeben wurde, gebe den Text in der Konsole aus, sonst im angegebenen Dateipfad

    :param text: Liste von Sätzen des Textes
    :param path: Optionaler Pfad, um den ergebenen Text auszugeben
    :return:
    """
    # 1)
    bpe_text = []
    for line in text:
        out = []
        for token in line.split():
            split_token = ' '.join(list(token)) + END_OF_TOKEN
            out.append(split_token)
        bpe_text.append(out)
    # 2)
    number_of_lines = len(bpe_text)

    bpe_map = {}

    for i, line in enumerate(bpe_text):
        print("\rBPE on sentence: {}/{}".format(i + 1, number_of_lines), end='')
        for j, token in enumerate(line):

            if token in bpe_map:
                bpe_token = bpe_map[token]
            else:
                bpe_token = token

                for pair in bpe_ops:
                    old_pat = " {0} {1} ".format(pair[0], pair[1])
                    new_pat = " {0}{1} ".format(pair[0], pair[1])

                    bpe_token = (" " + bpe_token + " ").replace(old_pat, new_pat).strip()

                bpe_map[token] = bpe_token

            line[j] = bpe_token

        bpe_text[i] = " ".join(line) + '\n'

    return bpe_text


def rev_bpe(text: list):
    """

    Dies Funktion macht eine BPE Zerlegung rückgängig

    1)  Fuer jeden Satz:
        1.1)  Füge die einzelnen Wörter so zusammen, dass zwischen Ihnen keine Leerzeichen sind
        1.2)  Teile die Wörter an den Trennzeichen("</w>"). Diese werden dabei entfernt
        1.3)  Für jedes Wort: Entferne die Leerzeichen innerhalb des Wortes, die die Tokens voneineander trennen
        1.4)  Füge die Liste von Wörtern wieder zusammen (und trenne sie mit leerzeichen voneinander)

    :param path: Optionaler Pfad, um den ergebenen Text auszugeben
    :param text: Liste von Sätzen des Textes in BPE Zerlegung
    :return: Verwandelter Text
    """

    # 1)
    for s_index, line in enumerate(text):
        # 1.1)
        sen = ''.join(line)
        # 1.2)
        words = sen.split('</w>')
        for w_index in range(len(words)):
            word = words[w_index]
            # 1.3)
            word = word.replace(' ', '')
            words[w_index] = word
        # 1.4)
        text[s_index] = ' '.join(words) # + '\n'

    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="bpe",
                                     usage="%(prog)s [-h]\n[-evl/--evaluate <path to source text> <path to target text> "
                                           "<src ops path prefix> <trg ops path prefix> <joint ops path prefix> "
                                           "<src bpe text path prefix> <trg bpe text path prefix> <joint bpe text path prefix> "
                                           "<skip training> <skip apply> <list of number of ops> <path to stats file>]\n"
                                           "[-l/--learn <path to text> <path to bpe ops> <number of ops>]\n"
                                           "[-lj/--learn_joint ""<path to src text> <path to trg text> <path to bpe ops> <number of ops>]\n"
                                           "[-apl/--apply <path to text> <path to bpe ops> <path to bpe text>]\n"
                                           "[-rev/--reverse <path to src text> <path to trg text>]")
    parser.add_argument("-evl", "--evaluate", nargs=12, help="Argumente fuer den Evaluierungs-Modus")
    parser.add_argument("-l", "--learn", nargs=3, help="Argumente fuer den Learn-Separated-Modus")
    parser.add_argument("-lj", "--learn_joint", nargs=4, help="Argumente fuer den Learn-Joint-Modus")
    parser.add_argument("-apl", "--apply", nargs=3, help="Argumente fuer den Apply-Modus")
    parser.add_argument("-rev", "--reverse", nargs=2, help="Argumente fue den Reverse_Modus")
    args = parser.parse_args()

    if args.reverse:
        text_path = args.reverse[0]
        target_path = args.reverse[1]

        print("Start reversing BPE subword partitioning on data txt='{0}'...".format(text_path))
        text = rev_bpe(data.read_gzip_text(text_path))
        print("Done reversing BPE subword partitioning on data txt='{0}'...".format(text_path))
        print("Saving text to '{0}'.".format(target_path))
        data.to_gzip_txt(target_path, text)

    if args.learn:
        text_path = args.learn[0]
        bpe_ops_path = args.learn[1]
        number_of_ops = eval(args.learn[2])

        text = data.read_gzip_text(text_path)

        print("Start training BPE subword partitioning on data txt='{0}' with nops={1} ..."
              .format(text_path, number_of_ops))
        bpe_ops = learn_bpe(text, number_of_ops)
        print("\rDone training BPE subword partitioning on data txt='{0}' with nops={1}."
              .format(text_path, number_of_ops))

        print("Saving separated BPEs to '{0}'.".format(bpe_ops_path))
        data.save_obj(bpe_ops_path, bpe_ops)

    if args.learn_joint:
        src_text_path = args.learn_joint[0]
        trg_text_path = args.learn_joint[1]
        bpe_ops_path = args.learn_joint[2]
        number_of_ops = eval(args.learn_joint[3])

        source = data.read_gzip_text(src_text_path)
        target = data.read_gzip_text(trg_text_path)
        joint = source + target

        print("Start training BPE subword partitioning on joint data src='{0}' and trg='{1}' with nops={2} ..."
              .format(src_text_path, trg_text_path, number_of_ops))
        bpe_ops = learn_bpe(joint, number_of_ops)
        print("\rDone training BPE subword partitioning on joint data src='{0}' and trg='{1}' with nops={2}."
              .format(src_text_path, trg_text_path, number_of_ops))

        print("Saving joint BPEs to '{0}'.".format(bpe_ops_path))
        data.save_obj(bpe_ops_path, bpe_ops)

    if args.apply:
        text_path = args.apply[0]
        bpe_ops_path = args.apply[1]
        bpe_text_path = args.apply[2]

        print("Loading txt='{0}' and BPE operations from '{1}' ...".format(text_path, bpe_ops_path), end='')
        text = data.read_gzip_text(text_path)
        bpe_ops = data.load_obj(bpe_ops_path)
        print("\rLoaded txt='{0}' and BPE operations from '{1}'.".format(text_path, bpe_ops_path))

        print("Start applying BPE on txt='{0}' ...".format(text_path))
        bpe_text = bpe(text, bpe_ops)
        print("\rDone applying BPE on txt='{0}'.".format(text_path))

        print("Saving BPE text to '{0}'.".format(bpe_text_path))
        data.to_gzip_txt(bpe_text_path, bpe_text)

    if args.evaluate:
        src_text_path = args.evaluate[0]
        trg_text_path = args.evaluate[1]

        src_bpe_ops_prefix = args.evaluate[2]
        trg_bpe_ops_prefix = args.evaluate[3]
        joint_bpe_ops_prefix = args.evaluate[4]

        src_bpe_text_prefix = args.evaluate[5]
        trg_bpe_text_prefix = args.evaluate[6]
        joint_bpe_text_prefix = args.evaluate[7]

        skip_training = eval(args.evaluate[8])
        skip_apply = eval(args.evaluate[9])
        evl_nops = eval(args.evaluate[10])
        stats_path = args.evaluate[11]

        source = data.read_gzip_text(src_text_path)
        target = data.read_gzip_text(trg_text_path)

        if not skip_training:
            print("Train separated BPE:\n")

            for nops in evl_nops:
                print("Start training BPE subword partitioning on data src='{0}' with nops={1} ..."
                      .format(src_text_path, nops))
                src_bpe_ops = learn_bpe(source, nops)
                print("\rDone training BPE subword partitioning on data src='{0}' with nops={1}."
                      .format(src_text_path, nops))

                print("Start training BPE subword partitioning on data trg='{0}' with nops={1} ..."
                      .format(trg_text_path, nops))
                trg_bpe_ops = learn_bpe(target, nops)
                print("\rDone training BPE subword partitioning on data trg='{0}' with nops={1}."
                      .format(trg_text_path, nops))

                src_path = src_bpe_ops_prefix + ".{0}ops.bpe".format(nops)
                trg_path = trg_bpe_ops_prefix + ".{0}ops.bpe".format(nops)

                print("Saving separated BPEs to src='{0}' and trg='{1}'.\n".format(src_path, trg_path))

                data.save_obj(src_path, src_bpe_ops)
                data.save_obj(trg_path, trg_bpe_ops)

            print("Train joint BPE:\n")

            joint_text = source + target

            for nops in evl_nops:
                print("Start training BPE subword partitioning on joint data src='{0}' and trg='{1}' with nops={2} ..."
                      .format(src_text_path, trg_text_path, nops))
                joint_bpe_ops = learn_bpe(joint_text, nops)
                print("\rDone training BPE subword partitioning on joint data src='{0}' and trg='{1}' with nops={2}."
                      .format(src_text_path, trg_text_path, nops))

                joint_path = joint_bpe_ops_prefix + ".{0}ops.bpe".format(nops)

                print("Saving joint BPEs to '{0}'.\n".format(joint_path))

                data.save_obj(joint_path, joint_bpe_ops)

        if not skip_apply:

            print("Applying separated BPE:\n")

            for nops in evl_nops:
                src_path = src_bpe_ops_prefix + ".{0}ops.bpe".format(nops)
                trg_path = trg_bpe_ops_prefix + ".{0}ops.bpe".format(nops)

                print("Loading BPE operations from src='{0}' and trg='{1}' ...".format(src_path, trg_path), end='')
                src_bpe_ops = data.load_obj(src_path)
                trg_bpe_ops = data.load_obj(trg_path)
                print("\rLoaded BPE operations from src='{0}' and trg='{1}'.".format(src_path, trg_path))

                print("Start applying BPE on src='{0}' ...".format(src_text_path))
                src_bpe_text = bpe(source, src_bpe_ops)
                print("\rDone applying BPE on src='{0}'.".format(src_text_path))
                print("Start applying BPE on trg='{0}' ...".format(trg_text_path))
                trg_bpe_text = bpe(target, trg_bpe_ops)
                print("\rDone applying BPE on trg='{0}'.".format(trg_text_path))

                src_bpe_text_path = src_bpe_text_prefix + ".{0}ops.bpe.txt.gz".format(nops)
                trg_bpe_text_path = trg_bpe_text_prefix + ".{0}ops.bpe.txt.gz".format(nops)

                print("Saving BPE texts to src='{0}' and trg='{1}'.\n".format(src_bpe_text_path, trg_bpe_text_path))
                data.to_gzip_txt(src_bpe_text_path, src_bpe_text)
                data.to_gzip_txt(trg_bpe_text_path, trg_bpe_text)

            print("Applying joint BPE:\n")

            for nops in evl_nops:
                joint_path = joint_bpe_ops_prefix + ".{0}ops.bpe".format(nops)

                print("Loading joint BPE operations from '{0}' ...".format(joint_path), end='')
                joint_bpe_ops = data.load_obj(joint_path)
                print("Loaded joint BPE operations from '{0}'.".format(joint_path))

                print("Start applying joint BPE on src='{0}' ...".format(src_text_path))
                src_bpe_text = bpe(source, joint_bpe_ops)
                print("\rDone applying joint BPE on src='{0}'.".format(src_text_path))
                print("Start applying joint BPE on trg='{0}' ...".format(trg_text_path))
                trg_bpe_text = bpe(target, joint_bpe_ops)
                print("\rDone applying joint BPE on trg='{0}'.".format(trg_text_path))

                src_bpe_text_path = src_bpe_text_prefix + ".{0}ops.joint-bpe.txt.gz".format(nops)
                trg_bpe_text_path = trg_bpe_text_prefix + ".{0}ops.joint-bpe.txt.gz".format(nops)

                print("Saving BPE texts to src='{0}' and trg='{1}'.\n".format(src_bpe_text_path, trg_bpe_text_path))
                data.to_gzip_txt(src_bpe_text_path, src_bpe_text)
                data.to_gzip_txt(trg_bpe_text_path, trg_bpe_text)

        print("Evaluating BPE:\n")

        col_nops = []
        col_src = []
        col_trg = []
        col_joint_src = []
        col_joint_trg = []

        for nops in evl_nops:
            src_bpe_text_path = src_bpe_text_prefix + ".{0}ops.bpe.txt.gz".format(nops)
            trg_bpe_text_path = trg_bpe_text_prefix + ".{0}ops.bpe.txt.gz".format(nops)
            src_joint_bpe_text_path = src_bpe_text_prefix + ".{0}ops.joint-bpe.txt.gz".format(nops)
            trg_joint_bpe_text_path = trg_bpe_text_prefix + ".{0}ops.joint-bpe.txt.gz".format(nops)

            print("Loading BPE texts for nops={0} ...".format(nops), end='')
            src_bpe_text = data.read_gzip_text(src_bpe_text_path)
            trg_bpe_text = data.read_gzip_text(trg_bpe_text_path)
            src_joint_bpe_text = data.read_gzip_text(src_joint_bpe_text_path)
            trg_joint_bpe_text = data.read_gzip_text(trg_joint_bpe_text_path)
            print("\rLoaded BPE texts for nops={0}.".format(nops))

            print("Creating Dictionaries for BPE texts for nops={0} ...".format(nops), end='')
            src_bpe_dict = Dictionary(src_bpe_text)
            trg_bpe_dict = Dictionary(trg_bpe_text)
            src_joint_bpe_dict = Dictionary(src_joint_bpe_text)
            trg_joint_bpe_dict = Dictionary(trg_joint_bpe_text)
            print("\rCreated Dictionaries for BPE texts for nops={0}.".format(nops))

            print("Gathering vocab stats for nops={0}.\n".format(nops))
            col_nops.append(nops)
            col_src.append(src_bpe_dict.get_size())
            col_trg.append(trg_bpe_dict.get_size())
            col_joint_src.append(src_joint_bpe_dict.get_size())
            col_joint_trg.append(trg_joint_bpe_dict.get_size())

        df = DataFrame(data={
            "number of operations": col_nops,
            "src vocab size": col_src,
            "trg vocab size": col_trg,
            "src vocab size (joint ops)": col_joint_src,
            "trg vocab size (joint ops)": col_joint_trg
        })

        print("Vocab stats:\n")

        src_dict = Dictionary(source)
        trg_dict = Dictionary(target)

        orig_src_str = "Original src vocab size: {0}".format(src_dict.get_size())
        orig_trg_str = "Original trg vocab size: {0}".format(trg_dict.get_size())
        orig_stats_str = "{}\n{}\n\n".format(orig_src_str, orig_trg_str)

        print(orig_stats_str, end="")

        with pd.option_context('display.max_rows', None, 'display.max_columns', None,
                               'display.expand_frame_repr', False):
            print(df)

        print("\nSaving vocab stats to '{0}'.".format(stats_path))
        data.to_txt(stats_path, orig_stats_str + df.to_string())

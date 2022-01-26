from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import numpy as np
import data
from preproc import batchify, Batch, BatchHandler, Dictionary, SimpleAlignment


def _construct_layer(predecessor, layer_cfg, training=None):
    """
    Dies ist eine Hilfsfunktion von ff_neural_network. Sie ist dafuer da um layer-Subgraphen anhand der uebergebenen
    layer Config-Info zu erstellen.

    :param predecessor: tf.Tensor welcher die Ausgabe des vorherigen layers enthalten soll.
    :param layer_cfg: Python-dict, welches die layer Config-Info enthaelt.
    :param training: tf.Tensor, welcher fuer BN-Layer angibt, ob sie sich im Traing-modus befinden oder nicht.
    :return: tf.Tensor welcher die Ausgabe des Layers enthalten soll.
    """
    type = layer_cfg['type']

    if type == "dense":
        activation = tf.layers.dense(predecessor,
                                     units=layer_cfg['units'],
                                     name=layer_cfg['name'],
                                     kernel_initializer=eval(layer_cfg['kernel_init']),
                                     activation=eval('tf.nn.{}'.format(layer_cfg['activation'])))
    elif type == "batch_normalization":
        activation = tf.layers.batch_normalization(predecessor,
                                                   training=training,
                                                   name=layer_cfg['name'])
    # with tf.name_scope(layer_cfg['name']):
    #     tf.summary.histogram("activations", activation)
    return activation


def ff_neural_network(src_window, trg_window, training, net_config: object, name=None):
    """
    Diese Funktion erzeugt den TensorFlow Computational Graph eines Feed-Forward Neural Networks der Architektur des
    Uebungsblatts 3. Zur Konstruktion des Graphen greift die Funktion auf uebergebene Config-Parameter zurueck.

    1)  Erschliesse einen neuen variable-scope, sodass das Netzwerk im Tensorboard schoen als ein Knoten angezeigt wird.
    2)  Stelle sicher, dass die uebergebenen Configdaten fuer ein Netzwerk der Architektur des Uebungsblatts 3 bestimmt
        sind. Dann rufe die Conigdaten ab.
    3)  Lege eine embedding-matrix an.
    4)  Die src- und trg-window Vektoren werden embedded, dabei wird jedem Index ein embedding-Vektor zugeordnet, d.h.
        der urspruengliche src- bzw. trg-window Vektor wird um eine weitere Dimension erweitert.
    5)  Da fuer die weitere Prozessierung src- und trg-window wieder eindimensional sein sollen, werden die embedding-
        Vektoren von src- bzw- trg-window konkateniert. Hier wird dies erreicht, in dem die 2D-Tensoren in 1D-Tensore
        umgeformt werden (*).
    6)  Konstruiere eine Reihe an fully-connected source layern, so wie es die Config-Info vorgibt.
    7)  Konstruiere eine Reihe an fully-connected target layern, so wie es die Config-Info vorgibt.
    8)  Konkateniere die Ausgaben des letzten fc src layers mit der des letzten fc trg layers.
    9)  Konstruiere eine Reihe an fully-connected layern, so wie es die Config-Info vorgibt.
    10) Uebergebe die Ausgabe des letzten fc layers dem projection layer und softmax-normalisiere dessen Ausgabe
    11) Nun uebergebe zum einen den tf.Tensor, welcher die Ausgabe des projection layers beinhalten wird, als auch
        den tf.Tensor welcher die softmax-normalisierte Ausgabe enthalten wird. Dies hat folgenden Grund: Fuer das
        Training des Netzwerks verwenden wir das cross-entropy loss. TensorFlow stellt davon eine Variante bereit,
        welche selbst eine softmax-Normalisierung vornimmt (dies hat Effizienzgruende). D.h. zu Trainingszeiten wollen
        wir von dem Netzwerk nur die Ausgabe des projection layers haben, wir direct dem softmax_cross_entropy loss
        uebergeben. Kommt es zum Einsatz des Netzwerks, dann wollen wir aber die softmax-normalisierte Ausgabe ablesen.
        Daher gibt diese Funktion zwei tf.Tensoren zurueck. Wir koennen diese Wechsel-Taktik nur nutzen, da softmax nur
        eine fixe Funktion ohne trainierbare Parameter ist.


    (*) Beachte: Spaeter werden dem Netzwerk ganze Batches auf einen Schlag uebergeben. Matrixarithmetik macht's moeglich:
        In unserem Fall hat ein Batch die Form einer Matrix (2D-Tensor). Jede Zeile der Matrix beinhaltet einen Feature-
        Vektor (in unserem Fall sind dass die src- bzw. trg-window Index-Vektoren). D.h. auf Batch-Ebene betrachtet
        wird die Matrix durch das embedding-lookup zu einem 3D-Tensor, welcher darauf wieder in einen 2D-Tensor
        umgewandelt wird. Um die Erklaerungen einfach zu halten, sind diese immer aus der Sicht geschrieben, als wuerden
        dem Netzwetzwerk immer nur ein Feature-Vektor aufeinmal uebergeben werden.

    :param src_window: tf.Tensor, welcher einen int-Vektor mit den Indizes der Token des source-windows enthalten soll.
    :param trg_window: tf.Tensor, welcher einen int-Vektor mit den Indizes der Token des target-windows enthalten soll.
    :param training: tf.Tensor, welcher einen boolean enthalten soll, welcher den Modus der BN-Layer angibt.
    :param net_config: Python-dict, welches die Netzwerk-Config Infos enthaelt.
    :param name: Optionaler Name des Netzwerks
    :return: tuple: (tf.Tensor mit Ausgabe des Projection-Layers, tf.Tensor mit softmax-normalisierter Ausgabe)
    """
    # 1)
    with tf.variable_scope(name, default_name="ffnn") as scope:
        # 2)
        assert net_config['architecture'] == "exercise-3", \
            "Die uebergebene Netzwerkkonfiguration ist nicht fuer Netzwerke" \
            "der Architektur aus Uebungsblatt 3 gedacht."

        window_size = net_config['window_size']
        embeddings_config = net_config['embeddings']
        src_vocab_size = net_config['src_vocab_size']
        trg_vocab_size = net_config['trg_vocab_size']
        fc_layer_src_cfg = net_config['fully_connected_src_layer']
        fc_layer_trg_cfg = net_config['fully_connected_trg_layer']
        fc_layer_cfg = net_config['fully_connected_layer']
        projection_layer_cfg = net_config['projection_layer']

        # 3)
        embedding_src_matrix = tf.get_variable(name="embedding_src_matrix",
                                               shape=[src_vocab_size, embeddings_config['embedding_size']],
                                               initializer=eval(embeddings_config['src_matrix_init']))
        embedding_trg_matrix = tf.get_variable(name="embedding_trg_matrix",
                                               shape=[trg_vocab_size, embeddings_config['embedding_size']],
                                               initializer=eval(embeddings_config['trg_matrix_init']))
        # 4)
        embedded_src_vectors = tf.nn.embedding_lookup(embedding_src_matrix, src_window, name="em_src_vectors")
        embedded_trg_vectors = tf.nn.embedding_lookup(embedding_trg_matrix, trg_window, name="em_trg_vectors")
        # 5)
        embedded_src_window = tf.reshape(embedded_src_vectors,
                                         [-1, embeddings_config['embedding_size'] * (2 * window_size + 1)],
                                         name="em_src_window")
        embedded_trg_window = tf.reshape(embedded_trg_vectors, [-1, embeddings_config['embedding_size'] * window_size],
                                         name="em_trg_window")

        # 6)
        fc_src_layer = []
        predecessor = embedded_src_window
        for layer_cfg in fc_layer_src_cfg:
            layer = _construct_layer(predecessor, layer_cfg)
            fc_src_layer.append(layer)
            predecessor = layer

        # 7)
        fc_trg_layer = []
        predecessor = embedded_trg_window
        for layer_cfg in fc_layer_trg_cfg:
            layer = _construct_layer(predecessor, layer_cfg)
            fc_trg_layer.append(layer)
            predecessor = layer

        # 8)
        concat_layer = tf.concat([fc_src_layer[-1], fc_trg_layer[-1]], 1, name="concat_layer")

        # 9)
        fc_layer = []
        predecessor = concat_layer
        for layer_cfg in fc_layer_cfg:
            layer = _construct_layer(predecessor, layer_cfg, training)
            fc_layer.append(layer)
            predecessor = layer

        # 10)
        projection_layer = tf.layers.dense(fc_layer[-1],
                                           units=trg_vocab_size,
                                           name="projection_layer",
                                           # activation=eval('tf.nn.{}'.format(projection_layer_cfg['activation'])),
                                           kernel_initializer=eval(projection_layer_cfg['kernel_init']))

        softmax_layer = tf.nn.softmax(projection_layer, dim=-1, name="softmax_layer")

        # with tf.name_scope("embedded_src_window"):
        #     tf.summary.histogram("activations", embedded_src_window)
        # with tf.name_scope("embedded_trg_window"):
        #     tf.summary.histogram("activations", embedded_trg_window)
        # with tf.name_scope("concat_layer"):
        #     tf.summary.histogram("activations", concat_layer)
        # with tf.name_scope("projection_layer"):
        #     tf.summary.histogram("activations", projection_layer)
        # with tf.name_scope("softmax_layer"):
        #    tf.summary.histogram("activations", softmax_layer)

        # 11)
        return projection_layer, softmax_layer


def load_dataset(batch_data_path: str):
    batch_data = data.load_obj(batch_data_path)

    src_windows = np.array(batch_data[0])
    trg_windows = np.array(batch_data[1])
    trg_labels = np.array(batch_data[2])

    return tf.data.Dataset.from_tensor_slices((src_windows, trg_windows, trg_labels))


if __name__ == '__main__':
    """
    Dieser Skript-Einstiegspunkt dient lediglich zum Testen des Modells.
    
    1)  Erstelle ein Dataset welches die Batchdaten (im (S,T,L)-Format) beinhaltet, welche unter dem angegebenen Pfad
        gespeichert sind. Zuerst beihaltet das Dataset pro Element ein Source- und Target-Window sowie ein Target-Label,
        wir sagen ein Element beinhaltet eine Batch-Zeile. Dann werden die Elemente durch shuffle() gemischt. Durch
        batch(N) werden die Elemente zu Batches zusammengefuehrt, ein Element beinhaltet nun N Batch-Zeilen. Durch 
        repeat() wird das Dataset unendlich wiederholt.        
    2)  Erstelle einen initializable Iterator ueber dem Dataset und lege fuer jedes Element einen next-Tensor an, in dem
        dann die naechsten Source-, Target-Windows, bzw. Target-Labels uebertragen werden.
    3)  Wir brauchen eine Variable die angibt ob sich das Netzwerk im Trainings- oder Anwendungsmodus befindet.
    4)  Lade die Netzwerkkonfiguration und erstelle den Netzwerkgraphen.
    5)  Lege eine Operation an, die alle Variablen initialisiert.
    6)  Speichere den erstellten Graphen ab.
    7)  Beginne eine Session und lasse den Iterator endlos ueber dem Dataset laufen.
    
    """

    batch_data_path = "./out/batch/multi30k.5000ops.joint-bpe.w1.separate-vocab.batch.dat"
    net_config_path = "./net-config.json"
    net_name = "proto-net-m"

    # 1)
    dataset = load_dataset(batch_data_path)
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.batch(64)
    dataset = dataset.repeat()

    # 2)
    iterator = dataset.make_initializable_iterator()
    next_src_window, next_trg_window, next_trg_label = iterator.get_next()

    # 3)
    training = tf.constant(False, dtype=tf.bool)

    # 4)
    net_config = data.read_json(net_config_path)[net_name]
    ffnn = ff_neural_network(next_src_window, next_trg_window, training, net_config, net_name)[1]

    # 5)
    init_vars = tf.global_variables_initializer()

    # 6)
    writer = tf.summary.FileWriter('./graphs/{}'.format(net_name))
    writer.add_graph(tf.get_default_graph())
    writer.flush()

    # 7)
    with tf.Session() as sess:
        sess.run(init_vars)
        sess.run(iterator.initializer)
        while True:
            out = sess.run(ffnn)
            print(out)

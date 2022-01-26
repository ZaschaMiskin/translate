import datetime
import re
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from ffnn import ff_neural_network
import data
import os

# Fehlermeldungen entfernen
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)


class Trainer:

    def __init__(self,
                 network_function, net_config: dict, net_name: str,
                 train_data: list, batch_size: int,
                 valid_data: list, valid_steps: int,
                 learning_rate: float, learning_rate_stagnation: float,
                 learning_rate_reduce_steps: int,
                 log_summary_path: str, log_summary_steps: int,
                 save_checkpoint_path: str, save_checkpoint_steps: int):
        """
        1)  Ueberfuehre die Paramatere in Attribute.
        2)  Falls eine Parameter bezueglich der Reduzierung der Learning-Rate bei Stagnation der Modell-Performance
            angegeben sind, dann uebernehme diese als Attribute und setze eine Flag, dass der Trainer sich um die
            Reduzierung der Lernrate keummert.
        3)  Initialisiere den Computational Graph.
        4)  Gib die Trainierbaren Parameter des Modells aus.
        5)  Initialisiere Summary-Writer und einen Saver der sich um das Speicher und Laden des Modells kuemmert.
        6)  Initialisiere weitere Variablen.

        :param network_function: Netzwerkerzeugende Funktion.
        :param net_config: Python-dict, welches die Netzwerk-Config Infos enthaelt.
        :param net_name: Name des Netzwerks
        :param train_data: Trainingsdaten (Ausgabe von batch.format_batch())
        :param batch_size: Batch-Size
        :param valid_data: Validierungsdaten (Ausgabe von batch.format_batch())
        :param valid_steps: Anzahl der Trainingsschritte, nach denen Validierung erfolgen soll
        :param learning_rate: Lernrate
        :param learning_rate_reduce_steps: Epochen ueber die Accuracy-Steigung m der Valdierung ermittelt wird
        :param learning_rate_stagnation: Steigungswert, bei dem die Lernrate reduziert wird, falls m diesen unterschreitet.
        :param log_summary_path: Pfad zum log-Verzeichnis
        :param log_summary_steps: Anzahl an Batches, nach denen Metriken ausgegeben werden sollen
        :param save_checkpoint_path: Pfad zum Checkpoint-Verzeichnis
        :param save_checkpoint_steps: Anzahl der Batches, nach denen Modell gespeichert werden soll
        """
        # 1)
        self.log_summary_path = log_summary_path
        self.log_summary_steps = log_summary_steps
        self.valid_data = valid_data
        self.valid_steps = valid_steps
        self.save_checkpoint_path = save_checkpoint_path
        self.save_checkpoint_steps = save_checkpoint_steps
        self.learning_rate = learning_rate

        # 2)
        if learning_rate_stagnation and learning_rate_reduce_steps:
            self.learning_rate_stagnation = learning_rate_stagnation
            self.learning_rate_reduce_steps = learning_rate_reduce_steps
            self.learning_rate_dynamic = True
        else:
            self.learning_rate_dynamic = False

        # 3)
        self._init_graph(network_function, net_config, net_name, train_data, valid_data, batch_size)

        # 4)
        trainable_variables = tf.trainable_variables()
        _print_trainable_variables(trainable_variables)

        # 5)
        self.train_writer = tf.summary.FileWriter(f"{self.log_summary_path}/train")
        self.valid_writer = tf.summary.FileWriter(f"{self.log_summary_path}/valid")
        self.saver = tf.train.Saver()

        # 6)
        self._init_storage()
        self.global_step = 0
        self._var_inited = False

    def _init_graph(self, network_function, net_config: dict, net_name: str,
                    train_data: list, valid_data: list, batch_size: int):
        """
        Diese Methode initialisiert den TensorFlow Computational Graph der fuer die Trainingsroutine verwendet wird.

        1)  Erzeuge tf.Dataset fuer die Trainingsdaten, einen Iterator fuer das Datenset und eine Initialisierungsoperation.
        2)  Sofer vorhanden, erzeuge tf.Dataset fuer die Validierungsdaten, einen Iterator fuer das Datenset und eine
            Initialisierungsoperation.
        3)  Erzeuge einen feedable Iterator und Operationen um das naechste Source- und Tragetwindow sowie Targetlabel
            zu erhalten. Dem feedable Iterator wird dann spaeter, je nachdem ob sich die Routine in Trainings- oder
            Validierungsphase befindet, ein Handle fuer den Trainings- oder Validierungsiterator uebergeben.
        4)  Erzeuge einen Platzhalter der angeben soll ob das Netzwerk trainiert oder getestet/validiert wird
            (fuer die Batchnorm. Layer)
        5)  Erzeuge den Computational Graph des Netzwerks und eine Operation, die dessen Vorhersage ermittelt.
        6)  Erzeuge eine Operation die den Cross-Entropy Loss des Netzwerks ermittelt.
        7)  Erzeuge Operationen und Platzhalter die fuer Trainingsoperationen ueber dem Netzwerk eingesetzt werden.
        8)  Erzeuge Operationen, die die Trainingsmetriken ermitteln.
        9)  Erzeuge Operationen, die die Metriken in Summaries ueberfuehren.
        10) Erzeuge eine Operation die die Variablen initialisiert.

        :param network_function: Netzwerkerzeugende Funktion
        :param net_config: config-dict welches die Netzwerkkonfiguration vorschreibt
        :param net_name: Name des Netzwerks
        :param train_data: Trainingsdaten im (S,T,L)-Batchformat
        :param valid_data: Validierungsdaten im (S,T,L)-Batchformat
        :param batch_size: Batch-Size
        """

        # 1)
        train_dataset = self._init_dataset(train_data, batch_size)
        self.training_iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                                 train_dataset.output_shapes)
        self.train_init_op = self.training_iterator.make_initializer(train_dataset, name='train_init_op')

        # 2)
        if valid_data:
            valid_dataset = self._init_dataset(valid_data, batch_size, drop_remainder=True)
            self.validation_iterator = tf.data.Iterator.from_structure(valid_dataset.output_types,
                                                                       valid_dataset.output_shapes)
            self.valid_init_op = self.validation_iterator.make_initializer(valid_dataset, name="valid_init_op")

        # 3)
        self.iterator_handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(self.iterator_handle, train_dataset.output_types,
                                                       train_dataset.output_shapes)
        self.next_src_window, self.next_trg_window, self.next_trg_label = iterator.get_next()
        self.training_handle = None
        self.validation_handle = None

        # 4)
        self.train_mode = tf.placeholder(dtype=bool, name='train_mode')

        # 5)
        projection, softmax = network_function(self.next_src_window, self.next_trg_window, self.train_mode, net_config,
                                               net_name)
        prediction = tf.argmax(projection, axis=1, name='prediction')

        # 6)
        self.loss_op = tf.losses.sparse_softmax_cross_entropy(labels=self.next_trg_label, logits=projection,
                                                              scope="loss")
        # 7)
        with tf.name_scope('train'):
            self.lr = tf.placeholder(dtype=tf.float32, name='learn_rate')
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='adam_optimizer')
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.train_op = optimizer.minimize(self.loss_op)
            self.train_op = tf.group([self.train_op, update_ops])

        # 8)
        with tf.name_scope('metrics'):
            correct_pred = tf.equal(tf.cast(prediction, dtype=tf.int32), self.next_trg_label, name='correct_prediction')
            self.accuracy_op = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32), name='accuracy')
            self.perplexity_op = tf.exp(self.loss_op, name='perplexity')

        # 9)
        tf.summary.scalar('loss/batch', self.loss_op)
        tf.summary.scalar('accuracy/batch', self.accuracy_op)
        tf.summary.scalar('perplexity/batch', self.perplexity_op)
        self.summary_op = tf.summary.merge_all(name="summary_op")

        # 10)
        self.init_op = tf.global_variables_initializer()

    def _init_dataset(self, data, batch_size, drop_remainder=False):
        """
        Diese Methode dient der Initialisierung der tf.Datasets von Trainings- und Validierungsdaten.

        :param data: Daten im (S,T,L)-Batchformat
        :param batch_size: Groesse der Batches
        :param drop_remainder: Gibt an, ob Ueberbleibsel nach der Aufteilung der Daten in Batches fester Groesse gedroppt werden sollen
        :return: tf.Dataset
        """
        src_windows = np.array(data[0])
        trg_windows = np.array(data[1])
        trg_labels = np.array(data[2])

        dataset = tf.data.Dataset.from_tensor_slices((src_windows, trg_windows, trg_labels))
        dataset = dataset.shuffle(buffer_size=1024)
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        # TODO: WIP
        # train_dataset = train_dataset.take(2)

        return dataset

    def _init_storage(self):
        """
        Diese Methode initialisiert Listen, die fuer die Zwischenspeicherung von Metriken benutzt werden.
        """
        self.accuracies_valid_store = []
        self.accuracies_epoch = []
        self.losses_epoch = []
        self.perplexities_epoch = []

    def initialize(self, sess):
        """
        Diese Methode initialisert die Variablen des Modells innerhalb einer Session. Alternativ kann ein bestehendes
        Modell in seiner aktuellen Variablenkonfiguration durch restore_checkpoint() geladen werden.

        :param sess: Session fuer die die Variablen initialisiert werden.
        """
        sess.run(self.init_op)
        self._var_inited = True

    def restore_checkpoint(self, sess, checkpoint_path):
        """
        Diese Methode laedt den Checkpoint eines Modells vom angegebenen Checkpointpfad.

        :param sess: Session in der das Modell geladen wird.
        :param checkpoint_path: Pfad zum Checkpoint
        """
        self.saver.restore(sess, checkpoint_path)
        print("\n[Checkpoint] Successfully restored model from path {0}.".format(checkpoint_path))
        self.global_step = int(checkpoint_path.split('-')[-1])
        self._var_inited = True

    def train(self, sess, epochs: int):
        """
        Diese Methode Trainiert das Modell des Trainers fuer 'epochs' Epochen und gibt das Ergebnis der anschliessenden
        Validierung zurueck.

        1)  Fetche die string-handle von training- und validation-iterator um diese nachher dem feedable iterator zu
            uebergeben.
        2)  Fuer jede Epoche:
          2.1)  (Re-)Initialisiere den Trainings-Iterator und setze den epoch_step counter zurueck.
          2.2)  Fuer jeden Batch des Trainingsdatensatzes:
            2.2.1)  Fuehre einen Trainingsschritt aus und fetche die daraus resultierenden Metriken.
            2.2.2)  Logge die Metriken.
            2.2.3)  Speichere das Modell in einem Checkpoint nach einer angegebenen Anzahl an Trainingsschritten.
            2.2.4)  Validiere das Modell nach nach einer angegebenen Anzahl an Trainingsschritten.
            2.2.5)  Erhoehe den global_step und epoch_step counter.
          2.3)  Nach jeder Epoche soll eine Zusammenfassung der Epoche geloggt werden.
        3)  Am Ende des Trainings wird das Modell nochmal validiert und anschliessend in einem Checkpoint gespeichert.
            Schliesslich werden die Ergebnisse der letzten Validierung zurueckgegeben.

        :param sess: Die Session unter der das Modell trainiert werden soll.
        :param epochs: Die Anzahl der Epochen, die das Modell trainiert werden soll.
        :return: accuracy, perplexity, loss der letzten Validierung.
        """
        assert self._var_inited, "Das Modell muss vor dem Training initialisiert werden. Rufe initialize() oder restore_checkpoint() auf!"

        # 1)
        self.training_handle = sess.run(self.training_iterator.string_handle())
        self.validation_handle = sess.run(self.validation_iterator.string_handle())

        # 2)
        print('\n[Training] Start:')
        for epoch in range(epochs):
            # 2.1)
            sess.run(self.train_init_op)
            epoch_step = 0

            # 2.2)
            print(f"\n[Training] Epoch {epoch + 1}/{epochs}:")
            while True:
                try:
                    # 2.2.1)
                    accuracy_batch, perplexity_batch, loss_batch, summary_batch, tr = sess.run(
                        fetches=(
                            self.accuracy_op, self.perplexity_op, self.loss_op, self.summary_op, self.train_op
                        ),
                        feed_dict={self.train_mode: True, self.lr: self.learning_rate,
                                   self.iterator_handle: self.training_handle}
                    )
                    # 2.2.2)
                    self._log_train_batch(accuracy_batch, perplexity_batch, loss_batch, epoch_step, summary_batch)
                    # 2.2.3)
                    self._save_if_checkpoint_step(sess)

                    # 2.2.4)
                    if self.valid_data and self.global_step % self.valid_steps == 0:
                        accuracy_valid, perplexity_valid, loss_valid = self.validate(sess)
                        self._augment_lr_if_performance_low(accuracy_valid, perplexity_valid, loss_valid)

                    # 2.2.5)
                    epoch_step += 1
                    self.global_step += 1

                except tf.errors.OutOfRangeError:
                    break

            # 2.3)
            self._log_train_epoch(epoch)

        # 3)
        print("\n[Training] Done! Performing final evaluation:")
        final_accuracy, final_perplexity, final_loss = self.validate(sess)
        self.save_checkpoint(sess)

        return final_accuracy, final_perplexity, final_loss

    def validate(self, sess, log=True):
        """
        Die Methode validiert das Modell des Trainers ueber den angebenen Validierungsdaten und gibt die Ergebnisse
        zurueck.

        1)  (Re-)Initialisiere den validation-iterator und fetche dessen string-handle sofern dieser noch nicht erzeugt
            wurde.
        2)  Deklariere Listen in denen die Metriken der Validierung zwischengespeichert werden sollen.
        3)  Fuer jeden Batch des Validierungsdatensatzes:
          3.1)  Uebergebe den Batch an das Netzwerk und fetche die dabei entstandenen Metriken.
          3.2)  Speichere die Metriken in den angelegten Listen.
        4)  Ermittle die Mittelwerte der erzeugten Metriken und gebe diese als Ergebnisse der Validierung zurueck.
        :param sess: Die Session unter der das Modell validiert werden soll.
        :param log: gibt an, ob die Vorgaenge und Ergebnisse der Validierung geloggt werden sollen.
        :return:
        """
        assert self._var_inited, "Das Modell muss vor der Validierung initialisiert werden. Rufe initialize() oder restore_checkpoint() auf!"
        assert self.valid_data, "Dem Trainer wurden keine Validierungsdaten uebergeben, daher kann er das Modell nicht validieren."

        if log:
            print('[Validation] Validating...', end='')

        # 1)
        sess.run(self.valid_init_op)

        if self.validation_handle is None:
            self.validation_handle = sess.run(self.validation_iterator.string_handle())

        # 2)
        accuracies = []
        losses = []
        perplexities = []

        # 3)
        while True:
            try:
                # 3.1)
                accuracy_batch, perplexity_batch, loss_batch = sess.run(
                    fetches=(self.accuracy_op, self.perplexity_op, self.loss_op),
                    feed_dict={self.train_mode: False, self.iterator_handle: self.validation_handle}
                )

                # 3.2)
                accuracies.append(accuracy_batch)
                losses.append(loss_batch)
                perplexities.append(perplexity_batch)

            except tf.errors.OutOfRangeError:
                break

        # 4)
        accuracy = np.mean(accuracies)
        loss = np.mean(losses)
        perplexity = np.mean(perplexities)

        if log:
            self._log_valid(accuracy, perplexity, loss)

        return accuracy, perplexity, loss

    def save_checkpoint(self, sess):
        """
        Diese Methode speichert das Modell in einem Checkpoint unter einem bei der Initialisierung des Trainers angegebenen
        Pfad.

        :param sess: Session deren Modellparameter gespeichert werden sollen.
        """
        print('[Checkpoint] Saving model...', end='')
        checkpoint_path = self.saver.save(sess, f"{self.save_checkpoint_path}",
                                          global_step=self.global_step)
        print("\r[Checkpoint] Saved model in path: '{0}'.".format(checkpoint_path))

    def _save_if_checkpoint_step(self, sess):
        """
        Waehrend des Trainings soll das Modell alle 'self.save_checkpoint_steps' in einem Checkpoint gespeichert werden.
        Diese Methode ueberprueft, ob der aktuelle global_step eine Speicherung vorsieht und fuehrt diese ggf. aus.

        :param sess: Session deren Modellparameter gespeichert werden sollen.
        """
        if (self.save_checkpoint_steps is not None) \
                and self.global_step != 0 and self.global_step % self.save_checkpoint_steps == 0:
            self.save_checkpoint(sess)

    def _augment_lr_if_performance_low(self, accuracy, perplexity, loss):
        """
        Diese Methode zeichnet die Performance des Modells ueber mehrere Validierungen hinweg auf und ueberprueft diese
        auf Stagnation. Falls die Performance des Modells stagniert, dann wird die learning_rate halbiert.

        1)  Fuege die Accuracy der aktuellen Validierung der self.accuracies_valid_store-list hinzu.
        2)  Falls die Anzahl der Aufzeichnungen in der list der Anzahl der learning_rate_reduce_steps entspricht:
          2.1)  Erstelle Koordinatenpunkte der Accuracies ueber die letzten learning_rate_reduce_steps Validierungen.
          2.2)  Ermittle die Steigung  m (y=mx+b) der Regressionsgeraden ueber die Koordinatenpunkte.
          2.3)  Falls m die Stagnationsgrenze unterschreitet, dann halbiere die Lernrate.
          2.4)  Loesche die Aufzeichnung, damit die Steigung erst in learning_rate_reduce_steps Validierungen erneut
                ermittelt wird.

        :param accuracy: Accuracy der aktuellen Validierung
        :param perplexity: Perplexity der aktuellen Validierung TODO: unbenutzt, evtl. zur spaeteren Anpassung des Kriteriums
        :param loss: Loss der aktuellen Validierung TODO: unbenutzt, evtl. zur spaeteren Anpassung des Kriteriums
        """
        if self.learning_rate_dynamic:
            # 1)
            self.accuracies_valid_store.append(accuracy)

            # 2)
            if len(self.accuracies_valid_store) == self.learning_rate_reduce_steps:
                # 2.1)
                x = range(self.learning_rate_reduce_steps)
                y = self.accuracies_valid_store

                # 2.2)
                m, _ = np.polyfit(x, y, deg=1)

                # 2.3)
                if m <= self.learning_rate_stagnation:
                    new_learning_rate = self.learning_rate / 2
                    print(
                        f"[Learning Rate] Reduziere die Lernrate von {self.learning_rate} auf {new_learning_rate}, "
                        f"aufgrund von Stagnation. Kuerzliche Steigung ueber {self.learning_rate_reduce_steps} "
                        f"Validierungen: m={m}, Stagnationsgrenze: m'={self.learning_rate_stagnation}.")
                    self.learning_rate = new_learning_rate

                # 2.4)
                self.accuracies_valid_store = []

    def _log_train_batch(self, accuracy, perplexity, loss, epoch_step, batch_summary):
        """
        Diese Methode zeichnet die Metriken eines Trainingsschritts auf und sorgt dafuer, dass diese sowohl als
        tf.Summary als auch in der Konsole geloggt werden.

        1)  Schreibe eine Summary fuer den letzten Trainigsschritt (-> Tensorboard)
        2)  Aktualisiere die Metriken- und Loss-Buffer.
        3)  Alle 'log_summary_steps' Epochenschritte: Mittle die neusten 'log_summary_steps' Metriken in
            den Buffern und erstelle dazu eine Summary. Gib diese zudem in der Konsole aus.

        :param accuracy: Accuracy des aktuellen Trainingsschritts
        :param perplexity: Perplexity des aktuellen Trainingsschritts
        :param loss: Loss des aktuellen Trainingsschritts
        :param epoch_step: Aktueller Trainingsschritt innerhalb der Epoche
        :param batch_summary: tf.Summary des aktuellen Trainingsschritts
        """
        # 1)
        self.train_writer.add_summary(batch_summary, self.global_step)

        # 2)
        self.accuracies_epoch.append(accuracy)
        self.losses_epoch.append(loss)
        self.perplexities_epoch.append(perplexity)

        # 3)
        if epoch_step % self.log_summary_steps == 0:
            accuracy_buffered = np.mean(self.accuracies_epoch[-self.log_summary_steps:])
            loss_buffered = np.mean(self.losses_epoch[-self.log_summary_steps:])
            perplexity_buffered = np.mean(self.perplexities_epoch[-self.log_summary_steps:])

            self._summarize_train_buffered(accuracy_buffered, perplexity_buffered, loss_buffered)

            print('[Buffer Summary] Global Step: {}, Accuracy: {:.5f}, '
                  'Perplexity: {:.5f}, Loss: {:.5f}'.format(
                str(self.global_step).zfill(4), accuracy_buffered, perplexity_buffered, loss_buffered))

    def _log_train_epoch(self, epoch):
        """
        Diese Methode mittelt die Metriken, die ueber die letzte Epoche aufgezeichnet wurden und sorgt dafür, dass diese
        sowohl in der Konsole als auch als tf.Summary ausgegeben werden. Zudem löscht sie den Metriken-Buffer nach jeder
        Epoche.

        :param epoch: aktuelle Epoche
        """
        accuracy_epoch = np.mean(self.accuracies_epoch)
        loss_epoch = np.mean(self.losses_epoch)
        perplexity_epoch = np.mean(self.perplexities_epoch)

        self._summarize_train_epoch(accuracy_epoch, perplexity_epoch, loss_epoch)

        print('[Epoch Summary] Epoch: {}, Accuracy: {:.5f}, '
              'Perplexity: {:.5f}, Loss: {:.5f}'.format(
            str(epoch + 1).zfill(4), accuracy_epoch, perplexity_epoch, loss_epoch))

        self.accuracies_epoch = []
        self.perplexities_epoch = []
        self.losses_epoch = []

    def _log_valid(self, accuracy, perplexity, loss):
        """
        Diese Methode sorgt dafür, dass die Metriken einer Validierung sowohl in der Konsole, als auch als tf.Summary
        geloggt werden.

        :param accuracy: Accuracy der aktuellen Validierung
        :param perplexity: Perplexity der aktuellen Validierung
        :param loss: Loss der aktuellen Validierung
        """
        self._summarize_validation(accuracy, perplexity, loss)

        print('\r[Validation Summary] Step: {}, Accuracy: {:.5f}, '
              'Perplexity: {:.5f}, Loss: {:.5f}'.format(
            str(self.global_step).zfill(4), accuracy, perplexity, loss))

    def _summarize_train_buffered(self, accuracy, perplexity, loss):
        buffered_summary = tf.Summary()
        buffered_summary.value.add(tag="accuracy/buffered", simple_value=accuracy)
        buffered_summary.value.add(tag="loss/buffered", simple_value=loss)
        buffered_summary.value.add(tag="perplexity/buffered", simple_value=perplexity)

        self.train_writer.add_summary(buffered_summary, self.global_step)

    def _summarize_train_epoch(self, accuracy, perplexity, loss):
        epoch_summary = tf.Summary()
        epoch_summary.value.add(tag="accuracy/epoch", simple_value=accuracy)
        epoch_summary.value.add(tag="perplexity/epoch", simple_value=perplexity)
        epoch_summary.value.add(tag="loss/epoch", simple_value=loss)

        self.train_writer.add_summary(epoch_summary, self.global_step)

    def _summarize_validation(self, accuracy, perplexity, loss):
        valid_summary = tf.Summary()
        valid_summary.value.add(tag="accuracy/epoch", simple_value=accuracy)
        valid_summary.value.add(tag="perplexity/epoch", simple_value=perplexity)
        valid_summary.value.add(tag="loss/epoch", simple_value=loss)

        self.valid_writer.add_summary(valid_summary, self.global_step)


def _print_trainable_variables(trainable_variables):
    """
    Diese Funktion gibt Infos zu den 'trainable_variables' tabellarisch in der Konsole aus.

    :param trainable_variables: list von trainierbaren Variablen (tf.Tensor)
    """
    var_name = []
    var_shape = []
    var_total = []

    for var in trainable_variables:
        var_name.append(var.name)
        var_shape.append(var.shape)
        var_total.append(np.prod(var.shape))

    var_name.append('')
    var_shape.append('')
    var_total.append(np.sum(var_total))

    dataframe = pd.DataFrame(data={'name': var_name, 'shape': var_shape, 'total': var_total})

    print("Trainable Variables:\n" + dataframe.to_string())


def train(network_function, net_config: dict, net_name: str,
          train_data: list, epochs: int, batch_size: int,
          learning_rate: float, learning_rate_reduce_steps: int, learning_rate_stagnation: float,
          valid_data: list, valid_steps: int,
          load_checkpoint_path: str,
          log_summary_path: str, log_summary_steps: int,
          save_checkpoint_path: str, save_checkpoint_steps: str):
    """
    :param network_function: Netzwerkerzeugende Funktion.
    :param net_config: Python-dict, welches die Netzwerk-Config Infos enthaelt.
    :param net_name: Name des Netzwerks
    :param train_data: Trainingsdaten (Ausgabe von batch.format_batch())
    :param epochs: Anzahl der Epochen, die Netzwerk trainiert werden soll.
    :param batch_size: Batch-Size.
    :param learning_rate: Lernrate
    :param learning_rate_reduce_steps: Epochen ueber die Accuracy-Steigung m der Valdierung ermittelt wird
    :param learning_rate_stagnation: Steigungswert, bei dem die Lernrate reduziert wird, falls m diesen unterschreitet.
    :param valid_data: Validierungsdaten (Ausgabe von batch.format_batch())
    :param valid_steps: Anzahl der Trainingsschritte, nach denen jeweils Validierungen erfolgen sollen
    :param load_checkpoint_path: Pfad zu einem Modell-Checkpoint
    :param log_summary_path: Pfad zum log-Verzeichnis
    :param log_summary_steps: Anzahl an Batches, nach denen Metriken ausgegeben werden sollen
    :param save_checkpoint_path: Pfad zum Checkpoint-Verzeichnis
    :param save_checkpoint_steps: Anzahl der Batches, nach denen Modell gespeichert werden soll
    """
    trainer = Trainer(network_function, net_config, net_name,
                      train_data, batch_size,
                      valid_data, valid_steps,
                      learning_rate, learning_rate_stagnation, learning_rate_reduce_steps,
                      log_summary_path, log_summary_steps,
                      save_checkpoint_path, save_checkpoint_steps)

    with tf.Session() as sess:

        if load_checkpoint_path:
            trainer.restore_checkpoint(sess, load_checkpoint_path)
        else:
            trainer.initialize(sess)

        trainer.train(sess, epochs=epochs)


if __name__ == '__main__':
    """
    Beispielargumente:
    
    "./out/batch/multi30k.5000ops.joint-bpe.w1.separate-vocab.batch.dat" 1000 256 
    "./net-config.json" "ref-net-epsilon"
    -lr 0.001 8 0.01
    -v  "./out/batch/multi30k.dev.5000ops.joint-bpe.w1.separate-vocab.batch.dat" 1
    -log "./logs" 100
    -s "./checkpoints" 10000
    """

    parser = argparse.ArgumentParser(prog='training')

    parser.add_argument('train', nargs=3, help='Pfad zu den Trainingsdaten (im Batch-Format), die Anzahl der '
                                               'Epochen, die das Modell trainiert werden soll und die Batch-Size')

    parser.add_argument('net_config', nargs=2, help='Pfad zur Konfigurationsdatei des Netzwerks gefolgt vom Namen des '
                                                    'Netzwerks.')

    parser.add_argument('-lr', '--learning_rate', nargs="+", default=0.001,
                        help='Lernrate des Optimizers. Zuzueglich optional, die Anzahl der stagnierenden Validierungen,'
                             ' bevor die Lernrate halbiert wird und die Steigungsgrenze welche nicht unterschritten'
                             ' werden soll.')

    parser.add_argument('-v', '--valid', nargs=2, default=None, help='Pfad zu Validierungsdaten (im Batch-Format) und '
                                                                     'die Anzahl der Epochen zwischen '
                                                                     'aufeinanderfolgenden Validierungen.')

    parser.add_argument('-log', '--log', nargs=2, default=None, help='Pfad zum uebergeordneten Log-Verzeichnis, in '
                                                                     'dem die tf.Summarys gespeichert werden. Der '
                                                                     'genaue Log-Pfad wird automatisch generiert.')

    parser.add_argument('-l', '--load', nargs=1, default=None, help='Pfad zum Checkpoint-file')

    parser.add_argument('-s', '--save', nargs=2, default=None, help='Pfad zum uebergeordneten Checkpoint-'
                                                                    'Verzeichnis mit Angabe nach wievielen Batches '
                                                                    'gespeichert werden soll. Der genaue Pfad wird '
                                                                    'automatisch generiert.')

    args = parser.parse_args()

    train_data = data.load_obj(args.train[0])
    epochs = int(args.train[1])
    batch_size = int(args.train[2])

    net_config = data.read_json(args.net_config[0])[args.net_config[1]]
    net_name = args.net_config[1]

    if net_config['architecture'] == 'exercise-3':
        network_function = ff_neural_network

    learning_rate = float(args.learning_rate[0])

    learning_rate_reduce_steps = None
    learning_rate_stagnation = None
    valid_data = None
    valid_steps = None
    load_checkpoint_path = None
    log_summary_path = None
    log_summary_steps = None
    save_checkpoint_path = None
    save_checkpoint_steps = None

    if len(args.learning_rate) == 3:
        learning_rate_reduce_steps = int(args.learning_rate[1])
        learning_rate_stagnation = float(args.learning_rate[2])

    if args.valid:
        valid_data = data.load_obj(args.valid[0])
        valid_steps = int(args.valid[1])

    if args.load:
        load_checkpoint_path = args.load[0]
        timestamp = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}', load_checkpoint_path)[0]
        if timestamp is None:
            timestamp = datetime.datetime.now().isoformat().replace(":", "-").split(".")[0]
    else:
        timestamp = datetime.datetime.now().isoformat().replace(":", "-").split(".")[0]

    if args.log:
        log_summary_path = f"{args.log[0]}/{net_name}/{timestamp}"
        log_summary_steps = int(args.log[1])

    if args.save:
        save_checkpoint_path = f"{args.save[0]}/{net_name}/{timestamp}"
        save_checkpoint_steps = int(args.save[1])

    train(network_function, net_config, net_name, train_data, epochs, batch_size, learning_rate,
          learning_rate_reduce_steps, learning_rate_stagnation, valid_data, valid_steps, load_checkpoint_path,
          log_summary_path, log_summary_steps, save_checkpoint_path, save_checkpoint_steps)
